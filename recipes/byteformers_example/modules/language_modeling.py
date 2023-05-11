import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.utilities import grad_norm
from tqdm import tqdm

from samantha.byteformers.benchmarks.utils import flops
from samantha.models import Llama, LlamaConfig
from samantha.utils.checkpoints_utils.convert_llama import convert_meta_llama_weights
from samantha.utils.hdfs_helper import get


def download_and_convert_llama(output_dir: str, llama_dir: Path, model_name: str):
    converted_llama_fp = Path(output_dir, model_name, "llama.pt")
    if not converted_llama_fp.exists():
        llama_model_path = Path(llama_dir, model_name)
        if not llama_model_path.exists():
            llama_model_path.mkdir(parents=True)
            get(
                "hdfs://harunava/home/byte_speech_sv/models/llama/7B/*",
                str(llama_model_path),
            )
        converted_model = convert_meta_llama_weights(
            str(llama_dir), output_dir, model_size=model_name
        )
        torch.save(converted_model, converted_llama_fp)
    return torch.load(Path(output_dir, model_name, "llama.pt"), map_location="cpu")


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sorted, dim=-1)
    mask = probs_sum - probs_sorted > p
    probs_sorted[mask] = 0.0
    probs_sorted.div_(probs_sorted.sum(dim=-1, keepdim=True))
    sampled = torch.multinomial(probs_sorted, num_samples=1)
    return torch.gather(probs_idx, -1, sampled)


class LanguageModelingModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        seq_len: int,
        learning_rate: float,
        warmup_steps: int,
        weight_decay: float,
        betas: Tuple[float],
        attention_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = LlamaConfig.from_name(model_name)
        self.config.vocab_size = vocab_size
        self.config.logit_num = vocab_size
        self.config.attention_kwargs = attention_kwargs
        self._tokens_seen = 0

        if model_name in ["7B", "13B", "30B", "65B"]:
            # LLaMA default is 1e-6
            self.config.rms_norm_epsilon = 1e-6
            self.config.attention_kwargs = {
                "enable_flash": False,
                "enable_mem_efficient": False,
                "max_seq_len": self.hparams.seq_len,
            }
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model = Llama(self.config)
            torch.set_default_tensor_type(torch.FloatTensor)
            print("Downloading pre-trained weights...")
            state_dict = download_and_convert_llama("output", "llama", model_name)
            print("Loading state dict...")
            self.model.load_state_dict(state_dict, strict=False)
            del state_dict
            self.model.eval()
        else:
            self.model = Llama(self.config)
            self.model.apply(self.model._init_weights)

        self.total_flops = flops(
            self.config, self.hparams.vocab_size, self.hparams.seq_len
        )["total"]
        # self.model = torch.compile(self.model)

    def on_before_optimizer_step(self, optimizer):
        # TODO: sync_dist=True will slow this down, but is more precise
        # consider doing this every n steps.
        self.log_dict(grad_norm(self, norm_type=2))

    def on_train_start(self) -> None:
        batch_size = self.trainer.train_dataloader.batch_size
        world_size = self.trainer.world_size
        effective_batch_size_tokens = world_size * batch_size * self.hparams.seq_len

        hparams = vars(self.hparams)
        hparams.update(
            {
                "batch_size": batch_size,
                "effective_batch_size": world_size * batch_size,
                "effective_batch_size_tokens": effective_batch_size_tokens,
            }
        )
        self.logger.log_hyperparams(
            hparams, {"loss/train": torch.inf, "loss/valid": torch.inf}
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model(src)

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], return_loss: bool
    ) -> torch.Tensor:
        src, targets = batch
        logits = self.forward(src)
        if return_loss:
            return self.loss(logits, targets)
        return logits

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        logits = rearrange(logits, "b n c -> b c n")
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return {"nll": loss, "perplexity": loss.exp()}

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch, return_loss=True)
        self._tokens_seen += (batch[0] >= 0).numel()
        self.log("loss/train", loss["nll"], rank_zero_only=True, prog_bar=True)
        self.log("ppl/train", loss["perplexity"], rank_zero_only=True)
        return loss["nll"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch, return_loss=True)
        self.log("loss/valid", loss["nll"], rank_zero_only=True, sync_dist=True)
        self.log("ppl/valid", loss["perplexity"], rank_zero_only=True, sync_dist=True)
        return loss["nll"]

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        return self.step(batch, return_loss=False)

    def configure_optimizers(self) -> Tuple[list]:
        # - Start with a warm up, ramp up then cosine
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            fused=True,
        )

        def update_lr(*_):
            config = self.hparams
            warmup_tokens = config.seq_len * config.warmup_steps
            final_tokens = config.seq_len * self.trainer.max_steps
            if self._tokens_seen < warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - warmup_tokens) / float(
                    max(1, final_tokens - warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=update_lr
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: bool = None,
        top_p: Optional[float] = 0.95,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        curr_inputs = (
            token_ids
            if token_ids.shape[1] <= self.hparams.seq_len
            else token_ids[:, : self.hparams.seq_len]
        )
        sampled_tokens_ids = torch.empty(
            (token_ids.shape[0], 0), device=token_ids.device, dtype=torch.long
        )
        kv_cache = {}
        for _ in tqdm(range(curr_inputs.shape[1], max_new_tokens), desc="Sampling"):
            logits = self.model(curr_inputs, kv_cache=kv_cache, last_logit_only=True)

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = logits.softmax(dim=-1)
            if do_sample:
                if top_p is not None:
                    idx_next = sample_top_p(probs, top_p)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            curr_inputs = idx_next
            sampled_tokens_ids = torch.cat((sampled_tokens_ids, idx_next), dim=1)
            yield curr_inputs
        # return sampled_tokens_ids

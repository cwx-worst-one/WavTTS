import pytorch_lightning as pl
import torch
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from samantha.utils.hparams import DotDict
from ...utils.sample import sample
from recipes.valle.lit_modules.lit_valle_coarse import sequence_mask

class ValleCoarseModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        tokenizer_len=50277,
        n_semantic=8192,
        checkpointing=True,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.tokenizer_len = tokenizer_len
        self.n_semantic = n_semantic
        self.extra_params = DotDict(extra_params)
        self.requires = {}

        torch._C._jit_set_bailout_depth(0)

        # hugging face setting
        if hasattr(self.model, "resize_token_embeddings"):
            print("Resize token embeddings...")
            self.model.resize_token_embeddings(tokenizer_len + n_semantic + 2)
            self.model.config.use_cache = False
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                input_tokens, seq_lens, pos_ids, seq_sen_ids, _, _ = batch
                b, t = input_tokens.shape
                loss_mask = sequence_mask(seq_lens, max_len=t, device="cuda")
                org_len = input_tokens.size(1)

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("VALLE: Coarse Training.")

        x = logits[:, 0:org_len - 1, :]
        targets = input_tokens[:, 1:org_len]
        loss = self.criterion(x.float(), targets, mask=loss_mask[:, 1:])
        accu = ((x.argmax(dim=-1) == targets).float() * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum() * 100
        self.log_dict(
            {
                "tr_loss": loss.item(),
                "accu": accu.item()
            },
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    @torch.no_grad()
    def prepare_feature(self, wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens):
        device = wavs_16k.device

        # pad text to 256
        if texts.size(1) >= 256:
            texts = texts[:, 0:256]
        else:
            texts = torch.nn.functional.pad(texts, (0, 256 - texts.size(1)))

        semantic_codes, semantic_length = self.get_semantic_codes(wavs_16k, wav_16k_lens)
        # add semantic bos token and eos token
        semantic_codes = torch.nn.functional.pad(semantic_codes, (1, 1))
        # bos
        semantic_codes[:, 0] = self.n_semantic
        # eos
        semantic_mask = torch.arange(semantic_codes.size(1)).unsqueeze(0).to(device) < (1 + semantic_length.unsqueeze(1))
        semantic_codes = torch.where(
            semantic_mask,
            semantic_codes,
            torch.zeros_like(semantic_codes) + self.n_semantic + 1,
        )
        # offset: BPE
        semantic_codes += self.tokenizer_len

        # concat input_tokens
        input_tokens = torch.cat([texts, semantic_codes], dim=1)

        # loss mask
        text_mask = torch.arange(texts.size(1)).unsqueeze(0).to(device) < (1 + text_lens.unsqueeze(1)) # add eos loss
        semantic_mask = torch.arange(semantic_codes.size(1)).unsqueeze(0).to(device) < (2 + semantic_length.unsqueeze(1))
        loss_mask = torch.cat([text_mask, semantic_mask], dim=1).float()
        return input_tokens, loss_mask

    @torch.no_grad()
    def get_semantic_codes(self, wavs_16k, wavs_16k_len):
        infer_fn = self.requires["semantic_infer_fn"]
        semantic_codes, semantic_length = infer_fn(
            self.requires["semantic"],
            self.requires["centroids"],
            wavs_16k,
            wavs_16k_len,
            wavs_16k.device,
        )
        return semantic_codes, semantic_length

    @torch.no_grad()
    def get_codec_codes(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2) # [b, t, n_code]
        return output

    @torch.no_grad()
    def inference_from_text(self, batch, tokenizer):
        seqs, seq_lens, pos_ids, seq_sen_ids, utts = batch
        b, t = seqs.shape   
        attention_mask = sequence_mask(seq_lens, max_len=None, device=seqs.device)
        input_tokens = seqs

        # inference
        semantic_outputs = []

        if hasattr(self.model, "resize_token_embeddings"):
            # hugging face style
            past_key_values = None
            for i in tqdm(range(1000)):
                model_outputs = self.model(input_tokens, past_key_values=past_key_values, use_cache=True)
                logits = model_outputs["logits"]
                past_key_values = model_outputs["past_key_values"]
                pred_logits = logits[:, -1, self.tokenizer_len: self.tokenizer_len + self.n_semantic + 2]
                pred_logits[:, self.n_semantic] = -1e3
                samples = sample(pred_logits, temp=0.9, mode="naive", device=device) # [b, 1]
                if samples.item() == self.n_semantic + 1:
                    break
                # next infer
                semantic_outputs.append(samples)
                input_tokens = samples + self.tokenizer_len # offset: token
            semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t]
        else:
            # llama style inference
            self.model.params.use_cache = True
            start_pos = 0
            for i in tqdm(range(1500)):
                # print("input_tokens: ", input_tokens.shape, input_tokens)
                # print("start_pos: ", start_pos)
                model_outputs = self.model(input_tokens, start_pos=start_pos)
                logits = model_outputs["logits"]
                start_pos += input_tokens.size(1)
                # print("logits: ", logits.shape, logits)
                # print("self.tokenizer_len: ", self.tokenizer_len)
                # print("self.n_semantic: ", self.n_semantic)
                pred_logits = logits[:, -1, self.tokenizer_len + 1: self.tokenizer_len + self.n_semantic + 3]
                pred_logits[:, self.n_semantic] = -1e3
                if i < 50:
                    pred_logits[:, self.n_semantic + 1] = -1e5

                # print("pred_logits: ", pred_logits.shape, pred_logits)
                samples = sample(pred_logits, temp=0.9, mode="naive", device=seqs.device) # [b, 1]
                # print("samples: ", samples.item())
                if samples.item() == self.n_semantic + 1:
                    break
                # next infer
                semantic_outputs.append(samples + self.tokenizer_len + 1)
                input_tokens = samples + self.tokenizer_len + 1 # offset: token
                pos_ids = torch.cat([pos_ids, pos_ids[:, -1:] + 1], dim=1)  # [b, t]
                seq_sen_ids = torch.cat(
                    [seq_sen_ids, torch.zeros_like(seq_sen_ids[:, -1:]) + 2], dim=1
                )

            semantic_outputs = torch.cat(semantic_outputs, dim=1) # [b, t]
            semantic_outputs = torch.cat([seqs, semantic_outputs], dim=1)
            self.model.params.use_cache = False

        return semantic_outputs, pos_ids, seq_sen_ids

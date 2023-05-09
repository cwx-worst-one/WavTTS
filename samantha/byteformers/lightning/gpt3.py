import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm

from ..benchmarks.utils import flops
from ..models.gpt2 import GPT2, GPT2Config


class GPT3Module(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        n_positions,
        n_embd,
        n_layer,
        n_head,
        n_inner,
        resid_pdrop,
        embd_pdrop,
        attn_pdrop,
        learning_rate,
        warmup_tokens,
        final_tokens,
        attention_kwargs={},
        weight_decay=0.1,
        betas=(0.9, 0.95),
    ):
        super().__init__()
        self.save_hyperparameters()

        config = GPT2Config(
            vocab_size,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_inner=n_inner,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            attention_kwargs=attention_kwargs,
        )
        self.model = GPT2(config)
        self._tokens_seen = 0
        self.total_flops = flops(config, vocab_size, n_positions)["total"]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2), sync_dist=True)

    def on_train_start(self):
        batch_size = self.trainer.train_dataloader.batch_size
        world_size = self.trainer.world_size
        effective_batch_size_tokens = (
            world_size * batch_size * self.model.config.n_positions
        )

        hparams = vars(self.hparams)
        hparams.update(
            {
                "batch_size": batch_size,
                "effective_batch_size": world_size * batch_size,
                "effective_batch_size_tokens": effective_batch_size_tokens,
            }
        )
        self.logger.log_hyperparams(
            hparams,
            {
                "loss/train": torch.inf,
                "loss/valid": torch.inf,
                "parameters": 0,
                "FLOPs": 0,
            },
        )

    def forward(self, src):
        self._tokens_seen += (src >= 0).numel()
        return self.model(src)

    def step(self, batch):
        src, targets = batch
        logits = self.forward(src)
        return self.loss(logits, targets)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        return {"nll": loss, "perplexity": loss.exp()}

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("loss/train", loss["nll"], rank_zero_only=True, prog_bar=True)
        self.log("ppl/train", loss["perplexity"], rank_zero_only=True)
        return loss["nll"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        # TODO: when trained using ddp, is accumulation of those metrics happen
        # correctly?
        self.log("loss/valid", loss["nll"], rank_zero_only=True, sync_dist=True)
        self.log("ppl/valid", loss["perplexity"], rank_zero_only=True, sync_dist=True)
        return loss["nll"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, return_loss=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # - Start with a warm up, ramp up then cosine
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            fused=True,
        )

        def update_lr(*_):
            config = self.hparams
            if self._tokens_seen < config.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, config.warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - config.warmup_tokens) / float(
                    max(1, config.final_tokens - config.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[update_lr, update_lr]
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.hparams.n_positions
                else idx[:, -self.hparams.n_positions :]
            )
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

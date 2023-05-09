import deepspeed
import pytorch_lightning as pl
import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.nn import functional as F
from torch.optim import Adam

from recipes.minGPT.modules.model import Block


class GPT(pl.LightningModule):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        dataset,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        learning_rate=3e-4,
        n_embd=768,
        block_size=128,
        embd_pdrop=0.1,
        n_layer=12,
        n_head=4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # input embedding stem
        self.hparams.vocab_size = dataset.vocab_size
        self.tok_emb = nn.Embedding(self.hparams.vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size

        self.blocks = nn.ModuleList(
            [Block(self.hparams) for _ in range(self.hparams.n_layer)]
        )

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
        # todo: need to enable deepspeed cpu adam only if offloading

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            return FusedAdam(
                optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
            )
        else:
            return Adam(
                optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = torch.utils.checkpoint.checkpoint(block, x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss)
        return loss

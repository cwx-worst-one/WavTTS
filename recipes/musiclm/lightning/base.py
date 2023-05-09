from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_lightning.utilities import grad_norm
from torch.distributions.one_hot_categorical import OneHotCategorical

from samantha.stages.module.byteformers.models import Llama, LlamaConfig


def generate_offsets(
    s: int, codebook_size: int, start_channel: int, end_channel: int, device
) -> torch.Tensor:
    """Generates a vector of offsets of the shape (q, s). For every
    quantizer q, we generate a list of sequence length s that adds the codebook size,
    so that each quantizer vector will have unique token ID's.

    Args:
        s (int): sequence length
        start_channel (int): start channel at which to begin the offsets
        start_channel (int): end channel at which to end the offset
        codebook_size (int): the codebook size of the original quantize,
            (in the AudioLM paper, it is 1024 in each RVQ layer)

    Returns:
        torch.Tensor: A vector of shape (q, s) that includes the indices by which
            we offset an (q, s) tensor containing the original codebook's token id's
    """

    offsets = torch.arange(start_channel, end_channel, device=device) * codebook_size
    return repeat(offsets, "q -> q s", s=s)


def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor]:
    """Compute the cross-entropy loss (negative log-likelihood) on the predicted
    and target token id's. We also compute the average accuracy.

    Args:
        preds (torch.Tensor): _description_
        targets (torch.Tensor): _description_

    Returns:
        Tuple[torch.Tensor]: _description_
    """
    # 8. Rearrange the tensor to N x C that is expected by nn.CrossEntropyLoss
    # and so that we don't have to flatten twice
    preds = rearrange(preds, "b n c -> b c n")
    loss = F.cross_entropy(preds, targets)
    accuracy = (preds.argmax(1) == targets).float().mean()
    perplexity = loss.exp()
    return {"loss": loss, "accuracy": accuracy, "ppl": perplexity}


def flatten_row_major(acoustic_tokens: torch.Tensor) -> torch.Tensor:
    """Flattens tokens of shape (b, q, s) in a row-major order to a
    sequence of tokens (b, (s q))

    Args:
        acoustic_tokens (torch.Tensor): _description_

    Returns:
        torch.Tensor: Flattened tensor
    """
    return rearrange(acoustic_tokens, "b q s -> b (s q)")


def unflatten_row_major(
    acoustic_tokens: torch.Tensor, channel_dim: int
) -> torch.Tensor:
    return rearrange(acoustic_tokens, "b (s q) -> b q s", q=channel_dim)


def temperature_sampling(preds: torch.Tensor, temperature: float) -> torch.Tensor:
    preds = (preds / temperature).softmax(dim=-1)
    output_onehot = OneHotCategorical(probs=preds).sample()
    return torch.argmax(output_onehot, dim=1)


def safe_log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -safe_log(-safe_log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LitModuleBase(pl.LightningModule):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        codebook_size: int,
        n_codebooks: int,
        max_sequence_length: int,
        optimizer_class,
        scheduler_class,
        attention_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = self.hparams.n_codebooks * self.hparams.codebook_size
        self.max_flattened_seq_len = (
            self.hparams.n_codebooks * self.hparams.max_sequence_length
        )

        config = LlamaConfig(
            vocab_size=self.vocab_size,
            n_layer=self.hparams.n_layer,
            n_head=self.hparams.n_head,
            n_embd=self.hparams.n_embd,
            attention_kwargs=self.hparams.attention_kwargs,
        )
        self.model = Llama(config)
        self.model.apply(self.model._init_weights)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2), sync_dist=True)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        batch_size = self.trainer.datamodule.batch_size
        world_size = self.trainer.world_size
        effective_batch_size_tokens = (
            world_size * batch_size * self.max_flattened_seq_len
        )

        optimizer = self.optimizers().optimizer
        lr_scheduler = self.lr_schedulers()
        optimizer_params = {f"optim.{k}": v for k, v in optimizer.defaults.items()}
        lr_scheduler_params = {
            f"lr_scheduler.{k}": v for k, v in lr_scheduler.hparams.items()
        }

        hparams = vars(self.hparams)
        hparams.update(
            {
                "batch_size": batch_size,
                "effective_batch_size": world_size * batch_size,
                "effective_batch_size_tokens": effective_batch_size_tokens,
                "vocab_size": self.vocab_size,
                "n_embd": self.model.config.n_embd,
                "n_head": self.model.config.n_head,
                "n_inner": self.model.config.n_inner,
                "n_layer": self.model.config.n_layer,
                "max_position_embeddings": self.max_flattened_seq_len,
                "optimizer": optimizer.__class__.__name__,
                **optimizer_params,
                "lr_scheduler": lr_scheduler.__class__.__name__,
                **lr_scheduler_params,
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

    def step(self, batch, return_loss: bool = True):
        """Forward pass of the model, given a batch of audio.

        Args:
            batch (_type_): Audio data
            return_loss (bool, optional): Whether to return the loss value or the
                predicted logits.

        Returns:
            _type_: Loss value or the predicted logits
        """
        audio = batch[0]

        input_token_ids = self.prepare_input_token_ids(audio)
        cond_token_ids = self.prepare_cond_token_ids(audio)

        prefixed_inputs = torch.cat((cond_token_ids, input_token_ids), dim=1)
        pred_tokens = self.forward(prefixed_inputs)

        cond_seq_len = cond_token_ids.shape[1]
        pred_tokens = pred_tokens[:, cond_seq_len:]
        if return_loss:
            return self.loss(pred_tokens, targets=input_token_ids)
        else:
            return pred_tokens

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return compute_loss(preds[:, :-1], targets[:, 1:])

    def offset(
        self,
        token_ids: torch.Tensor,
        start_channel: int,
        end_channel: int,
        codebook_size: int,
    ) -> torch.Tensor:
        offsets = generate_offsets(
            token_ids.shape[-1],
            codebook_size,
            start_channel,
            end_channel,
            device=token_ids.device,
        )
        token_ids = token_ids + offsets
        return token_ids

    def sample_within_bounds(
        self,
        input_token_ids,
        codebook_size: int,
        start_channel: int,
        end_channel: int,
        temperature: float,
    ):
        with torch.no_grad():
            last_pred = self.forward(input_token_ids)[:, -1]
        last_pred = last_pred[
            :, start_channel * codebook_size : end_channel * codebook_size,
        ]
        sampled = temperature_sampling(last_pred, temperature)
        sampled = rearrange(sampled, "b -> b 1")
        sampled = sampled + start_channel * codebook_size
        return sampled

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_token_ids)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/train", v, rank_zero_only=True, prog_bar=True)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/valid", v, rank_zero_only=True)
        return loss["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, return_loss=False)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.model.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

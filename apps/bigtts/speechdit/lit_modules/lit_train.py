import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.utilities import rank_zero_only  
from functools import lru_cache

from samantha.criterion.masked_loss import MaskedMAELoss, MaskedMSELoss, MaskedSSIMLoss, MaskedPseudoHuberLoss
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.model_metric import ModelMetric

from .utils import plot_mel

logger = logging.getLogger(__name__)

LOSS_DICT = {"l1": MaskedMAELoss, "l2": MaskedMSELoss, "ssim": MaskedSSIMLoss,
        "pseudohuber": MaskedPseudoHuberLoss}


class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterions,
        checkpointing=False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()

        self.criterion_dict = criterions

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        self.validation_step_outputs = []

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision, model_obj_or_objs=self.model
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, initializer in self.hparams.required_modules.items():
            self.requires[name] = initializer(rank=self.local_rank)

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        if "phone" in batch:
            batch["frontend"] = {
                "phone": batch["phone"],
                "tone": batch["tone"],
                "word_seg": batch["word_seg"],
            }
            if "lang" in batch:
                batch["frontend"]["lang"] = batch["lang"]

        ref = batch["bn"]
        feat_len = batch["bn_lens"]
        loss_mask = batch["bn_ctx_mask"]

        bsz, seqlen = ref.shape[0], ref.shape[1]
        batch_tokens = torch.sum(feat_len).item()

        self.model_metric.num_tokens += batch_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.model_metric.update(
                num_tokens=0,
                stage=self.trainer.state.stage,
                model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
            )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            out = self.model(batch)

            loss_dict = {}
            loss = 0
            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(out["pred_v"], out["target_v"], loss_mask)
                loss_dict[loss_type] = tmp_loss.item()
                loss += tmp_loss

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {"loss": loss.item(), "bsz": bsz, "seqlen": seqlen}
            log_dict.update(loss_dict)
            log_dict["training/loss"] = log_dict["loss"]
            metric = self.model_metric.compute(self.trainer.global_step)
            log_dict.update(metric)
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return loss

    # @rank_zero_only conflict with fsdp strategy
    def validation_step(self, batch, batch_idx):
        if "phone" in batch:
            batch["frontend"] = {
                "phone": batch["phone"],
                "tone": batch["tone"],
                "word_seg": batch["word_seg"]
                }
            if "lang" in batch:
                batch["frontend"]["lang"] = batch["lang"]

        ref = batch["bn"]
        feat_len = batch["bn_lens"]
        loss_mask = batch["bn_ctx_mask"]

        loss_dict = {}
        out = self.model(batch)

        loss = 0
        for loss_type, loss_func in self.criterion_dict.items():
            tmp_loss = loss_func(out["pred_v"], out["target_v"], loss_mask)
            loss_dict[loss_type] = tmp_loss.item()
            loss += tmp_loss
        self.validation_step_outputs.append(loss_dict)

        return loss

    @rank_zero_only
    def on_validation_epoch_end(self):
        loss_dict = {}
        valid_log_dict = {}
        keys = self.validation_step_outputs[0].keys()
        for k in keys:
            loss_dict[k] = 0
        for x in self.validation_step_outputs:
            for k in keys:
                loss_dict[k] += x[k]
        for k, v in loss_dict.items():
            valid_log_dict[f"valid/{k}"] = v / len(self.validation_step_outputs)
        self.log_dict(valid_log_dict, prog_bar=True, rank_zero_only=True)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @lru_cache()
    def need_custom_grad_clip(self, gradient_clip_algorithm):
        logger.info("enable custom gradient clipping")
        strategy_name = None
        if hasattr(self.trainer.strategy, "strategy_name"):
            strategy_name = self.trainer.strategy.strategy_name
        return strategy_name == "fsdp" and gradient_clip_algorithm == "norm"

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        if self.need_custom_grad_clip(gradient_clip_algorithm):
            self.trainer.model.clip_grad_norm_(max_norm=self.trainer.gradient_clip_val)
        else:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    @torch.no_grad()
    def inference(self, inputs, step):
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    predict = inference

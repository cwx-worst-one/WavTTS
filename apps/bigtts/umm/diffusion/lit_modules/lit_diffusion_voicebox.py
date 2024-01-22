import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.criterion.masked_loss import MaskedMAELoss, MaskedMSELoss, MaskedSSIMLoss
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.model_metric import ModelMetric

from .utils import plot_mel

logger = logging.getLogger(__name__)

LOSS_DICT = {
        "l1": MaskedMAELoss,
        "l2": MaskedMSELoss,
        "ssim": MaskedSSIMLoss
        }



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterions,
        checkpointing=True,
        resume_ckpt_path = None,
        umm_dropout = 0.0,
        umm_pad=16384
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()

        self.criterion_dict = {}
        for criterion in criterions:
            self.criterion_dict[criterion] = LOSS_DICT[criterion]()

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.umm_dropout = umm_dropout
        self.umm_pad = umm_pad


    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, initializer in self.hparams.required_modules.items():
            self.requires[name] = initializer(rank=self.local_rank)
            #if name == "umm_codebook":
            #    self.model.umm_pad_vector.data = self.requires[name].data.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def get_umm_token(self, wav):
        token = self.requires["umm"].wav2token(wav)
        return token

    def get_umm_embedding(self, token):
        embedding = F.embedding(token,
                torch.cat([self.requires["umm_codebook"], self.model.umm_pad_vector], dim=0))
        return embedding

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
                "word_seg": batch["word_seg"]
                }
            if "lang" in batch:
                batch["frontend"]["lang"] = batch["lang"]

        if self.umm_dropout > 0:
            drop_idx = torch.rand([batch["token"].shape[0],batch["token"].shape[1]]) < self.umm_dropout
            if torch.sum(drop_idx) > 0:
                batch["token"][drop_idx] = self.umm_pad

        if "umm_codebook" in self.requires:
            batch["token"] = self.get_umm_embedding(batch["token"])

        if "mel" in batch:
            ref = batch["mel"]
            feat_len = batch["mel_lens"]
            loss_mask = batch["mel_ctx_mask"]
        else:
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
            pred, target = self.model(batch)

            loss_dict = {}
            loss = 0
            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(pred, target, loss_mask)
                loss_dict[loss_type] = tmp_loss.item()
                loss += tmp_loss

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {
                "loss": loss.item(),
                "bsz": bsz,
                "seqlen": seqlen
                }
            log_dict.update(loss_dict)
            log_dict["training/loss"] = log_dict["loss"]
            metric = self.model_metric.compute(self.trainer.global_step)
            log_dict.update(metric)
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        # import pytorch_lightning.utilities
        # grad_norm_dict = pytorch_lightning.utilities.grad_norm(self.model, norm_type=2)
        grad_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2)
        self.log_dict({"training/grad_2_norm": grad_norm}, sync_dist=True, prog_bar=True)

    def log_mel(self, mels, t):
        for name, mel in mels.items():
            mel = mel.transpose(0,1).cpu().detach().numpy()
            self.loggers[1].log_image(name, [plot_mel(mel, t)])

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
    def inference(self, inputs, step):
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    predict = inference

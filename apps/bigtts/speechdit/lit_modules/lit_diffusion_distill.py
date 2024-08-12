import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from samantha.criterion.masked_loss import MaskedMAELoss, MaskedMSELoss, MaskedSSIMLoss
from samantha.models.llama_ldm import (
    BernoulliDistribution,
    UniformDistribution,
    extend_dim,
)
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.model_metric import ModelMetric

from .utils import plot_mel

logger = logging.getLogger(__name__)

LOSS_DICT = {"l1": MaskedMAELoss, "l2": MaskedMSELoss, "ssim": MaskedSSIMLoss}


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterions,
        checkpointing=True,
        resume_ckpt_path=None,
        umm_dropout=0.0,
        umm_pad=16384,
        min_t_diff=0.001,  # 1/1000
        max_t_diff=0.02,  # 1/50
        fix_t_boundary=True,
        min_guidance_scale=0,
        max_guidance_scale=7,
        guidance_dist="uniform",
        ema_decay=0.9999,
        ode_solver="ddim",
        p_target_x0=0,
        fine_tuning=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.teacher_model = model_cls()
        self.student_model = model_cls()

        if resume_ckpt_path is not None:
            self.load_from_pretrained(self.model, resume_ckpt_path)
            self.load_from_pretrained(self.teacher_model, resume_ckpt_path)
            self.load_from_pretrained(self.student_model, resume_ckpt_path)

        self.model.train()
        self.teacher_model.eval()
        self.student_model.train()
        self.teacher_model.requires_grad_(False)
        self.student_model.requires_grad_(False)

        self.criterion_dict = {}
        for criterion in criterions:
            self.criterion_dict[criterion] = LOSS_DICT[criterion]()

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.umm_dropout = umm_dropout
        self.umm_pad = umm_pad

        self.t_diff_distribution = UniformDistribution(vmin=min_t_diff, vmax=max_t_diff)
        if guidance_dist == "uniform":
            self.guidance_distribution = UniformDistribution(
                vmin=min_guidance_scale, vmax=max_guidance_scale
            )
        elif guidance_dist == "bernoulli":
            self.guidance_distribution = BernoulliDistribution(
                min_guidance_scale, max_guidance_scale
            )

    def load_from_pretrained(self, model, pretrained_path=None):
        rank_zero_info(f"Loading pre-trained model from checkpoint {pretrained_path}")
        ckpt_state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for k in ckpt_state_dict:
            new_k = k.replace("model.", "")  # saved model has prefix "model."
            if new_k in model_state_dict:
                if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                    rank_zero_info(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[new_k].shape}, "
                        f"loaded shape: {ckpt_state_dict[k].shape}"
                    )
                else:
                    new_state_dict[new_k] = ckpt_state_dict[k]
            else:
                rank_zero_info(f"Dropping parameter {k}")
        model.load_state_dict(new_state_dict, strict=False)

    def setup(self, stage: str) -> None:
        setattr(self.teacher_model, "flops_fn", FlopsProfiler(self.teacher_model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision, model_obj_or_objs=self.teacher_model
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, initializer in self.hparams.required_modules.items():
            self.requires[name] = initializer(rank=self.local_rank)

    @torch.no_grad()
    def get_umm_token(self, wav):
        token = self.requires["umm"].wav2token(wav)
        return token

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

        if self.umm_dropout > 0:
            drop_idx = (
                torch.rand([batch["token"].shape[0], batch["token"].shape[1]])
                < self.umm_dropout
            )
            if torch.sum(drop_idx) > 0:
                batch["token"][drop_idx] = self.umm_pad

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

        device = self.device
        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            x = batch["bn"]

            # Sample T
            t_diff = self.t_diff_distribution(1, device=device)

            if self.hparams.fix_t_boundary:
                t = (1 - t_diff) * torch.rand(bsz, device=device) + t_diff
                prev_t = t - t_diff
            else:
                t = torch.rand(bsz, device=device)
                prev_t = torch.clamp(t - t_diff, min=0)

            # Sample guidance scale
            guidance_scale = self.guidance_distribution(bsz, device)

            # Sample noise
            noise = torch.randn_like(x)
            t_batch = extend_dim(t, dim=x.ndim)
            alphas, betas = self.model.get_alpha_beta(t_batch)
            x_noisy = alphas * x + betas * noise

            pred_x0 = self.model.pred_x0(batch, x_noisy, t)

            with torch.no_grad():
                self.teacher_model.eval()
                batch = self.make_cfg_input(batch)
                if self.hparams.fine_tuning:
                    prev_t_batch = extend_dim(prev_t, dim=x.ndim)
                    prev_alphas, prev_betas = self.model.get_alpha_beta(prev_t_batch)
                    prev_x_noisy = prev_alphas * x + prev_betas * noise
                else:
                    prev_x_noisy = self.teacher_model.infer_one_step(
                        batch,
                        x_noisy,
                        t,
                        prev_t,
                        guidance_scale,
                        sampler=self.hparams.ode_solver,
                    )

                batch = self.unmake_cfg_input(batch)
                student_pred_x0 = self.student_model.pred_x0(
                    batch, prev_x_noisy, prev_t
                ).detach()

            zero_idx = prev_t == 0
            student_pred_x0[zero_idx] = x.transpose(1, 2)[zero_idx]

            if self.hparams.p_target_x0 > 0:
                drop_idx = torch.rand(bsz) < self.hparams.p_target_x0
                student_pred_x0[drop_idx] = x.transpose(1, 2)[drop_idx]

            loss_dict = {}
            loss = 0

            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(pred_x0, student_pred_x0, loss_mask)
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

    def make_cfg_input(self, inputs):
        B = inputs["token"].shape[0]
        inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
        inputs["frontend"]["phone"][B:, :] = 1
        inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
        inputs["frontend"]["tone"][B:, :] = 1
        inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
        inputs["frontend"]["word_seg"][B:, :] = 1
        return inputs

    def unmake_cfg_input(self, inputs):
        B = inputs["token"].shape[0]
        inputs["frontend"]["phone"] = inputs["frontend"]["phone"][:B, :]
        inputs["frontend"]["tone"] = inputs["frontend"]["tone"][:B, :]
        inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"][:B, :]
        return inputs

    def log_mel(self, mels, t):
        for name, mel in mels.items():
            mel = mel.transpose(0, 1).cpu().detach().numpy()
            self.loggers[1].log_image(name, [plot_mel(mel, t)])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def inference(self, inputs, step):
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    def on_train_batch_end(self, outputs, batch, batch_idx):
        update_ema(
            self.student_model.parameters(),
            self.model.parameters(),
            self.hparams.ema_decay,
        )

    predict = inference

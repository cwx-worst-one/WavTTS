import logging
from dataclasses import dataclass, field
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.utilities import rank_zero_only  
from pytorch_lightning.utilities.rank_zero import rank_zero_info 
from functools import lru_cache
from copy import deepcopy
import math

from samantha.criterion.masked_loss import MaskedMAELoss, MaskedMSELoss, MaskedSSIMLoss
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.model_metric import ModelMetric
from samantha.models.speechdit import UniformDistribution, BernoulliDistribution, LogitNormalDistribution, extend_dim, get_sigma

from .utils import plot_mel

logger = logging.getLogger(__name__)

def discriminator_loss(d_fake, d_real, loss_mask):
    loss_d_fake = 0
    loss_d_real = 0
    reduce_sum = torch.sum(loss_mask)
    n_loss = len(d_fake)
    for x_fake, x_real in zip(d_fake, d_real):
        x_fake = torch.sum(((1 + x_fake.squeeze(-1)) * loss_mask) ** 2)
        x_real = torch.sum(((1 - x_real.squeeze(-1)) * loss_mask) ** 2)
        loss_d_fake += x_fake / reduce_sum
        loss_d_real += x_real / reduce_sum
    return loss_d_fake/n_loss, loss_d_real/n_loss


def generator_loss(d_fake, loss_mask):
    loss_g_fake = 0.0
    reduce_sum = torch.sum(loss_mask)
    n_loss = len(d_fake)
    for x_fake in d_fake:
        x_fake = torch.sum(((1 - x_fake.squeeze(-1)) * loss_mask) ** 2)
        loss_g_fake += x_fake / reduce_sum
    return loss_g_fake/n_loss


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

@dataclass
class ConsistencyDistillArgs:
    N: int = 50
    min_guidance_scale: float = 3
    t_distribution: str = "uniform"
    max_guidance_scale: float = 5.5
    guidance_dist: str = "uniform"
    ema_decay: float = 0.99
    ode_solver: str = "ddim"
    ode_solver_use_cfg: bool = True
    p_target_x0: float = 0  # deprecated
    use_zero_groundtruth: bool = True
    use_weighted_loss: bool = False # To be test
    cfg_drop_rate: float = 0.0
    lambda_g_fake: float = 1
    lambda_d: float = 1

@dataclass
class DMDArgs:
    denoising_step: int = 4
    lambda_g_fake: float = 5e-3 # follow dmd
    lambda_d: float = 1e-2 # follow dmd

class BaseModule(pl.LightningModule):
    def __init__():
        super().__init__()

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision, model_obj_or_objs=self.model
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def load_from_pretrained(self, model, pretrained_path=None):
        rank_zero_info(f'Loading pre-trained model from checkpoint {pretrained_path}')
        ckpt_state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for k in ckpt_state_dict:
            new_k = k.replace("model.", "") # saved model has prefix "model."
            if new_k in model_state_dict:
                if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                    rank_zero_info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[new_k].shape}, "
                                f"loaded shape: {ckpt_state_dict[k].shape}")
                else:
                    new_state_dict[new_k] = ckpt_state_dict[k]
            else:
                rank_zero_info(f"Dropping parameter {k}")
        model.load_state_dict(new_state_dict, strict=False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()


class DMD(BaseModule):
    def __init__(
        self,
        model_cls,
        disc_cls,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        distill_args,
        criterions,
        g_clip_grad_norm=1,
        d_clip_grad_norm=1,
        g_cycle=1,
        d_cycle=1,
        resume_ckpt_path=None,
        save_ema_decay=0.9999
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["resume_ckpt_path"])
        hp = distill_args

        self.gen_model = model_cls()
        self.teacher_model = model_cls()
        self.fake_model = model_cls()
        self.ema_model = model_cls()
        self.disc = disc_cls()

        if resume_ckpt_path is not None:
            self.load_from_pretrained(self.gen_model, resume_ckpt_path)
            self.load_from_pretrained(self.teacher_model, resume_ckpt_path)
            self.load_from_pretrained(self.fake_model, resume_ckpt_path)
            self.load_from_pretrained(self.ema_model, resume_ckpt_path)

        self.gen_model.train()
        self.teacher_model.eval()
        self.fake_model.train()
        self.ema_model.eval()
        self.disc.train()

        self.teacher_model.requires_grad_(False)
        self.ema_model.requires_grad_(False)

        self.criterion_dict = criterions
        self.save_ema_decay = save_ema_decay

        self.automatic_optimization = False



class ConsistencyDistill(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        ode_model_cls,
        optimizer_cls,
        scheduler_cls,
        distill_args,
        required_modules,
        criterions,
        resume_ckpt_path=None,
        save_ema_decay=0.9999
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["resume_ckpt_path"])

        hp = distill_args
        self.save_hyperparameters()
        self.iter_model = model_cls()
        self.teacher_model = ode_model_cls()
        self.model = model_cls()
        self.ema_model = model_cls()

        if resume_ckpt_path is not None:
            self.load_from_pretrained(self.iter_model, resume_ckpt_path)
            self.load_from_pretrained(self.teacher_model, resume_ckpt_path)
            self.load_from_pretrained(self.model, resume_ckpt_path)
            self.load_from_pretrained(self.ema_model, resume_ckpt_path)

        self.iter_model.train()
        self.teacher_model.eval()
        self.model.train()
        self.ema_model.eval()

        self.teacher_model.requires_grad_(False)
        self.model.requires_grad_(False)
        self.ema_model.requires_grad_(False)

        self.criterion_dict = criterions

        self.N = hp.N
        self.distill_interval = 1 / self.N
        self.save_ema_decay = save_ema_decay

        if hp.guidance_dist == "uniform":
            self.guidance_distribution = UniformDistribution(vmin=hp.min_guidance_scale, vmax=hp.max_guidance_scale)
        elif guidance_dist == "bernoulli":
            self.guidance_distribution = BernoulliDistribution(hp.min_guidance_scale, hp.max_guidance_scale)

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision, model_obj_or_objs=self.model
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def load_from_pretrained(self, model, pretrained_path=None):
        rank_zero_info(f'Loading pre-trained model from checkpoint {pretrained_path}')
        ckpt_state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for k in ckpt_state_dict:
            new_k = k.replace("model.", "") # saved model has prefix "model."
            if new_k in model_state_dict:
                if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                    rank_zero_info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[new_k].shape}, "
                                f"loaded shape: {ckpt_state_dict[k].shape}")
                else:
                    new_state_dict[new_k] = ckpt_state_dict[k]
            else:
                rank_zero_info(f"Dropping parameter {k}")
        model.load_state_dict(new_state_dict, strict=False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        hp = self.hparams.distill_args
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

        bsz, seqlen, device = ref.shape[0], ref.shape[1], ref.device
        batch_tokens = torch.sum(feat_len).item()

        self.model_metric.num_tokens += batch_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.model_metric.update(
                num_tokens=0,
                stage=self.trainer.state.stage,
                model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
            )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            x = batch["bn"]
            
            n = torch.randint(1, self.N+1, [bsz], device=device)
            t, prev_t = n * self.distill_interval, (n - 1) * self.distill_interval

            guidance_scale = self.guidance_distribution(bsz, device).unsqueeze(1).unsqueeze(2)

            noise = torch.randn_like(x)
            t_batch = extend_dim(t, dim=x.ndim)
            alphas, betas = self.iter_model.get_alpha_beta(t_batch)
            target_v = alphas * noise - betas * x

            x_noisy = alphas * x + betas * noise

            prev_t_batch = extend_dim(prev_t, dim=x.ndim)
            prev_alphas, prev_betas = self.iter_model.get_alpha_beta(prev_t_batch)

            prev_sigma = prev_betas / prev_alphas

            train_batch = self.make_drop_input(batch, hp.cfg_drop_rate)

            train_batch["guidance_scale"] = guidance_scale.squeeze(1).squeeze(1)

            pred_v = self.iter_model(train_batch, t, x_noisy)["pred_v"].transpose(1, 2)
            pred_x0 = alphas * x_noisy - betas * pred_v

            if hp.use_weighted_loss:
                weight = 1 / (get_sigma(t) - get_sigma(prev_t))

            with torch.no_grad():
                self.teacher_model.eval()

                if hp.ode_solver_use_cfg:
                    teacher_cond_v = self.teacher_model(batch, t, x_noisy)["pred_v"].transpose(1, 2)
                    uncond_batch = self.make_uncond_input(batch) 
                    teacher_uncond_v = self.teacher_model(uncond_batch, t, x_noisy)["pred_v"].transpose(1, 2)
                    teacher_pred_v = guidance_scale * teacher_cond_v + (1 - guidance_scale) * teacher_uncond_v
                else:
                    teacher_pred_v = self.teacher_model(train_batch, t, x_noisy)["pred_v"].transpose(1, 2)

                if hp.ode_solver == "ddim":
                    teacher_pred_x0 = alphas * x_noisy - betas * teacher_pred_v
                    teacher_pred_noise = betas * x_noisy + alphas * teacher_pred_v
                    prev_x_noisy = prev_alphas * teacher_pred_x0 + prev_betas * teacher_pred_noise
                else:
                    raise NotImplementedError

                student_pred_v = self.model(train_batch, prev_t, prev_x_noisy)["pred_v"].transpose(1, 2)
                student_pred_x0 = prev_alphas * prev_x_noisy - prev_betas * student_pred_v

            zero_idx = (prev_t == 0) 
            if hp.use_zero_groundtruth:
                student_pred_x0[zero_idx] = x[zero_idx] 
            if hp.p_target_x0 > 0:
                drop_idx = torch.rand(bsz) < hp.p_target_x0
                student_pred_x0[drop_idx] = x[drop_idx]

            pred_x0 = pred_x0.transpose(1, 2)
            student_pred_x0 = student_pred_x0.transpose(1, 2)

            loss_dict = {}
            loss = 0
            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(pred_x0, student_pred_x0.detach(), loss_mask, weight if hp.use_weighted_loss else None)
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

    def make_uncond_input(self, inputs):
        uncond_inputs = deepcopy(inputs)
        unpad_mask = (uncond_inputs["frontend"]["phone"] != 0)
        uncond_inputs["frontend"]["phone"][unpad_mask] = 1
        uncond_inputs["frontend"]["tone"][unpad_mask] = 1
        uncond_inputs["frontend"]["word_seg"][unpad_mask] = 1
        uncond_inputs["bn_ctx"][:] = 0 # padding value is 0
        uncond_inputs["prompt_bn"][:] = 0 # padding value is 0
        uncond_inputs["flag_drop"][:] =True
        return uncond_inputs
    
    def make_drop_input(self, inputs, drop_rate):
        if drop_rate > 0:
            B = inputs["flag_drop"].shape[0]
            drop_idx = torch.randn([B]) < drop_rate
            drop_inputs = deepcopy(inputs)
            unpad_mask = (drop_inputs["frontend"]["phone"] != 0)
            unpad_mask[~drop_idx] = False
            drop_inputs["frontend"]["phone"][unpad_mask] = 1
            drop_inputs["frontend"]["tone"][unpad_mask] = 1
            drop_inputs["frontend"]["word_seg"][unpad_mask] = 1
            drop_inputs["bn_ctx"][drop_idx] = 0 # padding value is 0
            drop_inputs["prompt_bn"][drop_idx] = 0 # padding value is 0
            drop_inputs["flag_drop"][drop_idx] = True
            return drop_inputs
        else:
            return inputs


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.iter_model.parameters())
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
            self.trainer.iter_model.clip_grad_norm_(max_norm=self.trainer.gradient_clip_val)
        else:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    @torch.no_grad()
    def inference(self, inputs, step):
        x = self.model.inference(inputs, step)
        return x

    def on_train_batch_end(self, outputs, batch, batch_idx): 
        update_ema(self.model.parameters(), self.iter_model.parameters(), self.hparams.distill_args.ema_decay) 
        update_ema(self.ema_model.parameters(), self.iter_model.parameters(), self.save_ema_decay)

    predict = inference

class AdvDistill(ConsistencyDistill):
    def __init__(
        self,
        model_cls,
        disc_cls,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        t_dist,
        criterions,
        g_clip_grad_norm=1,
        d_clip_grad_norm=1,
        g_cycle=1,
        d_cycle=1,
        resume_ckpt_path=None,
    ):
        super(ConsistencyDistill, self).__init__()
        self.save_hyperparameters(ignore=["resume_ckpt_path"])

        self.model = model_cls()
        self.teacher = model_cls()
        self.disc = disc_cls()

        if resume_ckpt_path is not None:
            self.load_from_pretrained(self.teacher, resume_ckpt_path)
            self.load_from_pretrained(self.model, resume_ckpt_path)

        self.model.train()
        self.disc.train()
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self.criterion_dict = criterions

        self.t_dist = t_dist()

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g_cls(self.model.parameters())
        scheduler_g = self.hparams.scheduler_g_cls(optimizer_g)

        optimizer_d = self.hparams.optimizer_d_cls(self.disc.parameters())
        scheduler_d = self.hparams.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

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

        bsz, seqlen, device = ref.shape[0], ref.shape[1], ref.device
        batch_tokens = torch.sum(feat_len).item()

        self.model_metric.num_tokens += batch_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.model_metric.update(
                num_tokens=0,
                stage=self.trainer.state.stage,
                model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
            )

        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            x = batch["bn"]

            t = self.t_dist(bsz, device).float()
            noise = torch.randn_like(x)
            t_batch = extend_dim(t, dim=x.ndim)
            alphas, betas = self.model.get_alpha_beta(t_batch)
            x_noisy = alphas * x + betas * noise
            target_v = alphas * noise - betas * x

            pred_v = self.model(batch, t, x_noisy)["pred_v"].transpose(1, 2)
            pred_x0 = alphas * x_noisy - betas * pred_v
            pred_x0_noisy = alphas * pred_x0 + betas * noise

            feat_fake = self.teacher(batch, t, pred_x0_noisy, return_disc_feat=True)

            with torch.no_grad():
                feat_real = self.teacher(batch, t, x_noisy, return_disc_feat=True)

            if self.trainer.global_step % self.hparams.d_cycle == 0:
                # train discriminator
                self.toggle_optimizer(optim_d)
                
                fake_logits = self.disc([f_f.detach() for f_f in feat_fake], t, batch["text_mel_mask"])
                real_logits = self.disc([f_r.detach() for f_r in feat_real], t, batch["text_mel_mask"])

                fake_logits = [self.model.remove_text_prefix(fake_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for fake_logit in fake_logits]
                real_logits = [self.model.remove_text_prefix(real_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for real_logit in real_logits]

                loss_d_fake, loss_d_real = discriminator_loss(fake_logits, real_logits, loss_mask)

                loss_d = loss_d_fake + loss_d_real
                optim_d.zero_grad()
                self.manual_backward(loss_d)

                grad_norm_d = torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.hparams.d_clip_grad_norm)
                if math.isnan(grad_norm_d):
                    optim_d.zero_grad()
                else:
                    optim_d.step()
                scheduler_d.step()
                self.untoggle_optimizer(optim_d)

            if self.trainer.global_step % self.hparams.g_cycle == 0:
                # train student
                self.toggle_optimizer(optim_g)

                fake_logits = self.disc(feat_fake, t, batch["text_mel_mask"])
                fake_logits = [self.model.remove_text_prefix(fake_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for fake_logit in fake_logits]

                loss_g_fake = generator_loss(fake_logits, loss_mask)

                loss_g = 0
                loss_g += loss_g_fake
                loss_dict = dict()
                for loss_type, loss_func in self.criterion_dict.items():
                    tmp_loss = loss_func(pred_v.transpose(1,2), target_v.transpose(1,2), loss_mask, None)
                    loss_dict["training/v_"+loss_type] = tmp_loss.item()
                    loss_g += tmp_loss

                optim_g.zero_grad()
                self.manual_backward(loss_g)
                grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.g_clip_grad_norm)
                if math.isnan(grad_norm_g):
                    optim_g.zero_grad()
                else:
                    optim_g.step()
                scheduler_g.step()
                self.untoggle_optimizer(optim_g)

            loss_dict.update({
                    "training/loss_g_fake": loss_g_fake.item(),
                    "training/loss_g": loss_g.item(),
                    "training/loss_d": loss_d.item(),
                    "training/loss_d_fake": loss_d_fake.item(),
                    "training/loss_d_real": loss_d_real.item(),
                    "aux/optim_g_lr": optim_g.param_groups[0]["lr"],
                    "aux/optim_d_lr": optim_d.param_groups[0]["lr"],
                    "aux/grad_norm_d": grad_norm_d,
                    "aux/grad_norm_g": grad_norm_g,
                    })

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {"bsz": bsz, "seqlen": seqlen}
            log_dict.update(loss_dict)
            metric = self.model_metric.compute(self.trainer.global_step)
            log_dict.update(metric)
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx): 
        pass

class AdvConsistencyDistill(ConsistencyDistill):
    def __init__(
        self,
        model_cls,
        ode_model_cls,
        disc_cls,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        distill_args,
        criterions,
        g_clip_grad_norm=1,
        d_clip_grad_norm=1,
        g_cycle=1,
        d_cycle=1,
        resume_ckpt_path=None,
        save_ema_decay=0.9999
    ):
        super().__init__(
                model_cls,
                ode_model_cls,
                optimizer_g_cls,
                scheduler_g_cls,
                distill_args,
                None,
                criterions,
                resume_ckpt_path=resume_ckpt_path,
                save_ema_decay=save_ema_decay
                )
        self.save_hyperparameters(ignore=["resume_ckpt_path"])

        self.disc = disc_cls()
        self.disc.train()

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g_cls(self.iter_model.parameters())
        scheduler_g = self.hparams.scheduler_g_cls(optimizer_g)

        optimizer_d = self.hparams.optimizer_d_cls(self.disc.parameters())
        scheduler_d = self.hparams.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):

        hp = self.hparams.distill_args
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

        bsz, seqlen, device = ref.shape[0], ref.shape[1], ref.device
        batch_tokens = torch.sum(feat_len).item()

        self.model_metric.num_tokens += batch_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.model_metric.update(
                num_tokens=0,
                stage=self.trainer.state.stage,
                model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
            )
        
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers() 

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            loss_dict = {}
            x = batch["bn"]
            
            n = torch.randint(1, self.N+1, [bsz], device=device)
            t, prev_t = n * self.distill_interval, (n - 1) * self.distill_interval

            guidance_scale = self.guidance_distribution(bsz, device).unsqueeze(1).unsqueeze(2)

            noise = torch.randn_like(x)
            t_batch = extend_dim(t, dim=x.ndim)
            alphas, betas = self.iter_model.get_alpha_beta(t_batch)
            target_v = alphas * noise - betas * x

            x_noisy = alphas * x + betas * noise

            prev_t_batch = extend_dim(prev_t, dim=x.ndim)
            prev_alphas, prev_betas = self.iter_model.get_alpha_beta(prev_t_batch)

            prev_sigma = prev_betas / prev_alphas

            train_batch = self.make_drop_input(batch, hp.cfg_drop_rate)
            pred_v = self.iter_model(train_batch, t, x_noisy)["pred_v"].transpose(1, 2)
            pred_x0 = alphas * x_noisy - betas * pred_v
            pred_x0_noisy = alphas * pred_x0 + betas * noise 

            feat_fake = self.teacher_model(train_batch, t, pred_x0_noisy, return_disc_feat=True)  

            # train discrominator
            if self.trainer.global_step % self.hparams.d_cycle == 0:
                self.toggle_optimizer(optim_d)

                with torch.no_grad():
                     feat_real = self.teacher_model(train_batch, t, x_noisy, return_disc_feat=True)

                fake_logits = self.disc([f_f.detach() for f_f in feat_fake], t, batch["text_mel_mask"])
                real_logits = self.disc([f_r.detach() for f_r in feat_real], t, batch["text_mel_mask"])

                fake_logits = [self.model.remove_text_prefix(fake_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for fake_logit in fake_logits]
                real_logits = [self.model.remove_text_prefix(real_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for real_logit in real_logits]

                loss_d_fake, loss_d_real = discriminator_loss(fake_logits, real_logits, loss_mask)

                loss_d = loss_d_fake + loss_d_real
                loss_d = loss_d * hp.lambda_d
                optim_d.zero_grad()
                self.manual_backward(loss_d)

                grad_norm_d = torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.hparams.d_clip_grad_norm)
                if math.isnan(grad_norm_d):
                    optim_d.zero_grad()
                else:
                    optim_d.step()
                scheduler_d.step()

                self.untoggle_optimizer(optim_d)
                loss_dict.update({
                    "training/loss_d": loss_d.item(),
                    "training/loss_d_fake": loss_d_fake.item(),
                    "training/loss_d_real": loss_d_real.item(),
                    "aux/optim_d_lr": optim_d.param_groups[0]["lr"],
                    "aux/grad_norm_d": grad_norm_d,
                    })

            # train generator
            if self.trainer.global_step % self.hparams.g_cycle == 0:
                self.toggle_optimizer(optim_g)

                fake_logits = self.disc(feat_fake, t, batch["text_mel_mask"])
                fake_logits = [self.model.remove_text_prefix(fake_logit, x.shape[1], batch["text_lens"], batch["bn_lens"]) for fake_logit in fake_logits]

                loss_g_fake = generator_loss(fake_logits, loss_mask)

                loss_g = 0
                loss_g += loss_g_fake * hp.lambda_g_fake

                with torch.no_grad():
                    self.teacher_model.eval()

                    if hp.ode_solver_use_cfg:
                        teacher_cond_v = self.teacher_model(batch, t, x_noisy)["pred_v"].transpose(1, 2)
                        uncond_batch = self.make_uncond_input(batch) 
                        teacher_uncond_v = self.teacher_model(uncond_batch, t, x_noisy)["pred_v"].transpose(1, 2)
                        teacher_pred_v = guidance_scale * teacher_cond_v + (1 - guidance_scale) * teacher_uncond_v
                    else:
                        teacher_pred_v = self.teacher_model(train_batch, t, x_noisy)["pred_v"].transpose(1, 2)

                    if hp.ode_solver == "ddim":
                        teacher_pred_x0 = alphas * x_noisy - betas * teacher_pred_v
                        teacher_pred_noise = betas * x_noisy + alphas * teacher_pred_v
                        prev_x_noisy = prev_alphas * teacher_pred_x0 + prev_betas * teacher_pred_noise
                    else:
                        raise NotImplementedError

                    student_pred_v = self.model(train_batch, prev_t, prev_x_noisy)["pred_v"].transpose(1, 2)
                    student_pred_x0 = prev_alphas * prev_x_noisy - prev_betas * student_pred_v

                if hp.use_zero_groundtruth:
                    zero_idx = (prev_t == 0) 
                    student_pred_x0[zero_idx] = x[zero_idx] 
                if hp.p_target_x0 > 0:
                    drop_idx = torch.rand(bsz) < hp.p_target_x0
                    student_pred_x0[drop_idx] = x[drop_idx]

                pred_x0 = pred_x0.transpose(1, 2)
                student_pred_x0 = student_pred_x0.transpose(1, 2)

                for loss_type, loss_func in self.criterion_dict.items():
                    tmp_loss = loss_func(pred_x0, student_pred_x0.detach(), loss_mask, None)
                    loss_dict[loss_type] = tmp_loss.item()
                    loss_g += tmp_loss

                optim_g.zero_grad()
                self.manual_backward(loss_g)
                grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.g_clip_grad_norm)
                if math.isnan(grad_norm_g):
                    optim_g.zero_grad()
                else:
                    optim_g.step()
                scheduler_g.step()
                self.untoggle_optimizer(optim_g)
                loss_dict.update({
                    "training/loss_g": loss_g.item(),
                    "training/loss_g_fake": loss_g_fake.item(),
                    "aux/optim_g_lr": optim_g.param_groups[0]["lr"],
                    "aux/grad_norm_g": grad_norm_g,
                    })

                    
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {"bsz": bsz, "seqlen": seqlen}
            log_dict.update(loss_dict)
            log_dict["training/loss"] = log_dict["training/loss_g"]
            metric = self.model_metric.compute(self.trainer.global_step)
            log_dict.update(metric)
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler
from samantha.utils.model_metric import ModelMetric
from recipes.voicebox.modules.loss import sequence_mask
import logging


logger = logging.getLogger(__name__)



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterion_cls,
        checkpointing=True,
        use_len_mask = True,
        use_mask_loss = True,
        resume_ckpt_path = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()       
        self.criterion = criterion_cls()

        self.use_len_mask = use_len_mask
        self.use_mask_loss = use_mask_loss

    def setup(self, stage: str) -> None:
        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                frontend_inputs = {
                        "phone": batch["phone"],
                        "tone": batch["tone"],
                        "word_seg": batch["word_seg"]
                }
                ref = batch["phoneme_durations"].unsqueeze(-1) # [B, T, 1]
                ref = torch.clamp(ref, min=0, max=127)
                dur_ctx = batch["masked_phoneme_durations"]
                dur_ctx = torch.clamp(dur_ctx, min=0, max=127)
                dur_len = batch["text_lens"]

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            vt = self.model(dur_ctx, frontend_inputs, dur_len) # the nnet model should take x, t, and other conditional inputs

        if self.use_mask_loss and 'dur_ctx_mask' in batch:
            mask_loss = batch['dur_ctx_mask'].float() # [B, T]
            mask_loss = mask_loss.unsqueeze(-1) # [B, T, 1]
            vt = vt * mask_loss + ref * (1 - mask_loss)

        len_mask = sequence_mask(dur_len, device=ref.device)
        loss = self.criterion(vt, ref, len_mask)

        if torch.isnan(loss).any():
            loss = torch.tensor([0.0], requires_grad=True).to(loss.device)
        
        bsz, seqlen = ref.shape[0], ref.shape[1]
        self.log_dict(
            {
                "dur_loss": loss.item(),
                "bsz": bsz,
                "seqlen": seqlen
            },
            prog_bar=True,
            sync_dist=True)
        
        batch_tokens = torch.sum(dur_len).item()
        self.model_metric.update(
            num_tokens=batch_tokens,
            stage=self.trainer.state.stage,
            model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
        )
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.model_metric.compute(self.trainer.global_step)
            self.log_dict(
                metric, sync_dist=True, prog_bar=True
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
    def inference(self, batch, scale):
        frontend_inputs = {
            "phone": batch["phone"],
            "tone": batch["tone"],
            "word_seg": batch["word_seg"]
        }
        dur_ctx = torch.zeros_like(batch["phone"])
        dur_ctx[:, :batch['prompt_phone_durations'].shape[1]] = batch['prompt_phone_durations']
        dur_ctx = torch.clamp(dur_ctx, min=0, max=127)
        dur_len = batch["text_lens"]
        with torch.autocast(device_type="cuda", enabled=True):
            x = self.model(dur_ctx, frontend_inputs, dur_len)
            x /= scale
            x = torch.clamp(torch.round(x), min=1).long().squeeze()
        return x[batch['prompt_phone_durations'].shape[1]:]
    predict = inference

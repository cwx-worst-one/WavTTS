import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import wandb

from samantha.utils.hparams import DotDict
from recipes.bark.lit_modules.sample import sample
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric
import logging


logger = logging.getLogger(__name__)



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        logits_criterion_cls,
        dense_criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=True,
        stop_token_loss_weight=1.0,
        use_lang_id=False,
        use_spk_id=False,
        use_phoneme_loss=False,
        use_ctc_loss=False,
        freeze_text_encoder=False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.logits_criterion = logits_criterion_cls()
        self.dense_criterion = dense_criterion_cls()
        self.requires = {}

        if checkpointing:
            self.model.gradient_checkpointing_enable()

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
                mel = batch["mel"]
                mel_len = batch["mel_lens"]

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            ret_dict = self.model(frontend_inputs, mel, mel_len)
            mel_loss = F.mse_loss(ret_dict['pred_mel'].transpose(1, 2), mel)
        
        self.log_dict(
            {
                "mel_loss": mel_loss.item(),
            },
            prog_bar=True,
            sync_dist=True)
    
        return mel_loss

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
    def inference_from_text(self, batch, tokenizer):
        return None

    predict = inference_from_text
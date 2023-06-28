import pytorch_lightning as pl
import torch
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from einops import rearrange
from samantha.utils.hparams import DotDict


class Temperature(torch.nn.Module):

    def __init__(self, init_values=1.0/0.07, clip_values=100.0):
        super().__init__()
        self.clip_values = clip_values
        self.temp = torch.nn.Parameter(torch.FloatTensor([np.log(init_values)]))

    def forward(self):
        return self.temp.exp().clamp(0, self.clip_values)


class CLVPModule(pl.LightningModule):

    def __init__(
        self,
        text_model_cls,
        audio_model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        checkpointing=True,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.text_model = text_model_cls()
        self.audio_model = audio_model_cls()
        self.temp = Temperature()

        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}

        torch._C._jit_set_bailout_depth(0)
        if checkpointing:
            self.text_model.gradient_checkpointing_enable()
            self.audio_model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        return

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]AcousticModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wav_ids, text_ids, wav_lengths, text_lengths = batch

        with self.profiler.profile("[LightningModule]AcousticModule.model_forward"):
            wav_vec = self.audio_model(wav_ids, wav_lengths)
            text_vec = self.text_model(text_ids, text_lengths)
            #'''
            # gather all results
            wav_vec = self.all_gather(wav_vec, sync_grads=True) # [ws, bs, dim]
            text_vec = self.all_gather(text_vec, sync_grads=True) # [ws, bs, dim]
            # slice and compute loss
            ws, bs, dim = wav_vec.size()
            wav_vec = rearrange(wav_vec, "w b d -> (w b) d")
            text_vec = rearrange(text_vec, "w b d -> (w b) d")
            beg = self.global_rank * bs
            end = beg + bs
            # calculate loss
            temp = self.temp()
            logits1 = torch.matmul(wav_vec[beg:end], text_vec.t()) * temp # [bs, d] x [d, bs*ws] = [bs, bs*ws]
            logits2 = torch.matmul(text_vec[beg:end], wav_vec.t()) * temp # [bs, d] x [d, bs*ws] = [bs, bs*ws]
            loss1 = torch.nn.functional.cross_entropy(logits1, torch.arange(beg, end).to(logits1.device))
            loss2 = torch.nn.functional.cross_entropy(logits2, torch.arange(beg, end).to(logits2.device))
            loss = (loss1 + loss2) / 2
            # accuracy
            acc1 = logits1.argmax(dim=1) == torch.arange(beg, end).to(logits1.device)
            acc2 = logits2.argmax(dim=1) == torch.arange(beg, end).to(logits2.device)
            acc1 = acc1.sum() / bs * 100
            acc2 = acc2.sum() / bs * 100
            #'''
            '''
            temp = self.temp()
            logits = torch.mm(wav_vec, text_vec.t()) * temp
            targets = torch.arange(logits.size(0), device=logits.device)
            loss1 = torch.nn.functional.cross_entropy(logits, targets)
            loss2 = torch.nn.functional.cross_entropy(logits.t(), targets)
            loss = (loss1 + loss2) / 2
            acc1 = (logits.argmax(dim=1) == targets).float().mean() * 100
            acc2 = (logits.argmax(dim=0) == targets).float().mean() * 100
            '''

        if self.local_rank == 0:
            print("CLVP Training, V1...")

        self.log_dict(
            {"tr_loss": loss.item(), "temp": temp.item(), "acc1": acc1.item(), "acc2": acc2.item()},
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(
            list(self.text_model.parameters()) + list(self.audio_model.parameters()) + list(self.temp.parameters())
        )
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

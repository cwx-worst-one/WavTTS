import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
import deepspeed

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
from easydict import EasyDict

from samantha.utils.model_metric import ModelMetric
from recipes.umm_062.modules.criterion_ss import mel_spectrogram, MultiResolutionSTFTLoss
from recipes.umm_062.modules.criterion_ss import discriminator_loss, generator_loss, feature_loss
from recipes.umm_062.modules.criterion import STFTLoss


class UMMSSModule(pl.LightningModule):

    def __init__(
        self,
        g_model_cls,
        d_model_cls,
        scheduler_cls,
        extra_params=None,
        required_modules=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.g_model = g_model_cls()
        self.d_model = d_model_cls()
        self.scheduler_cls = scheduler_cls
        self.extra_params = EasyDict(extra_params)
        # disable automatic optimization for GAN training
        self.automatic_optimization = False
        self.rec_criterion = STFTLoss()
        self.metric_fn = MultiResolutionSTFTLoss()

    def setup(self, stage: str) -> None:
        pass

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_features(self, batch):
        audio = batch["audio"].squeeze(1)
        audio = self.g_model.preprocessing(audio)
        pad_len = audio.shape[-1] % self.extra_params.hop_length
        if pad_len > 0:
            pad_len = self.extra_params.hop_length - pad_len
        audio = torch.nn.functional.pad(audio, (0, pad_len))
        batch["audio"] = audio
        return batch
        
    def training_step(self, batch, batch_idx):
        self.prepare_features(batch)
        if self.extra_params.version == 'v1':
            self.training_step_v1(batch, batch_idx)
        elif self.extra_params.version == 'v2':
            self.training_step_v2(batch, batch_idx)
        else:
            raise Exception("Not supported version")

    def training_step_v1(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()
        log_dict = {}
        # generator output
        generator_out = self.g_model(batch)

        ### Train D ###
        wavs_r, wavs_g = generator_out['gt_wav'], generator_out['gen_wav']
        discriminator_out = self.train_discriminator_v1(wavs_r, wavs_g)
        loss_d = discriminator_out['loss_d']
        log_dict['loss_d'] = loss_d
        optim_d.zero_grad()
        self.manual_backward(loss_d)
        grad_d = torch.nn.utils.clip_grad_norm_(self.d_model.parameters(), 100.0)
        scheduler_d.step()
        optim_d.step()

        ### Train G ###
        generator_out = self.train_generator_v1(batch, generator_out)
        loss_g = generator_out['loss_g']
        w_kl = max(32 * (1 - self.trainer.global_step // 2 / 2000), 16.0)
        total_loss_g = 1 * loss_g + \
                       20 * generator_out['fmap_loss'] + \
                       50 * generator_out['mel_loss'] + \
                       w_kl * generator_out['kl_loss'] + \
                       10 * generator_out['mel_z_loss']
        log_dict['loss_g'] = generator_out['loss_g']
        log_dict['fmap'] = generator_out['fmap_loss']
        log_dict['mel'] = generator_out['mel_loss']
        log_dict['kl'] = generator_out['kl_loss'] * 32
        log_dict['mel_z'] = generator_out['mel_z_loss']
        log_dict['bsz'] = batch["audio"].shape[0]
        optim_g.zero_grad()
        self.manual_backward(total_loss_g)
        cnn_params = [p for name, p in self.g_model.named_parameters() if 'llama' not in name]
        gpt_params = [p for name, p in self.g_model.named_parameters() if 'llama' in name]
        grad_cnn = torch.nn.utils.clip_grad_norm_(cnn_params, 1000.0)
        grad_gpt = torch.nn.utils.clip_grad_norm_(gpt_params, 1.0)
        scheduler_g.step()
        optim_g.step()
        # grad norm log
        log_dict['grad_cnn'] = grad_cnn
        log_dict['grad_gpt'] = grad_gpt
        log_dict['grad_d'] = grad_d
        # old metric
        with torch.no_grad():
            mag, sc = self.metric_fn(wavs_r, wavs_g)
            log_dict['mag'] = mag
            log_dict['sc'] = sc
        # VAE monitor
        log_dict['logs'] = generator_out['logs'].detach().mean()
        log_dict['mean'] = generator_out['m'].detach().mean()

        log_dict['step'] = self.trainer.global_step // 2
        log_dict['training/loss'] = total_loss_g
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=False
        )

    def train_discriminator_v1(self, wavs_r, wavs_g):
        ret_dict = {}
        y_d_rs, y_d_gs, _, _ = self.d_model(wavs_r.detach(), wavs_g.detach())
        loss_d, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)
        ret_dict['loss_d'] = loss_d
        return ret_dict

    def train_generator_v1(self, batch, generator_out):
        wavs_r, wavs_g = generator_out['gt_wav'], generator_out['gen_wav']
        text_ids = batch["token"]
        text_lens = (text_ids > 0).sum(dim=1).long()
        # Mel loss
        mel_loss = 0.0
        n_mels = [5, 10, 20, 40, 80, 160, 320]
        n_mel_winlens= [32, 64, 128, 256, 512, 1024, 2048]
        for mel_dim, mel_winlen in zip(n_mels, n_mel_winlens):
            n_fft = mel_winlen
            hop_size = n_fft // 4
            mel_r = mel_spectrogram(wavs_r.squeeze(1),
                                    n_fft=n_fft,
                                    num_mels=mel_dim,
                                    sampling_rate=self.extra_params.sample_rate,
                                    hop_size=hop_size,
                                    win_size=mel_winlen,
                                    fmin=0,
                                    fmax=self.extra_params.sample_rate / 2).clamp(1e-5).log10()
            mel_g = mel_spectrogram(wavs_g.squeeze(1),
                                    n_fft=n_fft,
                                    num_mels=mel_dim,
                                    sampling_rate=self.extra_params.sample_rate,
                                    hop_size=hop_size,
                                    win_size=mel_winlen,
                                    fmin=0,
                                    fmax=self.extra_params.sample_rate / 2).clamp(1e-5).log10()
            loss = self.rec_criterion(mel_r, mel_g)['stft_loss']
            mel_loss += loss
        generator_out['mel_loss'] = mel_loss

        # GAN loss
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.d_model(wavs_r, wavs_g)
        loss_g, loss_g_items = generator_loss(y_d_gs)
        generator_out['loss_g'] = loss_g
        fmap_loss, fmap_loss_items = feature_loss(fmap_rs, fmap_gs, dynamic=True)
        generator_out['fmap_loss'] = fmap_loss
        # KL
        generator_out['kl_loss'] = generator_out['kl_loss']
        generator_out['mel_z_loss'] = self.rec_criterion(generator_out['gt_mel'], generator_out['mel_z'])['stft_loss']
        return generator_out

    def configure_optimizers(self):
        params1 = []
        params2 = []
        for name, params in self.g_model.named_parameters():
            if 'llama' in name or '_encoder' in name or 'flow' in name:
                params1.append(params)
            else:
                params2.append(params)
        g_params = [
            {"params": params1, "betas": [0.9, 0.95], "weight_decay": 0.1},
            {"params": params2, "betas": [0.8, 0.99], "weight_decay": 0.01},
        ]
        optim_g = deepspeed.ops.adam.FusedAdam(g_params, 2e-4, eps=1e-8)
        optim_d = deepspeed.ops.adam.FusedAdam(self.d_model.parameters(), 2e-4, betas=[0.8, 0.99], weight_decay=0.01, eps=1e-8)
        return [optim_g, optim_d] , [self.scheduler_cls(optim_g), self.scheduler_cls(optim_d)]

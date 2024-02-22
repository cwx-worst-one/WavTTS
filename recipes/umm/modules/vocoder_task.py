import argparse
import random
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_value_

from recipes.umm.models.voc_modules.commons.stft_loss import MultiResolutionSTFTLoss
from recipes.umm.models.voc_modules.hifigan.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    generator_loss, discriminator_loss
from recipes.umm.models.voc_modules.hifigan.mel_utils import mel_spectrogram
from recipes.umm.models.voc_modules.sa_melgan.modules import Generator, MultiPeriodDiscriminator
from recipes.umm.models.voc_modules.univnet.mrd import MultiResolutionDiscriminator
from recipes.umm.utils.mel_utils import torch_wav2spec


class MelGANVocoder(pl.LightningModule):
    def __init__(self, model_hp, save_hparams=True):
        super().__init__()
        if save_hparams:
            self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_hp = hparams = model_hp
        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)
        self.model_gen = Generator(
            args.n_mel_channels,
            hparams['ngf'],
            args.n_residual_layers,
            args.num_band,
            args,
            args.up_sample)
        self.model_disc = nn.ModuleDict()
        self.model_disc['mpd'] = MultiPeriodDiscriminator(hparams['mpd'])
        self.model_disc['msd'] = MultiScaleDiscriminator(hparams)
        self.model_disc['mrd'] = MultiResolutionDiscriminator(hparams)
        self.stft_loss = MultiResolutionSTFTLoss()

    def configure_optimizers(self):
        hparams = self.model_hp
        optimizer_gen = torch.optim.AdamW(self.model_gen.parameters(), lr=hparams['lr'],
                                          betas=[hparams['adam_b1'], hparams['adam_b2']])
        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(),
                                           lr=hparams.get('disc_lr', hparams['lr']),
                                           betas=[hparams['adam_b1'], hparams['adam_b2']])
        scheduler_gen = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_gen,
            **hparams["generator_scheduler_params"])
        scheduler_disc = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_disc,
            **hparams["discriminator_scheduler_params"])
        return [optimizer_gen, optimizer_disc], \
            [{"scheduler": scheduler_gen, "interval": "step"},
             {"scheduler": scheduler_disc, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        hparams = self.model_hp
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()
        t_begin = random.randint(0, batch['audio'].shape[2] - hparams['max_samples'])
        t_end = t_begin + hparams['max_samples']
        y = batch[['audio', 'audio_inst', 'audio_vocal'][random.randint(0, 2)]][:12, :, t_begin:t_end]
        f0 = None
        mel = torch_wav2spec(y[:, 0]).transpose(1, 2)[..., :hparams['max_samples'] // hparams['hop_size']]
        self.toggle_optimizer(optim_g)
        y_hat = self.model_gen(mel, f0)
        y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
        y_hat_mel = mel_spectrogram(y_hat.squeeze(1), hparams).transpose(1, 2)
        loss_dict = {}
        loss_dict['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
        _, y_mpd_fake, _, _ = self.model_disc['mpd'](y, y_hat)
        _, y_msd_fake, _, _ = self.model_disc['msd'](y, y_hat)
        loss_dict['a_p'] = generator_loss(y_mpd_fake) * hparams['lambda_adv'] * hparams.get('lambda_mpd', 1.0)
        loss_dict['a_s'] = generator_loss(y_msd_fake) * hparams['lambda_adv'] * hparams.get('lambda_msd', 1.0)
        if hparams['use_ms_stft']:
            loss_dict['sc'], loss_dict['mag'] = self.stft_loss(y.squeeze(1), y_hat.squeeze(1))
        y_hat = y_hat.detach()

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(sum([
            x for x in loss_dict.values()
            if isinstance(x, torch.Tensor) and x.requires_grad and x.grad_fn is not None
        ]))
        clip_grad_value_(self.model_gen.parameters(), 1.0)
        optim_g.step()
        sched_g.step()
        self.untoggle_optimizer(optim_g)
        losses_gen = loss_dict

        self.toggle_optimizer(optim_d)
        loss_dict = {}
        # MPD
        y_mpd_real, y_mpd_fake, _, _ = self.model_disc['mpd'](y, y_hat.detach())
        loss_dict['r_p'], loss_dict['f_p'] = discriminator_loss(y_mpd_real, y_mpd_fake)
        # MSD
        y_msd_real, y_msd_fake, _, _ = self.model_disc['msd'](y, y_hat.detach())
        loss_dict['r_s'], loss_dict['f_s'] = discriminator_loss(y_msd_real, y_msd_fake)
        # MRD
        if hparams['use_mrd']:
            y_mrd_real = [x[1] for x in self.model_disc['mrd'](y)]
            y_mrd_fake = [x[1] for x in self.model_disc['mrd'](y_hat.detach())]
            loss_dict['r_r'], loss_dict['f_r'] = discriminator_loss(y_mrd_real, y_mrd_fake)

        # disciminator backward
        optim_d.zero_grad()
        self.manual_backward(
            sum([
                x for x in loss_dict.values()
                if isinstance(x, torch.Tensor) and x.requires_grad and x.grad_fn is not None
            ]))
        clip_grad_value_(self.model_disc.parameters(), 1.0)
        optim_d.step()
        sched_d.step()
        self.untoggle_optimizer(optim_d)

        loss_dict.update(losses_gen)
        loss_dict = {f"tr/{k}": v for k, v in loss_dict.items()}
        # logging
        for param_group in optim_g.param_groups:
            lr = param_group['lr']
        loss_dict['aux/lr'] = lr
        loss_dict['aux/bs'] = y.shape[0]
        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

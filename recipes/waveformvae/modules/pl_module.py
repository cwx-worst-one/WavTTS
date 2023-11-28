import pytorch_lightning as pl
import torch
import time
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True

from recipes.waveformvae.utils import commons
from recipes.waveformvae.utils.audio_utils import (
    mel_spectrogram_torch,
    spectrogram_torch,
)
from recipes.waveformvae.utils.losses import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)


class WaveformVAEModule(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        train_cls,
        data_cls,
        val_output_samples_dir,
    ):
        super().__init__()
        print(generator)
        # all parameters in ctor will be saved to self.hparams
        # self.save_hyperparameters(ignore=["generator", "discriminator", "optimizer_cls", "scheduler_d_cls"])
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator, self.discriminator = generator, discriminator
        self.optimizer_g_cls, self.optimizer_d_cls = (
            self.hparams.optimizer_g_cls,
            self.hparams.optimizer_d_cls,
        )
        self.scheduler_g_cls, self.scheduler_d_cls = (
            self.hparams.scheduler_g_cls,
            self.hparams.scheduler_d_cls,
        )
        self.train_cls = self.hparams.train_cls
        self.data_cls = self.hparams.data_cls
        self.stft_loss = MultiResolutionSTFTLoss()

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # custom recorder for training step due to GAN training
        self.current_step = 0
        self.last_time = time.time()
        self.flops = 0
        device_name = torch.cuda.get_device_name()
        if 'A100' in device_name or 'A800' in device_name:
            self.device_FLOPS = 312e12
        elif 'H100' in device_name or 'H800' in device_name:
            self.device_FLOPS = 989e12
        else:
            raise RuntimeError('unknow cuda device name: ', device_name)

    def configure_optimizers(self):
        # generator
        optimizer_g = self.optimizer_g_cls(self.generator.parameters())
        scheduler_g = self.scheduler_g_cls(optimizer_g)
        # discriminator
        optimizer_d = self.optimizer_d_cls(self.discriminator.parameters())
        scheduler_d = self.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
        # return {
        #     "optimizer": [optimizer_g, optimizer_d],
        #     "lr_scheduler": {"scheduler": [scheduler_g, scheduler_d], "interval": "step"}
        # }
        # return [
        #     {'optimizer': optimizer_g, 'lr_scheduler': scheduler_g, 'frequency': 1, 'interval': 'epoch'},
        #     {'optimizer': optimizer_d, 'lr_scheduler': scheduler_d, 'frequency': 1, 'interval': 'epoch'}
        # ]

    def training_step_G_and_D(self, batch, batch_idx):
        # get optimizor and scheduler
        net_g, net_d = self.generator, self.discriminator
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        wav = batch
        mel = mel_spectrogram_torch(
            wav.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("n_mel_channels"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length"),
            self.data_cls.get("win_length"),
            self.data_cls.get("mel_fmin"),
            self.data_cls.get("mel_fmax"),
        )
        spec = spectrogram_torch(
            wav.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length"),
            self.data_cls.get("win_length"),
        )

        #######################
        #      Generator      #
        #######################
        self.toggle_optimizer(optim_g)
        wav_hat, mel_z, kl_loss, mean, logs = net_g(wav, spec, mel)
        mel_hat = mel_spectrogram_torch(
            wav_hat.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("n_mel_channels"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length"),
            self.data_cls.get("win_length"),
            self.data_cls.get("mel_fmin"),
            self.data_cls.get("mel_fmax"),
        )
        mel_40hz = mel_spectrogram_torch(
            wav.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("n_mel_channels"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length") * 2,  # 40hz
            self.data_cls.get("win_length"),
            self.data_cls.get("mel_fmin"),
            self.data_cls.get("mel_fmax"),
        )
        # generator loss
        loss_sc, loss_mag = self.stft_loss(wav.squeeze(1), wav_hat.squeeze(1))
        loss_mel = F.l1_loss(mel, mel_hat)
        loss_mel_z = F.l1_loss(mel_40hz, mel_z)
        loss_kl = kl_loss

        loss_gen_all = (
            self.train_cls.get("c_mel") * loss_mel_z
            + self.train_cls.get("c_mel") * loss_mel
            + self.train_cls.get("c_stft") * loss_sc
            + self.train_cls.get("c_stft") * loss_mag
            + self.train_cls.get("c_kl") * loss_kl
        )

        # adversarial loss
        if self.current_step > self.train_cls.get("discriminator_train_start_steps"):
            y_d_hat_r, y_d_hat_g, _, fmap_g = net_d(wav, wav_hat)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            with torch.no_grad():
                y_d_hat_r, y_d_hat_g, fmap_r, _ = net_d(wav, wav_hat)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen_all += (
                self.train_cls.get("c_gen") * loss_gen
                + self.train_cls.get("c_fm") * loss_fm
            )

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), 1.0)
        optim_g.step()
        scheduler_g.step()
        self.untoggle_optimizer(optim_g)

        #######################
        #    Discriminator    #
        #######################
        if self.current_step > self.train_cls.get("discriminator_train_start_steps"):
            self.toggle_optimizer(optim_d)
            with torch.no_grad():
                wav_hat, mel_z, kl_loss, mean, logs = net_g(wav, spec)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wav, wav_hat.detach())

            # discriminator loss
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            # disciminator backward
            optim_d.zero_grad()
            self.manual_backward(loss_disc_all)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), 1.0)
            optim_d.step()
            scheduler_d.step()
            self.untoggle_optimizer(optim_d)

        # log
        if self.current_step > self.train_cls.get("discriminator_train_start_steps"):
            self.log_dict(
                {
                    "train/loss_disc": loss_disc,
                    "train/loss_gen": loss_gen,
                    "train/loss_fm": loss_fm,
                    "train/loss_mel": loss_mel,
                    "train/loss_sc": loss_sc,
                    "train/loss_mag": loss_mag,
                    "train/loss_kl": loss_kl,
                    "train/grad_norm_d": grad_norm_d,
                    "train/grad_norm_g": grad_norm_g,
                    "train/loss_mel_z": loss_mel_z,
                    "stats/logs": logs.detach().mean(),
                    "stats/mean_mean": mean.detach().mean(),
                    "stats/mean_std": mean.detach().std(),
                    "aux/step": self.current_step,
                    "aux/opt_lr": optim_g.param_groups[0]["lr"],
                    "aux/sch_lr": scheduler_g.get_last_lr()[0],
                    "aux/batch": batch.size(0),
                },
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        else:
            self.log_dict(
                {
                    "train/loss_mel": loss_mel,
                    "train/loss_sc": loss_sc,
                    "train/loss_mag": loss_mag,
                    "train/loss_kl": loss_kl,
                    "train/grad_norm_g": grad_norm_g,
                    "train/loss_mel_z": loss_mel_z,
                    "stats/logs": logs.detach().mean(),
                    "stats/mean_mean": mean.detach().mean(),
                    "stats/mean_std": mean.detach().std(),
                    "aux/step": self.current_step,
                    "aux/opt_lr": optim_g.param_groups[0]["lr"],
                    "aux/sch_lr": scheduler_g.get_last_lr()[0],
                    "aux/batch": batch.size(0),
                },
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.current_step += 1

    def training_step_D_and_G(self, batch, batch_idx):
        flops = 0.0
        elapsed = time.time() - self.last_time
        mfu = self.flops * 3 / elapsed / self.device_FLOPS # extra 2x for backward.
        self.last_time = time.time()
        # get optimizor and scheduler
        net_g, net_d = self.generator, self.discriminator
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        wav = batch
        mel = mel_spectrogram_torch(
            wav.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("n_mel_channels"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length"),
            self.data_cls.get("win_length"),
            self.data_cls.get("mel_fmin"),
            self.data_cls.get("mel_fmax"),
        )

        spec = spectrogram_torch(
            wav.squeeze(1),
            self.data_cls.get("filter_length"),
            self.data_cls.get("sampling_rate"),
            self.data_cls.get("hop_length"),
            self.data_cls.get("win_length"),
        )

        # train discriminator
        wav_hat, mel_z, kl_loss, mean, logs = net_g(wav, spec, mel)
        net_g_flops, __ = net_g.flops(wav.shape, spec.shape, mel.shape)
        flops += net_g_flops

        mel_hat = mel_spectrogram_torch(wav_hat.squeeze(1),
                                        self.data_cls.get('filter_length'),
                                        self.data_cls.get('n_mel_channels'),
                                        self.data_cls.get('sampling_rate'),
                                        self.data_cls.get('hop_length'),
                                        self.data_cls.get('win_length'),
                                        self.data_cls.get('mel_fmin'),
                                        self.data_cls.get('mel_fmax'))
        mel_40hz = mel_spectrogram_torch(wav.squeeze(1),
                                         self.data_cls.get('filter_length'),
                                         self.data_cls.get('n_mel_channels'),
                                         self.data_cls.get('sampling_rate'),
                                         self.data_cls.get('hop_length') * 2,  # 40hz
                                         self.data_cls.get('win_length'),
                                         self.data_cls.get('mel_fmin'),
                                         self.data_cls.get('mel_fmax'))

        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = net_d(wav, wav_hat.detach())

        net_d_flops = net_d.flops(wav.shape, wav_hat.shape)
        flops += net_d_flops
        self.flops = flops

        # discriminator loss
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

        # disciminator backward
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        grad_norm_d = commons.clip_grad_value_v2_(net_d.parameters(), 1.0)
        optim_d.step()
        scheduler_d.step()
        self.untoggle_optimizer(optim_d)

        # train generator
        self.toggle_optimizer(optim_g)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, wav_hat)

        # generator loss
        loss_sc, loss_mag = self.stft_loss(wav.squeeze(1), wav_hat.squeeze(1))
        loss_mel = F.l1_loss(mel, mel_hat)
        loss_mel_z = F.l1_loss(mel_40hz, mel_z)
        loss_kl = kl_loss
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen = loss_gen
        loss_gen_all = (
            self.train_cls.get("c_gen") * loss_gen
            + self.train_cls.get("c_fm") * loss_fm
            + self.train_cls.get("c_mel") * loss_mel_z
            + self.train_cls.get("c_mel") * loss_mel
            + self.train_cls.get("c_stft") * loss_sc
            + self.train_cls.get("c_stft") * loss_mag
            + self.train_cls.get("c_kl") * loss_kl
        )

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        grad_norm_g = commons.clip_grad_value_v2_(net_g.parameters(), 1.0)
        optim_g.step()
        scheduler_g.step()
        self.untoggle_optimizer(optim_g)

        # log
        self.log_dict(
            {
                "train/loss_disc": loss_disc,
                "train/loss_gen": loss_gen,
                "train/loss_fm": loss_fm,
                "train/loss_mel": loss_mel,
                "train/loss_sc": loss_sc,
                "train/loss_mag": loss_mag,
                "train/loss_kl": loss_kl,
                "train/grad_norm_d": grad_norm_d,
                "train/grad_norm_g": grad_norm_g,
                "train/loss_mel_z": loss_mel_z,
                "stats/logs": logs.detach().mean(),
                "stats/mean_mean": mean.detach().mean(),
                "stats/mean_std": mean.detach().std(),
                "aux/step": self.current_step,
                "aux/opt_lr": optim_g.param_groups[0]["lr"],
                "aux/sch_lr": scheduler_g.get_last_lr()[0],
                "aux/batch": batch.size(0),
                "train/mfu": mfu,
                "train/max_memory_alloc": torch.cuda.max_memory_allocated() / 2**30,
                "train/malloc_retries": torch.cuda.memory_stats()["num_alloc_retries"],
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.current_step += 1

    def training_step(self, batch, batch_idx):
        if self.train_cls.get("train_mode") == "G_and_D":
            self.training_step_G_and_D(batch, batch_idx)
        elif self.train_cls.get("train_mode") == "D_and_G":
            self.training_step_D_and_G(batch, batch_idx)
        else:
            raise NotImplementedError

    # def forward(self, x):
    #     raise NotImplementedError

    # def validation_step(self, batch, batch_idx):
    #     raise NotImplementedError

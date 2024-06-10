import os

import pytorch_lightning as pl
import soundfile as sf
import torch
import torch_museval

from recipes.soundstream.utils.losses import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from recipes.soundstream.utils.balancer import Balancer, EMA


class SoundstreamModule(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        balancer,
        quant_token_num: int,
        generator_warmup_steps: int,
        sample_rate: int,
        val_output_samples_dir: str,
        optimizer_cls,
        gen_lr_scheduler_cls,
        dis_lr_scheduler_cls,
    ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        self.discriminator = discriminator

        # stft loss
        self.stft_criterion = MultiResolutionSTFTLoss()

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # custom recorder for training step due to GAN training
        self.current_step = 0

        if self.local_rank == 0:
            os.makedirs(val_output_samples_dir, exist_ok=True)
        

    def configure_optimizers(self):
        # generator
        optimizer_g = self.hparams.optimizer_cls(self.generator.parameters())
        scheduler_g = self.hparams.gen_lr_scheduler_cls(optimizer_g)
        # discriminator
        optimizer_d = self.hparams.optimizer_cls(self.discriminator.parameters())
        scheduler_d = self.hparams.dis_lr_scheduler_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def _get_token_usage_rate(self, quant_index):
        one_hot = torch.nn.functional.one_hot(
            quant_index[0].reshape(-1), self.hparams.quant_token_num
        )
        one_hot = self.all_gather(one_hot).sum(dim=0)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        rate = 100 * one_hot.sum() / self.hparams.quant_token_num
        return rate

    def training_step(self, batch, batch_idx):
        # get optimizor and scheduler
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # train discriminator
        wavs_g, quant_loss, quant_index, encoder_out = self.generator(
            batch["audio"], warmup=batch_idx < 1e4
        )
        # get token usage rate
        rate = self._get_token_usage_rate(quant_index)

        self.toggle_optimizer(opt_d)
        y_d_rs, y_d_gs, _, _ = self.discriminator(batch["audio"], wavs_g.detach())

        # d logit loss
        loss_d, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)
        total_loss_d = loss_d

        # warmup for generator
        if batch_idx > self.hparams.generator_warmup_steps:
            opt_d.zero_grad()
            self.manual_backward(total_loss_d)
            norm_d = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 1000.0
            )
            opt_d.step()
            sch_d.step()
        else:
            norm_d = torch.FloatTensor([0.0])

        self.untoggle_optimizer(opt_d)

        # train generator
        self.toggle_optimizer(opt_g)
        # mpd + mrd
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(batch["audio"], wavs_g)
        # g logit loss
        loss_g, loss_g_items = generator_loss(y_d_gs)

        sc_loss, mag_loss = self.stft_criterion(batch["audio"], wavs_g)

        # fmap loss
        fmap_loss, fmap_loss_items = feature_loss(fmap_rs, fmap_gs, dynamic=True)

        # total loss for generator
        if batch_idx > self.hparams.generator_warmup_steps:
            total_loss_g = {
                'generator_loss': loss_g,
                'quant_loss': quant_loss,
                'sc_loss': sc_loss,
                'mag_loss': mag_loss,
                'fmap_loss': fmap_loss,
                'wav_loss': (wavs_g - batch["audio"]).abs().mean()
            }
        else:
            total_loss_g = {
                'quant_loss': quant_loss,
                'sc_loss': sc_loss,
                'mag_loss': mag_loss,
                'wav_loss': (wavs_g - batch["audio"]).abs().mean()
            }

        opt_g.zero_grad()
        # self.manual_backward(total_loss_g)
        self.hparams.balancer.backward(total_loss_g, wavs_g)
        norm_g = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1000.0)
        opt_g.step()
        sch_g.step()
        self.untoggle_optimizer(opt_g)

        # log
        norm_w = (
            (self.generator.quant_vaes[0].embedding.weight.data ** 2)
            .sum(dim=1)
            .sqrt()
            .max()
        )
        self.log_dict(
            {
                "total_loss_d": total_loss_d,
                # "total_loss_g": total_loss_g,
                "sc_loss": sc_loss,
                "mag_loss": mag_loss,
                "fmap_loss": fmap_loss,
                "quant_loss": quant_loss,
                "norm_d": norm_d,
                "norm_g": norm_g,
                "norm_w": norm_w,
                "loss_g": loss_g,
                "loss_d": loss_d,
                "rate": rate,
                "step": self.current_step,
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.current_step += 1

    def validation_step(self, batch, batch_idx):
        wavs_g, _, _, _ = self.generator(batch["audio"], warmup=False)
        wavs_g = wavs_g.detach()

        # calculate SDR
        sdrs = []
        for wav_g, wav_o in zip(wavs_g, batch["audio"]):
            sdr, _, _, _ = torch_museval.evaluate(
                wav_g.T.unsqueeze(0).detach(),
                wav_o.T.unsqueeze(0).detach(),
            )
            sdr = torch.nanmedian(sdr)
            sdrs.append(sdr)
        
        # save the reconstructed wavs
        if not os.path.exists(f"{self.hparams.val_output_samples_dir}/origin"):
            os.makedirs(f"{self.hparams.val_output_samples_dir}/origin", exist_ok=True)
            for _id, wav in zip(batch["music_id"], batch["audio"]):
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/origin/{_id}.wav",
                    wav.cpu().numpy().T,
                    self.hparams.sample_rate,
                )

        for _id, wav in zip(batch["music_id"], wavs_g):
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.current_step}/{_id}.wav",
                wav.cpu().numpy().T,
                self.hparams.sample_rate,
            )

        results = {'ids': batch["music_id"], 'sdrs': sdrs}

        for _id, sdr in zip(batch["music_id"], sdrs):
            self.val_output_dict[_id] = {'sdr': sdr}

        return results

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # create folder based on current epoch
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}", exist_ok=True
        )
        self.val_output_dict = {}
        return


    def on_validation_epoch_end(self):
        # get results from all the gpu
        # NOTE: gather dict will get all the results from all the gpu under same key, might be redundant
        results = self.all_gather(self.val_output_dict).detach().cpu()

        # get median sdr
        # put values in result to list
        sdrs = []
        for k in results:
            sdrs.append(torch.mean(results[k]['sdr']))

        sdr = torch.median(torch.sort(torch.tensor(sdrs))[0])
        
        self.log_dict(
            {
                "val_sdr": sdr,
            },
            prog_bar=True,
            sync_dist=False,
            rank_zero_only=True,
        )
        if self.trainer.is_global_zero:
            print(f"Val SDR:", sdr)

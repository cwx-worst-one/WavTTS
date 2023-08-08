import os

import pytorch_lightning as pl
import soundfile as sf
import torch
import torch_museval

from recipes.soundstream.utils.losses import (
    MultiResolutionSTFTLoss,
    MelSpectrogramLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VocoderModule(pl.LightningModule):
    def __init__(
        self,
        seed: int,
        generator,
        discriminator,
        generator_warmup_steps: int,
        lambda_generator_loss: float,
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
        # for p in generator.encoder.parameters():
        #     p.requires_grad = False
        # for p in generator.mean_logvar_conv.parameters():
        #     p.requires_grad = False
        # self.generator.encoder.eval()
        # self.generator.mean_logvar_conv.eval()
        
        self.discriminator = discriminator

        # stft loss
        # self.stft_criterion = MultiResolutionSTFTLoss()
        self.mel_criterion = MelSpectrogramLoss(
            sample_rate=self.hparams.sample_rate,
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            loss_fn=torch.nn.L1Loss(),
            clamp_eps=1e-5,
            mag_weight=0.0,
            log_weight=1.0,
            pow=2.0,
            mel_fmins=[0, 0, 0, 0, 0, 0, 0],
            mel_fmaxes=[None, None, None, None, None, None, None],
        )

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # custom recorder for training step due to GAN training
        self.current_step = 0

        if self.local_rank == 0:
            os.makedirs(val_output_samples_dir, exist_ok=True)

    def on_fit_start(self):
        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)
    
    def _divide_params_group(self, model):
        no_decay = [
            "bn",
            "bias",
            "norm"
            "rotary",
            "embedding",
            ".g", # g in RMSNorm
        ]

        base_params = []
        no_decay_params = []
        for name, param in model.named_parameters(): 
            _found = False
            for k in no_decay:
                if k in name:
                    no_decay_params.append(param)
                    _found = True
                    break
            if not _found:
                base_params.append(param)

        return base_params, no_decay_params
        
    def configure_optimizers(self):
        # generator
        base_params_g, no_decay_params_g = self._divide_params_group(self.generator)

        optimizer_g = self.hparams.optimizer_cls(
            [{"params": base_params_g}, {"params": no_decay_params_g, "weight_decay": 0.0}],
        )
        scheduler_g = self.hparams.gen_lr_scheduler_cls(optimizer_g)
        # discriminator
        base_params_d, no_decay_params_d = self._divide_params_group(self.discriminator)
        optimizer_d = self.hparams.optimizer_cls(
            [{"params": base_params_d}, {"params": no_decay_params_d, "weight_decay": 0.0}],
        )
        scheduler_d = self.hparams.dis_lr_scheduler_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    
    def training_step(self, batch, batch_idx):
        if type(batch) is list:
            batch = {'audio': batch[0]}

        if len(batch['audio'].shape) == 2:
            batch['audio'] = batch['audio'].unsqueeze(1)
        
        # get optimizor and scheduler
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # train discriminator
        wavs_g, kl_loss, std_mean = self.generator(batch["audio"])
        self.toggle_optimizer(opt_d)
        y_d_rs, y_d_gs, _, _ = self.discriminator(batch["audio"], wavs_g.detach())

        # # d logit loss
        loss_d, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)
        total_loss_d = loss_d

        # # warmup for generator
        if batch_idx > self.hparams.generator_warmup_steps:
            opt_d.zero_grad()
            self.manual_backward(total_loss_d)
            norm_d = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 1.0
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

        mel_loss = self.mel_criterion(wavs_g, batch["audio"])

        # fmap loss
        fmap_loss, fmap_loss_items = feature_loss(fmap_rs, fmap_gs, dynamic=True)

        # total loss for generator
        if batch_idx > self.hparams.generator_warmup_steps:
            total_loss_g = (
                self.hparams.lambda_generator_loss * loss_g
                + 15 * mel_loss
                + 2 * fmap_loss
                + 4e-3 * kl_loss 
            )
            # total_loss_g = (
            #     # 7 * sc_loss
            #     7 * mel_loss
            #     + 2e-3 * kl_loss 
            # )
        else:
            total_loss_g = (
                15 * mel_loss
                + 4e-3 * kl_loss 
            )

        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        norm_g = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 100)
        opt_g.step()
        sch_g.step()
        self.untoggle_optimizer(opt_g)

        # log
        self.log_dict(
            {
                "total_loss_d": total_loss_d,
                "total_loss_g": total_loss_g,
                "mel": mel_loss,
                "fmap_loss": fmap_loss,
                "kl_loss": kl_loss,
                "std_mean": std_mean,
                "norm_d": norm_d,
                "norm_g": norm_g,
                "loss_g": loss_g,
                "loss_d": loss_d,
                "step": self.current_step,
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.current_step += 1

    def validation_step(self, batch, batch_idx):
        wavs_g, _, _ = self.generator(batch["audio"])

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
        results = self.all_gather(self.val_output_dict) 

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

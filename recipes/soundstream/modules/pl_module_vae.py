import os
import time

import pytorch_lightning as pl
import soundfile as sf
import torch
try:
    import torch_museval
except Exception as e:
    print('WARNING: torch_museval not installed. This is required if doing Soundstream training')
import torchaudio

from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.soundstream.utils.losses import (
    MultiResolutionSTFTLoss,
    MelSpectrogramLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from recipes.soundstream.utils.sample_pool import SamplePool
from recipes.soundstream.utils.flops import (
    FlopsProfiler,
    get_device_memory,
    get_device_flops,
    with_timeout,
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
        kl_weight: float,
        n_channels: int,
        dataloader_samplerate: int,
        encoder_samplerate: int,
        decoder_samplerate: int,
        sample_pool_size: int,
        train_batch_size: int,
        valid_batch_size: int,
        sample_length: int,
        val_output_samples_dir: str,
        optimizer_cls,
        gen_lr_scheduler_cls,
        dis_lr_scheduler_cls,
        precision=32,
        profiling_flops=True,
        mix_training=False
    ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.generator = generator
        if mix_training:
            for p in generator.encoder.parameters():
                p.requires_grad = False
            for p in generator.mean_logvar_conv.parameters():
                p.requires_grad = False
            self.generator.encoder.eval()
            self.generator.mean_logvar_conv.eval()

            if dataloader_samplerate != encoder_samplerate:
                self.enc_resampler = torchaudio.transforms.Resample(dataloader_samplerate, encoder_samplerate)
            if dataloader_samplerate != decoder_samplerate:
                self.dec_resampler = torchaudio.transforms.Resample(dataloader_samplerate, decoder_samplerate)

        self.discriminator = discriminator

        # stft loss
        # self.stft_criterion = MultiResolutionSTFTLoss()
        self.mel_criterion = MelSpectrogramLoss(
            sample_rate=decoder_samplerate,
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            loss_fn=torch.nn.L1Loss(),
            clamp_eps=1e-5,
            mag_weight=0.0,
            log_weight=1.0,
            pow=1.0,
            mel_fmins=[0, 0, 0, 0, 0, 0, 0],
            mel_fmaxes=[None, None, None, None, None, None, None],
        )

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # sample pool for efficient training
        self.sample_pool = SamplePool(
            data_samplerate=dataloader_samplerate,
            cache_size=sample_pool_size,
            batch_size=train_batch_size,
            length_samples=sample_length,
        )
        self.valid_batch_size = valid_batch_size

        # custom recorder for training step due to GAN training
        self.current_step = 0

        self.flops_prof = FlopsProfiler(self) if profiling_flops else None
        self.device_FLOPS = get_device_flops(precision=precision)
        self.total_flops = 0
        self.init_timestamp = time.time()
        self.last_timestamp = time.time()

    def setup(self, stage: str) -> None:
        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)

        if getattr(self, "enc_resampler", False):
            self.enc_resampler = self.enc_resampler.to(torch.device(f"cuda:{self.local_rank}"))

        if getattr(self, "dec_resampler", False):
            self.dec_resampler = self.dec_resampler.to(torch.device(f"cuda:{self.local_rank}"))

        if self.local_rank == 0:
            os.makedirs(self.hparams.val_output_samples_dir, exist_ok=True)
            os.makedirs(f"{self.hparams.val_output_samples_dir}/origin", exist_ok=True)

        self.init_timestamp = time.time()

    def _divide_params_group(self, model):
        no_decay = ["bn", "bias", "norm" "rotary", "embedding", ".g"]  # g in RMSNorm

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

    def load_state_dict(self, state_dict, strict: bool = True):
        new_state_dict = {}
        for name, weight in state_dict.items():
            if name.endswith("total_ops") or name.endswith("total_params"):
                continue
            new_state_dict[name] = weight
        return super().load_state_dict(new_state_dict, strict)

    def configure_optimizers(self):
        # generator
        base_params_g, no_decay_params_g = self._divide_params_group(self.generator)

        optimizer_g = self.hparams.optimizer_cls(
            [
                {"params": base_params_g},
                {"params": no_decay_params_g, "weight_decay": 0.0},
            ]
        )
        scheduler_g = self.hparams.gen_lr_scheduler_cls(optimizer_g)
        # discriminator
        base_params_d, no_decay_params_d = self._divide_params_group(self.discriminator)
        optimizer_d = self.hparams.optimizer_cls(
            [
                {"params": base_params_d},
                {"params": no_decay_params_d, "weight_decay": 0.0},
            ]
        )
        scheduler_d = self.hparams.dis_lr_scheduler_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    # @with_timeout(timeout=20)
    def training_step(self, batch, batch_idx):
        if self.flops_prof:
            self.flops_prof.start()

        begin_ts = time.time()
        self.trainer.strategy.barrier()
        barrier_time = time.time() - begin_ts

        if type(batch) is list:
            batch = {"audio": batch[0]}

        if len(batch["audio"].shape) == 2:
            batch["audio"] = batch["audio"].unsqueeze(1)

        # process batch
        batch["audio"] = self.sample_pool.process(batch["audio"])

        if getattr(self, "enc_resampler", False):
            input_audio = self.enc_resampler(batch["audio"])
            input_audio = input_audio.mean(dim=1, keepdims=True)
        else:
            input_audio = batch["audio"]

        if getattr(self, "dec_resampler", False):
            gt_audio = self.dec_resampler(batch["audio"])
        else:
            gt_audio = batch["audio"]

        # get optimizor and scheduler
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # train discriminator
        wavs_g, kl_loss, std_mean = self.generator(input_audio)
        self.toggle_optimizer(opt_d)
        y_d_rs, y_d_gs, _, _ = self.discriminator(gt_audio, wavs_g.detach())

        # # d logit loss
        loss_d, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)
        total_loss_d = loss_d

        # # warmup for generator
        # if batch_idx > self.hparams.generator_warmup_steps:
        opt_d.zero_grad()
        self.manual_backward(total_loss_d)
        norm_d = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), 1.0
        )
        opt_d.step()
        sch_d.step()
        # else:
        #     norm_d = torch.FloatTensor([0.0])

        self.untoggle_optimizer(opt_d)

        # train generator
        self.toggle_optimizer(opt_g)
        # mpd + mrd
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(gt_audio, wavs_g)

        # g logit loss
        loss_g, loss_g_items = generator_loss(y_d_gs)

        mel_loss = self.mel_criterion(wavs_g, gt_audio)

        # fmap loss
        fmap_loss, fmap_loss_items = feature_loss(fmap_rs, fmap_gs, dynamic=True)

        # total loss for generator
        if batch_idx > self.hparams.generator_warmup_steps:
            total_loss_g = (
                self.hparams.lambda_generator_loss * loss_g
                + 15 * mel_loss
                + 2 * fmap_loss
                + self.hparams.kl_weight * kl_loss
            )
            # total_loss_g = recipes/soundstream/modules/pl_module_vae.py(
            #     # 7 * sc_loss
            #     7 * mel_loss
            #     + 2e-3 * kl_loss
            # )
        else:
            total_loss_g = (
                self.hparams.lambda_generator_loss * loss_g
                + 15 * mel_loss
                + 2 * fmap_loss
                + 0 * kl_loss
            )

        opt_g.zero_grad()
        self.manual_backward(total_loss_g)
        norm_g = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 100)
        opt_g.step()
        sch_g.step()
        self.untoggle_optimizer(opt_g)

        stats_dict = {
            "training/loss": total_loss_d,
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
        }

        # mem & mfu stats
        if self.flops_prof:
            self.flops_prof.stop()
            self.total_flops += self.flops_prof.flops
            mem_gb, malloc_retries = get_device_memory()
            elapsed_time = time.time() - self.last_timestamp
            total_time = time.time() - self.init_timestamp

            stats_dict.update(
                {
                    "mem_gb": mem_gb,
                    "malloc_retries": malloc_retries,
                    "flops": self.flops_prof.flops,
                    "elapsed_time": elapsed_time,
                    "barrier_time": barrier_time,
                    "mfu": self.flops_prof.flops / elapsed_time / self.device_FLOPS,
                    "mfu_avg": self.total_flops / total_time / self.device_FLOPS,
                }
            )

        # log
        self.log_dict(stats_dict, prog_bar=True, sync_dist=True, rank_zero_only=True)

        self.current_step += 1
        self.last_timestamp = time.time()

    def validation_step(self, batch, batch_idx):
        if getattr(self, "enc_resampler", False):
            input_audio = self.enc_resampler(batch["audio"])
            input_audio = input_audio.mean(dim=1, keepdims=True)
        else:
            input_audio = batch["audio"]

        if getattr(self, "dec_resampler", False):
            gt_audio = self.dec_resampler(batch["audio"])
        else:
            gt_audio = batch["audio"]

        wavs_g, _, _ = self.generator(input_audio)
        # calculate SDR
        sdrs = []
        for wav_g, wav_o in zip(wavs_g, gt_audio):
            try:
                sdr, _, _, _ = torch_museval.evaluate(
                    wav_g.T.unsqueeze(0).detach(), wav_o.T.unsqueeze(0).detach(),
                    win=self.hparams.decoder_samplerate,
                    hop=self.hparams.decoder_samplerate,
                )
                sdr = torch.nanmedian(sdr)
                sdrs.append(sdr)
            except:
                # sometimes target is all zero
                sdrs.append(torch.zeros(1))

        # save the reconstructed wavs
        for meta_song_id, wav in zip(batch["meta_song_id"], gt_audio):
            file_name = (
                f"{self.hparams.val_output_samples_dir}/origin/{meta_song_id}.wav"
            )
            if not os.path.exists(file_name):
                try:
                    sf.write(file_name, wav.cpu().numpy().T, self.hparams.decoder_samplerate)
                except:
                    print(f"Error writing {file_name}")
                    continue

        for meta_song_id, wav in zip(batch["meta_song_id"], wavs_g):
            file_name = f"{self.hparams.val_output_samples_dir}/{self.current_step}/{meta_song_id}.wav"
            try:
                sf.write(file_name, wav.cpu().numpy().T, self.hparams.decoder_samplerate)
            except:
                print(f"Error writing {file_name}")
                continue

        results = {"ids": batch["meta_song_id"], "sdrs": sdrs}

        for meta_song_id, sdr in zip(batch["meta_song_id"], sdrs):
            self.val_output_dict[meta_song_id] = {"sdr": sdr}
        return results

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # create folder based on current epoch
        os.makedirs(f"{self.hparams.val_output_samples_dir}/origin", exist_ok=True)
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}", exist_ok=True
        )
        self.val_output_dict = {}
        return

    def on_validation_epoch_end(self):
        # get results from all the gpu
        # NOTE: gather dict will get all the results from all the gpu under same key, might be redundant
        # self.trainer.strategy.barrier()
        results = self.all_gather(self.val_output_dict)
        # get median sdr
        # put values in result to list
        sdrs = []
        # self.trainer.strategy.barrier()
        for k in results:
            sdr = torch.mean(results[k]["sdr"])
            if not torch.isnan(sdr).any():
                sdrs.append(sdr)

        sdr = torch.median(torch.sort(torch.tensor(sdrs))[0])

        # self.trainer.strategy.barrier()
        self.log_dict(
            {"val_sdr": sdr}, prog_bar=True, sync_dist=False, rank_zero_only=True
        )
        rank_zero_info(f"Val SDR: {sdr}")
        # self.trainer.strategy.barrier()

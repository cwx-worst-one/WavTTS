import math

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import transformers
from torch import nn

from typing import Type, Dict

from recipes.sacodec.models.components.loss import MelSpecReconstructionLoss, ChromaLoss, DescriptMelSpectrogramLoss
from recipes.sacodec.models.components.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, SNRLoss
from recipes.sacodec.models.sacodec import STFTEncoderVAE, ISTFTDecoder
from recipes.sacodec.models.sacodec_hierarchical import STFTEncoderVAEHierarchicalUMMLoss
from recipes.sacodec.models.sacodec_umm import STFTEncoderVAEPostUMM

from functools import partial
from recipes.sacodec.models.components.loss import amplitude_loss, phase_loss, phase_loss_channel, STFT_consistency_loss, melspec_stereo_loss
from recipes.sacodec.models.components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator # vocos 32 channels
from samantha.utils.utils import download_checkpoint
from recipes.musiclm.utils.dist import local_zero_first
import os

import wandb
import torch.nn.functional as F

try:
    import torch_museval
except Exception as e:
    print('WARNING: torch_museval not installed. This is required if doing codec training')
import torchaudio


from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_distortion_ratio

class SACodecModule(pl.LightningModule):
    def __init__(
        self,
        encoder_cls: Type[STFTEncoderVAE],
        decoder_cls: Type[ISTFTDecoder],
        sample_rate: int,
        initial_learning_rate: float,
        disc_learning_rate_ratio: float = 1,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 0.1,
        chroma_loss_coeff: float = 1.0,
        mag_phase_coeff: float = 1.0,
        sdr_loss_coeff: float = 0.0,
        pretrain_mel_steps: int = 0,
        kl_warmup_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_sdr: bool = True,
        n_fft: int = 1024,
        win_length: int = None,
        hop_length: int = 256,
        n_mels: int = 160,
        n_chroma: int = 12,
        audio_channels: int = 1,
        melspec_loss_type: str = "descript", # vocos, descript, descript_stereo
        pretrained_path: str = None,
        dynamic_vector_dropout_loss: bool = False,
        freeze_encoder: bool = False,
        **kwargs
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
        """
        super().__init__()
        self.save_hyperparameters()
        if kwargs:
            print('Extra kwargs not handled', kwargs)

        self.multiperioddisc = MultiPeriodDiscriminator(in_channels=audio_channels)
        self.multiresddisc = MultiResolutionDiscriminator(in_channels=audio_channels, channels=32)

        self.n_fft = n_fft
        win_length = n_fft if win_length is None else win_length
        self.win_length = win_length

        # TODO: remove this once new models are saved and remapped
        encoder_cls, decoder_cls = self.remap_legacy_class(encoder_cls, decoder_cls)
        self.encoder = encoder_cls()
        self.decoder = decoder_cls()

        self.melspec_loss_type = melspec_loss_type
        if "descript" in melspec_loss_type:
            if sample_rate == 24000:
                n_mels = [5, 10, 20, 40, 80, 160, 320]
                window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
            elif sample_rate >= 44100:
                n_mels = [5, 10, 20, 40, 80, 160, 320, 640]
                # n_mels = [3, 6, 12, 24, 48, 96, 192, 364]
                window_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        
            # descript multi resolution
            self.melspec_loss = DescriptMelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=n_mels,
                window_lengths=window_lengths,
                loss_fn=torch.nn.L1Loss(),
                clamp_eps=1e-5,
                mag_weight=0.0,
                log_weight=1.0,
                pow=1.0,
                mel_fmins=[0, 0, 0, 0, 0, 0, 0, 0],
                mel_fmaxes=[None, None, None, None, None, None, None, None],
            )
        else:
            # Vocos / AP codec does not use multi resolution melspec
            self.melspec_loss = MelSpecReconstructionLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )


        self.chroma_loss = ChromaLoss(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_chroma=n_chroma
        )

        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        self.automatic_optimization = False
        self.dynamic_vector_dropout_loss = dynamic_vector_dropout_loss

        self.freeze_encoder = freeze_encoder


        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.sdr_loss = SNRLoss()

        # set to False to fix umm weight loading error
        self.strict_loading = False

    def remap_legacy_class(self, encoder_cls, decoder_cls):
        ## LEGACY: update partials to use correct class
        if encoder_cls.func.__name__ == "STFTEncoderVAEMulti3DownEven":
            encoder_cls = partial(STFTEncoderVAE, **encoder_cls.keywords)
        if encoder_cls.func.__name__ == "STFTEncoderVAEMulti3DownEvenUMMPost":
            encoder_cls = partial(STFTEncoderVAEPostUMM, **encoder_cls.keywords)
        if encoder_cls.func.__name__ == "STFTEncoderVAEHierarchicalUMMLoss":
            encoder_cls = partial(STFTEncoderVAEHierarchicalUMMLoss, **encoder_cls.keywords)
        if decoder_cls.func.__name__ == "ISTFTDecoderMulti3UpEven":
            decoder_cls = partial(ISTFTDecoder, **decoder_cls.keywords)
        return encoder_cls, decoder_cls


    def setup(self, stage: str = None):
        if stage == "fit" and self.hparams.pretrained_path is not None:
            self.load_from_pretrained(self.hparams.pretrained_path)

    def load_from_pretrained(self, pretrained_path=None):
        print('Loading pre-trained model from checkpoint', pretrained_path)

        cache_dir = '.module_cache/sacodec'
        with local_zero_first():
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
            pretrained_path = download_checkpoint(pretrained_path, cache_dir)

        state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        # model_state_dict = self.state_dict()
        # self.load_state_dict(state_dict, strict=False)

        model_state_dict = self.state_dict()
        for k in state_dict:
            if k not in model_state_dict:
                print(f"Dropping parameter {k}")
                continue

            # if "multiperioddisc" in k: continue  # skip loading discriminator weights
            # if "multiresddisc" in k: continue  # skip loading discriminator weights
            
            # skip checking over non-tensor items. i.e. embedding_modules->set_extra_state
            if not torch.is_tensor(state_dict[k]): continue

            try:
                if state_dict[k].shape != model_state_dict[k].shape:
                    # special case for copying mono to stereo
                    if len(state_dict[k].shape) >= 2 and (state_dict[k].shape[1] * 2) == model_state_dict[k].shape[1]:
                        print(f"Weights found with different sizes. Copying subset of weights",
                            k, state_dict[k].shape, model_state_dict[k].shape)
                        model_state_dict[k][:, :state_dict[k].shape[1]] = state_dict[k]
                        model_state_dict[k][:, state_dict[k].shape[1]:] = state_dict[k]
                        state_dict[k] = model_state_dict[k]
                    elif (state_dict[k].shape[0] * 2) == model_state_dict[k].shape[0]:
                        print(f"Weights found with different sizes. Copying subset of weights",
                            k, state_dict[k].shape, model_state_dict[k].shape)
                        model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
                        model_state_dict[k][state_dict[k].shape[0]:] = state_dict[k]
                        state_dict[k] = model_state_dict[k]
                    else:
                        print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                        state_dict[k] = model_state_dict[k]
            except Exception as e:
                print(f"Error loading weights. Skipped: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]

        self.load_state_dict(state_dict, strict=False)


    def _divide_params_group(self, model):
        no_decay = ["bn", "bias", "norm", "norm1", "norm2", "embedder.weight", "rotary", "embedding", ".g"]  # g in RMSNorm

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

        base_params_mpd, no_decay_params_mpd = self._divide_params_group(self.multiperioddisc)
        base_params_mrd, no_decay_params_mrd = self._divide_params_group(self.multiresddisc)
        disc_params = [
            {"params": base_params_mpd},
            {"params": base_params_mrd},
            {"params": no_decay_params_mpd, "weight_decay": 0.0},
            {"params": no_decay_params_mrd, "weight_decay": 0.0},
        ]

        base_params_enc, no_decay_params_enc = self._divide_params_group(self.encoder)
        base_params_dec, no_decay_params_dec = self._divide_params_group(self.decoder)
        gen_params = [
            {"params": base_params_enc},
            {"params": base_params_dec},
            {"params": no_decay_params_enc, "weight_decay": 0.0},
            {"params": no_decay_params_dec, "weight_decay": 0.0},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate * self.hparams.disc_learning_rate_ratio, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps  # Max steps per optimizer
        disc_max_steps = max_steps - self.hparams.pretrain_mel_steps # offset by pretrain steps to keep schedulers in sync
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=disc_max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )
    
    def reconstruct_audio(self, audio: torch.Tensor):
        logamp, pha, rea, imag = self.encoder.audio_to_spec(audio)
        encoder_results = self.encoder(logamp, pha, return_loss=False)
        latent = encoder_results["latent"]
        logamp_g, pha_g, rea_g, imag_g, y_g = self.decoder(latent)
        audio_hat = y_g
        return audio_hat

    def get_latents(
        self,
        audio: torch.Tensor,
        audio_input_24k_mono: torch.Tensor=None,
        chunk_duration=None
    ) -> Dict[str, torch.Tensor]:
        """called by diffusion."""
        logamp, pha, rea, imag = self.encoder.audio_to_spec(audio)
        encoder_results = self.encoder(logamp, pha, return_loss=False, audio_input_24k_mono=audio_input_24k_mono)
        if isinstance(self.encoder, STFTEncoderVAEHierarchicalUMMLoss):
            # use hierarchical features for diffusion - which have smooth vae latents
            latent = encoder_results["features"]
        else:
            latent = encoder_results["latent"]
        return latent.transpose(1, 2)  # bs x emb x seq -> bs x seq x emb
    
    def decode_hierarchical_features(self, features: torch.Tensor, skip_idxs=None):
        encoder: STFTEncoderVAEHierarchicalUMMLoss = self.encoder
        features = features.transpose(1, 2) # B D L
        decoder_latents = encoder.features_to_decoder_latents(features, skip_idxs=skip_idxs)
        logamp_g, pha_g, rea_g, imag_g, y_g = self.decoder(decoder_latents)
        return y_g
    
    def normalize_features(self, features: torch.Tensor, mean, std):
        if mean == 0 and std == 1: return features
        if isinstance(self.encoder, STFTEncoderVAEHierarchicalUMMLoss):
            features = self.encoder.vectorizer.split_features(features, dim=-1)
            features = [(f - m) / s for f, m, s in zip(features, mean, std)]
            return torch.cat(features, dim=-1)
        else:
            return (features - mean) / std

    def denormalize_features(self, features: torch.Tensor, mean, std):
        if mean == 0 and std == 1: return features
        if isinstance(self.encoder, STFTEncoderVAEHierarchicalUMMLoss):
            features = self.encoder.vectorizer.split_features(features, dim=-1)
            features = [f * s + m for f, m, s in zip(features, mean, std)]
            return torch.cat(features, dim=-1)
        else:
            return features * std + mean


    def decode_latents(
        self,
        latents: torch.Tensor,
    ):
        # input: bs x seq x emb
        if isinstance(self.encoder, STFTEncoderVAEHierarchicalUMMLoss):
            # hierarchical features != decoder latents. Must go through upsample blocks and sum residuals first
            return self.decode_hierarchical_features(latents)
        latents = latents.transpose(1, 2) # bs x seq x emb -> bs x emb x seq. Vocos requires sequence last
        # x = self.backbone(latents)
        logamp_g, pha_g, rea_g, imag_g, y_g = self.decoder(latents)
        return y_g

    def training_step(self, batch, batch_idx, **kwargs):

        # get optimizor and scheduler
        opt_d, opt_g = self.optimizers()
        sch_d, sch_g = self.lr_schedulers()

        audio_input = batch['audio']
        audio_input_24k = batch['audio_24k'] if 'audio_24k' in batch else None

        return_encoder_loss = True
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            return_encoder_loss = False

        if 'audio_mp3_compress' in batch:
            audio_input_mp3 = batch['audio_mp3_compress']
            ## use data augmentation for encoder decoder
            logamp, pha, rea, imag = self.encoder.audio_to_spec(audio_input)
            logamp_mp3, pha_mp3, _, _ = self.encoder.audio_to_spec(audio_input_mp3) # input compressed audio into encoder / decoder
            encoder_results = self.encoder(logamp_mp3, pha_mp3, audio_input_24k_mono=audio_input_24k, return_loss=return_encoder_loss)
        else:
            logamp, pha, rea, imag = self.encoder.audio_to_spec(audio_input)
            encoder_results = self.encoder(logamp, pha, audio_input_24k_mono=audio_input_24k, return_loss=return_encoder_loss)
        latent = encoder_results["latent"]
        logamp_g, pha_g, rea_g, imag_g, y_g = self.decoder(latent)
        audio_hat = y_g


        ## Disc Update flags
        ### Gen Update flags
        train_discriminator = self.global_step >= self.hparams.pretrain_mel_steps and self.global_step % 2 == 0
        train_apcodec_loss = True
        train_gen_disc = self.global_step >= self.hparams.pretrain_mel_steps
        train_semantic_only = self.dynamic_vector_dropout_loss and "n_vector_dropout" in encoder_results and encoder_results["n_vector_dropout"] == 1

        # train discriminator
        if train_discriminator:
            self.toggle_optimizer(opt_d)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat.detach(), **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat.detach(), **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            if train_semantic_only:
                loss = loss * 0

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)

            opt_d.zero_grad()
            self.manual_backward(loss)
            opt_d.step()
            sch_d.step()
            self.untoggle_optimizer(opt_d)
            # return loss

        # train generator
        self.toggle_optimizer(opt_g)


        ## AP Codec losses ###
        if train_apcodec_loss:
            # Losses defined on log amplitude spectra
            L_A = amplitude_loss(logamp, logamp_g)

            L_IP, L_GD, L_PTD = phase_loss_channel(pha, pha_g, self.n_fft, pha.size()[-1])
            # Losses defined on phase spectra
            L_P = L_IP + L_GD + L_PTD

            _, _, rea_g_final, imag_g_final = self.encoder.audio_to_spec(y_g)
            L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
            L_R = F.l1_loss(rea, rea_g)
            L_I = F.l1_loss(imag, imag_g)
            # Losses defined on reconstructed STFT spectra
            L_S = L_C + 2.25 * (L_R + L_I)
            L_G = (2.25 * L_A + 5 * L_P + 1 * L_S) # original is multipled by 20
            self.log("generator_scaled_loss/L_G", self.hparams.mag_phase_coeff * L_G)
            self.log("generator/L_G", L_G, prog_bar=True)
            self.log("generator/L_A", L_A, prog_bar=False)
            self.log("generator/L_P", L_P, prog_bar=False)
            self.log("generator/L_S", L_S, prog_bar=False)
        else:
            L_G = 0

        if train_gen_disc:
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd)

            self.log("generator/multi_period_loss", loss_gen_mp)
            self.log("generator/multi_res_loss", loss_gen_mrd)
            self.log("generator/feature_matching_mp", loss_fm_mp)
            self.log("generator/feature_matching_mrd", loss_fm_mrd)

            self.log("generator_scaled_loss/gen_mp", loss_gen_mp)
            self.log("generator_scaled_loss/gen_mrd", self.hparams.mrd_loss_coeff * loss_gen_mrd)
            self.log("generator_scaled_loss/fm_mp", loss_fm_mp)
            self.log("generator_scaled_loss/fm_mrd", self.hparams.mrd_loss_coeff * loss_fm_mrd)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

        sdr_loss = self.sdr_loss(audio_hat, audio_input)

        B, CH, L = audio_input.shape
        if CH == 2 and "stereo" in self.melspec_loss_type: # stereo 2 channel
            mel_loss = melspec_stereo_loss(audio_hat, audio_input, self.melspec_loss)
        else:
            mel_loss = self.melspec_loss(audio_hat, audio_input)

        chroma_loss = self.chroma_loss(audio_hat, audio_input)

        loss = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.mel_loss_coeff * mel_loss
            + self.hparams.chroma_loss_coeff * chroma_loss
            + self.hparams.mag_phase_coeff * L_G
            + self.hparams.sdr_loss_coeff * sdr_loss
        )


        if train_semantic_only:
            loss = loss * 0 + self.hparams.chroma_loss_coeff * chroma_loss + self.mel_loss_coeff * mel_loss + self.hparams.sdr_loss_coeff * sdr_loss

        loss_kl = encoder_results["kl_loss"]
        if self.global_step >= self.hparams.kl_warmup_steps:
            loss += loss_kl

        if "umm_cosine_loss" in encoder_results:
            umm_cosine_loss = encoder_results["umm_cosine_loss"]
            scaled_cosine_loss = (1+umm_cosine_loss) * 10
            loss += scaled_cosine_loss
            self.log("generator/umm_cosine_loss", umm_cosine_loss, prog_bar=True)
            self.log("generator_scaled_loss/umm_cosine_loss", scaled_cosine_loss)
        if "umm_l1_loss" in encoder_results:
            umm_l1_loss = encoder_results["umm_l1_loss"]
            loss += umm_l1_loss
            self.log("generator/umm_l1_loss", umm_l1_loss, prog_bar=False)

        self.log("generator_scaled_loss/loss", loss)
        self.log("generator_scaled_loss/mel", self.mel_loss_coeff * mel_loss)
        self.log("generator_scaled_loss/chroma", self.hparams.chroma_loss_coeff * chroma_loss)
        self.log("generator_scaled_loss/kl", loss_kl)
        self.log("generator_scaled_loss/sdr", self.hparams.sdr_loss_coeff * sdr_loss)


        self.log("training/loss", loss, prog_bar=False) # for nan monitoring / shutdown
        self.log("generator/total_loss", loss, prog_bar=True)
        self.log("generator/mel_loss", mel_loss, prog_bar=True)
        self.log("generator/sdr_loss", sdr_loss, prog_bar=True)
        self.log("generator/chroma_loss", chroma_loss, prog_bar=False)
        self.log("generator/kl", loss_kl, prog_bar=False)

        opt_g.zero_grad()
        self.manual_backward(loss)
        opt_g.step()
        sch_g.step()
        self.untoggle_optimizer(opt_g)

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch['audio']
        audio_input_24k = batch['audio_24k'] if 'audio_24k' in batch else None

        logamp, pha, rea, imag = self.encoder.audio_to_spec(audio_input)
        encoder_results = self.encoder(logamp, pha, audio_input_24k_mono=audio_input_24k)
        latent = encoder_results["latent"]
        logamp_g, pha_g, rea_g, imag_g, y_g = self.decoder(latent)
        audio_hat = y_g

        sdr = 0
        si_sdr = 0
        if self.hparams.evaluate_sdr:
            si_sdrs = scale_invariant_signal_distortion_ratio(audio_hat, audio_input, zero_mean=True)
            si_sdr = torch.nanmedian(si_sdrs).item()

            sdrs = []
            for wav_g, wav_o in zip(audio_hat, audio_input):
                try:
                    sdr, _, _, _ = torch_museval.evaluate(
                        wav_g.T.unsqueeze(0).detach(), wav_o.T.unsqueeze(0).detach(),
                        win=self.hparams.sample_rate,
                        hop=self.hparams.sample_rate,
                        device=self.device
                    )
                    sdr = torch.nanmedian(sdr)
                    sdrs.append(sdr)
                except Exception as e:
                    # sometimes target is all zero
                    sdrs.append(torch.zeros(1))
                    print("Warning: torch_museval nan sdr:", e)
            sdr = torch.nanmedian(torch.tensor(sdrs)).item()
            
            ## Batched version produces memory leak
            # try:
            #     sdr, _, _, _ = torch_museval.evaluate(
            #         audio_hat.transpose(1, 2).detach(), audio_input.transpose(1,2).detach(),
            #         win=self.hparams.sample_rate,
            #         hop=self.hparams.sample_rate,
            #         device=self.device
            #     )
            #     # sdr, _ = torch.nanmedian(sdr, dim=1) # previous code was not batched. don't think it matters
            #     sdr = torch.nanmedian(sdr).detach().cpu()
            # except Exception as e:
            #     print('Could not evaluate sdr', e)

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        chroma_loss = self.chroma_loss(audio_hat, audio_input)
        total_loss = mel_loss + chroma_loss

        loss_kl = encoder_results["kl_loss"]

        outputs = {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "val_kl_loss": loss_kl,
            "sdr": sdr,
            "si_sdr": si_sdr,
            "latent": latent.detach(),
        }
        if "umm_cosine_loss" in encoder_results:
            outputs["umm_cosine_loss"] = encoder_results["umm_cosine_loss"]
        if "umm_l1_loss" in encoder_results:
            outputs["umm_l1_loss"] = encoder_results["umm_l1_loss"]
        if "features" in encoder_results:
            outputs["features"] = encoder_results["features"]

        # log audio outputs. use song describer instead
        self.log_val_audio(audio_input, audio_hat, batch["meta_song_id"], batch_idx)

        self.val_outputs.append(outputs)

    def log_val_audio(self, audio_input, audio_hat, meta_song_ids, batch_idx):
        if self.global_rank != 0: return
        if batch_idx > 5: return

        # Only log once for gt
        if self.global_step < 1000:
            gt_log_dict = {}
            for idx, (audio_in, audio_pred, meta_song_id) in enumerate(zip(audio_input, audio_hat, meta_song_ids)):
                song_id = meta_song_id[:10]
                filename = f"/tmp/demo_gt_{song_id}.wav" 
                caption = f"gt_{song_id}"
                torchaudio.save(filename, audio_in.cpu(), self.hparams.sample_rate)
                gt_log_dict[f"gt_preview_{song_id}"] = wandb.Audio(filename, sample_rate=self.hparams.sample_rate, caption=caption)
                
            logger_names = [logger.__class__.__name__ for logger in self.trainer.loggers]
            if "WandbLogger" not in logger_names: raise ValueError("WandbLogger is not in the trainer.loggers")
            wandb_logger = [logger for logger in self.trainer.loggers if logger.__class__.__name__ == "WandbLogger"][0]
            wandb_logger.experiment.log(gt_log_dict)
            
            
        # log audio outputs
        gen_log_dict = {}
        for idx, (audio_in, audio_pred, meta_song_id) in enumerate(zip(audio_input, audio_hat, meta_song_ids)):
            song_id = meta_song_id[:10]
            filename = f"/tmp/demo_gen_{song_id}_{self.global_step}.wav"
            caption = f"gen_{song_id}@{self.global_step}"
            torchaudio.save(filename, audio_pred.cpu(), self.hparams.sample_rate)
            gen_log_dict[f"gen_preview_{song_id}"] = wandb.Audio(filename, sample_rate=self.hparams.sample_rate, caption=caption)

        logger_names = [logger.__class__.__name__ for logger in self.trainer.loggers]
        if "WandbLogger" not in logger_names:
            raise ValueError("WandbLogger is not in the trainer.loggers")
        wandb_logger = [logger for logger in self.trainer.loggers if logger.__class__.__name__ == "WandbLogger"][0]
        wandb_logger.experiment.log(gen_log_dict)

    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        if "umm_cosine_loss" in outputs[0]:
            umm_cosine_loss = torch.stack([x["umm_cosine_loss"] for x in outputs]).mean()
            self.log("val/umm_cosine_loss", umm_cosine_loss, sync_dist=True)
        if "umm_l1_loss" in outputs[0]:
            umm_l1_loss = torch.stack([x["umm_l1_loss"] for x in outputs]).mean()
            self.log("val/umm_l1_loss", umm_l1_loss, sync_dist=True)
        sdr = torch.median(torch.sort(torch.tensor([x["sdr"] for x in outputs], device=self.device))[0])
        si_sdr = np.array([x["si_sdr"] for x in outputs]).mean()

        latents = torch.cat([x["latent"] for x in outputs], dim=0)
        std, mean = torch.std_mean(latents)

        if "features" in outputs[0]:
            features = torch.cat([x["features"] for x in outputs], dim=0)
            if isinstance(self.encoder, STFTEncoderVAEHierarchicalUMMLoss):
                features = self.encoder.vectorizer.split_features(features)
                for idx, f in enumerate(features):
                    features_std, features_mean = torch.std_mean(f)
                    self.log(f"val/features_mean_{idx}", features_mean, sync_dist=True)
                    self.log(f"val/features_std_{idx}", features_std, sync_dist=True)
            else:
                features_std, features_mean = torch.std_mean(features)
                self.log("val/features_mean", features_mean, sync_dist=True)
                self.log("val/features_std", features_std, sync_dist=True)


        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/total_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/sdr", sdr, sync_dist=True, prog_bar=True)
        self.log("val/si_sdr", si_sdr, sync_dist=True)
        self.log("val/mean", mean, sync_dist=True)
        self.log("val/std", std, sync_dist=True)
        del self.val_outputs

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)

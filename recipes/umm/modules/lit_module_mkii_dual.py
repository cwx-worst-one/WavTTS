import argparse
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.utils import clip_grad_value_

from recipes.umm.models.patchgan_disc2D import PatchGANDisc2D
from recipes.umm.models.voc_modules.pitch_predictor.model import PitchPredictor
from recipes.umm.models.voc_modules.utils import mel2wav
from recipes.umm.modules.lit_module import Stage0
from recipes.umm.requires.model_initializer import init_stage3_dual_voc
from recipes.umm.utils.mel_utils import torch_wav2spec
from recipes.umm.utils.ssim import ssim


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


def build_disc(c_in=80, num_disc=3, disc_hidden_size=32, disc_hidden_size_max=256):
    disc_norm_type = "sn"
    mel_disc = PatchGANDisc2D(
        freq_length=c_in,
        time_length=-1,
        hidden_size=disc_hidden_size,
        max_hidden_size=disc_hidden_size_max,
        norm_type=disc_norm_type,
        same_clip_batch=True,
        num_layers=4,
        num_disc=num_disc,
    )
    return mel_disc


def get_vocoder(hparams, local_rank):
    vocoder_module = hparams.required_modules["vocoder"]
    voc_ckpt = vocoder_module["ckpt_path"].strip()
    if voc_ckpt != "":
        cache_dir = vocoder_module["cache_dir"].strip()
        vocoder = init_stage3_dual_voc(voc_ckpt, local_rank, cache_dir)["mel_vocoder"]
    return vocoder


def get_pitchpdt(hparams, local_rank):
    pass


class DualUMMUtilsMixin:
    """
    Utility Mixin for DualUMM lit module.
    1 April 2024: @hanoihantrakul
    Many of these can be refactored into individual functions for next MR.
    """

    def add_pitch_loss(self, recon_f0, f0, recon_vuv, vuv, losses, postfix=""):
        # F0
        recon_f0 = recon_f0.contiguous().float()
        f0 = (f0.contiguous().float() + 1).log()
        f0_loss = (torch.abs(recon_f0 - f0) * vuv).sum() / (torch.sum(vuv) + 1)
        losses[f"f0{postfix}"] = f0_loss

        # vuv
        recon_vuv = recon_vuv.contiguous().float()
        vuv = vuv.contiguous().float()
        vuv_loss = F.binary_cross_entropy_with_logits(recon_vuv, vuv)
        losses[f"vuv{postfix}"] = vuv_loss

    def add_mel_loss(
        self, mel_out, target, losses, postfix="", mel_losses={"l1": 1, "ssim": 1}
    ):
        for loss_name, lambd in mel_losses.items():
            losses[f"{loss_name}{postfix}"] = (
                getattr(self, f"{loss_name}_loss")(mel_out, target) * lambd
            )

    def l1_loss(self, decoder_output, target, *args, **kwargs):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, decoder_output, target, *args, **kwargs):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction="none")
        weights = weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def ssim_loss(self, decoder_output, target, *args, **kwargs):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        bias = 6.0
        assert decoder_output.shape == target.shape
        weights = weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def pitch_percep_loss(self, decoder_output, target, *args, **kwargs):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        self.pitchpdt(decoder_output)
        h_pred = [] + self.pitchpdt.h
        self.pitchpdt(target)
        h_gt = [] + self.pitchpdt.h
        percep_loss = F.mse_loss(h_pred[-1], h_gt[-1])
        return percep_loss

    def add_LSGAN_losses(self, p, target, ret, name="A"):
        for i, p_i in enumerate(p):
            ret[f"{name}{i}"] = F.mse_loss(p_i, p_i.new_ones(p_i.size()) * target)

    def process_tgt_mel(self, batch, max_T=None):
        """
        Run the mel transform based on `torch_wav2spec()`.

        Implementation Notes
        2 April 2024 @hanoihantrakul:
        This is different from ConformerUMM and DualUMM where the underlying
        audio transform was SpeechTransform(). This transformation from
        audio to mel spectrogram was originally handled in the `preprocessing()`
        method of the model, not the lit_module. In DualUMM, this transformation
        happens in the lit_module, not the model.

        Could possibly refactor in next MR, but not going to do it right now.
        """
        target_dict = {}
        for f in self.branches:
            target_dict[f] = torch_wav2spec(
                F.pad(
                    batch[f"audio_{f}".replace("audio_full", "audio")][:, 0],
                    [0, 1200 * 4],
                )
            )[:, :max_T]
        if "audio_vocal_perturb" in batch:
            target_dict["vocal_ptb"] = torch_wav2spec(
                F.pad(batch["audio_vocal_perturb"][:, 0], [0, 1200 * 4])
            )[:, :max_T]
        return target_dict

    def mel_torch2wav_np(self, m):
        wav = mel2wav(m, None, self.vocoder)
        wav = wav.cpu().numpy()
        return wav

    def get_code_rate(self, target_tokens):
        """
        29 March 2024
        Copied over from Stage3.get_code_rate() so this class only needs Stage0.
        @hanoihantrakul to refactor in next MR. Note that Stage3.get_code_rate()
        replaces default Stage1.get_code_rate() in original ConformerUMM. I am
        trying to remove this confusion.
        @hanoihantrakul to refactor in next MR into a separate function.
        """
        code_rate = (
            sum(
                [
                    len(target_tokens[i, :].unique())
                    for i in range(target_tokens.size(0))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
        )
        return code_rate

    def get_quant_rate(self, quant_index, quant_token_num):
        """
        29 March 2024
        Copied over from Stage3MSS, which uses `get_quant_rate`. However
        there is also another method `get_quant_rates` which is favored and used in Stage1.
        @hanoihantrakul to refactor and use the correct one.

        Despite `get_quant_rate` being labelled as deprecated, and to use `get_quant_rates`,
        it is clear all the implementations use `get_quant_rate`. I keep this for now
        so that values reported to wandb do not get messed up.

        @hanoihantrakul to refactor in next MR into a separate function.
        """
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = one_hot.sum() / quant_token_num
        return quant_rate


class DualUMMv2(Stage0, DualUMMUtilsMixin):
    """
    Implementation notes:
    1 April 2024 @hanoihantrakul
    I refactored the code to make DualUMM depend on older ConformerUMM and ConvUMM
    as little as possible. Right now the functionality is divided between Ren Yi's
    `DualUMMUtilsMixin` and `Stage0` lit module. Ideally, I will remove the dependancy
    on Stage0.
    """

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        disc_optimizer_cls,
        criterion_config,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        load_required_modules_in_init=False,
        **kwargs,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=lambda: None,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        if "branches" in kwargs:
            self.branches = set(kwargs.pop("branches"))
        else:
            self.branches = {"vocal"}
        if "adv_branches" in kwargs:
            self.adv_branches = set(kwargs.pop("adv_branches"))
        else:
            self.adv_branches = self.branches
        self.criterion_config = self.config = config = criterion_config
        self.mel_discs = nn.ModuleDict(
            {
                "full": build_disc(
                    160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
                ),
                "vocal": build_disc(
                    160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
                ),
                "inst": build_disc(
                    160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
                ),
            }
        )
        self.ctc_loss_fn = nn.CTCLoss(
            blank=config.ctc_blank_id,
            reduction=config.ctc_loss_reduction,
            zero_infinity=config.ctc_zero_infinity,
        )
        self.automatic_optimization = False
        if load_required_modules_in_init:
            self.load_required_modules()

    def load_required_modules(self):
        super().load_required_modules()
        if "pitchpdt" in self.hparams.required_modules:
            self.pitchpdt = get_pitchpdt(self.hparams, self.local_rank)
        if "vocoder" in self.hparams.required_modules:
            self.vocoder = get_vocoder(self.hparams, self.local_rank)

    def configure_optimizers(self):
        """
        DualUMM training needs to setup separate optimizers for:
        1. Model Training
        2. Discriminator Training
        """

        def _get_params_for_decoder_only():
            """Handle specific case where only decoder is trained. Turn off VQ layers and Mel-160 Audio Encoder."""
            decoder_only_params = []
            skip_names = ["vq", "audio_encoder"]
            skip_names += [
                f"encoder_layers.{x}." for x in range(self.config.vq_layer_idx)
            ]
            for name, params in self.model.named_parameters():
                skip = False
                for sname in skip_names:
                    if sname in name:
                        skip = True
                        break
                if skip:
                    continue
                print("| find trainable params: ", name)
                decoder_only_params.append(params)
            return decoder_only_params

        # Append param groups to this list
        params_group = []
        if self.config.get("train_decoder_only", False):
            params_group.append(_get_params_for_decoder_only())
        else:
            normal_params = []
            special_params = []
            for name, params in self.model.named_parameters():
                if "vq.embedding.weight" in name:
                    print("Key {} use zero WD".format(name))
                    special_params.append(params)
                else:
                    normal_params.append(params)
            params_group.append({"params": special_params, "weight_decay": 0.0})
            params_group.append({"params": normal_params})
        # Configure model optimizer and discriminator optimizer separately
        optimizer = self.hparams.optimizer_cls(params_group)
        optimizer_disc = self.hparams.disc_optimizer_cls(self.mel_discs.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        scheduler_disc = self.hparams.scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]

    def on_train_batch_start(self, batch, batch_idx):
        """
        Implementation Notes:
        1 April 2024 @hanoihantrakul
        I learned from @Li Tang that VQ is sensitive to normalization. Therefore
        TTS team allows BatchNorm parameters to update for the first 20k steps.
        Then afer that, the BatchNorm parameters are kept constant. This is an
        optimization trick at the time of writing. This logic is invoked at the
        start of each batch training step.
        """

        def _has_a_vocal_vq_layer():
            """Check if model has a VQ layer applied to vocal."""
            return hasattr(self.model.vq_proj_in_vocal, "__len__")

        def _model_uses_batchnorm_as_projection_layer():
            """Check if model uses BatchNorm as projection layer into VQ."""
            # This is hack we saw in Li Tang's code for checking if it is BN
            return len(self.model.vq_proj_in_vocal) == 3

        def _keep_constant_model_batchnorm_params():
            """Keep BatchNorm parameters constant. Do not update the params."""
            module = self.model.vq_proj_in_vocal[2]
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                self.model.vq_proj_in_vocal[2].eval()
                self.model.vq_proj_in_inst[2].eval()
                if self.trainer.global_step % 1000 == 0:
                    print(module.running_mean)

        if self.trainer.global_step >= 20_000:
            if not _has_a_vocal_vq_layer():
                return  # no Vocal VQ so skip
            if not _model_uses_batchnorm_as_projection_layer():
                return  # not a BatchNorm layer so skip
            else:
                # make sure BatchNorm params are not updated
                _keep_constant_model_batchnorm_params()
        return

    def training_step(self, batch, batch_idx):
        def _is_a_tensor_and_should_be_updated(x):
            """Helper function for extracting trainable parameters in generator and discriminator"""
            return (
                isinstance(x, torch.Tensor)
                and x.requires_grad
                and x.grad_fn is not None
            )

        # manually get optimizer and scheduler for GAN step
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        # prepare input/target data
        batch.update(self.prepare_feature(batch))
        # see comment for this method to understand why it is separated outside of `self.prepare_feature()`
        batch.update(self.process_tgt_mel(batch, batch["f0"].shape[1]))
        T = batch["vocal"].shape[1]
        ds = self.model.config.get("downsampling", 4)
        T_split = random.randint(T // 4, 3 * T // 4) // ds * ds
        if "vocal_ptb" in batch:
            batch["mel_vocal_f"] = batch["vocal_ptb"][:, :T_split]
            batch["mel_vocal_b"] = batch["vocal_ptb"][:, T_split:]
            batch["mel_vocal_ref_f"] = batch["vocal"][:, :T_split]
            batch["mel_vocal_ref_b"] = batch["vocal"][:, T_split:]
        else:
            batch["mel_vocal_f"] = batch["vocal"][:, :T_split]
            batch["mel_vocal_b"] = batch["vocal"][:, T_split:]

        #########################
        # generator step
        #########################
        losses_gen = {}
        self.toggle_optimizer(optim_g)
        output_dict = self.training_step_gen(batch, losses_gen)

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(
            sum(
                [
                    x
                    for x in losses_gen.values()
                    if _is_a_tensor_and_should_be_updated(x)
                ]
            )
        )
        clip_grad_value_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)
        output_dict = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        #########################
        # disciminator step
        #########################
        losses_disc = {}
        loss_dict = {f"tr/{k}": v for k, v in losses_gen.items()}
        if self.model.config.w_loss_adv > 0:
            self.toggle_optimizer(optim_d)
            self.training_step_disc(output_dict, batch, losses_disc)
            # discriminator backward
            optim_d.zero_grad()

            self.manual_backward(
                sum(
                    [
                        x
                        for x in losses_disc.values()
                        if _is_a_tensor_and_should_be_updated(x)
                    ]
                )
            )
            clip_grad_value_(self.mel_discs.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)
            loss_dict.update({f"tr/{k}": v for k, v in losses_disc.items()})

        # logging
        text_ids = batch["text_ids"]
        for t in {"vocal", "inst"} & self.branches:
            if self.trainer.global_step % 100 == 0:
                code_rate = self.get_code_rate(output_dict[f"vq_ids_{t}"])
                loss_dict[f"aux/code_rate_{t}"] = code_rate
            quant_rate = self.get_quant_rate(
                output_dict[f"vq_ids_{t}"].long(), self.model.config.vq_codebook_size
            )
            loss_dict[f"aux/quant_rate_{t}"] = quant_rate
            if t == "vocal":
                if getattr(self.model.vq_vocal, "entropy", None) is not None:
                    loss_dict[f"aux/entropy_{t}"] = self.model.vq_vocal.entropy()
            if t == "inst":
                if getattr(self.model.vq_inst, "entropy", None) is not None:
                    loss_dict[f"aux/entropy_{t}"] = self.model.vq_inst.entropy()
        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        k = list(self.branches)[0]
        loss_dict["aux/num_mel_frames"] = batch[k].size(0) * batch[k].size(1)
        loss_dict["aux/bs"] = text_ids.shape[0]

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def training_step_gen(self, batch, loss_dict):
        output_dict = self.model(batch, self.branches)
        text_ids = batch["text_ids"]
        # pitch loss
        w_f0vuv = self.model.config.get("w_loss_f0vuv", 0.1)
        if self.model.config.add_pitch:
            for t in {"vocal", "full"} & self.branches:
                self.add_pitch_loss(
                    output_dict[f"f0_out_{t}"].squeeze(-1),
                    batch["f0"],
                    output_dict[f"vuv_out_{t}"].squeeze(-1),
                    batch["vuv"],
                    loss_dict,
                    postfix=f"_{t}",
                )
                loss_dict[f"f0_{t}"] = loss_dict[f"f0_{t}"] * w_f0vuv
                loss_dict[f"vuv_{t}"] = loss_dict[f"vuv_{t}"] * w_f0vuv
        # VQ loss
        for t in {"vocal", "inst"} & self.branches:
            if output_dict.get(f"vq_loss_{t}") is not None:
                loss_dict[f"vq_{t}"] = output_dict[f"vq_loss_{t}"]

        # Mel recon loss
        for t in self.branches:
            if output_dict.get(f"mel_out_{t}") is not None:
                self.add_mel_loss(
                    output_dict[f"mel_out_{t}"], batch[t], loss_dict, postfix=f"_{t}"
                )

        if self.model.config.add_chroma:
            for t in {"full", "inst"} & self.branches:
                loss_dict[f"chroma_{t}"] = F.mse_loss(
                    output_dict[f"chroma_out_{t}"], batch["chroma"]
                )
        # ASR LAS loss
        w_las = self.model.config.get("w_loss_las", 0.1)
        for t in {"full"} & self.branches:
            loss_dict[f"las_{t}"] = F.cross_entropy(
                output_dict[f"text_out_{t}"].transpose(1, 2), text_ids
            )
            loss_dict[f"las_{t}"] = loss_dict[f"las_{t}"] * w_las

        if self.model.config.w_loss_adv > 0:
            losses_adv = {}
            for t in self.adv_branches:
                o_ = self.mel_discs[t](output_dict[f"mel_out_{t}"])
                p_, h_p_, start_frames = o_["y"], o_.get("h"), o_.get("start_frames")
                self.add_LSGAN_losses(p_, 1, losses_adv, f"A{t}")
            for k in losses_adv:
                losses_adv[k] = losses_adv[k] * self.model.config.w_loss_adv
            loss_dict.update(losses_adv)
        return output_dict

    def training_step_disc(self, model_out, target_dict, loss_dict):
        for t in self.adv_branches:
            disc_out_real = self.mel_discs[t](target_dict[t])
            p_real, start_frames = disc_out_real["y"], disc_out_real.get("start_frames")
            mel_p = model_out[f"mel_out_{t}"]
            disc_out_fake = self.mel_discs[t](mel_p, None, start_frames)
            p_fake = disc_out_fake["y"]
            self.add_LSGAN_losses(p_real, 1, loss_dict, f"R{t}")
            self.add_LSGAN_losses(p_fake, 0, loss_dict, f"F{t}")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # prepare input/target data
            batch.update(self.prepare_feature(batch))
            batch.update(self.process_tgt_mel(batch, batch["f0"].shape[1]))
            ds = self.model.config.get("downsampling", 4)
            T_mid = batch["vocal"].shape[1] // 2 // ds * ds
            batch["mel_vocal_ref_f"] = batch["mel_vocal_f"] = batch["vocal"][:, :T_mid]
            batch["mel_vocal_ref_b"] = batch["mel_vocal_b"] = batch["vocal"][:, T_mid:]
            output_dict = self.model(batch, self.branches)
            for t in self.branches:
                mel_pred = output_dict[f"mel_out_{t}"]
                mel_tgt = batch[t]

                step_cur = self.global_step // 2
                for i, (v_pred, v_gt) in enumerate(zip(mel_pred, mel_tgt)):
                    if i == 5:
                        break
                    wav_g = self.mel_torch2wav_np(v_gt)
                    self.logger.experiment.add_audio(
                        f"{t}_g{i:02d}", wav_g, step_cur, 24000
                    )
                    wav_p = self.mel_torch2wav_np(v_pred)
                    self.logger.experiment.add_audio(
                        f"{t}_p{i:02d}", wav_p, step_cur, 24000
                    )

                    fig = plt.figure(figsize=(16, 8))
                    plt.pcolor(v_pred.cpu().T, vmin=-6, vmax=0.5)
                    self.logger.experiment.add_figure(
                        f"val/{t}_p{i:02d}", fig, global_step=step_cur
                    )
                    fig = plt.figure(figsize=(16, 8))
                    plt.pcolor(v_gt.cpu().T, vmin=-6, vmax=0.5)
                    self.logger.experiment.add_figure(
                        f"val/{t}_g{i:02d}", fig, global_step=step_cur
                    )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        """
        Implementation Notes:
        29 March 2024 @hanoihantrakul:
        Copied over from `lit_module.Stage3MSS.prepare_feature()`
        so this class only needs Stage0 to be inherited. Inheriting
        from Stage3MSS is difficult to maintain.

        @hanoihantrakul to refactor in next MR.
        """
        # Prepare tokens
        input_dict = {"text_ids": batch["token"].long()}
        # Prepare MSS audio tracks
        _audio_dict = {k: batch[k] for k in ["audio", "audio_vocal", "audio_inst"]}
        _audio_dict = {k: v.squeeze(dim=1).float() for k, v in _audio_dict.items()}
        _audio_dict = {k: self.pad_audio(v) for k, v in _audio_dict.items()}
        """
        @hanoihantrakul 10/10/2023
        Problem: Superclass Stage0.preprocessing() assumes 1 fixed audio argument `x`, but there are 3 audio tracks.
        Solution: Pass in a single dict instead of audio directly. Then handle dict in self.model.preprocessing()
        """
        # This next line calls the lit_module model preprocessing `self.model.preprocessing()`
        # Look at dualumm.DualUMMv2.preprocessing() to understand what happens.
        preprocessed_feats = self.preprocessing(_audio_dict)
        input_dict.update(preprocessed_feats)
        return input_dict


def run_dualMSS_decode(requires, samples, params, token_type="vocal"):
    if "batch" in params:
        batch = params["batch"]
        prompt_audio = batch.get("audio_vocal", batch.get("style_audio"))
    else:
        prompt_audio = None
    vocoder = requires["mel_vocoder"]
    umm_model = requires["umm"]
    wavs = []
    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
        mels = umm_model.token2mel(samples, prompt_audio, token_type)
    for mel in mels:
        wavs.append(mel2wav(mel, None, vocoder))
    return torch.stack(wavs, 0)


class DualUMMv2Inst(Stage0, DualUMMUtilsMixin):
    """
    Configure DualUMM for instrumental branch training only.

    Implementation notes @hanoihantrakul
    1 April 2024:
    I deliberately did not inherit from DualUMMv2 because
    the differences in managing the adversarial loss are large enough
    that this should be its own class.

    DualUMM trains on MSS data. When adapting this to instrumental-only
    data, the preprocessing code is actually closer to ConformerUMM and
    ConvUMM. There is code that is copied from these systems here. I
    have not yet refactored this code to be more general.
    """

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        disc_optimizer_cls,
        criterion_config,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        load_required_modules_in_init=False,
        **kwargs,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=lambda: None,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        """
        Implementation Notes:
        1 April 2024 @hanoihantrakul
        Although there is only one branch and self.branches = ['inst']
        it makes maintaining and comparing the code to DualUMMv2
        easier to just follow the key naming convention and use of {t}
        to access the correct branch.
        """
        self.branches = ["inst"]
        self.adv_branches = ["inst"]
        self.criterion_config = self.config = config = criterion_config
        self.mel_discs = nn.ModuleDict(
            {
                "inst": build_disc(
                    160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
                )
            }
        )
        self.automatic_optimization = False  # Need to set this for GAN portion
        if load_required_modules_in_init:
            self.load_required_modules()

    def load_required_modules(self):
        super().load_required_modules()
        if "pitchpdt" in self.hparams.required_modules:
            self.pitchpdt = get_pitchpdt(self.hparams, self.local_rank)
        if "vocoder" in self.hparams.required_modules:
            self.vocoder = get_vocoder(self.hparams, self.local_rank)

    def configure_optimizers(self):
        """
        DualUMM training needs to setup separate optimizers for:
        1. Model Training
        2. Discriminator Training
        """

        """
        @hanoihantrakul 1 April 2024
        This whole function can be refactorized and reused via composition in 
        both DualUMMv2 and DualUMMv2InstOnly.
        """

        def _get_params_for_decoder_only():
            """Handle specific case where only decoder is trained. Turn off VQ layers and Mel-160 Audio Encoder."""
            decoder_only_params = []
            skip_names = ["vq", "audio_encoder"]
            skip_names += [
                f"encoder_layers.{x}." for x in range(self.config.vq_layer_idx)
            ]
            for name, params in self.model.named_parameters():
                skip = False
                for sname in skip_names:
                    if sname in name:
                        skip = True
                        break
                if skip:
                    continue
                print("| find trainable params: ", name)
                decoder_only_params.append(params)
            return decoder_only_params

        # Append param groups to this list
        params_group = []
        if self.config.get("train_decoder_only", False):
            params_group.append(_get_params_for_decoder_only())
        else:
            normal_params = []
            special_params = []
            for name, params in self.model.named_parameters():
                if "vq.embedding.weight" in name:
                    print("Key {} use zero WD".format(name))
                    special_params.append(params)
                else:
                    normal_params.append(params)
            params_group.append({"params": special_params, "weight_decay": 0.0})
            params_group.append({"params": normal_params})
        # Configure model optimizer and discriminator optimizer separately
        optimizer = self.hparams.optimizer_cls(params_group)
        optimizer_disc = self.hparams.disc_optimizer_cls(self.mel_disc.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        scheduler_disc = self.hparams.scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]

    def training_step(self, batch, batch_idx):
        def _is_a_tensor_and_should_be_updated(x):
            """Helper function for extracting trainable parameters in generator and discriminator"""
            return (
                isinstance(x, torch.Tensor)
                and x.requires_grad
                and x.grad_fn is not None
            )

        # manually get optimizer and scheduler for GAN step
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        # prepare input/target data
        batch.update(self.prepare_feature(batch))
        # see comment for this method to understand why it is separated outside of `self.prepare_feature()`
        batch.update(self.process_tgt_mel(batch, batch["f0"].shape[1]))

        #########################
        # generator step
        #########################
        losses_gen = {}
        self.toggle_optimizer(optim_g)
        output_dict = self.training_step_gen(batch, losses_gen)

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(
            sum(
                [
                    x
                    for x in losses_gen.values()
                    if _is_a_tensor_and_should_be_updated(x)
                ]
            )
        )
        clip_grad_value_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)
        output_dict = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        #########################
        # disciminator step
        #########################
        losses_disc = {}
        loss_dict = {f"tr/{k}": v for k, v in losses_gen.items()}
        if self.model.config.w_loss_adv > 0:
            self.toggle_optimizer(optim_d)
            self.training_step_disc(output_dict, batch, losses_disc)
            # discriminator backward
            optim_d.zero_grad()

            self.manual_backward(
                sum(
                    [
                        x
                        for x in losses_disc.values()
                        if _is_a_tensor_and_should_be_updated(x)
                    ]
                )
            )
            clip_grad_value_(self.mel_discs.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)
            loss_dict.update({f"tr/{k}": v for k, v in losses_disc.items()})

        # logging
        for t in self.branches:
            if self.trainer.global_step % 100 == 0:
                code_rate = self.get_code_rate(output_dict[f"vq_ids_{t}"])
                loss_dict[f"aux/code_rate_{t}"] = code_rate
            quant_rate = self.get_quant_rate(
                output_dict[f"vq_ids_{t}"].long(), self.model.config.vq_codebook_size
            )
            loss_dict[f"aux/quant_rate_{t}"] = quant_rate
            if getattr(self.model.vq_inst, "entropy", None) is not None:
                loss_dict[f"aux/entropy_{t}"] = self.model.vq_inst.entropy()
        k = list(self.branches)[0]
        loss_dict["aux/num_mel_frames"] = batch[k].size(0) * batch[k].size(1)

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def training_step_gen(self, batch, loss_dict):
        output_dict = self.model(batch)

        # VQ loss
        for t in self.branches:
            if output_dict.get(f"vq_loss_{t}") is not None:
                loss_dict[f"vq_{t}"] = output_dict[f"vq_loss_{t}"]

        # Mel recon loss
        for t in self.branches:
            if output_dict.get(f"mel_out_{t}") is not None:
                self.add_mel_loss(
                    output_dict[f"mel_out_{t}"], batch[t], loss_dict, postfix=f"_{t}"
                )

        # Chroma loss
        if self.model.config.add_chroma:
            for t in self.branches:
                loss_dict[f"chroma_{t}"] = F.mse_loss(
                    output_dict[f"chroma_out_{t}"], batch["chroma"]
                )

        # Adversarial Loss
        if self.model.config.w_loss_adv > 0:
            losses_adv = {}
            for t in self.adv_branches:
                o_ = self.mel_discs[t](output_dict[f"mel_out_{t}"])
                p_, h_p_, start_frames = o_["y"], o_.get("h"), o_.get("start_frames")
                self.add_LSGAN_losses(p_, 1, losses_adv, f"A{t}")
            for k in losses_adv:
                losses_adv[k] = losses_adv[k] * self.model.config.w_loss_adv
            loss_dict.update(losses_adv)
        return output_dict

    def training_step_disc(self, model_out, target_dict, loss_dict):
        for t in self.adv_branches:
            disc_out_real = self.mel_discs[t](target_dict[t])
            p_real, start_frames = disc_out_real["y"], disc_out_real.get("start_frames")
            mel_p = model_out[f"mel_out_{t}"]
            disc_out_fake = self.mel_discs[t](mel_p, None, start_frames)
            p_fake = disc_out_fake["y"]
            self.add_LSGAN_losses(p_real, 1, loss_dict, f"R{t}")
            self.add_LSGAN_losses(p_fake, 0, loss_dict, f"F{t}")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        """
        Implementation Notes
        2 April 2024 @hanoihantrakul
        If you look at the original DualUMMv2.prepare_feature() code you'll see
        it is copied from Stage3MSS.prepare_feature(). This is because DualUMMv2 needs
        to prepare features from 3 keys. For DualUMMv2Inst model, the code needs to use
        the original non-MSS data from the older ConformerUMM and ConvUMM pipeline. I have
        copy-pasted and modified the code from `lit_module.Stage2.prepare_feature()` to
        accomplish this.
        """
        input_dict = {}
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        # This next line calls the lit_module model preprocessing `self.model.preprocessing()`
        # Look at dualumm.DualUMMv2Inst.preprocessing() to understand what happens.
        feature = self.preprocessing(
            audio
        )  # this is only adding chroma under the hood. Recommend refactoring in next MR.
        input_dict.update(feature)
        return input_dict

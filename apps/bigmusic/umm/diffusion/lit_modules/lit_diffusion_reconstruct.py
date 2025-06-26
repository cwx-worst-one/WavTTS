import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torchaudio.functional import resample
from functools import partial
import librosa
from pytorch_lightning import LightningModule
import math
from typing import Any
from apps.bigmusic.umm.diffusion.lit_modules.lit_diffusion_voicebox import (
    VoiceBoxModule as pl_module,
)
from samantha.utils import groundtruth
from samantha.dataio.lite.utils.mel import mel_spectrogram
from apps.bigtts.umm.diffusion.lit_modules.infer_utils import set_seed, save_wav
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from hyperpyyaml import load_hyperpyyaml

import logging

logger = logging.getLogger(__name__)


def prepare_diffusion_model(diffusion_ckpt_path, device):
    model = pl_module.load_from_checkpoint(
        diffusion_ckpt_path, map_location="cpu"
    ).model.to(device)
    model.eval()
    return model


def prepare_umm(umm_ckpt_path, device):
    rank = int(device[-1])
    from recipes.umm.requires.model_initializer import init_stage3

    token_model = init_stage3(umm_ckpt_path, rank, "./")["Stage3"].eval()
    return token_model


def prepare_ummv2(umm_ckpt_path, device):
    from recipes.umm.modules.lit_module_mk3 import Stage3

    model = Stage3.load_from_checkpoint(umm_ckpt_path).eval().to(device)
    return model


def prepare_usm(usm_ckpt_path, device):
    from recipes.umm.modules.lit_module_mk3 import USMStage3

    model = USMStage3.load_from_checkpoint(usm_ckpt_path).eval().to(device)
    return model


def prepare_umm_music(umm_ckpt_path, device):
    from recipes.umm.modules.lit_module import Stage3

    model = Stage3.load_from_checkpoint(umm_ckpt_path).eval().to(device)
    return model


def prepare_umm_conv(umm_ckpt_path, device):
    from recipes.umm.requires.model_initializer import init_stage3_conv1d

    model = (
        init_stage3_conv1d(umm_ckpt_path, 0, "./.test_umm_conv1d_cache")["Stage3Conv1D"]
        .eval()
        .to(device)
    )
    return model


def prepare_umm_dualconv(umm_ckpt_path, device):
    from recipes.umm.requires.model_initializer import init_dualumm

    model = init_dualumm(umm_ckpt_path, device=device)["Stage3"].eval().to(device)
    return model


def prepare_umm_convgan(umm_ckpt_path, device):
    from recipes.umm.requires.model_initializer import init_convumm_gan

    model = init_convumm_gan(umm_ckpt_path, device=device)["Stage3"].eval().to(device)
    return model


def prepare_umm_codebook(umm_codebook_path, device):
    codebook = torch.load(umm_codebook_path).to(device)
    return codebook


def prepare_mel_transform(mel_config):
    mel_transform = partial(
        mel_spectrogram,
        n_fft=mel_config["n_fft"],  # 2048,
        num_mels=mel_config["num_mels"],  # 80,
        hop_size=mel_config["hop_size"],
        win_size=mel_config["win_size"],
        sampling_rate=24000,
        fmin=0,
        fmax=12000,
    )
    return mel_transform


def prepare_vocoder(vocoder_ckpt_path, device):
    vocoder = torch.jit.load(vocoder_ckpt_path, map_location=device).eval()
    return vocoder


class MelNorm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def denorm_mel(self, mel):
        return (mel * self.std) + self.mean

    def norm_mel(self, mel):
        return (mel - self.mean) / self.std


class DiffusionU2SInfer(LightningModule):
    def __init__(
        self,
        diffusion_ckpt_path,
        umm_ckpt_path,
        vocoder_ckpt_path,
        output_dir,
        infer_type,
        umm_frame_rate,
        mel_frame_rate,
        mel_config,
        seed=1996,
        save_prompt=False,
        umm_type="UMM",  # UMM or USM
        umm_codebook_path=None,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        text_cfg_w=1,
        rescale_factor=0.7,
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        use_phone_lang=False,
        without_prefix=True,
        **kwargs,
    ):
        super().__init__()

        seed = set_seed(seed)
        self.vocoder_ckpt_path = vocoder_ckpt_path
        self.umm_ckpt_path = umm_ckpt_path
        self.mel_transform = prepare_mel_transform(mel_config)
        self.output_dir = output_dir
        self.infer_type = infer_type
        self.mel_config = mel_config
        self.umm_frame_rate = umm_frame_rate
        self.mel_frame_rate = mel_frame_rate
        self.save_prompt = save_prompt
        self.umm_type = umm_type
        self.umm_codebook_path = umm_codebook_path

        self.diffusion_precision = diffusion_precision
        self.diffusion_nfe = diffusion_nfe
        self.diffusion_sampler = diffusion_sampler
        self.text_cfg_w = text_cfg_w
        self.rescale_factor = rescale_factor
        self.use_wvae_vocoder = use_wvae_vocoder

        self.use_phone_lang = use_phone_lang

        self.without_prefix = without_prefix
        if self.infer_type != "vocoder":
            self.model = prepare_diffusion_model(diffusion_ckpt_path, self.device)

        self.bn_config = bn_config
        if self.use_wvae_vocoder:
            self.bn_norm = MelNorm(bn_config["bn_norm_mean"], bn_config["bn_norm_std"])
        else:
            self.mel_norm = MelNorm(
                mel_config["mel_norm_mean"], mel_config["mel_norm_std"]
            )

        if token_config is None:
            self.token_config = {"token_cfg": True, "token_padding": 32767}
        else:
            self.token_config = token_config

        self.mel_norm = MelNorm(mel_config["mel_norm_mean"], mel_config["mel_norm_std"])
        self.mel_mask_value = mel_config["mel_mask_value"]  # -5
        self.token_sample_rate = 24000

        os.makedirs(output_dir, exist_ok=True)

        logger.info("DiffusionU2SInfer:")
        args_dict = locals()
        del args_dict["self"]
        logger.info(", ".join(f"{k}={v}\n" for k, v in args_dict.items()))

    def wvae_decode(self, pred_emb):
        pred_emb = pred_emb.float()
        duration = pred_emb.shape[-1] // self.mel_frame_rate
        with torch.autocast(device_type="cuda", enabled=False):
            if duration > 30:
                wavs_g = vocode_in_chunks(
                    pred_emb,
                    self.wvae,
                    mini_bs=1,
                    chunk_size=self.bn_config["chunk_size"],
                )
            else:
                wavs_g = vocode_in_chunks(
                    pred_emb, self.wvae, mini_bs=self.bn_config["mini_bs"], chunk_size=1
                )
        return wavs_g

    # align wav to make sure wav length could be divided by `umm_frame_rate` and `mel_frame_rate` evenly
    def align_wav(
        self, wav, umm_sampling_rate, mel_sampling_rate, umm_frame_rate, mel_frame_rate
    ):
        umm_hop = umm_sampling_rate // umm_frame_rate
        mel_hop = mel_sampling_rate // mel_frame_rate
        align_block_len = abs(umm_hop * mel_hop) // math.gcd(umm_hop, mel_hop)
        crop_wav_len = wav.shape[1] % align_block_len
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, align_block_len - crop_wav_len), "constant", 0)
        return wav, align_block_len

    # align wav to make sure padded wav length could be divided by `wav_divide` evenly
    def align_wav2(self, wav, wav_divide, sampling_rate, umm_frame_rate, umm_add_len):
        umm_hop = sampling_rate // umm_frame_rate
        wav_add_len = umm_add_len * umm_hop
        crop_wav_len = (wav_add_len + wav.shape[1]) % wav_divide
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, wav_divide - crop_wav_len), "constant", 0)
        return wav, wav_divide

    def wav2token(self, wav):
        if self.umm_type == "USM":
            token_len = wav.shape[1] // 960
            umm_token = self.umm.wav2token(
                resample(wav, 24000, 16000), dtype=torch.bfloat16
            )
            umm_token = umm_token[:, :token_len]
        elif self.umm_type in ["UMM", "UMM_conv"]:
            umm_token = self.umm.wav2token(wav)
        elif self.umm_type in ["UMM_dualconv"]:
            token_vocal = self.umm.wav2token(wav, "vocal").squeeze()
            token_inst = self.umm.wav2token(wav, "inst").squeeze()
            umm_token = torch.stack([token_vocal, token_inst], 1)
            umm_token = umm_token.reshape(-1).unsqueeze(0)
        elif self.umm_type in ["UMM_dualconvV1", "UMM_dualconvV3"]:
            from recipes.umm.utils.mss import MSSPredictor

            predictor = MSSPredictor().to(self.device)
            voc, acc = predictor(wav)
            token_vocal = self.umm.wav2token(voc[:, None], "vocal")
            token_acc = self.umm.wav2token(acc[:, None], "inst")
            umm_token = torch.stack([token_vocal, token_acc], 2).flatten(1, 2)
        elif self.umm_type in ["UMMv2", "UMM_music"]:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                umm_token = self.umm.wav2token(wav)
        elif self.umm_type == "UMM2":
            umm_token = self.umm.forward({"audio":wav})['vq_ids']
        else:
            raise NotImplementedError

        return umm_token

    def prepare_features(self, batch):
        (
            batched_uttid,
            batched_prompt_text_id,
            batched_prompt_wav_path,
            batched_syn_text_id,
            batched_syn_wav_path,
            batched_syn_umm_token,
            batched_scale,
        ) = self.batched_data(batch)
        bs = len(batched_uttid)

        # if prompt_text_id is None or syn_text_id is None:
        #     return None
        batched_text_id = None
        device = f"cuda:{self.local_rank}"
        inputs = dict()

        # if len(batch) == 6:
        #     inputs["scale"] = batch[5]
        # else:
        #     inputs["scale"] = None

        # Syn
        if self.infer_type == "diffusion-vocoder":
            assert batched_syn_wav_path is not None
            inputs["gt_wav"] = []
            scales = []
            syn_wavs = []
            syn_wavlens = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                wav, _ = librosa.load(
                    syn_wav_path, sr=self.token_sample_rate, mono=True
                )
                inputs["gt_wav"].append(wav)
                wav = torch.FloatTensor(wav).unsqueeze(0)
                if self.bn_config["wav_norm"]:
                    if batched_scale is None:
                        scale = max(0.001, torch.max(torch.abs(wav)))
                        wav = wav / scale * 0.95
                        scales.append(scale.item())
                    else:
                        scales.append(batched_scale[bidx])
                wav = wav.to(device)
                syn_wav = wav
                syn_wavs.append(syn_wav)
                syn_wavlen = int(
                    syn_wav.shape[-1]
                    * 1.0
                    * self.mel_config["sampling_rate"]
                    / self.token_sample_rate
                )
                syn_wavlens.append(syn_wavlen)

            batched_scale = scales
            max_wavlen = max([syn_wav.shape[-1] for syn_wav in syn_wavs])

            syn_wavs = torch.cat(
                [
                    F.pad(syn_wav, [0, max_wavlen - syn_wav.shape[-1]], "constant", 0)
                    for syn_wav in syn_wavs
                ],
                dim=0,
            )
            syn_wavs, _ = self.align_wav(
                syn_wavs,
                self.token_sample_rate,
                self.mel_config["sampling_rate"],
                self.umm_frame_rate,
                self.mel_frame_rate,
            )
            batched_syn_umm_token = self.wav2token(syn_wavs)
            inputs["scale"] = scales
            inputs["syn_wavlen"] = syn_wavlens
        elif self.infer_type == "ar-diffusion-vocoder":
            assert batched_syn_umm_token is not None
            batched_syn_umm_token_len = batched_syn_umm_token[0].shape[-1]
            for bidx, syn_umm_token in enumerate(batched_syn_umm_token):
                assert syn_umm_token.shape[-1] == batched_syn_umm_token_len
                if len(syn_umm_token.shape) == 1:
                    batched_syn_umm_token[bidx] = syn_umm_token.unsqueeze(0)
            batched_syn_umm_token = torch.cat(batched_syn_umm_token, dim=0)
            inputs["scale"] = batched_scale
        else:
            raise NotImplementedError
        if hasattr(self, "umm_codebook"):
            batched_syn_umm_token = F.embedding(
                batched_syn_umm_token, self.umm_codebook
            )

        # Prompt
        if batched_prompt_wav_path is None:
            batched_prompt_umm_token = None
            batched_prompt_wav = None
            target_bn_len = 6 * self.mel_frame_rate
        else:
            batched_prompt_wavlen = []
            batched_prompt_wav = []
            for bidx, prompt_wav_path in enumerate(batched_prompt_wav_path):

                wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True)
                # inputs["gt_wav"] = wav
                wav = torch.FloatTensor(wav).unsqueeze(0)
                wav = wav.to(device)
                batched_prompt_wav.append(wav)
                batched_prompt_wavlen.append(wav.shape[-1])

            max_wavlen = max(batched_prompt_wavlen)
            batched_prompt_wav = torch.cat(
                [
                    F.pad(
                        prompt_wav,
                        [0, max_wavlen - prompt_wav.shape[-1]],
                        "constant",
                        0,
                    )
                    for prompt_wav in batched_prompt_wav
                ],
                dim=0,
            )

            if getattr(self.model.hp, "use_window_mask", False):
                batched_prompt_wav, target_bn_len = self.align_wav_for_chunk(
                    batched_prompt_wav,
                    self.mel_config["sampling_rate"],
                    self.umm_frame_rate,
                    self.mel_frame_rate,
                    0 if batched_text_id is None else batched_text_id.shape[-1],
                )
            else:
                if self.infer_type == "diffusion-vocoder":
                    batched_prompt_wav, target_bn_len = self.align_wav(
                        batched_prompt_wav,
                        self.token_sample_rate,
                        self.mel_config["sampling_rate"],
                        self.umm_frame_rate,
                        self.mel_frame_rate,
                    )
                elif self.infer_type == "ar-diffusion-vocoder":
                    wav_divide = (
                        4800
                        if (self.umm_frame_rate == 25 and self.mel_frame_rate == 40)
                        else 600
                    )
                    batched_prompt_wav, target_bn_len = self.align_wav2(
                        batched_prompt_wav,
                        wav_divide,
                        self.mel_config["sampling_rate"],
                        self.umm_frame_rate,
                        batched_syn_umm_token.shape[1],
                    )
            if self.umm is not None:
                batched_prompt_umm_token = self.wav2token(batched_prompt_wav)
                if hasattr(self, "umm_codebook"):
                    batched_prompt_umm_token = F.embedding(
                        batched_prompt_umm_token, self.umm_codebook
                    )

            else:
                batched_prompt_umm_token = None

        # TODO: Text
        # if prompt_text_id is not None and syn_text_id is not None:
        #     text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
        #     text_id = F.pad(text_id, (0, 1), "constant", 1)
        #     inputs["frontend"] = {
        #             "phone": text_id[0:1, :].to(device),
        #             "tone": text_id[1:2, :].to(device),
        #             "word_seg": text_id[2:3, :].to(device),
        #         }
        #     #null_text_id = torch.ones_like(text_id)
        #     #inputs["null_frontend"] = {
        #     #        "phone": null_text_id[0:1, :].to(device),
        #     #        "tone": null_text_id[1:2, :].to(device),
        #     #        "word_seg": null_text_id[2:3, :].to(device),
        #     #    }
        #     if self.use_phone_lang:
        #         inputs["frontend"]["lang"] = text_id[3:4, :].to(device)
        #     #    inputs["null_frontend"]["lang"] = null_text_id[3:4, :].to(device)
        batched_prompt_text_id = None

        if self.infer_type in ["diffusion-vocoder", "ar-diffusion-vocoder"]:

            if self.use_wvae_vocoder:
                if batched_prompt_wav is None:
                    inputs["prompt_bn"] = (
                        torch.ones(
                            [bs, self.bn_config["bn_dim"], target_bn_len], device=device
                        )
                        * self.bn_config["bn_padding"]
                    )
                else:
                    if len(batched_prompt_wav.shape) == 2:
                        batched_prompt_wav = batched_prompt_wav.unsqueeze(1)
                    batched_encoder_out = self.wvae.encode(batched_prompt_wav)
                    batched_crop_bn, _, _ = self.wvae.sample(
                        batched_encoder_out, deterministic=False
                    )
                    batched_crop_bn = self.bn_norm.norm_mel(batched_crop_bn)
                    inputs["prompt_bn"] = batched_crop_bn[:, :, :target_bn_len]

                inputs["prompt_length"] = target_bn_len
                if not self.without_prefix:
                    inputs["bn_ctx"] = inputs["prompt_bn"].transpose(1, 2)
                else:
                    if getattr(self.model.hp, "use_window_mask", False):
                        inputs["bn_ctx"] = (
                            torch.ones([bs, 0, self.bn_config["bn_dim"]], device=device)
                            * self.bn_config["bn_padding"]
                        )
                    else:
                        inputs["prompt_length"] = 0
                        inputs["bn_ctx"] = (
                            torch.ones(
                                [
                                    bs,
                                    int(
                                        batched_syn_umm_token.shape[1]
                                        / self.umm_frame_rate
                                        * self.mel_frame_rate
                                    ),
                                    self.bn_config["bn_dim"],
                                ],
                                device=device,
                            )
                            * self.bn_config["bn_padding"]
                        )
            else:
                # TODO: mel
                raise NotImplementedError
        else:
            raise NotImplementedError
        inputs["uttid"] = batched_uttid

        return inputs, batched_prompt_umm_token, batched_syn_umm_token

    def make_cfg_input(self, inputs):
        if self.text_cfg_w != 1:
            if "frontend" in inputs:
                inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
                inputs["frontend"]["phone"][1, :] = 1
                inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
                inputs["frontend"]["tone"][1, :] = 1
                inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(
                    2, 1
                )
                inputs["frontend"]["word_seg"][1, :] = 1
                if "lang" in inputs["frontend"]:
                    inputs["frontend"]["lang"] = inputs["frontend"]["lang"].repeat(2, 1)
                    inputs["frontend"]["lang"][1, :] = 1

            token_key = "token"
            if "all_token" in inputs:
                token_key = "all_token"
            if self.token_config["token_cfg"]:
                inputs[token_key] = torch.cat(
                    (
                        inputs[token_key],
                        torch.ones_like(inputs[token_key])
                        * self.token_config["token_padding"],
                    ),
                    0,
                )
            else:
                inputs[token_key] = inputs[token_key].repeat(2, 1)

            if self.use_wvae_vocoder:
                bn_ctx_key = "bn_ctx"
                if "all_bn_ctx" in inputs:
                    bn_ctx_key = "all_bn_ctx"
                if "prompt_bn" in inputs:
                    inputs["prompt_bn"] = inputs["prompt_bn"].repeat(2, 1, 1)
                inputs[bn_ctx_key] = inputs[bn_ctx_key].repeat(2, 1, 1)
            else:
                inputs["prompt_mel"] = inputs["prompt_mel"].repeat(2, 1, 1)
                inputs["mel_ctx"] = inputs["mel_ctx"].repeat(2, 1, 1)
        return inputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.infer_type == "vocoder":
            # uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch
            self.wvae_reconstruct(batch)
        else:
            inputs, prompt_umm_token, syn_umm_token = self.prepare_features(batch)
            # for key in inputs:
            #    if torch.is_tensor(inputs[key]):
            #        print(f"\t {key} -> shape={inputs[key].shape}")
            #    else:
            #        print(f"\t {key} -> {inputs[key]}")
            if inputs is None:
                return
            inputs["token"] = syn_umm_token
            if not self.without_prefix:
                inputs["token"] = torch.cat([prompt_umm_token, inputs["token"]], dim=1)
            if self.text_cfg_w != 1:
                inputs = self.make_cfg_input(inputs)
            if self.diffusion_precision == "bf16":
                dtype = torch.bfloat16
            elif self.diffusion_precision == "fp16":
                dtype = torch.float16
            elif self.diffusion_precision == "fp32":
                dtype = torch.float32
            else:
                raise NotImplementedError
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                out_mel = self.model.inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    rescale_factor=self.rescale_factor,
                )
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                out_mel = out_mel[:, :, inputs["prompt_length"] :]
            out_mel = out_mel.float()
            if self.use_wvae_vocoder:
                z = out_mel
                z = self.bn_norm.denorm_mel(z)
                output_wavs = self.wvae_decode(z)

            else:
                out_mel = self.mel_norm.denorm_mel(out_mel)
                out_mel = torch.clamp(out_mel, min=-8.5, max=3.5)
                output_wavs = self.vocoder(out_mel)

            batched_audio = output_wavs.cpu().numpy()
            if self.bn_config["wav_norm"] and inputs["scale"] is not None:
                batched_audio = batched_audio * torch.as_tensor(inputs["scale"]) / 0.95
            batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)

            for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
                if self.save_prompt:
                    prompt_wav = inputs["gt_wav"][bidx]
                    audio = np.concatenate([prompt_wav, np.ones([10]), audio])

                output_path = os.path.join(self.output_dir, uttid + ".wav")
                if "syn_wavlen" in inputs:
                    audio = audio[..., : inputs["syn_wavlen"][bidx]]
                save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])

            return torch.from_numpy(batched_audio)

    def wvae_reconstruct(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        for item in batch:
            uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = item
            wav, _ = librosa.load(
                syn_wav_path,
                sr=self.mel_config["sampling_rate"],
                mono=not getattr(self.mel_config, "stereo", False),
            )
            wav = torch.FloatTensor(wav).unsqueeze(0)
            if self.bn_config["wav_norm"]:
                scale = max(0.001, torch.max(torch.abs(wav)))
                wav = wav / scale * 0.95
            wav = wav.to(device)
            if len(wav.shape) == 2:
                wav = wav.unsqueeze(1)
            reconstruct_wav, _, _ = self.wvae(wav)
            output_path = os.path.join(self.output_dir, uttid + ".wav")
            save_wav(reconstruct_wav.cpu().numpy(), output_path)

    def setup(self, stage):
        device = f"cuda:{self.local_rank}"
        if self.infer_type != "vocoder" and self.umm_ckpt_path:
            if self.umm_type == "USM":
                self.umm = prepare_usm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM":
                self.umm = prepare_umm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMMv2":
                self.umm = prepare_ummv2(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM_music":
                self.umm = prepare_umm_music(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM_conv":
                self.umm = prepare_umm_conv(self.umm_ckpt_path, device)
            elif self.umm_type in ["UMM_dualconv", "UMM_dualconvV1", "UMM_dualconvV3"]:
                self.umm = prepare_umm_dualconv(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM2":
                hparams = load_hyperpyyaml(open(self.umm_ckpt_path, "r", encoding="utf-8"))
                # model = hparams["frontend"]
                model = hparams["model"]
                required_modules = hparams["required_modules"]
                for module_name, loader_config in hparams["required_modules"].items():
                    print(f"loading module {module_name}...")
                    _args = {k: v for k, v in loader_config.items() if k != "loader" and k != "initializer"}
                    loader = loader_config["loader"](**_args)
                    self.umm = loader.nn_load_model(model)
            else:
                raise NotImplementedError
        else:
            self.umm = None

        if self.use_wvae_vocoder:
            self.wvae = self.bn_config["vocoder_model"](local_rank=self.local_rank)[
                "vocoder"
            ]
            # self.wvae_encoder = load_torch_script(
            #     model_path=self.bn_config['wvae_encoder_path'],
            #     rank=self.trainer.local_rank,
            #     cache_dir=self.bn_config['wvae_cache_dir']
            # )
            # self.wvae_decoder = load_torch_script(
            #     model_path=self.bn_config['wvae_decoder_path'],
            #     rank=self.trainer.local_rank,
            #     cache_dir=self.bn_config['wvae_cache_dir']
            # )
            # self.wvae = Wave(self.wvae_encoder, self.wvae_decoder,
            #     version=self.bn_config['wvae_version'],
            #     hop_size=self.bn_config['wvae_encoder_hop_size'],
            #     win_size=self.bn_config['wvae_encoder_win_size'],
            #     bn_hop_size=self.bn_config['hop_size'])
        else:
            self.vocoder = prepare_vocoder(self.vocoder_ckpt_path, device)

        if self.umm_codebook_path is not None:
            self.umm_codebook = prepare_umm_codebook(self.umm_codebook_path, device)

    def align_wav_for_chunk(
        self, wav, sampling_rate, umm_frame_rate, mel_frame_rate, text_len
    ):
        raise NotImplementedError

    def batched_data(self, batch):
        batched_prompt_text_id = None
        batched_prompt_wav_path = None
        batched_syn_text_id = None
        batched_syn_wav_path = None
        batched_syn_umm_token = None
        batched_uttid = None
        batched_scale = None
        assert self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]

        def batching(container, data):
            if data is None or data is "":
                if container is not None:
                    raise ValueError("try to batching data 'None'")
            else:
                if container is None:
                    container = []
                container.append(data)
            return container

        def assert_same_batchsize_if_exsist(container, bs):
            assert (container is None) or (
                container is not None and len(container) == bs
            )

        for item in batch:
            (
                prompt_text_id,
                prompt_wav_path,
                syn_text_id,
                syn_wav_path,
                syn_umm_token,
                uttid,
            ) = (None, None, None, None, None, None)
            if self.infer_type == "ar-diffusion-vocoder":
                (prompt_text_id, syn_text_id, prompt_wav_path, syn_umm_token, uttid) = (
                    item[:5]
                )
            elif self.infer_type == "diffusion-vocoder":
                (uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id) = (
                    item[:5]
                )

            if prompt_wav_path:
                if not os.path.isfile(prompt_wav_path):
                    prompt_wav_path = None
                elif os.path.isfile(prompt_wav_path) ^ self.model.hp.use_prompt:
                    logger.warning(
                        f"prompt wav {os.path.isfile(prompt_wav_path)}/{prompt_wav_path} mismatch with config use_prompt={self.model.hp.use_prompt}"
                    )

            batched_uttid = batching(batched_uttid, uttid)
            batched_prompt_wav_path = batching(batched_prompt_wav_path, prompt_wav_path)
            batched_prompt_text_id = batching(batched_prompt_text_id, prompt_text_id)
            batched_syn_text_id = batching(batched_syn_text_id, syn_text_id)
            batched_syn_wav_path = batching(batched_syn_wav_path, syn_wav_path)
            batched_syn_umm_token = batching(batched_syn_umm_token, syn_umm_token)
            if len(batch) == 6:
                batched_scale = batching(batched_scale, item[-1])
        assert batched_uttid is not None
        bs = len(batched_uttid)
        assert_same_batchsize_if_exsist(batched_prompt_wav_path, bs)
        assert_same_batchsize_if_exsist(batched_prompt_text_id, bs)
        assert_same_batchsize_if_exsist(batched_syn_text_id, bs)
        assert_same_batchsize_if_exsist(batched_syn_wav_path, bs)
        assert_same_batchsize_if_exsist(batched_syn_umm_token, bs)
        assert_same_batchsize_if_exsist(batched_scale, bs)
        return (
            batched_uttid,
            batched_prompt_text_id,
            batched_prompt_wav_path,
            batched_syn_text_id,
            batched_syn_wav_path,
            batched_syn_umm_token,
            batched_scale,
        )


class ChunkInfer(DiffusionU2SInfer):
    def __init__(
        self,
        diffusion_ckpt_path,
        umm_ckpt_path,
        vocoder_ckpt_path,
        output_dir,
        infer_type,
        umm_frame_rate,
        mel_frame_rate,
        mel_config,
        seed=1996,
        save_prompt=False,
        umm_type="UMM",  # UMM or USM
        umm_codebook_path=None,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        text_cfg_w=1,
        rescale_factor=0.7,
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        use_phone_lang=False,
        token_chunk_size=25,
        token_chunk_overlap=0,
        without_prefix=False,
        **kwargs,
    ):
        super().__init__(
            diffusion_ckpt_path,
            umm_ckpt_path,
            vocoder_ckpt_path,
            output_dir,
            infer_type,
            umm_frame_rate,
            mel_frame_rate,
            mel_config,
            seed,
            save_prompt,
            umm_type,
            umm_codebook_path,
            diffusion_precision,
            diffusion_nfe,
            diffusion_sampler,
            text_cfg_w,
            rescale_factor,
            use_wvae_vocoder,
            bn_config,
            token_config,
            use_phone_lang,
            without_prefix=without_prefix,
        )
        self.token_chunk_size = token_chunk_size
        self.token_chunk_overlap = token_chunk_overlap

        if hasattr(self.model.hp, "window_size"):
            self.attention_window_size = self.model.hp.window_size[-1]
        else:
            self.attention_window_size = None

        logger.info("ChunkInfer:")
        args_dict = locals()
        del args_dict["self"]
        logger.info(", ".join(f"{k}={v}\n" for k, v in args_dict.items()))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, prompt_umm_token, syn_umm_token = self.prepare_features(batch)
        if inputs is None:
            return

        if self.diffusion_precision == "bf16":
            dtype = torch.bfloat16
        elif self.diffusion_precision == "fp16":
            dtype = torch.float16
        elif self.diffusion_precision == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError

        out_mel = None
        bs = len(syn_umm_token)
        device = syn_umm_token.device
        start_list = np.array(range(0, syn_umm_token.shape[1], self.token_chunk_size))

        outputs = []

        # To ensure reproduciable
        if not self.without_prefix:
            total_frame = int(
                (prompt_umm_token.shape[1] + syn_umm_token.shape[1])
                / self.umm_frame_rate
                * self.mel_frame_rate
            )
            assert prompt_umm_token is not None
            inputs["all_token"] = torch.cat([prompt_umm_token, syn_umm_token], dim=1)
        else:
            total_frame = int(
                (syn_umm_token.shape[1]) / self.umm_frame_rate * self.mel_frame_rate
            )
            inputs["all_token"] = syn_umm_token

        self.model.clear_cache(self.diffusion_nfe, total_frame, bs=bs)

        if self.text_cfg_w != 1:
            inputs = self.make_cfg_input(inputs)

        for i, start in enumerate(start_list):
            if i < len(start_list) - 1:
                end = min(
                    start_list[i + 1] + self.token_chunk_overlap, syn_umm_token.shape[1]
                )
                if end <= syn_umm_token.shape[1]:
                    current_overlap = self.token_chunk_overlap
                else:
                    end = syn_umm_token.shape[1]
                    current_overlap = end - start_list[i + 1]
                valid_chunk_size = self.token_chunk_size
            else:
                end = syn_umm_token.shape[1]
                valid_chunk_size = end - start
                current_overlap = 0
            valid_chunk_size = int(
                valid_chunk_size / self.umm_frame_rate * self.mel_frame_rate
            )

            if i == 0:
                inputs["token"] = inputs["all_token"][:, start:end]
            else:
                inputs["token"] = torch.cat(
                    [
                        inputs["token"][:, :-prev_overlap],
                        inputs["all_token"][:, start:end],
                    ],
                    dim=1,
                )

            token_len = inputs["token"].shape[1]
            mel_len = int(token_len / self.umm_frame_rate * self.mel_frame_rate)

            if out_mel is not None:
                prompt_length += out_mel.shape[2]
            else:
                prompt_length = inputs["bn_ctx"].shape[1]
            pad_len = valid_chunk_size
            inputs["bn_ctx"] = F.pad(
                inputs["bn_ctx"],
                (0, 0, 0, pad_len),
                "constant",
                self.bn_config["bn_padding"],
            )

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                out_mel = self.model.inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    rescale_factor=self.rescale_factor,
                    use_cache=True,
                    cached_v_len=prompt_length,
                )
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                out_mel = out_mel[
                    :, :, prompt_length : prompt_length + valid_chunk_size
                ]
            out_mel = out_mel.float()  # [B, C, T]
            outputs.append(out_mel)
            prev_overlap = current_overlap

        self.model.clear_cache(self.diffusion_nfe)
        outputs = torch.cat(outputs, dim=-1)
        if self.use_wvae_vocoder:
            z = outputs
            z = self.bn_norm.denorm_mel(z)
            output_wavs = self.wvae_decode(z)
        else:
            out_mel = self.mel_norm.denorm_mel(out_mel)
            out_mel = torch.clamp(out_mel, min=-8.5, max=3.5)
            output_wavs = self.vocoder(out_mel)

        batched_audio = output_wavs.cpu().numpy()
        if self.bn_config["wav_norm"] and inputs["scale"] is not None:
            batched_audio = batched_audio * torch.as_tensor(inputs["scale"]) / 0.95
        batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)

        groundtruth.emit("vocoder", data={"out_mel": outputs, "audio": batched_audio})

        for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            if self.save_prompt:
                prompt_wav = inputs["gt_wav"][bidx]
                audio = np.concatenate([prompt_wav, np.ones([10]), audio])

            output_path = os.path.join(self.output_dir, uttid + ".wav")
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])
        return torch.from_numpy(batched_audio)

    def align_wav_for_chunk(
        self, wav, sampling_rate, umm_frame_rate, mel_frame_rate, text_len
    ):
        umm_hop = sampling_rate // umm_frame_rate
        mel_hop = sampling_rate // mel_frame_rate
        current_bn_len = wav.shape[-1] // mel_hop
        # print(f"{wav.shape=}, {current_bn_len=}")
        pad_bn_len = (current_bn_len + text_len) % self.attention_window_size
        if pad_bn_len == 0:
            target_bn_len = current_bn_len
        else:
            target_bn_len = (self.attention_window_size - pad_bn_len) + current_bn_len
        target_umm_len = np.ceil(target_bn_len * umm_frame_rate / mel_frame_rate)
        target_wav_len = int(target_umm_len * umm_hop)
        pad_wav_len = target_wav_len - wav.shape[-1]
        if pad_wav_len > 0:
            wav = F.pad(wav, (0, pad_wav_len), "constant", 0)
        return wav, target_bn_len


"""
    def prepare_features(self, batch):
        (
            batched_uttid, 
            batched_prompt_text_id, 
            batched_prompt_wav_path, 
            batched_syn_text_id,
            batched_syn_wav_path,
            batched_syn_umm_token,
            batched_scale
        ) = self.batched_data(batch)

        bs = len(batched_uttid)
        device = f"cuda:{self.local_rank}"
        inputs = dict()
        # if len(batch) == 6:
        #     inputs["scale"] = batch[5]
        # else:
        #     inputs["scale"] = None

        # TODO: Text
        # if prompt_text_id is not None and  syn_text_id is not None:
        #     text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
        #     text_id = F.pad(text_id, (0, 1), "constant", 1)
        #     inputs["frontend"] = {
        #             "phone": text_id[0:1, :].to(device),
        #             "tone": text_id[1:2, :].to(device),
        #             "word_seg": text_id[2:3, :].to(device),
        #     }
        #     if self.use_phone_lang:
        #         inputs["frontend"]["lang"] = text_id[3:4, :].to(device)
        # else:
        #     text_id = None
        batched_text_id = None
        
        # Syn
        if self.infer_type == "diffusion-vocoder": 
            assert batched_syn_wav_path is not None
            inputs["gt_wav"] = []
            scales = []
            syn_wavs = []
            syn_wavlens = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
                inputs["gt_wav"].append(wav)
                wav = torch.FloatTensor(wav).unsqueeze(0)
                if self.bn_config['wav_norm']:
                    if batched_scale is None:
                        scale = max(0.001, torch.max(torch.abs(wav)))
                        wav = wav / scale * 0.95
                        scales.append(scale.item())
                    else:
                        scales.append(batched_scale[bidx])
                wav = wav.to(device)
                #syn_wav = self.align_wav(wav, self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate)
                syn_wav = wav
                syn_wavs.append(syn_wav)
                syn_wavlens.append(syn_wav.shape[-1])

            batched_scale = scales
            max_wavlen = max(syn_wavlens)
            syn_wavs = torch.cat([F.pad(syn_wav,[0,max_wavlen-syn_wav.shape[-1]],"constant",0) for syn_wav in syn_wavs],dim=0)
            batched_syn_umm_token = self.wav2token(syn_wavs)
            inputs["scale"] = scales
            inputs["syn_wavlen"] = syn_wavlens
        elif self.infer_type == "ar-diffusion-vocoder":
            assert batched_syn_umm_token is not None

            batched_syn_umm_token_len = batched_syn_umm_token[0].shape[-1]
            for bidx, syn_umm_token in enumerate(batched_syn_umm_token):
                assert syn_umm_token.shape[-1] == batched_syn_umm_token_len
                if len(syn_umm_token.shape) == 1:
                    batched_syn_umm_token[bidx] = syn_umm_token.unsqueeze(0)
            batched_syn_umm_token = torch.cat(batched_syn_umm_token,dim=0)
            inputs["scale"] = batched_scale
            # np.savetxt(f"ar-diffusion-vocoder.syn_umm_token.txt", syn_umm_token.detach().cpu().numpy().reshape(-1,1),fmt="%d")
        else:
            raise NotImplementedError
        
        if hasattr(self,"umm_codebook"):
            batched_syn_umm_token = F.embedding(batched_syn_umm_token, self.umm_codebook)

        # if os.path.isfile(prompt_wav_path) ^ self.model.hp.use_prompt:
        #     logger.warning(f"prompt wav {os.path.isfile(prompt_wav_path)}/{prompt_wav_path} mismatch with config use_prompt={self.model.hp.use_prompt}")

        # Prompt
        if batched_prompt_wav_path is None:
            batched_prompt_umm_token = None
            batched_prompt_wav = None
            target_bn_len = 6 * self.mel_frame_rate
        else:
            batched_prompt_wavlen = []
            batched_prompt_wav = []
            for bidx,prompt_wav_path in enumerate(batched_prompt_wav_path):
                assert os.path.isfile(prompt_wav_path)
                wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True) 
                # inputs["gt_wav"] = wav
                wav = torch.FloatTensor(wav).unsqueeze(0)
                wav = wav.to(device)
                batched_prompt_wav.append(wav)
                batched_prompt_wavlen.append(wav.shape[-1])

            max_wavlen = max(batched_prompt_wavlen)
            batched_prompt_wav= torch.cat([F.pad(prompt_wav,[0,max_wavlen-prompt_wav.shape[-1]],"constant",0) for prompt_wav in batched_prompt_wav],dim=0)
            batched_prompt_wav, target_bn_len = self.align_wav_for_chunk(batched_prompt_wav, 
                self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate,
                0 if batched_text_id is None else batched_text_id.shape[-1]
            )

            if self.umm != None:     
                batched_prompt_umm_token = self.wav2token(batched_prompt_wav)
                if hasattr(self, "umm_codebook"):
                    batched_prompt_umm_token = F.embedding(batched_prompt_umm_token, self.umm_codebook)
            else:
                batched_prompt_umm_token = None
        

        if self.infer_type in ["diffusion-vocoder", "ar-diffusion-vocoder"]: 
            #inputs["token"] = torch.cat([prompt_umm_token, syn_umm_token], dim=1)
            #token_len = inputs["token"].shape[1]
            #mel_len = int(token_len / self.umm_frame_rate * self.mel_frame_rate)

            if self.use_wvae_vocoder:
                if batched_prompt_wav is None:
                    inputs["prompt_bn"] = torch.ones([bs, self.bn_config["bn_dim"],target_bn_len],device=device)*self.bn_config["bn_padding"]
                else:
                    if len(batched_prompt_wav.shape) == 2:
                        batched_prompt_wav = batched_prompt_wav.unsqueeze(1)
                    batched_encoder_out = self.wvae.encode(batched_prompt_wav)
                    batched_crop_bn, _, _ = self.wvae.sample(batched_encoder_out, deterministic=False)
                    batched_crop_bn = self.bn_norm.norm_mel(batched_crop_bn)
                    inputs["prompt_bn"] = batched_crop_bn[:, :, :target_bn_len]
                    
                inputs["prompt_length"] = target_bn_len
                if not self.without_prefix:
                    inputs["bn_ctx"] = inputs["prompt_bn"].transpose(1, 2)
                else: 
                    # inputs["bn_ctx"] = torch.ones([1,int(syn_umm_token.shape[1] / self.umm_frame_rate * self.mel_frame_rate) ,self.bn_config["bn_dim"]],device=device)*self.bn_config["bn_padding"]
                    inputs["bn_ctx"] = torch.ones([bs, 0, self.bn_config["bn_dim"]],device=device)*self.bn_config["bn_padding"]
            else:
                # TODO: mel
                raise NotImplementedError

        else:
            raise NotImplementedError

        inputs["uttid"] = batched_uttid
        
        return inputs, batched_prompt_umm_token, batched_syn_umm_token
"""


class ChunkInfer2(DiffusionU2SInfer):
    def __init__(
        self,
        diffusion_ckpt_path,
        umm_ckpt_path,
        vocoder_ckpt_path,
        output_dir,
        infer_type,
        umm_frame_rate,
        mel_frame_rate,
        mel_config,
        seed=1996,
        save_prompt=False,
        umm_type="UMM",
        umm_codebook_path=None,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        text_cfg_w=1,
        rescale_factor=0.7,
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        use_phone_lang=False,
        without_prefix=True,
        padding_mode="token_padding",
        **kwargs,
    ):
        super().__init__(
            diffusion_ckpt_path,
            umm_ckpt_path,
            vocoder_ckpt_path,
            output_dir,
            infer_type,
            umm_frame_rate,
            mel_frame_rate,
            mel_config,
            seed,
            save_prompt,
            umm_type,
            umm_codebook_path,
            diffusion_precision,
            diffusion_nfe,
            diffusion_sampler,
            text_cfg_w,
            rescale_factor,
            use_wvae_vocoder,
            bn_config,
            token_config,
            use_phone_lang,
            without_prefix,
        )
        self.padding_mode = padding_mode
        if self.infer_type == "ar-diffusion-vocoder":
            assert self.padding_mode == "token_padding"
        assert (
            self.model.hp.window_type == "blockwise"
            and self.model.hp.window_size[1] >= 1
        )
        self.bn_chunk_size = self.model.hp.window_size[1]
        self.token_chunk_size = (
            self.bn_chunk_size * self.umm_frame_rate // self.mel_frame_rate
        )
        self.token_overlap = int(np.prod(self.model.hp.token_downscales))
        self.mem_efficient = kwargs.get("mem_efficient", False)

        print("enable memory efficient:", self.mem_efficient)
        self.context_duration = int(kwargs.get("context_duration", 60))
        if self.infer_type == "ar-diffusion-vocoder":
            self.context_duration = 0
        self.concat_context = (not self.without_prefix) and kwargs.get("concat_context", False) and self.context_duration > 0

    def prepare_features(self, batch) -> None:
        (
            batched_uttid,
            batched_prompt_text_id,
            batched_prompt_wav_path,
            batched_syn_text_id,
            batched_syn_wav_path,
            batched_syn_umm_token,
            batched_scale,
        ) = self.batched_data(batch)

        bs = len(batched_uttid)
        batched_text_id = None
        device = f"cuda:{self.local_rank}"
        inputs = dict()

        # syn
        if self.infer_type == "diffusion-vocoder":
            assert batched_syn_wav_path is not None
            inputs["gt_wav"] = []
            scales = []
            ori_syn_wavlens = []
            syn_wavs = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                syn_wav, sr = librosa.load(
                    syn_wav_path, sr=None, mono=True
                )
                inputs["gt_wav"].append(syn_wav)
                ori_syn_wavlens.append(syn_wav.shape[-1])
                if sr != self.token_sample_rate:
                    syn_wav, sr = librosa.load(
                        syn_wav_path, sr=self.token_sample_rate, mono=True
                    )
                syn_wavlens.append(syn_wav.shape[-1])
                if self.bn_config["wav_norm"]:
                    if batched_scale is None:
                        scale = max(0.001, torch.max(torch.abs(syn_wav)))
                        syn_wav = syn_wav / scale * 0.95
                        scales.append(scale.item())
                    else:
                        scales.append(batched_scale[bidx])
                syn_wavs.append(syn_wav)

            if self.padding_mode == "wav_padding":
                padding_value = 0.0
                max_wavlen = max(syn_wavlens)
                syn_wavs = torch.stack(
                    [
                        F.pad(
                            syn_wav,
                            [0, max_wavlen - syn_wavlen],
                            "constant",
                            padding_value,
                        )
                        for syn_wav, syn_wavlen in zip(syn_wavs, syn_wavlens)
                    ],
                    dim=0,
                )
                batched_syn_umm_token = self.wav2token(syn_wavs)

            elif self.padding_mode == "token_padding":
                padding_value = self.token_config["token_padding"]
                batched_syn_umm_token = [
                    self.wav2token(syn_wav.unsqueeze(0)) for syn_wav in syn_wavs
                ]
                max_syn_umm_token_len = max(
                    [syn_umm_token.shape[-1] for syn_umm_token in batched_syn_umm_token]
                )
                batched_syn_umm_token = torch.cat(
                    [
                        F.pad(
                            syn_wav,
                            [0, max_syn_umm_token_len - syn_umm_token.shape[-1]],
                            "constant",
                            padding_value,
                        )
                        for syn_umm_token in batched_syn_umm_token
                    ],
                    dim=0,
                )
            else:
                raise ValueError(f"no padding implement named '{self.padding_mode}'")

            inputs["scale"] = scales
            inputs["syn_wavlen"] = ori_syn_wavlens
        elif self.infer_type == "ar-diffusion-vocoder":
            assert (
                batched_syn_umm_token is not None
                and self.padding_mode == "token_padding"
            )
            padding_value = self.token_config["token_padding"]
            syn_umm_token_lens = [
                syn_umm_token.shape[-1] for syn_umm_token in batched_syn_umm_token
            ]
            max_syn_umm_token_len = max(syn_umm_token_lens)
            syn_wavlens = [
                syn_umm_token_len * (self.bn_config["sample_rate"]  // self.umm_frame_rate)
                for syn_umm_token_len in syn_umm_token_lens
            ]
            batched_syn_umm_token = torch.stack(
                [
                    F.pad(
                        syn_umm_token,
                        [0, max_syn_umm_token_len - syn_umm_token.shape[-1]],
                        "constant",
                        padding_value,
                    )
                    for syn_umm_token in batched_syn_umm_token
                ],
                dim=0,
            )
            if batched_syn_umm_token.ndim == 3:
                batched_syn_umm_token = batched_syn_umm_token.squeeze(1)
            inputs["scale"] = batched_scale
            inputs["syn_wavlen"] = syn_wavlens
        else:
            raise NotImplementedError(f"{self.infer_type=}")

        # prompt
        assert self.use_wvae_vocoder
        if self.without_prefix:
            batched_prompt_umm_token = None
            inputs["prompt_bn"] = torch.zeros(
                [bs, self.bn_config["bn_dim"], 0], device=device
            )
            inputs["prompt_length"] = 0
        else:
            assert batched_prompt_wav_path is not None
            prompt_wavlens = []
            prompt_wavs = []
            for bidx, prompt_wav_path in enumerate(batched_prompt_wav_path):
                prompt_wav, _ = librosa.load(
                    prompt_wav_path, sr=self.token_sample_rate, mono=True
                )
                prompt_wavlens.append(prompt_wav.shape[-1])
                prompt_wavs.append(prompt_wav)

            if self.padding_mode == "wav_padding":
                padding_value = 0.0
                max_wavlen = max(prompt_wavlens)
                prompt_wavs = torch.stack(
                    [
                        F.pad(
                            prompt_wav,
                            [0, max_wavlen - prompt_wavlen],
                            "constant",
                            padding_value,
                        )
                        for prompt_wav, prompt_wavlen in zip(
                            prompt_wavs, prompt_wavlens
                        )
                    ],
                    dim=0,
                )  # B,T
                # prompt token
                batched_prompt_umm_token = self.wav2token(prompt_wavs)
                prompt_token_lens = [batched_prompt_umm_token.shape[-1]] * bs
                # prompt bn
                target_bn_len = (
                    batched_prompt_umm_token.shape[-1]
                    * self.mel_frame_rate
                    // self.umm_frame_rate
                )
                prompt_wavs = prompt_wavs.unsqueeze(1)  # B,1,T
                batched_encoder_out = self.wvae.encode(prompt_wavs)
                batched_crop_bn, _, _ = self.wvae.sample(
                    batched_encoder_out, deterministic=False
                )
                batched_crop_bn = self.bn_norm.norm_mel(batched_crop_bn)

            elif self.padding_mode == "token_padding":
                # prompt token
                padding_value = self.token_config["token_padding"]
                batched_prompt_umm_token = [
                    self.wav2token(prompt_wav.unsqueeze(0))
                    for prompt_wav in prompt_wavs
                ]
                prompt_token_lens = [
                    prompt_umm_token.shape[-1]
                    for prompt_umm_token in batched_prompt_umm_token
                ]
                max_prompt_umm_token_len = max(prompt_token_lens)
                batched_prompt_umm_token = torch.cat(
                    [
                        F.pad(
                            prompt_umm_token,
                            [0, max_prompt_umm_token_len - prompt_umm_token.shape[-1]],
                            "constant",
                            padding_value,
                        )
                        for prompt_umm_token in batched_prompt_umm_token
                    ],
                    dim=0,
                )

                # prompt bn
                padding_value = self.bn_config["bn_padding"]
                batched_crop_bn = []
                prompt_bn_lens = []
                for prompt_wav in prompt_wavs:
                    prompt_wav = prompt_wav.unsqueeze(0).unsqueeze(1)  # 1,1,T
                    encoder_out = self.wvae.encode(prompt_wav)
                    crop_bn, _, _ = self.wvae.sample(encoder_out, deterministic=False)
                    crop_bn = self.bn_norm.norm_mel(crop_bn)
                    prompt_bn_lens.append(crop_bn.shape[-1])
                    batched_crop_bn.append(crop_bn)
                target_bn_len = max(prompt_bn_lens)
                batched_crop_bn = torch.cat(
                    [
                        F.pad(
                            crop_bn,
                            [0, target_bn_len - crop_bn.shape[-1]],
                            "constant",
                            padding_value,
                        )
                        for crop_bn in batched_crop_bn
                    ],
                    dim=0,
                )
            else:
                raise ValueError(f"no padding implement named '{self.padding_mode}'")
            inputs["prompt_bn"] = batched_crop_bn[:, :, :target_bn_len]
            inputs["prompt_length"] = target_bn_len

        inputs["all_bn_ctx"] = inputs["prompt_bn"].transpose(1, 2)
        inputs["uttid"] = batched_uttid
        return inputs, batched_prompt_umm_token, batched_syn_umm_token

    def prepare_features_prefix(self, batch) -> None:
        (
            batched_uttid,
            batched_prompt_text_id,
            batched_prompt_wav_path,
            batched_syn_text_id,
            batched_syn_wav_path,
            batched_syn_umm_token,
            batched_scale,
        ) = self.batched_data(batch)

        bs = len(batched_uttid)
        batched_text_id = None
        device = f"cuda:{self.local_rank}"
        inputs = dict()

        """ compute umm token and bn """
        # syn
        if self.infer_type == "diffusion-vocoder":
            assert batched_syn_wav_path is not None
            ori_syn_wavlens =[]
            inputs["gt_wav"] = []
            scales = []
            syn_umm_token_lens = []
            batched_syn_umm_token = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                syn_wav, sr = librosa.load(
                    syn_wav_path, sr=None, mono=False
                )
                inputs["gt_wav"].append(syn_wav)
                ori_syn_wavlens.append(syn_wav.shape[-1])
                if sr != self.token_sample_rate:
                    syn_wav, sr = librosa.load(
                        syn_wav_path, sr=self.token_sample_rate, mono=True
                    )
                if self.bn_config["wav_norm"]:
                    if batched_scale is None:
                        scale = max(0.001, torch.max(torch.abs(syn_wav)))
                        syn_wav = syn_wav / scale * 0.95
                        scales.append(scale.item())
                    else:
                        scales.append(batched_scale[bidx])
                # syn_wavs.append(syn_wav)
                syn_wav = torch.from_numpy(syn_wav).to(device)
                syn_umm_token = self.wav2token(syn_wav.unsqueeze(0))
                syn_umm_token = syn_umm_token[..., self.context_duration * self.umm_frame_rate:]
                syn_umm_token_lens.append(syn_umm_token.shape[-1])
                batched_syn_umm_token.append(syn_umm_token.squeeze(0))
            inputs["scale"] = scales
            inputs["syn_wavlen"] = ori_syn_wavlens

        else:
            assert (
                batched_syn_umm_token is not None
                and self.padding_mode == "token_padding"
            )
            # padding_value = self.token_config["token_padding"]
            syn_umm_token_lens = []
            ori_syn_wavlens =[]
            for bidx in range(bs):
                if batched_syn_umm_token[bidx].ndim > 1:
                    batched_syn_umm_token[bidx] = batched_syn_umm_token[bidx].squeeze(0)
                syn_umm_token_lens.append(batched_syn_umm_token[bidx].shape[-1])
                ori_syn_wavlens.append(batched_syn_umm_token[bidx].shape[-1] * self.bn_config["sample_rate"]  // self.umm_frame_rate)
            inputs["scale"] = batched_scale
            inputs["syn_wavlen"] = ori_syn_wavlens

        # prompt
        assert self.use_wvae_vocoder
        assert batched_prompt_wav_path is not None
        batched_prompt_umm_token = []
        prompt_umm_token_lens = []
        prompt_bn_lens = []
        batched_crop_bn = []
        batched_prompt_wav = []
        for bidx, prompt_wav_path in enumerate(batched_prompt_wav_path):
            prompt_wav, _ = librosa.load(
                prompt_wav_path, sr=self.token_sample_rate, mono=True
            )
            prompt_wav = torch.from_numpy(prompt_wav).to(device)
            prompt_umm_token = self.wav2token(prompt_wav.unsqueeze(0))
            prompt_umm_token_lens.append(prompt_umm_token[-1])
            batched_prompt_umm_token.append(prompt_umm_token.squeeze(0))
            
            prompt_wav, sr = librosa.load(
                prompt_wav_path, sr=None, mono=False
            )
            batched_prompt_wav.append(prompt_wav)
            prompt_wav = torch.from_numpy(prompt_wav).to(device)
            if len(prompt_wav.shape)==1:
                prompt_wav = prompt_wav.unsqueeze(0)
            encoder_out = self.wvae.encode(prompt_wav.unsqueeze(0))
            crop_bn, _, _ = self.wvae.sample(encoder_out, deterministic=False)
            crop_bn = self.bn_norm.norm_mel(crop_bn)
            prompt_bn_lens.append(crop_bn.shape[-1])
            batched_crop_bn.append(crop_bn.transpose(-1,-2).squeeze(0))
        
        inputs["prompt_bn"] = batched_crop_bn
        inputs["prompt_length"] = prompt_bn_lens
        """ padding to batch """
        # umm
        assert len(batched_syn_umm_token) == len(batched_prompt_wav_path)
        inputs["prompt_bn"] = pad_sequence(
            inputs["prompt_bn"],
            batch_first=True,
            padding_value=self.bn_config["bn_padding"],
        )
        inputs["all_bn_ctx"] = inputs["prompt_bn"]
        inputs["uttid"] = batched_uttid

        return inputs, batched_prompt_umm_token, batched_syn_umm_token

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if not self.without_prefix:
            return self.predict_step_prefix(batch, batch_idx, dataloader_idx)
        
        inputs, prompt_umm_token, syn_umm_token = self.prepare_features(batch)
        if inputs is None:
            return

        if self.diffusion_precision == "bf16":
            dtype = torch.bfloat16
        elif self.diffusion_precision == "fp16":
            dtype = torch.float16
        elif self.diffusion_precision == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError

        bs = len(syn_umm_token)
        device = syn_umm_token.device

        if not self.without_prefix:
            assert prompt_umm_token is not None
            inputs["all_token"] = torch.cat([prompt_umm_token, syn_umm_token], dim=1)
        else:
            inputs["all_token"] = syn_umm_token

        # align for n*token_chunk_size+chunk_overlap (B,T)
        token_len = inputs["all_token"].shape[1]
        n_chunks = math.ceil(token_len / self.token_chunk_size)
        aligned_token_len = (
            math.ceil(token_len / self.token_chunk_size) * self.token_chunk_size
            + self.token_overlap
        )
        token_pad_len = aligned_token_len - token_len
        if token_pad_len > 0:
            inputs["all_token"] = F.pad(
                inputs["all_token"],
                [0, token_pad_len],
                mode="constant",
                value=self.token_config["token_padding"],
            )
        print(
            f"{batch_idx=} token align from {token_len} to {aligned_token_len} ({token_pad_len=} {n_chunks=})"
        )

        # pad bn_ctx (B,T,C)
        bn_ctx_len = inputs["all_bn_ctx"].shape[1]
        aligned_bn_ctx_len = self.bn_chunk_size * n_chunks
        bn_ctx_pad_len = aligned_bn_ctx_len - bn_ctx_len
        print(
            f"{batch_idx=} bn_ctx align from {bn_ctx_len} to {aligned_bn_ctx_len} ({bn_ctx_pad_len=} {n_chunks=})"
        )
        if bn_ctx_pad_len > 0:
            inputs["all_bn_ctx"] = F.pad(
                inputs["all_bn_ctx"],
                [0, 0, 0, bn_ctx_pad_len],
                mode="constant",
                value=self.bn_config["bn_padding"],
            )

        total_frame = math.ceil(token_len * self.mel_frame_rate / self.umm_frame_rate)
        print(f"{batch_idx=} {total_frame=}")
        # apply text cfg
        if self.text_cfg_w != 1:
            inputs = self.make_cfg_input(inputs)

        # chunk inference
        full_mel = []
        self.model.clear_infer_params(
            self.diffusion_nfe, total_frame, bs, self.text_cfg_w, self.mem_efficient
        )
        for chunk_idx in range(n_chunks):
            token_start_index = chunk_idx * self.token_chunk_size - (
                0 if chunk_idx == 0 else self.token_overlap
            )
            token_end_index = (
                chunk_idx + 1
            ) * self.token_chunk_size + self.token_overlap

            bn_start_index = chunk_idx * self.bn_chunk_size
            bn_end_index = min((chunk_idx + 1) * self.bn_chunk_size, total_frame)

            inputs["token"] = inputs["all_token"][:, token_start_index:token_end_index]
            inputs["bn_ctx"] = inputs["all_bn_ctx"][:, bn_start_index:bn_end_index]
            # print(f"{chunk_idx=} token={(token_start_index,token_end_index, token_end_index-token_start_index)}({inputs['all_token'].shape[1]}, {inputs['token'].shape[1]})")
            # print(f"{chunk_idx=} bn_ctx={(bn_start_index, bn_end_index, bn_end_index-bn_start_index)}({inputs['all_bn_ctx'].shape[1]} {inputs['bn_ctx'].shape[1]})")
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                chunk_mel = self.model.chunk_inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    use_infer_params=True,
                    first=chunk_idx == 0,
                )
            # print(f"{chunk_idx=} mel={chunk_mel.shape=}")
            self.model.update_infer_params(
                bn_end_index - bn_start_index, last=chunk_idx + 1 >= n_chunks - 1
            )
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                full_mel.append(chunk_mel)
        self.model.clear_infer_params(self.diffusion_nfe)

        full_mel = torch.cat(full_mel, dim=-1)
        full_mel = full_mel[:, inputs["prompt_length"] :, :]
        if self.use_wvae_vocoder:
            z = full_mel
            z = self.bn_norm.denorm_mel(z)
            output_wavs = self.wvae_decode(z)
        else:
            full_mel = self.mel_norm.denorm_mel(full_mel)
            full_mel = torch.clamp(full_mel, min=-8.5, max=3.5)
            output_wavs = self.vocoder(full_mel)

        batched_audio = output_wavs.cpu().numpy()
        if self.bn_config["wav_norm"] and inputs["scale"] is not None:
            batched_audio = batched_audio * torch.as_tensor(inputs["scale"]) / 0.95
        batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)

        groundtruth.emit("vocoder", data={"out_mel": full_mel, "audio": batched_audio})

        for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            if self.save_prompt:
                prompt_wav = inputs["gt_wav"][bidx]
                audio = np.concatenate([prompt_wav, np.ones([10]), audio])

            output_path = os.path.join(self.output_dir, uttid + ".wav")
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])
        return torch.from_numpy(batched_audio)

    def predict_step_prefix(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, prompt_umm_token, syn_umm_token = self.prepare_features_prefix(batch)
        if inputs is None:
            return

        if self.diffusion_precision == "bf16":
            dtype = torch.bfloat16
        elif self.diffusion_precision == "fp16":
            dtype = torch.float16
        elif self.diffusion_precision == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError

        if not prompt_umm_token:
            return 
        if not syn_umm_token:
            return 
        
        bs = len(syn_umm_token)
        device = syn_umm_token[0].device

        inputs["all_token"] = [torch.cat([prompt_umm_token[bidx], syn_umm_token[bidx]], dim=-1) for bidx in range(bs)]
        inputs["all_token"] = pad_sequence(inputs["all_token"], batch_first=True, padding_value=self.token_config["token_padding"])
        # align for n*token_chunk_size+chunk_overlap (B,T)
        token_len = inputs["all_token"].shape[1]
        n_chunks = math.ceil(token_len / self.token_chunk_size)
        aligned_token_len = (
            math.ceil(token_len / self.token_chunk_size) * self.token_chunk_size
            + self.token_overlap
        )
        token_pad_len = aligned_token_len - token_len
        if token_pad_len > 0:
            inputs["all_token"] = F.pad(
                inputs["all_token"],
                [0, token_pad_len],
                mode="constant",
                value=self.token_config["token_padding"],
            )
        print(
            f"{batch_idx=} token align from {token_len} to {aligned_token_len} ({token_pad_len=} {n_chunks=})"
        )

        # pad bn_ctx (B,T,C)
        bn_ctx_len = inputs["all_bn_ctx"].shape[1]
        aligned_bn_ctx_len = self.bn_chunk_size * n_chunks
        bn_ctx_pad_len = aligned_bn_ctx_len - bn_ctx_len
        print(
            f"{batch_idx=} bn_ctx align from {bn_ctx_len} to {aligned_bn_ctx_len} ({bn_ctx_pad_len=} {n_chunks=})"
        )
        if bn_ctx_pad_len > 0:
            inputs["all_bn_ctx"] = F.pad(
                inputs["all_bn_ctx"],
                [0, 0, 0, bn_ctx_pad_len],
                mode="constant",
                value=self.bn_config["bn_padding"],
            )

        total_frame = math.ceil(token_len * self.mel_frame_rate / self.umm_frame_rate)
        print(f"{batch_idx=} {total_frame=}")
        # apply text cfg
        if self.text_cfg_w != 1:
            inputs = self.make_cfg_input(inputs)
        

        # chunk inference
        full_mel = []
        self.model.clear_infer_params(
            self.diffusion_nfe, total_frame, bs, self.text_cfg_w, self.mem_efficient
        )
        for chunk_idx in range(n_chunks):
            token_start_index = chunk_idx * self.token_chunk_size - (
                0 if chunk_idx == 0 else self.token_overlap
            )
            token_end_index = (
                chunk_idx + 1
            ) * self.token_chunk_size + self.token_overlap

            bn_start_index = chunk_idx * self.bn_chunk_size
            bn_end_index = min((chunk_idx + 1) * self.bn_chunk_size, total_frame)

            inputs["token"] = inputs["all_token"][:, token_start_index:token_end_index]
            inputs["bn_ctx"] = inputs["all_bn_ctx"][:, bn_start_index:bn_end_index]
            # print(f"{chunk_idx=} token={(token_start_index,token_end_index, token_end_index-token_start_index)}({inputs['all_token'].shape[1]}, {inputs['token'].shape[1]})")
            # print(f"{chunk_idx=} bn_ctx={(bn_start_index, bn_end_index, bn_end_index-bn_start_index)}({inputs['all_bn_ctx'].shape[1]} {inputs['bn_ctx'].shape[1]})")
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                chunk_mel = self.model.chunk_inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    use_infer_params=True,
                    first=chunk_idx == 0,
                )
            # print(f"{chunk_idx=} mel={chunk_mel.shape=}")
            self.model.update_infer_params(
                bn_end_index - bn_start_index, last=chunk_idx + 1 >= n_chunks - 1
            )
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                full_mel.append(chunk_mel)
        self.model.clear_infer_params(self.diffusion_nfe)

        full_mel = torch.cat(full_mel, dim=-1)
        batched_audio = []
        for bidx in range(bs):
            prompt_length = inputs["prompt_length"][bidx]
            if self.use_wvae_vocoder:
                z = full_mel[bidx, :, prompt_length:].unsqueeze(0)
                z = self.bn_norm.denorm_mel(z)
                out_wav = self.wvae_decode(z)
            else:
                _full_mel = self.mel_norm.denorm_mel(full_mel[bidx,:, prompt_length:])
                _full_mel = torch.clamp(_full_mel, min=-8.5, max=3.5)
                out_wav = self.vocoder(_full_mel)

            audio = out_wav
            if self.bn_config["wav_norm"] and inputs["scale"] is not None:
                audio = audio * torch.as_tensor(inputs["scale"][bidx]) / 0.95
            audio = torch.clip(audio,min=-1,max=1).squeeze(0)
            audio = audio.cpu().numpy()
            uttid = inputs["uttid"][bidx]
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]

            batched_audio.append(torch.from_numpy(audio))
            output_path = os.path.join(self.output_dir, uttid + ".wav")
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])

            if self.concat_context:
                output_path = os.path.join(self.output_dir, uttid + ".full.wav")
                full_audio = inputs['gt_wav'][bidx]
                context_audio = full_audio[...,:self.context_duration * self.mel_config["sampling_rate"]]
                print(f"{context_audio.shape=} {audio.shape=}")
                audio = np.concatenate([context_audio, audio], axis=-1)
                save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])

        return pad_sequence(batched_audio, batch_first=True)

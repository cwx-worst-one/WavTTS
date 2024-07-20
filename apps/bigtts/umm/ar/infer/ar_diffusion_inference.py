import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from apps.bigtts.umm.ar.lightning import (
    SemanticModule_MergeV2,
    SemanticModule_MergeV2_1,
)
from apps.bigtts.umm.diffusion.lit_modules.infer_utils import (
    load_torch_script,
    save_wav,
)
from apps.bigtts.umm.diffusion.lit_modules.lit_diffusion_reconstruct import MelNorm
from apps.bigtts.umm.diffusion.lit_modules.lit_diffusion_voicebox import VoiceBoxModule
from apps.bigtts.umm.diffusion.lit_modules.wvae import Wave
from samantha.utils.audio import load_wav
from samantha.utils.common import set_seed
from samantha.utils.cuda import torch_allow_tf32
from samantha.utils.hparams import DotDict

from ..data.lyrics import AddConditionsTransform, LyricsTokenTransform
from ..utility.umm_initializer import init_umm

logger = logging.getLogger(__name__)


@dataclass
class AROptions:
    ckpt_path: str = None
    precision: str = "fp16-mixed"
    output_path: str = None
    temperature: float = 0.9
    thresh: float = 0.9
    mode: str = "naive"
    max_blank_length: int = 5
    max_repeat_times: int = 1
    step_out_blank: str = "v2"
    version: str = "merge_v2"
    icl_mode: str = "continuation"
    eos_weight: float = 1.0
    cfg_config: tuple = (False, 0.0)


@dataclass
class DiffusionOptions:
    ckpt_path: str = None
    precision: str = "bf16-mixed"
    output_path: str = None
    mel_frame_rate: int = 40
    save_prompt: bool = False
    nfe: int = 10
    sampler: str = "ddim"
    text_cfg_w: int = 1
    bn_config: dict = None
    use_phone_lang: bool = False


@dataclass
class UMMOptions:
    ckpt_path: str = None
    version: str = "0.6.2"
    frame_rate: int = 25


class ARDiffusionInference(LightningModule):
    def __init__(
        self,
        sample_rate,
        seed,
        src_lang,
        tgt_lang,
        umm_opts: UMMOptions,
        ar_opts: AROptions,
        diffusion_opts: DiffusionOptions,
    ):
        super().__init__()
        self.save_hyperparameters()
        logger.info(
            f"{sample_rate=}, {seed=}, {src_lang=}, {tgt_lang=}, {umm_opts=}, {ar_opts=}, {diffusion_opts=}"
        )

        self.lang2id = {"en": 0, "zh": 1, "zh_en": 2}

        set_seed(seed)
        self.eos_weight = float(ar_opts.eos_weight)

        self.ar_model_hp = DotDict(
            {
                "duration": 60,
                "semantic_temperature": ar_opts.temperature,
                "semantic_thresh": ar_opts.thresh,
                "sample_mode": ar_opts.mode,
                "max_blank_length": ar_opts.max_blank_length,
                "max_repeat_times": ar_opts.max_repeat_times,
                "step_out_blank": ar_opts.step_out_blank,
                "icl_mode": ar_opts.icl_mode,
                "cfg_config": ar_opts.cfg_config,
                "eos_weight": ar_opts.eos_weight,
            }
        )

        self.bn_norm = MelNorm(
            diffusion_opts.bn_config["bn_norm_mean"],
            diffusion_opts.bn_config["bn_norm_std"],
        )
        os.makedirs(ar_opts.output_path, exist_ok=True)
        os.makedirs(diffusion_opts.output_path, exist_ok=True)

    def setup(self, stage):
        umm_opts, ar_opts, diffusion_opts = (
            self.hparams.umm_opts,
            self.hparams.ar_opts,
            self.hparams.diffusion_opts,
        )

        self.ar_model = prepare_semantic_model(
            ar_opts.ckpt_path, self.device, version=ar_opts.version
        )
        lyrics_max_seq_len = self.ar_model.extra_params.get("lyrics_max_seq_len", 600)
        self.lyrics_token_transform = LyricsTokenTransform.init_sami_tokenizer(
            lyrics_max_seq_len=lyrics_max_seq_len,
            truncate_long_lyrics=True,
            test_wer=True,
        )
        self.add_conditions_transform = AddConditionsTransform("lyrics_tokens")

        self.diffusion_model = prepare_diffusion_model(
            diffusion_opts.ckpt_path, self.device
        )
        self.umm = init_umm(umm_opts.ckpt_path, self.device, umm_opts.version)

        bn_config = diffusion_opts.bn_config
        wvae_encoder = load_torch_script(
            model_path=bn_config["wvae_encoder_path"],
            rank=self.device.index,
            cache_dir=bn_config["wvae_cache_dir"],
        )
        wvae_decoder = load_torch_script(
            model_path=bn_config["wvae_decoder_path"],
            rank=self.device.index,
            cache_dir=bn_config["wvae_cache_dir"],
        )
        self.wvae = Wave(
            wvae_encoder,
            wvae_decoder,
            version=bn_config["wvae_version"],
            hop_size=bn_config["wvae_encoder_hop_size"],
            win_size=bn_config["wvae_encoder_win_size"],
        )

    def preprocess_prompt_wav(self, prompt_wav_path, device):
        wav, sr = load_wav(prompt_wav_path, sr=24000, pad_width=(960 * 1, 960 * 2))
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)).item())
        wav = wav / scale * 0.95
        wav = wav.to(device)
        return wav

    @lru_cache(32)
    def convert_precision_to_amp_dtype(self, precision):
        if precision in ["bf16", "bf16-mixed"]:
            return torch.bfloat16
        if precision in ["fp16", "fp16-mixed"]:
            return torch.float16
        if precision in ["fp32"]:
            return torch.float32
        raise ValueError(f"Unsupported precision {precision=}")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        set_seed(self.hparams.seed)
        uttid = batch[0]

        try:
            infer_umm_token = self.text2token(batch)
            output_path = os.path.join(self.hparams.ar_opts.output_path, f"{uttid}.npy")
            np.save(output_path, infer_umm_token.cpu().numpy(), allow_pickle=False)
        except Exception as e:
            logger.error(f"error on text2token", exc_info=e)
            return
        try:
            audio = self.token2wav(batch, infer_umm_token)
            output_path = os.path.join(
                self.hparams.diffusion_opts.output_path, f"{uttid}.wav"
            )
            save_wav(audio, output_path)
        except Exception as e:
            logger.error(f"error on token2wav", exc_info=e)
            return

    def text2token(self, batch):
        uttid, prompt_lab, prompt_text, prompt_wav_path, infer_lab, infer_text = batch
        ar_batch = self.prepare_ar_inputs(prompt_wav_path, prompt_lab, infer_lab)
        dtype = self.convert_precision_to_amp_dtype(self.hparams.ar_opts.precision)
        with (
            torch_allow_tf32(enable_matmul=False),
            torch.autocast(device_type="cuda", dtype=dtype, enabled=True),
        ):
            infer_umm_token = self.ar_model.predict(ar_batch, self.ar_model_hp)

        return infer_umm_token

    def token2wav(self, batch, inferred_token):
        inputs = self.prepare_diffusion_inputs(batch, inferred_token)
        if inputs is None:
            logger.warning("diffusion inputs is None")
            return
        diffusion_opts = self.hparams.diffusion_opts
        dtype = self.convert_precision_to_amp_dtype(diffusion_opts.precision)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out_mel = self.diffusion_model.inference(
                inputs,
                diffusion_opts.nfe,
                diffusion_opts.sampler,
                text_cfg_w=diffusion_opts.text_cfg_w,
            )
        out_mel = out_mel[:, :, inputs["prompt_length"] :].float()
        z = self.bn_norm.denorm_mel(out_mel)
        output_wav = self.wvae.decoder_from_z(z.transpose(1, 2)).squeeze().cpu().numpy()
        audio = output_wav * inputs["scale"] / 0.95
        audio = np.clip(audio, a_min=-1, a_max=1)
        if diffusion_opts.save_prompt:
            prompt_wav = inputs["gt_wav"]
            audio = np.concatenate([prompt_wav, np.ones([10]), audio])
        return audio

    def prepare_ar_inputs(self, prompt_wav_path, prompt_lab, infer_lab):
        prompt_wav = self.preprocess_prompt_wav(prompt_wav_path, self.device)
        ar_batch = {}

        # prepare umm token
        # copy 3次，取前1/3
        prompt_umm_token = self.umm.wav2token(
            torch.cat([prompt_wav, prompt_wav, prompt_wav], -1)
        )
        prompt_umm_token = prompt_umm_token[:, : prompt_umm_token.size(-1) // 3]
        # hard-code from @kainan.
        prompt_umm_token = prompt_umm_token[:, :-1]
        ar_batch["audio_prompt"] = prompt_umm_token

        # TODO: @jiawei.chen add prepare mel

        # prepare text-ids.
        ## 续写
        if self.hparams.ar_opts.icl_mode == "continuation":
            ar_batch["lyrics"] = prompt_lab + "|" + infer_lab
        else:
            ar_batch["lyrics"] = infer_lab  # 注意这里改了
        ar_batch = self.lyrics_token_transform(ar_batch)
        ar_batch = self.add_conditions_transform(ar_batch)
        # batching.
        for key in [
            "lyrics_tokens",
            "prompt_text_lens",
            "phones",
            "tones",
            "wordsegs",
            # cfg config
            "phones_cfg",
            "tones_cfg",
            "wordsegs_cfg",
        ]:
            ar_batch[key] = ar_batch[key].unsqueeze(0)
        # language.
        ar_batch["src_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.src_lang)]
        )
        ar_batch["tgt_lang"] = torch.LongTensor(
            [self.lang2id.get(self.hparams.tgt_lang)]
        )
        return ar_batch

    def prepare_diffusion_inputs(self, batch, inferred_token):
        from samantha.dataio.lite.utils.phone_to_id import PhoneToId

        uttid, prompt_lab, prompt_text, prompt_wav_path, infer_lab, infer_text = batch
        p2i = PhoneToId()
        prompt_text_id, *_ = p2i.convert_tacolab_to_text_id_infer(
            prompt_lab.strip().split("\n")
        )
        syn_text_id, *_ = p2i.convert_tacolab_to_text_id_infer(
            infer_lab.strip().split("\n")
        )

        if prompt_text_id is None or syn_text_id is None:
            return None

        device = self.device
        prompt_text_id = torch.as_tensor(prompt_text_id, device=device)
        syn_text_id = torch.as_tensor(syn_text_id, device=device)
        inputs = {}

        # Syn
        inferred_token = inferred_token.view(1, -1)

        # Prompt
        wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=False)
        if len(wav.shape) > 1:
            wav = wav[0]
        inputs["gt_wav"] = wav
        wav = torch.as_tensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)).item())
        wav = wav / scale * 0.95
        wav = wav.to(device)
        wav_divide = (
            4800
            if (
                self.hparams.umm_opts.frame_rate == 25
                and self.hparams.diffusion_opts.mel_frame_rate == 40
            )
            else 600
        )
        prompt_wav = self.align_wav2(
            wav,
            wav_divide,
            self.hparams.sample_rate,
            self.hparams.umm_opts.frame_rate,
            inferred_token.shape[1],
        )
        inputs["scale"] = scale

        prompt_umm_token = self.umm.wav2token(prompt_wav)

        # Text
        text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
        text_id = F.pad(text_id, (0, 1), "constant", 1)
        inputs["frontend"] = {
            "phone": text_id[0:1, :].to(device),
            "tone": text_id[1:2, :].to(device),
            "word_seg": text_id[2:3, :].to(device),
        }
        if self.hparams.diffusion_opts.use_phone_lang:
            inputs["frontend"]["lang"] = text_id[3:4, :].to(device)

        inputs["token"] = torch.cat([prompt_umm_token, inferred_token], dim=1)
        token_len = inputs["token"].shape[1]
        mel_len = int(
            token_len
            / self.hparams.umm_opts.frame_rate
            * self.hparams.diffusion_opts.mel_frame_rate
        )

        crop_bn = self.wvae.encode(prompt_wav)
        m, logs = torch.split(crop_bn, 64, dim=-1)
        crop_bn = m + torch.randn_like(m) * torch.exp(logs)
        crop_bn = self.bn_norm.norm_mel(crop_bn)
        inputs["prompt_bn"] = crop_bn.transpose(1, 2)  # [B,C,T]
        inputs["bn_ctx"] = (
            torch.ones([1, mel_len, crop_bn.shape[2]], device=device)
            * self.hparams.diffusion_opts.bn_config["bn_padding"]
        )
        inputs["bn_ctx"][:, : inputs["prompt_bn"].shape[-1], :] = inputs[
            "prompt_bn"
        ].transpose(1, 2)
        inputs["prompt_length"] = inputs["prompt_bn"].shape[-1]

        inputs["uttid"] = uttid

        inputs = self.make_cfg_input(inputs)

        return inputs

    def make_cfg_input(self, inputs):
        if self.hparams.diffusion_opts.text_cfg_w != 1:
            inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
            inputs["frontend"]["phone"][1, :] = 1
            inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
            inputs["frontend"]["tone"][1, :] = 1
            inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
            inputs["frontend"]["word_seg"][1, :] = 1
            if "lang" in inputs["frontend"]:
                inputs["frontend"]["lang"] = inputs["frontend"]["lang"].repeat(2, 1)
                inputs["frontend"]["lang"][1, :] = 1
        return inputs

    def align_wav2(self, wav, wav_divide, sampling_rate, umm_frame_rate, umm_add_len):
        umm_hop = sampling_rate // umm_frame_rate
        wav_add_len = umm_add_len * umm_hop
        crop_wav_len = (wav_add_len + wav.shape[1]) % wav_divide
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, wav_divide - crop_wav_len), "constant", 0)
        return wav


def prepare_semantic_model(semantic_model_path, device, version):
    logger.info(f"Loading version: {version}")

    model_cls = {
        "merge_v2": SemanticModule_MergeV2,
        "merge_v2_1": SemanticModule_MergeV2_1,
    }
    if version not in model_cls:
        raise ValueError(
            f"Unknown semantic {version=}, valid versions are {list(model_cls.keys())}"
        )

    return (
        model_cls[version]
        .load_from_checkpoint(semantic_model_path, map_location=device)
        .eval()
    )


def prepare_diffusion_model(diffusion_ckpt_path, device):
    model = VoiceBoxModule.load_from_checkpoint(
        diffusion_ckpt_path, map_location=torch.device(device)
    ).model.eval()
    return model

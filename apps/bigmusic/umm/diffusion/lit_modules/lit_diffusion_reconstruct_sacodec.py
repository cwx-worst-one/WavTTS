import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import resample
from functools import partial
import librosa
from pytorch_lightning import LightningModule
import math
from typing import Any
# from apps.bigmusic.umm.diffusion.lit_modules.lit_diffusion_voicebox import (
#     VoiceBoxModule as pl_module,
# )
from recipes.voicebox.lit_modules.lit_diffusion_voicebox_sacodec import VoiceBoxModule as pl_module
from samantha.utils import groundtruth
from samantha.dataio.lite.utils.mel import mel_spectrogram
from apps.bigtts.umm.diffusion.lit_modules.infer_utils import set_seed, save_wav
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from hyperpyyaml import load_hyperpyyaml
from recipes.sacodec.modules.sacodec_module_umm import SACodecModule
from samantha.utils.hparams import DotDict

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

def prepare_umm_codebook(umm_codebook_path, device):
    codebook = torch.load(umm_codebook_path).to(device)
    return codebook


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
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        audio_norm_type="clip",
        extra_params=None,
        **kwargs,
    ):
        super().__init__()

        seed = set_seed(seed)
        self.vocoder_ckpt_path = vocoder_ckpt_path
        self.umm_ckpt_path = umm_ckpt_path
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
        self.use_wvae_vocoder = use_wvae_vocoder

        # if self.infer_type != "vocoder":
        self.model = prepare_diffusion_model(diffusion_ckpt_path, self.device)

        self.bn_config = bn_config
        self.extra_params = DotDict(extra_params)
        self.save_hyperparameters()

        ## TODO: add back bn_norm, sacodec uses mean/std of 0/1, so not needed for now
        self.bn_norm = MelNorm(bn_config["bn_norm_mean"], bn_config["bn_norm_std"])

        if token_config is None:
            self.token_config = {"token_cfg": True, "token_padding": 32767}
        else:
            self.token_config = token_config
        self.mel_mask_value = mel_config["mel_mask_value"]  # -5
        self.token_sample_rate = 24000
        self.audio_norm_type = audio_norm_type

        os.makedirs(output_dir, exist_ok=True)

        logger.info("DiffusionU2SInfer:")
        args_dict = locals()
        del args_dict["self"]
        logger.info(", ".join(f"{k}={v}\n" for k, v in args_dict.items()))

    @torch.no_grad()
    def sacodec_embs_to_wav(self, latents):
        latents = self.bn_norm.denorm_mel(latents)
        audio_hat = self.sacodec.decode_latents(latents.transpose(1, 2))
        # logamp_g, pha_g, rea_g, imag_g, y_g = self.sacodec.decoder(latents)
        # audio_hat = y_g
        return audio_hat # B x CH x L


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

    def wav2token(self, wav):
        if self.umm_type in ["UMM", "UMM_conv"]:
            umm_token = self.umm.model.wav2token_alloutputs(wav)['vq_ids']
        else:
            raise NotImplementedError

        return umm_token

    def prepare_features(self, batch):
        (
            batched_uttid,
            batched_syn_wav_path,
            batched_syn_umm_token,
        ) = self.batched_data(batch)
        bs = len(batched_uttid)

        # if prompt_text_id is None or syn_text_id is None:
        #     return None
        device = f"cuda:{self.local_rank}"
        inputs = dict()

        # Syn
        if self.infer_type == "diffusion-vocoder":
            assert batched_syn_wav_path is not None
            syn_wavs = []
            syn_wavlens = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                wav, _ = librosa.load(
                    syn_wav_path, sr=self.token_sample_rate, mono=True
                )
                wav = torch.FloatTensor(wav).unsqueeze(0)
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
            inputs["syn_wavlen"] = syn_wavlens
            inputs["syn_wavs"] = torch.tensor(syn_wavs)
        elif self.infer_type == "ar-diffusion-vocoder":
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
            inputs["syn_wavlen"] = syn_wavlens
        else:
            raise NotImplementedError
        if hasattr(self, "umm_codebook"):
            batched_syn_umm_token = F.embedding(
                batched_syn_umm_token, self.umm_codebook
            )


        if self.infer_type in ["diffusion-vocoder", "ar-diffusion-vocoder"]:
            batched_prompt_umm_token = None
            target_bn_len = 6 * self.mel_frame_rate
            inputs["prompt_bn"] = (
                torch.ones(
                    [bs, self.bn_config["bn_dim"], target_bn_len], device=device
                )
                * self.bn_config["bn_padding"]
            )
            inputs["prompt_length"] = target_bn_len
            if getattr(self.model.hp, "use_window_mask", False):
                inputs["bn_ctx"] = (
                    torch.ones([bs, 0, self.bn_config["bn_dim"]], device=device)
                )
            else:
                raise Exception("Use window mask must be True")
        else:
            raise NotImplementedError
        inputs["uttid"] = batched_uttid

        return inputs, batched_prompt_umm_token, batched_syn_umm_token

    def make_cfg_input(self, inputs):
        if self.text_cfg_w != 1:
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

            
            if "all_bn_ctx" in inputs:
                inputs["all_bn_ctx"] = inputs["all_bn_ctx"].repeat(2, 1, 1)
            elif "bn_ctx" in inputs:
                inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
            else:
                raise Exception("Not hangled make_cfg_input")
        return inputs

    def sacodec_reconstruct(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        for item in batch:
            uttid, syn_wav_path = item
            wav, _ = librosa.load(
                syn_wav_path,
                sr=self.mel_config["sampling_rate"],
                mono=False,
            )
            wav = torch.FloatTensor(wav).unsqueeze(0)
            wav = wav.to(device)
            if len(wav.shape) == 2:
                wav = wav.unsqueeze(1)
            reconstruct_wav = self.sacodec.reconstruct_audio(wav)
            output_path = os.path.join(self.output_dir, uttid + ".wav")
            save_wav(reconstruct_wav.cpu().numpy(), output_path)

    def setup(self, stage):
        device = f"cuda:{self.local_rank}"
        if self.infer_type != "vocoder" and self.umm_ckpt_path:
            if self.umm_type == "UMM":
                self.umm = prepare_umm(self.umm_ckpt_path, device)
            else:
                raise NotImplementedError
        else:
            self.umm = None

        self.sacodec: SACodecModule = self.bn_config["vocoder_model"](local_rank=self.local_rank)[
            "sacodec"
        ]

        if self.umm_codebook_path is not None:
            self.umm_codebook = prepare_umm_codebook(self.umm_codebook_path, device)

    def batched_data(self, batch):
        batched_syn_wav_path = None
        batched_syn_umm_token = None
        batched_uttid = None
        assert self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]

        def batching(container, data):
            if data is None or data == "":
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
                uttid,
                syn_wav_path,
                syn_umm_token,
            ) = (None, None, None)
            # if self.infer_type == "diffusion-vocoder":
            #     (uttid, syn_wav_path) = item[:2]

            if self.infer_type == "ar-diffusion-vocoder":
                (prompt_text_id, syn_text_id, prompt_wav_path, syn_umm_token, uttid) = (
                    item[:5]
                )
            elif self.infer_type == "diffusion-vocoder":
                (uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id) = (
                    item[:5]
                )

            batched_uttid = batching(batched_uttid, uttid)
            batched_syn_wav_path = batching(batched_syn_wav_path, syn_wav_path)
            batched_syn_umm_token = batching(batched_syn_umm_token, syn_umm_token)
        assert batched_uttid is not None
        bs = len(batched_uttid)
        assert_same_batchsize_if_exsist(batched_syn_wav_path, bs)
        assert_same_batchsize_if_exsist(batched_syn_umm_token, bs)
        return (
            batched_uttid,
            batched_syn_wav_path,
            batched_syn_umm_token,
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
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        token_chunk_size=25,
        token_chunk_overlap=0,
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
            use_wvae_vocoder,
            bn_config,
            token_config,
            **kwargs,
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
        # if batch_idx < 19: 
        #     print('Skipping batch', batch_idx)
        #     return None
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
            prompt_length = inputs["bn_ctx"].shape[1]
            # print("Prompt length", prompt_length)
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

        with torch.autocast(device_type="cuda", enabled=False):
            output_wavs = self.sacodec_embs_to_wav(outputs.float()).cpu()

        batched_audio = output_wavs.cpu().numpy()
        if self.audio_norm_type == "clip":
            batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)
        elif self.audio_norm_type == "norm":
            batched_audio = (output_wavs / output_wavs.abs().amax(dim=(1, 2), keepdim=True)).cpu().numpy()

        groundtruth.emit("vocoder", data={"out_mel": outputs, "audio": batched_audio})

        for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            # if self.save_prompt:
            #     prompt_wav = inputs["gt_wav"][bidx]
            #     audio = np.concatenate([prompt_wav, np.ones([10]), audio])

            output_path = os.path.join(self.output_dir, uttid + ".wav")
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])
            print("saving to", output_path)

        return torch.from_numpy(batched_audio)


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
        save_meta=True,
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
            # rescale_factor,
            use_wvae_vocoder,
            bn_config,
            token_config,
            # use_phone_lang,
            # without_prefix,
            **kwargs,
        )
        self.save_meta = save_meta
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
        # self.without_prefix = True
        # self.concat_context = (not self.without_prefix) and kwargs.get("concat_context", False) and self.context_duration > 0

    def prepare_features(self, batch) -> None:
        # (
        #     batched_uttid,
        #     batched_prompt_text_id,
        #     batched_prompt_wav_path,
        #     batched_syn_text_id,
        #     batched_syn_wav_path,
        #     batched_syn_umm_token,
        #     batched_scale,
        # ) = self.batched_data(batch)

        # bs = len(batched_uttid)
        (
            batched_uttid,
            batched_syn_wav_path,
            batched_syn_umm_token,
        ) = self.batched_data(batch)
        bs = len(batched_uttid)
        batched_text_id = None
        device = f"cuda:{self.local_rank}"
        inputs = dict()

        # syn
        if self.infer_type == "diffusion-vocoder":
            assert batched_syn_wav_path is not None
            inputs["gt_wav"] = []
            ori_syn_wavlens = []
            syn_wavlens = []
            syn_wavs = []
            for bidx, syn_wav_path in enumerate(batched_syn_wav_path):
                syn_wav, sr = librosa.load(
                    syn_wav_path, sr=None, mono=True
                )
                syn_wav = torch.FloatTensor(syn_wav).unsqueeze(0).to(device)

                # for gt wav saving accurately
                gt_wav, _ = librosa.load(
                    syn_wav_path, sr=self.mel_config["sampling_rate"], mono=False
                )
                
                inputs["gt_wav"].append(gt_wav)
                ori_syn_wavlens.append(syn_wav.shape[-1])
                if sr != self.token_sample_rate:
                    syn_wav, sr = librosa.load(
                        syn_wav_path, sr=self.token_sample_rate, mono=True
                    )
                    syn_wav = torch.FloatTensor(syn_wav).unsqueeze(0).to(device)
                syn_wavlens.append(syn_wav.shape[-1])
                syn_wavs.append(syn_wav)

            if self.padding_mode == "token_padding":
                padding_value = self.token_config["token_padding"]
                batched_syn_umm_token = [
                    self.wav2token(syn_wav) for syn_wav in syn_wavs
                ]
                max_syn_umm_token_len = max(
                    [syn_umm_token.shape[-1] for syn_umm_token in batched_syn_umm_token]
                )
                batched_syn_umm_token = torch.cat(
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
            else:
                raise ValueError(f"no padding implement named '{self.padding_mode}'")

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
            inputs["syn_wavlen"] = syn_wavlens
        else:
            raise NotImplementedError(f"{self.infer_type=}")

        # prompt
        batched_prompt_umm_token = None
        inputs["prompt_bn"] = torch.zeros(
            [bs, self.bn_config["bn_dim"], 0], device=device
        )
        inputs["prompt_length"] = 0

        inputs["all_bn_ctx"] = inputs["prompt_bn"].transpose(1, 2)
        inputs["uttid"] = batched_uttid
        return inputs, batched_prompt_umm_token, batched_syn_umm_token

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

        bs = len(syn_umm_token)
        device = syn_umm_token.device

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
        with torch.autocast(device_type="cuda", enabled=False):
            output_wavs = self.sacodec_embs_to_wav(full_mel.float()).cpu()

        batched_audio = output_wavs.cpu().numpy()
        if self.audio_norm_type == "clip":
            batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)
        elif self.audio_norm_type == "norm" and output_wavs.abs().max() > 1:
            batched_audio = (output_wavs / output_wavs.abs().amax(dim=(1, 2), keepdim=True)).cpu().numpy()

        groundtruth.emit("vocoder", data={"out_mel": full_mel, "audio": batched_audio})

        for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            if self.save_prompt:
                prompt_wav = inputs["gt_wav"][bidx]
                # audio = np.concatenate([prompt_wav, np.ones([10]), audio])
                gt_path = os.path.join(self.output_dir, uttid + ".target_audio.wav")
                print(prompt_wav.shape, "prompt_wav")
                if len(prompt_wav.shape) > 1:
                    gt_wav = prompt_wav.T 
                else:
                    gt_wav = prompt_wav
                save_wav(gt_wav, gt_path, sr=self.mel_config["sampling_rate"])
            if self.save_meta:
                metadata = {
                            'file_name': uttid,
                        }
                meta_fp = os.path.join(self.output_dir, f"{uttid}.metadata.json")
                with open(meta_fp, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            output_path = os.path.join(self.output_dir, uttid + ".generated.wav")
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])

        return torch.from_numpy(batched_audio)

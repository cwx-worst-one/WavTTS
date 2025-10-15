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
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks, vocode_in_ovl_chunks_v2, process_eos_indexes, pad_sequence_dim
from hyperpyyaml import load_hyperpyyaml
from recipes.sacodec.modules.sacodec_module_umm import SACodecModule
from samantha.utils.hparams import DotDict

import logging

logger = logging.getLogger(__name__)

def prepare_diffusion_model(diffusion_ckpt_path, device) -> pl_module:
    model = pl_module.load_from_checkpoint(
        diffusion_ckpt_path, map_location="cpu"
    ).model.to(device)
    model.eval()
    return model


def prepare_umm(umm_ckpt_path, device, umm_type):
    rank = int(device[-1])
    if umm_type in ["UMM", "UMM_conv"]:
        from recipes.umm.requires.model_initializer import init_stage3
        token_model = init_stage3(umm_ckpt_path, rank, cache_dir="./")["Stage3"].eval()
    elif umm_type in ["RVQ"]:
        from recipes.umm2.scripts.model_initializer import init_stage3
        token_model = init_stage3(umm_ckpt_path, rank, cache_dir="./")["Stage3"].eval()
    else:
        raise ValueError(f"umm_type {umm_type} not supported")
    
    return token_model

def prepare_umm_codebook(umm_codebook_path, device):
    codebook = torch.load(umm_codebook_path).to(device)
    return codebook

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

        if token_config is None:
            self.token_config = {"token_cfg": True, "token_padding": 32768, "token_eos": 32769, "frame_rate": 25 }
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

    def sacodec_embs_to_wav(self, latents, eos_index_list=None):
        """Truncate latents to EOS and decode sequentially to prevent OOM."""
        latents = latents.transpose(1, 2) # B C T -> B T C
        latents = self.sacodec.encoder.denormalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
        if eos_index_list is None:
            return self.sacodec.decode_latents()

        output_wavs = []
        for latent, eos_index in zip(latents, eos_index_list):
            latent = latent.unsqueeze(0) # T C -> B T C
            output_wav = self.sacodec.decode_latents(latent[..., :eos_index, :])
            output_wav = output_wav.squeeze(0) # B C L -> C L
            output_wavs.append(output_wav.cpu())

        output_wavs = pad_sequence_dim(output_wavs, dim=0)
        return output_wavs
        
    @torch.no_grad()
    def sacodec_embs_to_wav_chunked(self, latents, eos_index_list=None):
        """Chunked decoding uses less memory and gives results similar to online streaming. Use sacodec_embs_to_wav for best results"""
        # TODO: support eos_index_list
        latents = self.sacodec.encoder.denormalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
        audio_hat = vocode_in_ovl_chunks_v2(latents, self.sacodec,
                            mini_bs=1,
                            chunk_size=self.bn_config.get("bn_chunk_size", 100),
                            overlap=self.bn_config.get("bn_bothside_overlap_size", 8),
                            vocoder_frame_rate=self.bn_config["frame_rate"],
                            sample_rate=self.bn_config["sample_rate"],
                            vocoder_type="v2")
        return audio_hat


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
            # umm_token = self.umm.wav2token(wav)
            umm_token = self.umm.wav2requires(wav, 24000, slice_method="even")
        elif self.umm_type in ['RVQ']:
            wav = wav.unsqueeze(1)
            umm_token = self.umm.wav2requires(wav, slice_method="even", chunk_size=60, requires=['token'])['token']
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

    def setup(self, stage):
        device = f"cuda:{self.local_rank}"
        if self.infer_type != "vocoder" and self.umm_ckpt_path:
            self.umm = prepare_umm(self.umm_ckpt_path, device, self.umm_type)
        else:
            self.umm = None

        if self.infer_type == "ar-diffusion-vocoder":
            self.umm = None

        self.sacodec: SACodecModule = self.bn_config["vocoder_model"](local_rank=self.local_rank)[
            "sacodec"
        ]
        self.sacodec.precision = self.bn_config["precision"]

        if self.infer_type == "ar-diffusion-vocoder":
            # self.sacodec.encoder = None # encoder is needed for normalize and denormalize features
            self.sacodec.melspec_loss = None
            self.sacodec.multiperioddisc = None
            self.sacodec.chroma_loss = None
            self.sacodec.multiresddisc = None

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
            logger.info("saving to", output_path)

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
        rescale_factor=-1,
        use_wvae_vocoder=False,
        bn_config=None,
        token_config=None,
        use_phone_lang=False,
        without_prefix=True,
        padding_mode="token_padding",
        save_meta=True,
        save_diffusion_output=False,
        vocoder_chunk_infer=False,
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
        self.save_diffusion_output = save_diffusion_output
        self.save_meta = save_meta
        self.padding_mode = padding_mode
        self.vocoder_chunk_infer = vocoder_chunk_infer
        self.rescale_factor = rescale_factor
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
                syn_umm_token.shape[1] for syn_umm_token in batched_syn_umm_token
            ]
            max_syn_umm_token_len = max(syn_umm_token_lens)
            syn_wavlens = [
                syn_umm_token_len * (self.bn_config["sample_rate"]  // self.umm_frame_rate)
                for syn_umm_token_len in syn_umm_token_lens
            ]
            pad_shape = [[0, max_syn_umm_token_len - syn_umm_token_len]
                         for syn_umm_token_len in syn_umm_token_lens]
            if batched_syn_umm_token[0].ndim == 3:  # hierarchical tokens
                pad_shape = [[0, 0, 0, max_syn_umm_token_len - syn_umm_token_len]
                             for syn_umm_token_len in syn_umm_token_lens]
            batched_syn_umm_token = torch.stack(
                [
                    F.pad(
                        batched_syn_umm_token[b],
                        pad_shape[b], 
                        "constant",
                        padding_value,
                    )
                    for b in range(len(batched_syn_umm_token))
                ],
                dim=0,
            )
            if batched_syn_umm_token.ndim >= 3:
                batched_syn_umm_token = batched_syn_umm_token.squeeze(1)
            inputs["syn_wavlen"] = syn_wavlens
        else:
            raise NotImplementedError(f"{self.infer_type=}")

        if self.extra_params.get('diffusion_drop_rvq_token', -1) > 0 and batched_syn_umm_token.ndim == 3:
            logger.info(f"Dropping rvq token, index=[{self.extra_params['diffusion_drop_rvq_token']}:-1]")
            batched_syn_umm_token[:, :, self.extra_params['diffusion_drop_rvq_token']:] = self.token_config["token_padding"]
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
            if inputs["all_token"].ndim == 2:
                inputs["all_token"] = F.pad(
                    inputs["all_token"],
                    [0, token_pad_len],
                    mode="constant",
                    value=self.token_config["token_padding"],
                )
            elif inputs["all_token"].ndim == 3:
                inputs["all_token"] = F.pad(syn_umm_token,[0, 0, 0, token_pad_len],mode="constant",value=self.token_config["token_padding"],)
            else:
                raise NotImplementedError(f"Padding unimplemented for {inputs['all_token'].shape=}")
        logger.info(
            f"{batch_idx=} token align from {token_len} to {aligned_token_len} ({token_pad_len=} {n_chunks=})"
        )

        # pad bn_ctx (B,T,C)
        bn_ctx_len = inputs["all_bn_ctx"].shape[1]
        aligned_bn_ctx_len = self.bn_chunk_size * n_chunks
        bn_ctx_pad_len = aligned_bn_ctx_len - bn_ctx_len
        logger.info(
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
        logger.info(f"{batch_idx=} {total_frame=}")
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
            # logger.info(f"{chunk_idx=} token={(token_start_index,token_end_index, token_end_index-token_start_index)}({inputs['all_token'].shape[1]}, {inputs['token'].shape[1]})")
            # logger.info(f"{chunk_idx=} bn_ctx={(bn_start_index, bn_end_index, bn_end_index-bn_start_index)}({inputs['all_bn_ctx'].shape[1]} {inputs['bn_ctx'].shape[1]})")
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                chunk_mel = self.model.chunk_inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    use_infer_params=True,
                    first=chunk_idx == 0,
                    rescale_factor=self.rescale_factor
                )
            # logger.info(f"{chunk_idx=} mel={chunk_mel.shape=}")
            self.model.update_infer_params(
                bn_end_index - bn_start_index, last=chunk_idx + 1 >= n_chunks - 1
            )
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                full_mel.append(chunk_mel)
        self.model.clear_infer_params(self.diffusion_nfe)

        full_mel = torch.cat(full_mel, dim=-1) # B x C x T
        full_mel = full_mel[:, :, inputs["prompt_length"]:] # Currently, prompt_length=0

        eos_id = self.token_config["token_eos"]
        semantic_frame_rate = self.token_config["frame_rate"]
        vocoder_frame_rate = self.bn_config["frame_rate"]
        eos_index_list = process_eos_indexes(
            semantic_samples=syn_umm_token, eos_id=eos_id,
            semantic_frame_rate=semantic_frame_rate, output_frame_rate=vocoder_frame_rate
        )
        if self.vocoder_chunk_infer:
            output_wavs = self.sacodec_embs_to_wav_chunked(full_mel, eos_index_list)
        else:
            output_wavs = self.sacodec_embs_to_wav(full_mel, eos_index_list)
        
        batched_audio = output_wavs.cpu().numpy()
        if self.audio_norm_type == "clip":
            batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)
        elif self.audio_norm_type == "norm" and output_wavs.abs().max() > 1:
            batched_audio = (output_wavs / output_wavs.abs().amax(dim=(1, 2), keepdim=True)).cpu().numpy()

        groundtruth.emit("vocoder", data={"out_mel": full_mel, "audio": batched_audio})

        save_output_dir = os.path.join(self.output_dir, "default")
        os.makedirs(save_output_dir, exist_ok=True)
        for bidx, (uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            if self.save_prompt:
                prompt_wav = inputs["gt_wav"][bidx]
                # audio = np.concatenate([prompt_wav, np.ones([10]), audio])
                gt_path = os.path.join(save_output_dir, uttid + ".target_audio.wav")
                logger.info(f"{bidx=} {prompt_wav.shape=}")
                if len(prompt_wav.shape) > 1:
                    gt_wav = prompt_wav.T 
                else:
                    gt_wav = prompt_wav
                save_wav(gt_wav, gt_path, sr=self.mel_config["sampling_rate"])
            if self.save_meta:
                metadata = {
                            'file_name': uttid,
                        }
                meta_fp = os.path.join(save_output_dir, f"{uttid}.metadata.json")
                with open(meta_fp, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            output_path = os.path.join(save_output_dir, uttid + ".generated.wav")
            if "syn_wavlen" in inputs:
                audio = audio[..., : inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])
            logger.info(f'{output_path=}')
            if self.save_diffusion_output:
                latent = full_mel[bidx]
                latent_path = os.path.join(save_output_dir, uttid + ".diffusion_output.pt")
                torch.save(latent, latent_path)

        return torch.from_numpy(batched_audio)
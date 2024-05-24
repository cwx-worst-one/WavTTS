import os
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchaudio.functional import resample
from functools import partial
import librosa
from pytorch_lightning import LightningModule
import math
from typing import Any
from .lit_diffusion_voicebox import VoiceBoxModule as pl_module
from samantha.utils import groundtruth
from samantha.dataio.lite.utils.mel import mel_spectrogram
from apps.bigtts.umm.diffusion.lit_modules.infer_utils import set_seed, save_wav, load_torch_script
from apps.bigtts.umm.diffusion.lit_modules.wvae import Wave, mel_spectrogram_torch, spectrogram_torch
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks


import logging
logger = logging.getLogger(__name__)


def prepare_diffusion_model(diffusion_ckpt_path, device):
    model = pl_module.load_from_checkpoint(
        diffusion_ckpt_path, device=torch.device(device)
    ).model.to(device)
    model.eval()
    return model

def prepare_umm(umm_ckpt_path, device):
    rank = int(device[-1])
    from recipes.umm.requires.model_initializer import init_stage3
    token_model = init_stage3(umm_ckpt_path, rank, "./")["Stage3"].eval()
    return token_model

def prepare_ummv2(umm_ckpt_path, device):
    rank = int(device[-1])
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
    model = init_stage3_conv1d(umm_ckpt_path, 0, "./.test_umm_conv1d_cache")["Stage3Conv1D"].eval().to(device)
    return model

def prepare_umm_dualconv(umm_ckpt_path, device):
    from recipes.umm.requires.model_initializer import init_dualumm
    model = init_dualumm(umm_ckpt_path, device=device)['Stage3'].eval().to(device)
    return model


def prepare_umm_codebook(umm_codebook_path, device):
    codebook = torch.load(umm_codebook_path).to(device)
    return codebook

def prepare_mel_transform(mel_config):
    mel_transform = partial(mel_spectrogram,
            n_fft=mel_config['n_fft'], # 2048,
            num_mels=mel_config['num_mels'], # 80,
            hop_size=mel_config["hop_size"],
            win_size=mel_config["win_size"],
            sampling_rate=24000,
            fmin=0, fmax=12000
            )
    return mel_transform

def prepare_vocoder(vocoder_ckpt_path, device):
    vocoder = torch.jit.load(vocoder_ckpt_path, map_location=device).eval()
    return vocoder


class MelNorm():
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
        umm_type="UMM", # UMM or USM
        umm_codebook_path=None,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        text_cfg_w=1,
        use_wvae_vocoder=False,
        bn_config=None,
        use_phone_lang=False,
        without_prefix=True,
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
        self.use_wvae_vocoder = use_wvae_vocoder

        self.use_phone_lang = use_phone_lang
        
        self.without_prefix = without_prefix
        if self.infer_type != "vocoder":
            self.model = prepare_diffusion_model(diffusion_ckpt_path, self.device)
        
        self.bn_config = bn_config
        if self.use_wvae_vocoder:
            self.bn_norm = MelNorm(
                bn_config["bn_norm_mean"], bn_config["bn_norm_std"])
        else:
            self.mel_norm = MelNorm(
                mel_config["mel_norm_mean"], mel_config["mel_norm_std"])

        self.mel_norm = MelNorm(
            mel_config["mel_norm_mean"], mel_config["mel_norm_std"])
        self.mel_mask_value = mel_config["mel_mask_value"] # -5
        self.token_sample_rate = 24000
        
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"DiffusionU2SInfer:")
        args_dict = locals()
        del args_dict['self']
        logger.info(", ".join(f"{k}={v}\n" for k, v in args_dict.items()))

    
    def wvae_decode(self, pred_emb):
        pred_emb = pred_emb.float()
        duration = pred_emb.shape[-1] // self.mel_frame_rate
        with torch.autocast(device_type="cuda", enabled=False):
            if duration > 30:
                wavs_g = vocode_in_chunks(pred_emb, self.wvae, mini_bs=1, chunk_size=1)
            else:
                wavs_g = vocode_in_chunks(pred_emb, self.wvae, mini_bs=4, chunk_size=1)
        return wavs_g

    # align wav to make sure wav length could be divided by `umm_frame_rate` and `mel_frame_rate` evenly
    def align_wav(self, wav, umm_sampling_rate, mel_sampling_rate, umm_frame_rate, mel_frame_rate):
        umm_hop = umm_sampling_rate // umm_frame_rate
        mel_hop = mel_sampling_rate // mel_frame_rate
        align_block_len = abs(umm_hop*mel_hop) // math.gcd(umm_hop, mel_hop)
        crop_wav_len = wav.shape[1] % align_block_len
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, align_block_len-crop_wav_len), "constant", 0)
        return wav, align_block_len

    # align wav to make sure padded wav length could be divided by `wav_divide` evenly
    def align_wav2(self, wav, wav_divide, sampling_rate, umm_frame_rate, umm_add_len):
        umm_hop = sampling_rate // umm_frame_rate
        wav_add_len = umm_add_len * umm_hop
        crop_wav_len = (wav_add_len + wav.shape[1]) % wav_divide
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, wav_divide-crop_wav_len), "constant", 0)
        return wav, wav_divide
    
    def wav2token(self, wav):
        if self.umm_type == "USM":
            token_len = wav.shape[1] // 960
            umm_token = self.umm.wav2token(resample(wav, 24000, 16000), dtype=torch.bfloat16)
            umm_token = umm_token[:, :token_len]
        elif self.umm_type in ["UMM", "UMM_conv"]:
            umm_token = self.umm.wav2token(wav)
        elif self.umm_type in ["UMM_dualconv"]:
            token_vocal = self.umm.wav2token(wav, 'vocal').squeeze()
            token_inst = self.umm.wav2token(wav, 'inst').squeeze()
            umm_token = torch.stack([token_vocal, token_inst], 1)
            umm_token = umm_token.reshape(-1).unsqueeze(0)
        elif self.umm_type in ["UMM_dualconvV1", "UMM_dualconvV3"]:
            from recipes.umm.utils.mss import MSSPredictor
            predictor = MSSPredictor().to(self.device)
            voc, acc = predictor(wav)
            token_vocal = self.umm.wav2token(voc[:, None], 'vocal')
            token_acc = self.umm.wav2token(acc[:, None], 'inst')
            umm_token = torch.stack([token_vocal, token_acc], 2).flatten(1, 2)
        elif self.umm_type in ["UMMv2","UMM_music"]:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                umm_token = self.umm.wav2token(wav)
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
            batched_scale
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
                wav, _ = librosa.load(syn_wav_path, sr=self.token_sample_rate, mono=True) 
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
                syn_wav = wav
                syn_wavs.append(syn_wav)
                syn_wavlen = int(syn_wav.shape[-1]*1.0*self.mel_config["sampling_rate"]/self.token_sample_rate)
                syn_wavlens.append(syn_wavlen)
            
            batched_scale = scales
            max_wavlen = max([syn_wav.shape[-1] for syn_wav in syn_wavs])

            syn_wavs = torch.cat([F.pad(syn_wav,[0,max_wavlen-syn_wav.shape[-1]],"constant",0) for syn_wav in syn_wavs],dim=0)
            syn_wavs, _ = self.align_wav(syn_wavs, self.token_sample_rate, self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate)
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
        else:
            raise NotImplementedError
        if hasattr(self, "umm_codebook"):
            batched_syn_umm_token = F.embedding(batched_syn_umm_token, self.umm_codebook)
        


        
        # Prompt
        if batched_prompt_wav_path is None:
            batched_prompt_umm_token = None
            batched_prompt_wav = None
            target_bn_len = 6 * self.mel_frame_rate
        else:
            batched_prompt_wavlen=[]
            batched_prompt_wav = []
            for bidx,prompt_wav_path in enumerate(batched_prompt_wav_path):

                wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True)
                # inputs["gt_wav"] = wav
                wav = torch.FloatTensor(wav).unsqueeze(0)
                wav = wav.to(device)
                batched_prompt_wav.append(wav)
                batched_prompt_wavlen.append(wav.shape[-1])

            max_wavlen = max(batched_prompt_wavlen)
            batched_prompt_wav= torch.cat([F.pad(prompt_wav,[0,max_wavlen-prompt_wav.shape[-1]],"constant",0) for prompt_wav in batched_prompt_wav],dim=0)

            if getattr(self.model.hp,"use_window_mask", False):
                 batched_prompt_wav, target_bn_len = self.align_wav_for_chunk(batched_prompt_wav, 
                    self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate,
                    0 if batched_text_id is None else batched_text_id.shape[-1]
                )
            else:
                if self.infer_type == "diffusion-vocoder":
                    batched_prompt_wav, target_bn_len = self.align_wav(batched_prompt_wav, self.token_sample_rate, self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate)
                elif self.infer_type == "ar-diffusion-vocoder":
                    wav_divide = 4800 if (self.umm_frame_rate==25 and self.mel_frame_rate==40) else 600
                    batched_prompt_wav, target_bn_len = self.align_wav2(batched_prompt_wav, wav_divide, self.mel_config["sampling_rate"], self.umm_frame_rate, batched_syn_umm_token.shape[1])
            if self.umm != None:
                batched_prompt_umm_token = self.wav2token(batched_prompt_wav)
                if hasattr(self, "umm_codebook"):
                    batched_prompt_umm_token = F.embedding(batched_prompt_umm_token, self.umm_codebook)
            
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
                    if getattr(self.model.hp,"use_window_mask",False):
                        inputs["bn_ctx"] = torch.ones([bs, 0, self.bn_config["bn_dim"]],device=device)*self.bn_config["bn_padding"]
                    else:
                        inputs["prompt_length"] = 0
                        inputs["bn_ctx"] = torch.ones([bs, int(batched_syn_umm_token.shape[1] / self.umm_frame_rate * self.mel_frame_rate), self.bn_config["bn_dim"]],device=device)*self.bn_config["bn_padding"]
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
                inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
                inputs["frontend"]["word_seg"][1, :] = 1
                if "lang" in inputs["frontend"]:
                    inputs["frontend"]["lang"] = inputs["frontend"]["lang"].repeat(2, 1)
                    inputs["frontend"]["lang"][1, :] = 1
            inputs["token"] = inputs["token"].repeat(2, 1)
            if self.use_wvae_vocoder:
                if "prompt_bn" in inputs:
                    inputs["prompt_bn"] = inputs["prompt_bn"].repeat(2, 1, 1)
                inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
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
            inputs['token'] = syn_umm_token
            if not self.without_prefix:
                inputs['token'] = torch.cat([prompt_umm_token,inputs["token"]],dim=1)
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
                out_mel = self.model.inference(inputs, 
                        self.diffusion_nfe,
                        self.diffusion_sampler,
                        text_cfg_w=self.text_cfg_w)
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                out_mel = out_mel[:, :, inputs["prompt_length"]:]
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
            if self.bn_config['wav_norm'] and inputs["scale"] is not None:
                batched_audio = batched_audio * torch.as_tensor(inputs["scale"]) / 0.95
            batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)

            for bidx,(uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
                if self.save_prompt:
                    prompt_wav = inputs["gt_wav"][bidx]
                    audio = np.concatenate([prompt_wav, np.ones([10]), audio])
                
                output_path = os.path.join(self.output_dir, uttid+".wav")
                if "syn_wavlen" in inputs:
                    audio = audio[...,:inputs["syn_wavlen"][bidx]]
                save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])

            return torch.from_numpy(batched_audio)
    
    def wvae_reconstruct(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        for item in batch:
            uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = item
            wav, _ = librosa.load(syn_wav_path, sr=self.mel_config["sampling_rate"], mono=not getattr(self.mel_config,"stereo", False)) 
            wav = torch.FloatTensor(wav).unsqueeze(0)
            if self.bn_config['wav_norm']:
                scale = max(0.001, torch.max(torch.abs(wav)))
                wav = wav / scale * 0.95
            wav = wav.to(device)
            if len(wav.shape) == 2:
                wav = wav.unsqueeze(1)
            reconstruct_wav, _, _ = self.wvae(wav)
            output_path = os.path.join(self.output_dir, uttid+".wav")
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
            else:
                raise NotImplementedError
        else:
            self.umm = None
        
        if self.use_wvae_vocoder:
            self.wvae = self.bn_config['vocoder_model'](local_rank=self.local_rank)['vocoder']
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

    def align_wav_for_chunk(self, wav, sampling_rate, umm_frame_rate, mel_frame_rate, 
            text_len):
            raise NotImplementedError

    def batched_data(self, batch):
        batched_prompt_text_id=None
        batched_prompt_wav_path=None
        batched_syn_text_id=None
        batched_syn_wav_path=None
        batched_syn_umm_token=None
        batched_uttid=None
        batched_scale=None
        assert self.infer_type in ["ar-diffusion-vocoder","diffusion-vocoder"]
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
            assert (container is None) or (container is not None and len(container) == bs)


        for item in batch:
            prompt_text_id, prompt_wav_path,syn_text_id,syn_wav_path, syn_umm_token, uttid =None,None,None,None,None,None
            if self.infer_type == "ar-diffusion-vocoder":
                prompt_text_id, syn_text_id, prompt_wav_path, syn_umm_token, uttid = item[:5]
            elif self.infer_type=="diffusion-vocoder":
                uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = item[:5]

            if prompt_wav_path:
                if not os.path.isfile(prompt_wav_path):
                    prompt_wav_path = None
                elif os.path.isfile(prompt_wav_path) ^ self.model.hp.use_prompt:
                        logger.warning(f"prompt wav {os.path.isfile(prompt_wav_path)}/{prompt_wav_path} mismatch with config use_prompt={self.model.hp.use_prompt}")

            
            batched_uttid=batching(batched_uttid, uttid)
            batched_prompt_wav_path=batching(batched_prompt_wav_path, prompt_wav_path)
            batched_prompt_text_id=batching(batched_prompt_text_id, prompt_text_id)
            batched_syn_text_id=batching(batched_syn_text_id, syn_text_id)
            batched_syn_wav_path=batching(batched_syn_wav_path, syn_wav_path)
            batched_syn_umm_token=batching(batched_syn_umm_token, syn_umm_token)
            if len(batch)==6:
                batched_scale=batching(batched_scale, item[-1])
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
            batched_scale
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
        umm_type="UMM", # UMM or USM
        umm_codebook_path=None,
        diffusion_precision="bf16",
        diffusion_nfe=10,
        diffusion_sampler="ddim",
        text_cfg_w=1,
        use_wvae_vocoder=False,
        bn_config=None,
        use_phone_lang=False,
        token_chunk_size=25,
        token_chunk_overlap=0,
        without_prefix=False,
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
            use_phone_lang,
            without_prefix=without_prefix,
        )
        self.token_chunk_size = token_chunk_size
        self.token_chunk_overlap = token_chunk_overlap

        if hasattr(self.model.hp, "window_size"):
            self.attention_window_size = self.model.hp.window_size[-1]
        else:
            self.attention_window_size = None

        logger.info(f"ChunkInfer:")
        args_dict = locals()
        del args_dict['self']
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

        if not self.without_prefix:
            assert prompt_umm_token is not None
            inputs["token"] = prompt_umm_token
        out_mel = None
        bs = len(syn_umm_token)
        device = syn_umm_token.device
        start_list = np.array(range(0, syn_umm_token.shape[1], self.token_chunk_size))

        outputs = []

        # To ensure reproduciable
        if not self.without_prefix:
            total_frame = int((prompt_umm_token.shape[1] + syn_umm_token.shape[1])/self.umm_frame_rate*self.mel_frame_rate)
        else:
            total_frame = int((syn_umm_token.shape[1])/self.umm_frame_rate*self.mel_frame_rate)
        self.model.clear_cache(self.diffusion_nfe, total_frame, bs=bs)
        for i, start in enumerate(start_list):
            if i < len(start_list) - 1:
                end = min(start_list[i+1]+self.token_chunk_overlap, syn_umm_token.shape[1])
                if end <= syn_umm_token.shape[1]:
                    current_overlap = self.token_chunk_overlap
                else:
                    end = syn_umm_token.shape[1]
                    current_overlap = end - start_list[i+1]
                valid_chunk_size = self.token_chunk_size
            else:
                end = syn_umm_token.shape[1]
                valid_chunk_size = end - start
                current_overlap = 0
            valid_chunk_size = int(valid_chunk_size / self.umm_frame_rate * self.mel_frame_rate)
            
            if i == 0:
                if not self.without_prefix:
                    inputs["token"] = torch.cat([
                        inputs["token"], 
                        syn_umm_token[:, start:end]], dim=1)
                else:
                    inputs["token"] = syn_umm_token[:, start:end]
            else:
                inputs["token"] = torch.cat([
                    inputs["token"][:, :-prev_overlap], 
                    syn_umm_token[:, start:end].repeat(2 if self.text_cfg_w!=1 else 1,1)], dim=1)

            token_len = inputs["token"].shape[1]
            mel_len = int(token_len / self.umm_frame_rate * self.mel_frame_rate)

            if out_mel is not None:
                prompt_length += out_mel.shape[2]
            else:
                prompt_length = inputs["bn_ctx"].shape[1]
            pad_len = valid_chunk_size
            inputs["bn_ctx"] = F.pad(inputs["bn_ctx"], (0, 0, 0, pad_len), "constant", self.bn_config["bn_padding"])

            if self.text_cfg_w != 1 and i == 0:
                inputs = self.make_cfg_input(inputs)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True): 
                out_mel = self.model.inference(inputs, 
                        self.diffusion_nfe,
                        self.diffusion_sampler,
                        text_cfg_w=self.text_cfg_w,
                        use_cache=True,
                        cached_v_len=prompt_length)
            if self.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
                out_mel = out_mel[:, :, prompt_length:prompt_length+valid_chunk_size]
            out_mel = out_mel.float() # [B, C, T]
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
        if self.bn_config['wav_norm'] and inputs["scale"] is not None:
            batched_audio = batched_audio * torch.as_tensor(inputs["scale"]) / 0.95
        batched_audio = np.clip(batched_audio, a_min=-1, a_max=1)

        groundtruth.emit('vocoder', data={
            'out_mel': outputs,
            'audio': batched_audio
        })

        for bidx,(uttid, audio) in enumerate(zip(inputs["uttid"], batched_audio)):
            if self.save_prompt:
                prompt_wav = inputs["gt_wav"][bidx]
                audio = np.concatenate([prompt_wav, np.ones([10]), audio])
            
            output_path = os.path.join(self.output_dir, uttid+".wav")
            if "syn_wavlen" in inputs:
                audio = audio[...,:inputs["syn_wavlen"][bidx]]
            save_wav(audio.T, output_path, sr=self.mel_config["sampling_rate"])
        return torch.from_numpy(batched_audio)

    def align_wav_for_chunk(self, wav, sampling_rate, umm_frame_rate, mel_frame_rate, 
            text_len):
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

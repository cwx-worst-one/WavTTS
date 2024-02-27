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
from recipes.musiclm.utils.dist import local_zero_first
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.voicebox.utils.infer_utils import set_seed, save_wav, load_torch_script
from recipes.voicebox.utils.infer_utils import set_seed, save_wav
from recipes.voicebox.lit_modules.lit_diffusion_voicebox import VoiceBoxModule as pl_module
from recipes.voicebox.vocoder.BigVGAN.meldataset import mel_spectrogram
from recipes.voicebox.vocoder.wvae import Wave
import logging
logger = logging.getLogger(__name__)

def prepare_diffusion_model(diffusion_ckpt_path, device):
    model = pl_module.load_from_checkpoint(
        diffusion_ckpt_path, device=torch.device(device)
    ).model.to(device)
    model.eval()
    return model

def prepare_diffusion_model_for_bigmusic(checkpoint_path, local_rank, cache_dir, params):
    diffusion_ckpt_path = checkpoint_path
    umm_ckpt_path = params["umm_ckpt_path"]

    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        diffusion_ckpt_path = download_checkpoint(diffusion_ckpt_path, cache_dir)
        umm_ckpt_path = download_checkpoint(umm_ckpt_path, cache_dir)
    

    bn_config = params.get('bn_config', None)
    mel_config = params.get('mel_config', None)


    infer_type=params.get("infer_type", "diffusion-vocoder")
    umm_frame_rate=params.get("umm_frame_rate", 25)
    mel_frame_rate=params.get("mel_frame_rate", 40)
    seed=params.get("seed", 1996)
    umm_type=params.get("umm_type", "UMM_music")
    diffusion_precision=params.get("diffusion_precision", "bf16")
    diffusion_nfe=params.get("diffusion_nfe", 10)
    diffusion_sampler=params.get("diffusion_sampler", "ddim")
    text_cfg_w=params.get("text_cfg_w", 0)
    use_wvae_vocoder=params.get("use_wvae_vocoder", False)


    diffusion = DiffusionU2SInfer(
        diffusion_ckpt_path=diffusion_ckpt_path,
        umm_ckpt_path=umm_ckpt_path,
        output_dir="./",
        bn_config=bn_config,
        mel_config=mel_config,
        infer_type=infer_type,
        umm_frame_rate=umm_frame_rate,
        mel_frame_rate=mel_frame_rate,
        seed=seed,
        umm_type=umm_type,
        diffusion_precision=diffusion_precision,
        diffusion_nfe=diffusion_nfe,
        diffusion_sampler=diffusion_sampler,
        text_cfg_w=text_cfg_w,
        use_wvae_vocoder=use_wvae_vocoder,
        device = f"cuda:{local_rank}",
        )

    diffusion.setup_from_local_rank(local_rank)
    return { "diffusion": diffusion }

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
        output_dir,
        infer_type,
        umm_frame_rate,
        mel_frame_rate,
        mel_config,
        vocoder_ckpt_path=None,
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
        device="cuda",
    ):
        super().__init__()

        seed = set_seed(seed)
        self.vocoder_ckpt_path = vocoder_ckpt_path
        self.umm_ckpt_path = umm_ckpt_path
        self.mel_transform = prepare_mel_transform(mel_config)
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
        
        if self.infer_type != "vocoder":
            self.model = prepare_diffusion_model(diffusion_ckpt_path, device)
        
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
        
        os.makedirs(output_dir, exist_ok=True)
    

    # align wav for umm & mel feature length.
    def align_wav(self, wav, sampling_rate, umm_frame_rate, mel_frame_rate):
        umm_hop = sampling_rate // umm_frame_rate
        mel_hop = sampling_rate // mel_frame_rate * 4
        align_block_len = abs(umm_hop*mel_hop) // math.gcd(umm_hop, mel_hop)
        crop_wav_len = wav.shape[1] % align_block_len
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, align_block_len-crop_wav_len), "constant", 0)
        return wav

    def align_wav2(self, wav, wav_divide, sampling_rate, umm_frame_rate, umm_add_len):
        umm_hop = sampling_rate // umm_frame_rate
        wav_add_len = umm_add_len * umm_hop
        crop_wav_len = (wav_add_len + wav.shape[1]) % wav_divide
        if crop_wav_len > 0:
            wav = F.pad(wav, (0, wav_divide-crop_wav_len), "constant", 0)
        return wav

    def prepare_features(self, batch):
        if self.infer_type == "ar-diffusion-vocoder":
            prompt_text_id, syn_text_id, prompt_wav_path, syn_umm_token, uttid = batch
        elif self.infer_type == "diffusion-vocoder":
            uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch
        else:
            raise NotImplementedError

        # if prompt_text_id is None or syn_text_id is None:
        #     return None
        
        device = f"cuda:{self.trainer.local_rank}"
        inputs = dict()

        # Syn
        if self.infer_type == "diffusion-vocoder": 
            wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
            inputs["gt_wav"] = wav
            wav = torch.FloatTensor(wav).unsqueeze(0)
            scale = max(0.001, torch.max(torch.abs(wav)).item())
            wav = wav / scale * 0.95
            wav = wav.to(device)
            inputs["scale"] = scale
            syn_wav = self.align_wav(wav, self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate)
            if self.umm_type == "USM":
                token_len = syn_wav.shape[1] // 960
                syn_umm_token = self.umm.wav2token(resample(syn_wav, 24000, 16000), dtype=torch.bfloat16)
                syn_umm_token = syn_umm_token[:, :token_len]
            elif self.umm_type == "UMM":
                syn_umm_token = self.umm.wav2token(syn_wav)
            elif self.umm_type in ["UMMv2","UMM_music"]:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    syn_umm_token = self.umm.wav2token(syn_wav)
            else:
                raise NotImplementedError
        elif self.infer_type == "ar-diffusion-vocoder":
             syn_umm_token = syn_umm_token.unsqueeze(0)
        else:
            raise NotImplementedError
        if hasattr(self, "umm_codebook"):
            syn_umm_token = F.embedding(syn_umm_token, self.umm_codebook)
        

        if os.path.isfile(prompt_wav_path) ^ self.model.hp.use_prompt:
            logger.error(f"prompt wav {os.path.isfile(prompt_wav_path)}/{prompt_wav_path} mismatch with config use_prompt={self.model.hp.use_prompt}")

        
        # Prompt
        if os.path.isfile(prompt_wav_path):
            wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True)
            inputs["gt_wav"] = wav
            wav = torch.FloatTensor(wav).unsqueeze(0)
            scale = max(0.001, torch.max(torch.abs(wav)).item())
            wav = wav / scale * 0.95
            wav = wav.to(device)
            if self.infer_type == "diffusion-vocoder":
                prompt_wav = self.align_wav(wav, self.mel_config["sampling_rate"], self.umm_frame_rate, self.mel_frame_rate)
            elif self.infer_type == "ar-diffusion-vocoder":
                wav_divide = 4800 if (self.umm_frame_rate==25 and self.mel_frame_rate==40) else 600
                prompt_wav = self.align_wav2(wav, wav_divide, self.mel_config["sampling_rate"], self.umm_frame_rate, syn_umm_token.shape[1])
            inputs["scale"] = scale

            if self.umm_type == "USM":
                token_len = prompt_wav.shape[1] // 960
                prompt_umm_token = self.umm.wav2token(resample(prompt_wav, 24000, 16000), dtype=torch.bfloat16)
                prompt_umm_token = prompt_umm_token[:, :token_len]
            elif self.umm_type == "UMM":
                prompt_umm_token = self.umm.wav2token(prompt_wav)
            elif self.umm_type in ["UMMv2","UMM_music"]:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    prompt_umm_token = self.umm.wav2token(prompt_wav)
            else:
                raise NotImplementedError

            if hasattr(self, "umm_codebook"):
                prompt_umm_token = F.embedding(prompt_umm_token, self.umm_codebook)
        else:
            prompt_umm_token = None
            prompt_wav = None
         


        # Text
        if prompt_text_id is not None and syn_text_id is not None:
            text_id = torch.cat([prompt_text_id, syn_text_id[:, 1:]], dim=-1)
            text_id = F.pad(text_id, (0, 1), "constant", 1)
            inputs["frontend"] = {
                    "phone": text_id[0:1, :].to(device),
                    "tone": text_id[1:2, :].to(device),
                    "word_seg": text_id[2:3, :].to(device),
                }
            #null_text_id = torch.ones_like(text_id)
            #inputs["null_frontend"] = {
            #        "phone": null_text_id[0:1, :].to(device),
            #        "tone": null_text_id[1:2, :].to(device),
            #        "word_seg": null_text_id[2:3, :].to(device),
            #    }
            if self.use_phone_lang:
                inputs["frontend"]["lang"] = text_id[3:4, :].to(device)
            #    inputs["null_frontend"]["lang"] = null_text_id[3:4, :].to(device)

        if self.infer_type in ["diffusion-vocoder", "ar-diffusion-vocoder"]: 
            if prompt_umm_token is None:
                inputs["token"] = syn_umm_token
            else:
                inputs["token"] = torch.cat([prompt_umm_token, syn_umm_token], dim=1)
            token_len = inputs["token"].shape[1]
            mel_len = int(token_len / self.umm_frame_rate * self.mel_frame_rate)

            if self.use_wvae_vocoder:
                inputs["bn_ctx"] = torch.ones([1, mel_len, self.bn_config['bn_dim']], device=device) * self.bn_config['bn_padding']

                if prompt_wav is not None:
                    crop_bn = self.wvae.encode(prompt_wav)
                    m, logs = torch.split(crop_bn, 64, dim=-1)
                    crop_bn = m + torch.randn_like(m) * torch.exp(logs)
                    crop_bn = self.bn_norm.norm_mel(crop_bn)

                    inputs["prompt_bn"] = crop_bn.transpose(1, 2) # [B,C,T]
                    inputs["prompt_length"] = inputs["prompt_bn"].shape[-1]
                    inputs["bn_ctx"][:, :inputs["prompt_bn"].shape[-1], :] = inputs["prompt_bn"].transpose(1, 2)
                else:
                    inputs["prompt_length"] = 0

                
            else:
                prompt_mel = self.mel_transform(prompt_wav)
                prompt_mel = self.mel_norm.norm_mel(prompt_mel)
                inputs["prompt_mel"] = prompt_mel
                inputs["mel_ctx"] = torch.ones([1, mel_len, self.mel_config["num_mels"]], device=device) * self.mel_mask_value
                inputs["mel_ctx"][:, :inputs["prompt_mel"].shape[-1],] = inputs["prompt_mel"].transpose(1, 2)
                inputs["prompt_length"] = inputs["prompt_mel"].shape[-1]
        else:
            raise NotImplementedError
        inputs["uttid"] = uttid

        if self.text_cfg_w != 1:
            inputs = self.make_cfg_input(inputs)
        
        return inputs
    
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

            inputs["token"] = inputs["token"].repeat(2, 1) # 1， 500 -> 2, 500
            if self.use_wvae_vocoder:
                if "prompt_bn" in inputs:
                    inputs["prompt_bn"] = inputs["prompt_bn"].repeat(2, 1, 1) # 1,64,401->2, 64, 401
                inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
            else:
                inputs["prompt_mel"] = inputs["prompt_mel"].repeat(2, 1, 1)
                inputs["mel_ctx"] = inputs["mel_ctx"].repeat(2, 1, 1)
        return inputs
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.infer_type == "vocoder":
            uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch
            self.wvae_reconstruct(batch)
        else:
            inputs = self.prepare_features(batch)
            # for key in inputs:
            #    if torch.is_tensor(inputs[key]):
            #        print(f"\t {key} -> shape={inputs[key].shape}")
            #    else:
            #        print(f"\t {key} -> {inputs[key]}")
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
                output_wav = self.wvae.decoder_from_z(z.transpose(1, 2))
            else:
                out_mel = self.mel_norm.denorm_mel(out_mel)
                out_mel = torch.clamp(out_mel, min=-8.5, max=3.5)
                output_wav = self.vocoder(out_mel)

            audio = output_wav.squeeze().cpu().numpy()
            audio = audio * inputs["scale"] / 0.95
            audio = np.clip(audio, a_min=-1, a_max=1)

            if self.save_prompt:
                prompt_wav = inputs["gt_wav"]
                audio = np.concatenate([prompt_wav, np.ones([10]), audio])
            output_path = os.path.join(self.output_dir, inputs["uttid"]+".wav")
            save_wav(audio, output_path)
    
    def wvae_reconstruct(self, batch):
        device = f"cuda:{self.trainer.local_rank}"
        uttid, prompt_wav_path, syn_wav_path, prompt_text_id, syn_text_id = batch
        wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
        wav = torch.FloatTensor(wav).unsqueeze(0)
        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        wav = wav.to(device)

        reconstruct_wav = self.wvae.reconstruct(wav)
        output_path = os.path.join(self.output_dir, uttid+".wav")
        save_wav(reconstruct_wav.cpu().numpy(), output_path)


    def setup(self, stage):
        device = f"cuda:{self.trainer.local_rank}"
        if self.infer_type != "vocoder":
            if self.umm_type == "USM":
                self.umm = prepare_usm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM":
                self.umm = prepare_umm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMMv2":
                self.umm = prepare_ummv2(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM_music":
                self.umm = prepare_umm_music(self.umm_ckpt_path, device)
        
        if self.use_wvae_vocoder:
            self.wvae_encoder = load_torch_script(
                model_path=self.bn_config['wvae_encoder_path'],
                rank=self.trainer.local_rank,
                cache_dir=self.bn_config['wvae_cache_dir']
            )
            self.wvae_decoder = load_torch_script(
                model_path=self.bn_config['wvae_decoder_path'],
                rank=self.trainer.local_rank,
                cache_dir=self.bn_config['wvae_cache_dir']
            )
            self.wvae = Wave(self.wvae_encoder, self.wvae_decoder, 
                version=self.bn_config['wvae_version'],
                hop_size=self.bn_config['wvae_encoder_hop_size'],
                win_size=self.bn_config['wvae_encoder_win_size'])
        else:
            self.vocoder = prepare_vocoder(self.vocoder_ckpt_path, device)


        if self.umm_codebook_path is not None:
            self.umm_codebook = prepare_umm_codebook(self.umm_codebook_path, device)


    def setup_from_local_rank(self, local_rank):
        device=f"cuda:{local_rank}"
        if self.infer_type != "vocoder":
            if self.umm_type == "USM":
                self.umm = prepare_usm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM":
                self.umm = prepare_umm(self.umm_ckpt_path, device)
            elif self.umm_type == "UMMv2":
                self.umm = prepare_ummv2(self.umm_ckpt_path, device)
            elif self.umm_type == "UMM_music":
                self.umm = prepare_umm_music(self.umm_ckpt_path, device)
        
        if self.use_wvae_vocoder:
            self.wvae_encoder = load_torch_script(
                model_path=self.bn_config['wvae_encoder_path'],
                rank=local_rank,
                cache_dir=self.bn_config['wvae_cache_dir']
            )
            self.wvae_decoder = load_torch_script(
                model_path=self.bn_config['wvae_decoder_path'],
                rank=local_rank,
                cache_dir=self.bn_config['wvae_cache_dir']
            )
            self.wvae = Wave(self.wvae_encoder, self.wvae_decoder, 
                version=self.bn_config['wvae_version'],
                hop_size=self.bn_config['wvae_encoder_hop_size'],
                win_size=self.bn_config['wvae_encoder_win_size'])
        else:
            self.vocoder = prepare_vocoder(self.vocoder_ckpt_path, device)


        if self.umm_codebook_path is not None:
            self.umm_codebook = prepare_umm_codebook(self.umm_codebook_path, device)



def norm_wav(diffusion, wav, is_prompt=False):
    scale = max(0.001, torch.max(torch.abs(wav)).item())
    wav = wav / scale * 0.95
    if diffusion.infer_type == "diffusion-vocoder" or is_prompt:
        wav = diffusion.align_wav(wav, diffusion.mel_config["sampling_rate"], diffusion.umm_frame_rate, diffusion.mel_frame_rate)
    elif diffusion.infer_type == "ar-diffusion-vocoder":
        wav_divide = 4800 if (diffusion.umm_frame_rate==25 and diffusion.mel_frame_rate==40) else 600
        wav = diffusion.align_wav2(wav, wav_divide, diffusion.mel_config["sampling_rate"], diffusion.umm_frame_rate, syn_umm_token.shape[1])
    return wav, scale

def wav2token(diffusion, wav, device, sr=24000, is_prompt=False):
    inputs = {}
    if sr != 24000:
        raise NotImplementedError
    
    wav, scale = norm_wav(diffusion, wav)
    wav = wav.to(device)
    if diffusion.umm_type == "USM":
        token_len = wav.shape[1] // 960
        umm_token = diffusion.umm.wav2token( dtype=torch.bfloat16)
        umm_token = umm_token[:, :token_len]
    elif diffusion.umm_type == "UMM":
        umm_token = diffusion.umm.wav2token(wav)
    elif diffusion.umm_type in ["UMMv2","UMM_music"]:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            umm_token = diffusion.umm.wav2token(wav)
    else:
        raise NotImplementedError

    if hasattr(diffusion, "umm_codebook"):
        umm_token = F.embedding(umm_token, diffusion.umm_codebook)
    
    return umm_token, scale, wav


def token2wav(diffusion, umm_token, scale=1.0, prompt_wav=None):
    inputs = dict()
    device = umm_token.device

    # Prepare diffusion inputs
    inputs["scale"] = scale
    if prompt_wav is None:
        inputs["token"] = umm_token
    else:
        prompt_umm_token, _, prompt_wav = wav2token(diffusion, prompt_wav, is_prompt=True, device=device)
        inputs["token"] = torch.cat([prompt_umm_token, umm_token], dim=1)
        
    token_len = inputs["token"].shape[1]
    batch_size = inputs["token"].shape[0]
    mel_len = int(token_len / diffusion.umm_frame_rate * diffusion.mel_frame_rate)

    if diffusion.use_wvae_vocoder:
        inputs["bn_ctx"] = torch.ones([batch_size, mel_len, diffusion.bn_config['bn_dim']], device=device) * diffusion.bn_config['bn_padding']
        if prompt_wav is not None:
            crop_bn = diffusion.wvae.encode(prompt_wav) # 1， 401， 128
            m, logs = torch.split(crop_bn, 64, dim=-1)
            crop_bn = m + torch.randn_like(m) * torch.exp(logs)
            crop_bn = diffusion.bn_norm.norm_mel(crop_bn) # 1，401，64

            inputs["prompt_bn"] = crop_bn.transpose(1, 2) # [B,C,T]
            inputs["prompt_length"] = inputs["prompt_bn"].shape[-1]
            inputs["bn_ctx"][:, :inputs["prompt_bn"].shape[-1], :] = inputs["prompt_bn"].transpose(1, 2)
        else:
            inputs["prompt_length"] = 0

    if diffusion.text_cfg_w != 1:
        inputs = diffusion.make_cfg_input(inputs)

    # 2,500 / 2, 800,64/ 2,64,401
    # 4,500 / 4,800,64 / 4,64,401
    # Predict diffusion
    if diffusion.diffusion_precision == "bf16":
        dtype = torch.bfloat16
    elif diffusion.diffusion_precision == "fp16":
        dtype = torch.float16
    elif diffusion.diffusion_precision == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError 
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True): 
        out_mel = diffusion.model.inference(inputs, 
                diffusion.diffusion_nfe,
                diffusion.diffusion_sampler,
                text_cfg_w=diffusion.text_cfg_w)
        if diffusion.infer_type in ["ar-diffusion-vocoder", "diffusion-vocoder"]:
            out_mel = out_mel[:, :, inputs["prompt_length"]:]
        out_mel = out_mel.float()

    with torch.no_grad():
        if diffusion.use_wvae_vocoder:
            z = out_mel
            z = diffusion.bn_norm.denorm_mel(z) # 2,64,399
            output_wav = diffusion.wvae.decoder_from_z(z.transpose(1, 2))
        else:
            out_mel = diffusion.mel_norm.denorm_mel(out_mel)
            out_mel = torch.clamp(out_mel, min=-8.5, max=3.5)
            output_wav = diffusion.vocoder(out_mel)

        output_wav = output_wav * inputs["scale"] / 0.95
        output_wav = torch.clamp(output_wav, min=-1, max=1)
    return output_wav

@torch.no_grad()
def run_diffusion_vocoder(requires, samples, params, prompt_wav=None):
    diffusion = requires['diffusion']
    output_wav = token2wav(diffusion, 
              samples,
              prompt_wav=prompt_wav)
    output_wav = output_wav.detach()
    return output_wav

def test():
    hparams_file = "recipes/voicebox/conf/infer_reconstruction_25hzMel_test.yaml"
    syn_wav_path = "/mnt/bn/lab-speechsv-zhongyi-lq/voicebox/generated_output/0208_beam4_zh_musicinput_fullsong_xuyao_baseline/test_courage2000.generated.wav"
    prompt_wav_path = "/mnt/bn/lab-speechsv-zhongyi-lq/voicebox/generated_output/0208_debug_SFT_master_pretrain_CL/cxz-110001.wav.style_audio.wav"
    output_wav1 = "test1.wav"
    output_wav2 = "test2.wav"
    local_rank = 0
    device = f"cuda:0"
    cache_dir = ".module_cache/"

    from hyperpyyaml import load_hyperpyyaml
    from samantha.utils.hparams import DotDict
    with open(hparams_file, "r", encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin)
    checkpoint_path = params["diffusion_ckpt_path"]
    requires = prepare_diffusion_model_for_bigmusic(checkpoint_path, local_rank, cache_dir, params)

    prompt_wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True) 
    syn_wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
    prompt_wav = torch.FloatTensor(prompt_wav).unsqueeze(0)#.to(device)
    syn_wav = torch.FloatTensor(syn_wav).unsqueeze(0)#.to(device)
    umm_token, scale, _ = wav2token(requires["diffusion"], wav=syn_wav, device=device)
    rerender_audio = run_diffusion_vocoder(requires, samples=umm_token, params=params, prompt_wav=prompt_wav)
    rerender_audio = rerender_audio.squeeze().cpu().numpy()
    save_wav(rerender_audio, output_wav1)

    prompt_wav, _ = librosa.load(prompt_wav_path, sr=24000, mono=True) 
    syn_wav, _ = librosa.load(syn_wav_path, sr=24000, mono=True) 
    prompt_wav = torch.FloatTensor(prompt_wav).unsqueeze(0)#.to(device)
    syn_wav = torch.FloatTensor(syn_wav).unsqueeze(0)#.to(device)
    prompt_wav = torch.tile(prompt_wav, [2,1])
    syn_wav = torch.tile(syn_wav, [2,1])
    umm_token, scale, _ = wav2token(requires["diffusion"], wav=syn_wav, device=device)
    rerender_audio = run_diffusion_vocoder(requires, samples=umm_token, params=params, prompt_wav=prompt_wav)
    rerender_audio = rerender_audio.squeeze().cpu().numpy()[0]
    save_wav(rerender_audio, output_wav2)


if __name__ == "__main__":
    test()
import os
from tqdm import tqdm
import math
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Tuple
from einops import rearrange, repeat

import numpy as np
import time
import gradio as gr
import glob

from recipes.musiclm.lightning.modules import SemanticModule
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
    mulan_rvq_indexs,
)
from recipes.diffusion.modules.pl_module import DiffusionModule
from recipes.diffusion.models.tnt import TNTDiffusionNetwork
from recipes.diffusion.models.semantic_model.utils import (
    load_torch_script_module,
    w2v_bert_tokenization
)
from recipes.soundstream.models.vqgan import VQGAN_KL, VQGAN_KL_mix
from recipes.soundstream.models.modules.discriminator import MultiScaleSTFTDiscriminator
from recipes.soundstream.modules.pl_module_vae import VocoderModule


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = extend_dim(scale, x.ndim)
        x = x.clamp(-scale, scale) / scale
        return x

def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

class ARVSampler(nn.Module):
    def __init__(self, in_channels: int, length: int, num_splits: int):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
    
    def set_device(self, device: torch.device):
        self.device = device

    def sample_loop(
            self, 
            model: nn.Module,
            positive_mulan_context: torch.tensor, 
            negative_mulan_context: torch.tensor, 
            semantic_context: torch.tensor,
            current: Tensor, 
            num_steps: int = 20, 
            bf16_portion: float = 0.,
            chunk_index: int = -1, 
            show_progress: bool = False,
            angle_schedule: str = 'linear', 
            schdeule_slope: float = 2.0,
            classifier_free_guidance: int = 1,
    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        if angle_schedule == 'linear':
            angle_schedule = np.linspace(schdeule_slope, 1., num_steps)
            angle_schedule /= angle_schedule.sum()
        elif angle_schedule == 'uniform':
            angle_schedule = [1 / num_steps,] * num_steps

        B, C, T = current.shape

        sigma = 1.
        for i in progress_bar:
            if chunk_index < 0:
                sigma_i = torch.ones(B, 1, T, device=self.device) * sigma
            else:
                # continuation
                sigma_i = torch.ones(B, 1, 2*self.split_length, device=self.device)
                sigma_i = torch.cat([torch.zeros(B, 1, T-2*self.split_length, device=self.device), sigma_i], -1)
                sigma_i = sigma_i * sigma

            if i < int(num_steps*bf16_portion):
                enabled = True
            else:
                enabled = False

            if sigma_i[0,0,0] <= 0.25:
                key = '2-0'
            elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                key = '2-1'
            elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                key = '2-2'
            elif sigma_i[0,0,0] > 0.75:
                key = '2-3'

            # model prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                v_pred = model[key].model(
                    current, 
                    timesteps=sigma_i, 
                    mulan_context=positive_mulan_context, 
                    semantic_context=semantic_context,
                    mulan_force_cfg=0,
                    semantic_force_cfg=0,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model[key].model(
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=positive_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=1,
                            semantic_force_cfg=1,
                        )
                    v_negative = v_uncond
                    if negative_mulan_context is not None:
                        v_negative = model[key].model(
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=negative_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=0,
                            semantic_force_cfg=1,
                        )
                    guidance_scale = classifier_free_guidance
                    v_pred = v_uncond + guidance_scale * (v_pred - v_negative)
            omega = math.pi / 2. * angle_schedule[i]
            sigma = sigma - angle_schedule[i]
            current = np.cos(omega) * current - np.sin(omega) * v_pred
            progress_bar.set_description(f"Sampling {i}")
        current = clip(current)
        return current

    def sample_start(self, 
        model, 
        num_items: int, 
        num_steps: int, 
        bf16_portion: float,
        positive_mulan_context: torch.tensor, 
        negative_mulan_context: torch.tensor, 
        semantic_context: torch.tensor,
        **kwargs
    ) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Sample start
        return self.sample_loop(
            model=model,
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context,
            current=torch.randn(b, c, t, device=self.device),
            num_steps=num_steps, 
            bf16_portion=bf16_portion,
            chunk_index=-1, 
            **kwargs
        )

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        positive_mulan_context: torch.tensor,
        negative_mulan_context: torch.tensor,
        semantic_context: torch.tensor,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        bf16_portion: float,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        angle_schedule: str = 'linear',
        schdeule_slope: float = 2.0,
        classifier_free_guidance = 1,
    ) -> Tensor:
        #assert_message = f"required at least {self.num_splits} chunks"
        #assert num_chunks >= self.num_splits, assert_message
        self.num_chunks = num_chunks

        # Sample initial chunks
        start = self.sample_start(
            model=model,
            num_items=num_items,
            num_steps=num_steps,
            bf16_portion=bf16_portion,
            angle_schedule=angle_schedule,
            classifier_free_guidance=classifier_free_guidance,
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.detach().cpu().numpy()

def generate_audio(
    text_prompt,
    semantic_token_sequence,
    num_iters,
    bf16_portion,
    schdeule_slope,
    guidance,
):
    negative_prompt = ''

    global mulan_model
    global mulan_centers
    global semantic_module
    global diffusion_model
    global sampler
    global vocoder_model
    global device

    num_iters = int(num_iters)
    bf16_portion = float(bf16_portion)
    schdeule_slope = float(schdeule_slope)
    guidance = float(guidance)
    print('-----------------------------')
    print("text_prompt", text_prompt)
    print("num_iters", num_iters)
    print("bf16_portion", bf16_portion)
    print("guidance", guidance)
    print('-----------------------------')
    text_prompt = [text_prompt for i in range(1)]

    with torch.no_grad():
        time_start = time.time()
        # text prompt
        mulan_emb = mulan_inference(mulan_model, text=text_prompt, device=device)
        postive_mulan_ids, ds = mulan_rvq_indexs(mulan_emb, mulan_centers)

        if semantic_token_sequence != "":
            semantic_samples = torch.tensor(
                eval(semantic_token_sequence), device=device
            )[None, :].repeat(1, 1)
        else:
            semantic_samples = semantic_module.predict(postive_mulan_ids, ExtraParams())  # .repeat(3, 1)

        # negative prompt
        if negative_prompt == '':
            negative_mulan_ids = None
        else:
            mulan_emb = mulan_inference(mulan_model, text=negative_prompt, device=device)
            negative_mulan_ids, ds = mulan_rvq_indexs(mulan_emb, mulan_centers)
            negative_mulan_ids.repeat(1, 1)

        text2semantic_time = time.time() - time_start

        time_start = time.time()
        pred_emb = sampler(
            model=diffusion_model,
            positive_mulan_context=postive_mulan_ids,
            negative_mulan_context=negative_mulan_ids,
            semantic_context=semantic_samples,
            num_items=semantic_samples.shape[0],
            num_chunks=1,
            num_steps=num_iters,
            bf16_portion=bf16_portion,
            start=None,
            show_progress=True,
            angle_schedule='linear',
            schdeule_slope=schdeule_slope,
            classifier_free_guidance=guidance,
        )

        wavs_g = vocoder_model.decode(pred_emb.float())
        diffusion2audio_time = time.time() - time_start

        wav_g = wavs_g[0]

        output_audio = []
        # for wav_g in wavs_g:
        wav_g = wav_g / wav_g.abs().max()
        wav_g = torch_fp32_to_numpy_int16(wav_g)
        output_audio.append(wav_g)

        return (
            (24000, output_audio[0]),
            text2semantic_time,
            diffusion2audio_time,
            f"{list(semantic_samples[0].detach().cpu().numpy())}",
        )
REMOTE_PATHS = {
    'mulan_model_path': '/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/gpt_all/checkpoints/mulan-step=036000-median_rank_0=127-kaggle.ckpt',
    'mulan_center_path': '/home/byte_speech_sv/dongguo/mulan/codebook/kmeans_minibatch_codebook-mulan1b_g4_mix_127-1024x12.npy',
    'semantic_model_path': '/mnt/bn/audio-diffusion/ducle/logs/semantic_flash_llama/mcc40m_filtered_705m/checkpoints/step=081000-tr_loss=2.4451-val_loss_0=2.8561.ckpt',
    'semantic_center_path': '/home/byte_speech_sv/zongyu.yin/ckpts/w2v/2.1/centroids_epoch_10.npy',
    'diffusion_model_path': '/home/byte_speech_sv/weitsung.lu/diffusion/model_9_part5/checkpoints/diffusion-step=267999.ckpt',
    'vocoder_model_path': '/home/byte_speech_sv/weitsung.lu/soundstream/yongye_vae_dac_fintune_v2/checkpoints/soundstream-step=486399-val_sdr=11.5247.ckpt',
    'vocoder_model_path_2': '/home/byte_speech_sv/weitsung.lu/soundstream/yongye/1000k_ckpt.pyt'
}

LOCAL_PATHS = {
    'mulan_model_path': 'recipes/diffusion/assets/mulan-step=036000-median_rank_0=127-kaggle.ckpt',
    'mulan_center_path': 'recipes/diffusion/assets/kmeans_minibatch_codebook-mulan1b_g4_mix_127-1024x12.npy',
    'semantic_model_path': 'recipes/diffusion/assets/step=081000-tr_loss=2.4451-val_loss_0=2.8561.ckpt',
    'semantic_center_path': 'recipes/diffusion/assets/centroids_epoch_10.npy',
    'diffusion_model_path': 'recipes/diffusion/assets/diffusion-step=267999.ckpt',
    'vocoder_model_path': 'recipes/diffusion/assets/soundstream-step=486399-val_sdr=11.5247.ckpt',
    'vocoder_model_path_2': 'assets/1000k_ckpt.pyt' # different path due to soundstream code
}

if __name__ == "__main__":
     # settings
    device = torch.device('cuda:0')
    
    # Download models
    asset_path = 'recipes/diffusion/assets'
    os.makedirs(asset_path, exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    for k, v in LOCAL_PATHS.items():
        if not os.path.exists(v):
            if k == 'vocoder_model_path_2':
                os.system(f'hdfs dfs -get {REMOTE_PATHS[k]} assets')
            else:
                if '/home/' in REMOTE_PATHS[k]:
                    os.system(f'hdfs dfs -get {REMOTE_PATHS[k]} {asset_path}')
                elif '/mnt/' in REMOTE_PATHS[k]:
                    os.system(f'cp {REMOTE_PATHS[k]} {asset_path}')
    

    diffusion_model_path = {
        '2-0': 'recipes/diffusion/assets/diffusion_2.0/diffusion-step=024999.ckpt',
        '2-1': 'recipes/diffusion/assets/diffusion_2.1/diffusion-step=022999.ckpt',
        '2-2': 'recipes/diffusion/assets/diffusion_2.2/diffusion-step=025999.ckpt',
        '2-3': 'recipes/diffusion/assets/diffusion_2.3/diffusion-step=024999.ckpt',

    }

    # Mulan
    mulan_model = create_mulan_model(LOCAL_PATHS['mulan_model_path'], device=device)
    mulan_centers = np.load(LOCAL_PATHS['mulan_center_path'])
    mulan_centers = torch.from_numpy(mulan_centers).float().to(device)

    # Semantic model
    class ExtraParams:
        sample_rate = 24000
        duration = 10
        num_rounds = 3
        semantic_duration = 10
        coarse_duration = 10
        fine_duration = 4
        semantic_stride = 5
        coarse_stride = 5
        fine_stride = 3
        wav2vec_codebook_size = 1024
        mulan_codebook_size = 1024
        mulan_num_rvq = 12
        soundstream_codebook_size = 1024
        wav2vec_frame_rate = 25
        soundstream_frame_rate = 50
        num_coarse = 4
        num_fine = 8
        semantic_temperature = 1.0
        coarse_temperature = 0.9
        fine_temperature = 0.8
        sample_mode = "gumbel"


    semantic_module = SemanticModule.load_from_checkpoint(LOCAL_PATHS['semantic_model_path']).to(device).eval()
    semantic_centers = np.load(LOCAL_PATHS['semantic_center_path'])
    semantic_centers = torch.from_numpy(semantic_centers).float().to(device)

    # diffusion model
    diffusion_model = {}
    for key in diffusion_model_path:
        diffusion_model[key] = DiffusionModule.load_from_checkpoint(
            diffusion_model_path[key],
            diffusion_model=TNTDiffusionNetwork(
                input_dim=16,
                feature_dim=1024,
                context_dim=1,
                depth=16,
                segment_size=64,
                segment_stride=64,
                dropout=0,
                mulan_cfg_prob=0.10,
                semantic_cfg_prob=0.10,
                use_checkpoint=False
            ),
            target_dim=16,
            num_chunks=1,
            chunk_length=1250,
        )
        diffusion_model[key].eval()
        diffusion_model[key].to(device)
        diffusion_model[key].sampler.set_device(device)

    sampler = ARVSampler(16, 2500, 1)
    sampler.set_device(device)

    vocoder_model_pl = VocoderModule.load_from_checkpoint(
        LOCAL_PATHS['vocoder_model_path'],
        generator=VQGAN_KL_mix(
            latent_dim=16,
            downsample_rates=[2, 3, 4, 8],
            upsample_rates=[4, 4, 3, 2],
            encoder_base_dim=96,
            decoder_base_dim=2560,
        ),
        discriminator=MultiScaleSTFTDiscriminator(
            filters=48,
        ),
        strict=False
    )
    vocoder_model = vocoder_model_pl.generator.eval().to(device)

    # gradio
    text_prompt = gr.Textbox(
        value="smooth hiphop", label="Text prompt (str)"
    )
    semantic_token_sequence = gr.Textbox(
        label="Semantic token sequence (Optional)",
        placeholder="Enter a sequence of semantic tokens as a list, so [712, 333, 236, ...]",
    )
    num_iters = gr.Textbox(
        value=25,
        label="Number of iterations for the diffusion process (defalut: (int) 20))"
    )
    bf16_portion = gr.Textbox(
        value=0.9,
        label="First N% of iterations will run in half precision. Experimental feature, may hurt quality. (default: (float) 0.7)"
    )
    schdeule_slope = gr.Textbox(
        value=2.5,
        label="Schedule slope for the diffusion process (default: (float) 2.0)"
    )
    guidance = gr.Textbox(
        value=3.5,
        label="Guidance for the classifier-free diffusion (default: (float) 2.5)"
    )


    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            text_prompt,
            semantic_token_sequence,
            num_iters,
            bf16_portion,
            schdeule_slope,
            guidance,
        ],
        outputs=[
            gr.Audio(label="Output 1"),
            gr.Textbox(label="Text to semantic time (sec)"),
            gr.Textbox(label="Diffusion to audio time (sec)"),
            gr.Textbox(label="Semantic tokens 1"),
        ],
        title="Diffusion demo",
        description="Interactive demo for the diffusion model.",
    )
    demo.queue(concurrency_count=4)
    demo.launch(share=True)


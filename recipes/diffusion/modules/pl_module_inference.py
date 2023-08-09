import os
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

import torchaudio

import numpy as np
import pytorch_lightning as pl
from einops import rearrange
from recipes.musiclm.lightning.modules import SemanticModule
from recipes.musiclm.lightning.rlhf import SemanticSequenceTrainingModule
from recipes.musiclm.lightning.inference import BaseModule
from recipes.diffusion.modules.pl_module import load_ema_checkpoint
from recipes.diffusion.models.tnt import TNTDiffusionNetwork
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import format_name


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
            condition_signal: list = ['semantic', 'mulan'],
    ) -> Tensor:
        
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        enable_semantic_condition, enable_mulan_condition = 1, 1

        if 'semantic' in condition_signal:
            enable_semantic_condition = 0
        elif 'mulan' in condition_signal:
            enable_mulan_condition = 0

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
                key = '2_0'
            elif sigma_i[0,0,0] > 0.25 and sigma_i[0,0,0] <= 0.5:
                key = '2_1'
            elif sigma_i[0,0,0] > 0.5 and sigma_i[0,0,0] <= 0.75:
                key = '2_2'
            elif sigma_i[0,0,0] > 0.75:
                key = '2_3'

            # model prediction
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                v_pred = model[key](
                    current, 
                    timesteps=sigma_i, 
                    mulan_context=positive_mulan_context, 
                    semantic_context=semantic_context,
                    mulan_force_cfg=enable_mulan_condition,
                    semantic_force_cfg=enable_semantic_condition,
                )

                if classifier_free_guidance != 1:
                    v_uncond = model[key](
                            current, 
                            timesteps=sigma_i, 
                            mulan_context=positive_mulan_context, 
                            semantic_context=semantic_context, 
                            mulan_force_cfg=1,
                            semantic_force_cfg=1,
                        )
                    v_negative = v_uncond
                    if negative_mulan_context is not None:
                        v_negative = model[key](
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
        condition_signal: list = ['semantic', 'mulan'],
        **kwargs
    ) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Sample start
        return self.sample_loop(
            model=model,
            positive_mulan_context=positive_mulan_context,
            negative_mulan_context=negative_mulan_context,
            semantic_context=semantic_context,
            current=torch.randn(b, c, t, device=self.device,),
            num_steps=num_steps, 
            bf16_portion=bf16_portion,
            chunk_index=-1, 
            condition_signal=condition_signal,
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
        condition_signal: list = ['semantic', 'mulan'],
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
            semantic_context=semantic_context,
            condition_signal=condition_signal,
        )
        # Return start if only num_splits chunks
        if num_chunks <= self.num_splits:
            return start[..., :self.num_chunks * self.split_length]

class InferenceModule(BaseModule):
    def __init__(
        self,
        required_modules,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        if self.extra_params.get("rl_finetuned_semantic", False):
            print(f"Loading RL-finetuned semantic decoder: {self.extra_params.semantic_ckpt}")
            self.semantic_module = SemanticSequenceTrainingModule.load_from_checkpoint(
                self.extra_params.semantic_ckpt
            ).eval()
        else:
            print(f"Loading semantic decoder: {self.extra_params.semantic_ckpt}")
            self.semantic_module = SemanticModule.load_from_checkpoint(
                self.extra_params.semantic_ckpt
            ).eval()
        self.requires = {}
        self.load_required_modules()

    def on_predict_start(self):
        self.load_diffusion_model()
        self.load_vocoder()

    def setup(self, stage: str) -> None:
        self.load_required_modules()

    def load_diffusion_model(self):
        diffusion_model_path = {}
        diffusion_model = {}
        for key in self.extra_params:
            if 'diffusion_ckpt' in key: 
                # get path tem 
                local_path = os.path.join(self.extra_params['model_dir'], key)
                diffusion_model_path[key.split('ckpt_')[-1]] = os.path.join(local_path, os.path.basename(self.extra_params[key]))

                if self.trainer.local_rank == 0:
                    if not os.path.exists(os.path.join(local_path, os.path.basename(self.extra_params[key]))):
                        print(f'Downloading {self.extra_params[key]}')
                        os.makedirs(local_path, exist_ok=True)
                        if '/home/' in self.extra_params[key]:
                            os.system(f'hdfs dfs -get {self.extra_params[key]} {local_path}')
                        elif '/mnt/' in self.extra_params[key]:
                            os.system(f'cp {self.extra_params[key]} {local_path}')
                self.trainer.strategy.barrier()

        for key in diffusion_model_path:
            diffusion_model[key] = load_ema_checkpoint(
                diffusion_model_path[key],
                TNTDiffusionNetwork(
                    input_dim=32,
                    feature_dim=1024,
                    context_dim=1,
                    depth=16,
                    segment_size=32,
                    segment_stride=32,
                    dropout=0,
                    mulan_cfg_prob=0.10,
                    semantic_cfg_prob=0.10,
                    use_checkpoint=False
                ),
            )
            diffusion_model[key].eval()
            diffusion_model[key].to(self.device)

        self.diffusion_model = diffusion_model

        sampler = ARVSampler(32, 1250, 1)
        sampler.set_device(self.device)

        self.sampler = sampler

    def load_vocoder(self):
        # TODO: need ema load func in next version
        self.vocoder_model = init_vocoder(
            trainer=self.trainer,
            path=self.extra_params['vocoder_ckpt'],
            device=self.device,
            cache_dir=self.extra_params['model_dir'],
        )

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))
            if name == "mulan_centers":
                assert self.extra_params.mulan_num_rvq == self.requires["mulan_centers"].shape[0]
    
    def _predict_step(self, batch, round):
        text_embs = []
        prompts = batch["text"]
        categories = batch["category"]
        bs = len(prompts)

        for text in batch["text"]:
            text_emb = self.requires["mulan_infer_fn"](
                self.requires["mulan"], text=text, device=f"cuda:{self.local_rank}"
            )
            text_embs.append(text_emb)

        mulan_embeds = torch.cat(text_embs, dim=0)
        mulan_ids, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        
        semantic_samples = self.semantic_module.predict(mulan_ids, self.extra_params)
        # to wav with diffusion
        pred_emb = self.sampler(
                model=self.diffusion_model,
                positive_mulan_context=mulan_ids,
                negative_mulan_context=None,
                semantic_context=semantic_samples,
                num_items=semantic_samples.shape[0],
                num_chunks=1,
                num_steps=self.extra_params['diffusion_steps'],
                bf16_portion=self.extra_params['bf16_portion'],
                start=None,
                show_progress=True,
                angle_schedule='linear',
                schdeule_slope=self.extra_params['schedule_slope'],
                classifier_free_guidance=self.extra_params['guidance_scale'],
                condition_signal=self.extra_params['diffusion_condition'],
        ).detach()
        wavs = self.vocoder_model["model"].decode(pred_emb.float()).detach()


        if self.extra_params.save_cosine_similarity:
            mulan_audio_embeds = self.requires["mulan_infer_fn"](model=self.requires["mulan"], music=wavs.float(), device=wavs.device)
            cs = torch.nn.functional.cosine_similarity(mulan_embeds, mulan_audio_embeds, dim=1)

        for i, wav in enumerate(wavs):
            wav_dir = os.path.join(self.extra_params.output_dir, categories[i])
            os.makedirs(wav_dir, exist_ok=True)
            fp = os.path.join(wav_dir, f"{format_name(prompts[i])}.{round}")
            print(f"[Saving] {fp}")
            if self.extra_params.save_cosine_similarity:
                fp = fp + f".cs{cs[i]:.4f}"
            if self.extra_params.save_semantic_samples:
                torch.save(semantic_samples[i].cpu(), f"{fp}.pt")

            torchaudio.save(
                f"{fp}.wav",
                wav.cpu(),
                24000,
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.extra_params.num_rounds):
            self._predict_step(batch, i)

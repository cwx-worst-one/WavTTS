from recipes.umm2.models.base import BaseStage, SpanModel, PipelineModel
from recipes.umm2.models.umm_fm import \
    ConformerEncoderLayer, \
    Conv2dSubsampling, \
    Conv2dUpsampling, \
    ConformerRotaryPositionalEmbedding, \
    UMM
import math
from recipes.umm2.models.tasks.autoencoder import AE_UMM
from recipes.umm2.models.tasks.rvq_module import EMAVectorQuantizerRPSimple
from recipes.umm2.models.tasks.discriminator import MultiFreqDiscriminator
import torch
from recipes.umm2.modules.metric.common import STFTLoss
from recipes.umm2.models.umm_dual import AE_UMM_Dual, AudioEncoder, DualEncoder, Spec_Head

from dataclasses import dataclass
from typing import Dict, List, Union, Callable, Optional
from functools import partial
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DualTokenizerConfig:
    sample_rate: int = 32000
    encoder_block: int = 6
    decoder_block: int = 6
    rvq: int = 2
    win: int = 2048
    stride: int = 441 # 100Hz for 44100Hz
    emb_dim: int = 1024
    group: int = 5
    num_bins: int = 1025
    vq_codebook_size: int = 32768
    vq_codebook_dim: int = 32
    vq_decay: float = 0.99
    vq_stale_count: int = 100
    vq_type: str = 'RP_simple'
    frame_rate: int = 25


class MixVectorQuantization(nn.Module):
    def __init__(self, config, dt_config, dist=True):
        super().__init__()
        # rvq proj
        self.config = config
        self.dt_config = dt_config
        self.vq_proj_in = nn.Linear(
                config.hidden_size, dt_config.vq_codebook_dim, bias=False
            )
        self.vq_proj_out = nn.Linear(
            dt_config.vq_codebook_dim * 2, config.hidden_size, bias=False
        )
        
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        # rvq
        self.rvq = dt_config.rvq
        self.eps = 1e-5
        self.RVQ = nn.ModuleList([])
        self.stale_tolerance = dt_config.vq_stale_count
        for r in range(self.rvq):
            if dt_config.vq_type != 'RP_simple':
                raise NotImplementedError
            self.RVQ.append(
                EMAVectorQuantizerRPSimple(r=r,    
                                           stale_tolerance=self.stale_tolerance,
                                            codebook_size=dt_config.vq_codebook_size, 
                                            codebook_dim=dt_config.vq_codebook_dim,
                                            same_index_shape=True, 
                                            decay=dt_config.vq_decay, 
                                            dist=dist)
                                        )

    def forward(self, enc_out_dict, dual_enc_out_dict, inference_R=None):
        hidden_states = dual_enc_out_dict['latent']
        hidden_states = self.vq_proj_in(hidden_states)
        noise_scale = 0
        if self.config.get("vq_proj_noise", 0) > 0:
            noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(0) / self.config.vq_proj_noise
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            self.cnt.add_(1)

        vq_output_dict = self.forward_vq(hidden_states)
        vq_emb = vq_output_dict["vq_embs"]
        R0_emb = enc_out_dict["vq_embs"][..., 0]

        dropout_rvq_embs = vq_emb[..., -1]
        if self.training:
            if torch.rand(1) < 0.5:
                batch_size = vq_emb.shape[0]
                # here, quantizer dropout is applied to each [sample]
                quantizer_dropout = torch.randint(vq_emb.shape[-1], size=(batch_size,))
                # TODO: dropout of token layers should be sin curve instead uniform
                dropout_rvq_embs = vq_emb[range(batch_size), ..., quantizer_dropout]
        else:
            if inference_R is not None:
                dropout_rvq_embs = vq_emb[..., inference_R]
                
        concat_emb = torch.cat((R0_emb, dropout_rvq_embs), dim=-1)
        hidden_states = self.vq_proj_out(concat_emb)

        output_dict = {
            "latent": hidden_states,
            "vq_ids": vq_output_dict["ids"],
            "loss": vq_output_dict["loss"],
            "noise_scale": noise_scale,
        } 
        if "entropy" in vq_output_dict.keys():
            output_dict["vq_entropy"] = vq_output_dict["entropy"]
        if "ppl" in vq_output_dict.keys():
            output_dict["ppl"] = vq_output_dict["ppl"]

        return output_dict
    

    def forward_vq(self, z): #, enc_out_dict):
        z_unitnorm = z * torch.rsqrt(z.pow(2).sum(-1, keepdim=True) + self.eps) # [B, T, D]
        
        residual = z_unitnorm
        quantized, indices, rvq_loss, ppl = [], [], [], []
        for r in range(len(self.RVQ)):
            this_z_q, this_indices, this_vq_loss, this_ppl = self.RVQ[r](residual)
            residual = residual - this_z_q
            if r == 0:
                quantized.append(this_z_q)
            else:
                quantized.append(quantized[-1] + this_z_q)
            indices.append(this_indices)
            rvq_loss.append(this_vq_loss)
            ppl.append(this_ppl)
            # print(r, this_vq_loss)
        
        quantized = torch.stack(quantized, -1)
        indices = torch.stack(indices, -1)
        rvq_loss = torch.stack(rvq_loss, -1)
        ppl = torch.stack(ppl, -1)
        # straight-through estimator (have been called in each VQ)
        # quantized = (quantized - z_unitnorm.unsqueeze(-1)).detach() + z_unitnorm.unsqueeze(-1)

        output_dict = {
            "vq_embs": quantized,
            "ids": indices,
            "loss": rvq_loss,
            "ppl": ppl,
        }
        return output_dict


class DualTokenizerModel(PipelineModel):
    def __init__(
        self,
        stages: List[Union[BaseStage, SpanModel, Callable]],
        input_names: List[str],
        output_names: List[str],
    ):
        super(DualTokenizerModel, self).__init__(stages, input_names, output_names)


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_msr_wav(self, batch):
        def get_nframe(audio_len, nhop):
            return math.ceil(audio_len / nhop)
        
        sample_rate1, sample_rate2 = self.stages[0].config.sample_rate, self.stages[1].sample_rate
        audio_key1, audio_key2 = "audio", 'audio_'+str(sample_rate2)
        wav1, wav2 = batch[audio_key1], batch[audio_key2]
        if wav1.dim() == 3:
            wav1 = wav1.squeeze(dim=1)
        if wav2.dim() == 3:
            wav2 = wav2.squeeze(dim=1)

        frame_rate1 = self.stages[0].config.frame_rate
        rate1 = int(sample_rate1 / frame_rate1)

        hop_length1, hop_length2 = self.stages[0].config.hop_length, self.stages[1].dt_config.stride
        if wav1.size(-1) % rate1 > 0:
            wav1 = F.pad(wav1, (0, rate1 - (wav1.size(-1) % rate1)), "constant", 0)
            
        frame1 = get_nframe(wav1.size(-1), hop_length1) 
        wav2_len = frame1 * hop_length2
        if wav2_len > wav2.size(-1):
            wav2 = F.pad(wav2, (0, wav2_len - wav2.size(-1)), "constant", 0)
        else:
            wav2 = wav2[..., :wav2_len]
        frame2 = get_nframe(wav2.size(-1), hop_length2) 

        if frame1 != frame2:
            wav1 = F.pad(wav1, (0, int(30*sample_rate1) - wav1.size(-1)), "constant", 0)
            wav2 = F.pad(wav2, (0, int(30*sample_rate2) - wav2.size(-1)), "constant", 0)
        
        batch[audio_key1] = wav1
        batch[audio_key2] = wav2
        return batch

    # @torch.no_grad()
    # @torch.cuda.amp.autocast(enabled=False)
    # def wav2token(self, batch):
    #     r"""Perform forward computation.
    #     This method will call :func:`forward <BaseStage.forward>` function of
    #     :attr:`stages <self.stages>` sequentially.
    #     Args:
    #         batch (List[Tensor]): input batch which contains
    #          ``len(input_names)`` tensors.
    #     """
    #     stage1, stage4 = self.stages

        
    #     stage1_enc_out_dict = stage1.wav2token(batch["audio"])
    #     stage4_enc_out_dict = stage4.encoder(batch)

    #     # if stage1_enc_out_dict["latent"].shape[1] != stage4_enc_out_dict["latent"].shape[1]:
    #     #     import pdb; pdb.set_trace()

    #     # dict_keys(['latent', 'vq_ids', 'loss', 'noise_scale', 'ppl', 'position_embeddings'])
    #     quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict)



    def forward(self, batch):
        r"""Perform forward computation.

        This method will call :func:`forward <BaseStage.forward>` function of
        :attr:`stages <self.stages>` sequentially.

        Args:
            batch (List[Tensor]): input batch which contains
             ``len(input_names)`` tensors.

        """
        stage1, stage4 = self.stages
        
        """
        Return:
            - latent: (will not be used)
            - ppl, loss_rvq, loss:  (will not be used)
            - vq_ids: [B, 25T, R]
            - vq_embs: [B, 25T, H, R]
            - mel: mel spec of 24k audio [B, 100T, 128]
            - audio: padded audio [B, 44100T]
        """

        batch = self.prepare_msr_wav(batch)
        stage1_enc_out_dict = stage1.wav2token(batch["audio"])
        
        # dict_keys(['audio_44100', 'spec', 'latent'])
        stage4_enc_out_dict = stage4.encoder(batch)

        # if stage1_enc_out_dict["latent"].shape[1] != stage4_enc_out_dict["latent"].shape[1]:
        #     import pdb; pdb.set_trace()

        # dict_keys(['latent', 'vq_ids', 'loss', 'noise_scale', 'ppl', 'position_embeddings'])
        inference_R = None
        if 'inference_R' in batch:
            inference_R = batch['inference_R']
        quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict, inference_R)
        quant_out_dict.update({
            "position_embeddings": stage1_enc_out_dict["position_embeddings"],
        })
        # dict_keys(['latent'])
        dec_out_dict = stage1.decoder(quant_out_dict)
        # dict_keys(['spec_out'])
        output_dict = stage4.spec_head(dec_out_dict)

        # include task-wise losses for display purpose
        quant_out_dict.update(output_dict)

        sample_rate2 = stage4.sample_rate
        audio_key2 = 'audio_'+str(sample_rate2)

        audio_recon = stage4.wave_decode(quant_out_dict["spec_out"], nsample=batch[audio_key2].shape[-1])
        quant_out_dict.update({
            'audio_recon_'+str(sample_rate2): audio_recon,
            'spec': stage4_enc_out_dict["spec"]})
        
        return quant_out_dict
    

if __name__ == "__main__":
    import torch
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig(
        vq_stale_count=100,
        vq_type='RP_simple',
        vq_decay=0.99,
        rvq=2,
        hidden_size=1024,
        vq_codebook_dim=32,
        vq_codebook_size=32768
        )

    dt_config = DualTokenizerConfig(
        sample_rate=32000,
        stride=320,
    )
    
    dummy_input = {
        "audio": torch.zeros([1, 24000 * 30]), 
        "audio_32000": torch.zeros([1, 32000 * 30]),
    }
    quantization = MixVectorQuantization(config, dt_config, dist=False)
    spec_head = Spec_Head(config, dt_config)
    stage4 = DualEncoder(config, dt_config, quantization, spec_head)

    stage1_enc_out_dict = {
        'vq_embs': torch.rand(1, 30*25, 32, 2),
        'vq_ids': torch.randint(0, config.vq_codebook_size, (1, 30*25, 2))
    }
    stage4_enc_out_dict = stage4.encoder(dummy_input)
    quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict)
    print(quant_out_dict.keys(), stage4_enc_out_dict.keys())
    stage4_enc_out_dict.update(quant_out_dict)
    import pdb; pdb.set_trace()
    out_dict = stage4.spec_head(stage4_enc_out_dict)
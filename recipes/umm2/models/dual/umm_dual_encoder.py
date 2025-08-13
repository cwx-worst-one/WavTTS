from recipes.umm2.models.base import BaseStage, SpanModel, PipelineModel
from recipes.umm2.models.umm_fm import \
    ConformerEncoderLayer, \
    Conv2dSubsampling, \
    Conv2dUpsampling, \
    ConformerRotaryPositionalEmbedding, \
    UMM, \
    AudioEncoder
import math
import inspect
from recipes.umm2.models.tasks.autoencoder import AE_UMM
from recipes.umm2.models.tasks.rvq_module import EMAVectorQuantizerRPSimple, EMAVectorQuantizerEntropy
from recipes.umm2.models.tasks.discriminator import MultiFreqDiscriminator
import torch
from recipes.umm2.modules.metric.common import STFTLoss
from recipes.umm2.models.umm_dual import Spec_Head


from dataclasses import dataclass
from typing import Dict, List, Union, Callable, Optional
from functools import partial
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DualTokenizerConfig:
    sample_rate: int = 24000
    encoder_block: int = 6
    decoder_block: int = 12
    rvq: int = 4
    win: int = 2048
    stride: int = 240 # 100Hz for 44100Hz
    emb_dim: int = 1024
    group: int = 1
    num_bins: int = 1025
    vq_codebook_size: int = 32768
    vq_codebook_dim: int = 32
    vq_decay: float = 0.99
    vq_stale_count: int = 100
    vq_type: str = 'VQEntropy'
    frame_rate: int = 25
    vq_proj_noise: int = 10000


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
            dt_config.vq_codebook_dim, config.hidden_size, bias=False
        )
        
        if dt_config.vq_proj_noise > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        # rvq
        self.rvq = dt_config.rvq
        self.eps = 1e-5
        self.RVQ = nn.ModuleList([])
        self.stale_tolerance = dt_config.vq_stale_count
        for r in range(self.rvq):
            if dt_config.vq_type == 'RP_simple':
                self.RVQ.append(EMAVectorQuantizerRPSimple(r=r,    
                                stale_tolerance=self.stale_tolerance,
                                codebook_size=dt_config.vq_codebook_size, 
                                codebook_dim=dt_config.vq_codebook_dim,
                                same_index_shape=True, 
                                decay=dt_config.vq_decay, 
                                dist=dist)
                            )
            elif dt_config.vq_type == 'VQEntropy':
                self.RVQ.append(EMAVectorQuantizerEntropy(
                                codebook_size=dt_config.vq_codebook_size,
                                codebook_dim=dt_config.vq_codebook_dim,
                                same_index_shape=True,
                                decay=dt_config.vq_decay,
                                dist=dist,)
                            )

    def forward(self, enc_out_dict, dual_enc_out_dict, inference_R=None):
        """
        enc_out_dict: 
            - vq_embs: [B, T, h]
            - prevq_embs: [B, T, h]
            - vq_ids: [B, T]
        dual_enc_out_dict:
            - latent: [B, T, H]
        """
        hidden_states = dual_enc_out_dict['latent'] # [B, T, H]
        hidden_states = self.vq_proj_in(hidden_states)  # [B, T, h]
        noise_scale = 0

        if self.dt_config.vq_proj_noise > 0 and self.training:
            noise_scale = (self.dt_config.vq_proj_noise - self.cnt).clamp(0) / self.dt_config.vq_proj_noise
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            self.cnt.add_(1)

        if "e_scale" in inspect.getfullargspec(self.RVQ[0].forward).args:   # with entropy loss
            e_scale = 0.
            if hasattr(self, "cnt") and self.cnt < 30000 and self.training:
                e_scale = 1.0
        else:
            e_scale = None


        vq_output_dict = self.forward_vq(hidden_states, enc_out_dict, e_scale=e_scale)
        vq_emb = vq_output_dict["vq_embs"]  # [B, T, h, R]
        vq_ids = vq_output_dict["vq_ids"]
        dropout_rvq_embs = vq_emb[..., -1]
        if self.training:
            if torch.rand(1) < 0.2:
                batch_size = vq_emb.shape[0]
                # here, quantizer dropout is applied to each [sample]
                quantizer_dropout = torch.randint(vq_emb.shape[-1], size=(batch_size,))
                # TODO: dropout of token layers should be sin curve instead uniform
                dropout_rvq_embs = vq_emb[range(batch_size), ..., quantizer_dropout]
        else:
            if inference_R is not None:
                R_idx = min(max(inference_R-1, 0), vq_emb.shape[-1]-1)
                # print("R_idx", R_idx)
                dropout_rvq_embs = vq_emb[..., R_idx]
                vq_ids = vq_ids[..., :R_idx+1]
                
        hidden_states = self.vq_proj_out(dropout_rvq_embs)

        output_dict = {
            "latent": hidden_states,
            "vq_ids": vq_ids,
            "vq_embs": vq_emb,
            "loss": vq_output_dict["loss"],
            "noise_scale": noise_scale,
        } 
        if "entropy" in vq_output_dict.keys():
            output_dict["vq_entropy"] = vq_output_dict["entropy"]
        if "ppl" in vq_output_dict.keys():
            output_dict["ppl"] = vq_output_dict["ppl"]

        return output_dict
    

    def forward_vq(self, z, enc_out_dict, e_scale=None):
        z_1 = enc_out_dict['prevq_embs']
        z_q1, id_R1, loss_R1, entropy_R1 = enc_out_dict['vq_embs'], enc_out_dict['vq_ids'], \
            enc_out_dict["loss_vq"], enc_out_dict["vq_entropy"]

        # according to implementation by Li Tang, the residual is defined between:
        # 1) prevq_embs (output of vq_proj_in / z) of the dual encoder 
        # 2) postvq_embs (quantized prevq_embs / z_1) of the encoder
        residual = z - z_q1
        # z_unitnorm = z * torch.rsqrt(z.pow(2).sum(-1, keepdim=True) + self.eps) # [B, T, D]
        quantized, indices, rvq_loss, entropy = [z_q1], [id_R1], [loss_R1], [entropy_R1]
        for r in range(len(self.RVQ)):
            if e_scale is not None:
                r_output_dict = self.RVQ[r](residual, e_scale)
                this_z_q, this_indices, this_vq_loss, this_entropy = \
                    r_output_dict["embs"], r_output_dict["ids"], r_output_dict["loss"], r_output_dict["entropy"]
                entropy.append(this_entropy)
            else:
                this_z_q, this_indices, this_vq_loss, this_ppl = self.RVQ[r](residual)
                entropy.append(this_ppl)
            
            residual = residual - this_z_q
            # if r == 0:
            #     quantized.append(this_z_q)
            # else:
            quantized.append(quantized[-1] + this_z_q)
            indices.append(this_indices)
            rvq_loss.append(this_vq_loss)

        # straight-through estimator (have been called in each VQ)
        # quantized = (quantized - z_unitnorm.unsqueeze(-1)).detach() + z_unitnorm.unsqueeze(-1)

        output_dict = {
            "vq_embs": torch.stack(quantized, -1),
            "vq_ids": torch.stack(indices, -1),
            "loss": torch.stack(rvq_loss, -1),
            "entropy": torch.stack(entropy, -1),
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
        # manually set stage1 encoder & VQ params as non-trainable
        stage1 = self.stages[0]
        stage1.audio_encoder.requires_grad_(False)
        stage1.embed_positions.requires_grad_(False)
        # stage1.encoder_input_dropout.requires_grad(False)
        stage1.audio_transform.requires_grad_(False)
        stage1.encoder_layers[:stage1.insert_layer_nums[0]].requires_grad_(False)
        stage1.insert_modules.requires_grad_(False)
        stage1.insert_modules[0].vq.embedding.update = False

    def forward(self, batch):
        r"""Perform forward computation.

        This method will call :func:`forward <BaseStage.forward>` function of
        :attr:`stages <self.stages>` sequentially.

        Args:
            batch (List[Tensor]): input batch which contains
             ``len(input_names)`` tensors.

        """
        stage1, stage2, stage4 = self.stages
        
        """
        Return:
            - latent: (will not be used)
            - ppl, loss_rvq, loss:  (will not be used)
            - vq_ids: [B, 25*frames, R]
            - vq_embs: [B, 25*frames, 32, R]
            - loss_vq / loss: [R]
            - prevq_embs: [B, 25*frames, 32]
            - prevq_latent: [B, 25*frames, 1024]
            - mel: mel spec of 24k audio [B, 100*frames, 128]
            - audio: padded audio [B, 24000*secs]
        """

        # with torch.no_grad():
        # 'loss_vq', 'latent', 'loss', 'vq_entropy', 'vq_ids', 'prevq_embs', 'vq_embs', 'audio', 'mel', 'prevq_latent', 'position_embeddings'
        stage1_enc_out_dict = stage1.wav2token(batch["audio"])
        
        # "latent"
        stage4_enc_out_dict = stage4.encoder(stage1_enc_out_dict)
        # dict_keys(['latent', 'vq_ids', 'loss', 'noise_scale', 'ppl', 'position_embeddings'])
        inference_R = batch['inference_R'] if "inference_R" in batch else None
        quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict, inference_R)
        quant_out_dict.update({
            "position_embeddings": stage1_enc_out_dict["position_embeddings"],
        })

        # dict_keys(['latent'])
        dec_out_dict = stage1.decoder(quant_out_dict)
        # dict_keys(['spec_out'])
        # output_dict = stage4.spec_head(dec_out_dict)
        dec_out_dict.update({
            "mel": stage1_enc_out_dict["mel"],
            "audio": stage1_enc_out_dict["audio"],
            "token": batch["token"], 
        })

        output_dict = stage2(dec_out_dict)

        # include task-wise losses for display purpose
        model_output_dict = {
            # loss
            "loss": output_dict["loss"] + quant_out_dict["loss"].sum(),
            "loss_rvq": quant_out_dict["loss"],
            "loss_mel": output_dict["loss_mel"],
            "loss_ctc": output_dict["loss_ctc"],
            "loss_chroma": output_dict["loss_chroma"],
            "loss_f0_vuv": output_dict["loss_f0_vuv"],
            # rvq
            "vq_entropy": quant_out_dict["vq_entropy"],
            "vq_ids": quant_out_dict["vq_ids"],
            "noise_scale": quant_out_dict["noise_scale"],
            # output
            "mel": stage1_enc_out_dict["mel"],
            "audio": stage1_enc_out_dict["audio"],
            "flops": output_dict["flops"],
            "text_ids": output_dict["text_ids"],
            "ctc_out": output_dict["ctc_out"],
        }

        return model_output_dict
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav, inference_R=None):
        stage1, stage2, stage4 = self.stages
        stage1_enc_out_dict = stage1.wav2token(wav)
        stage4_enc_out_dict = stage4.encoder(stage1_enc_out_dict)
        # dict_keys(['latent', 'vq_ids', 'loss', 'noise_scale', 'ppl', 'position_embeddings'])
        quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict, inference_R)
        return quant_out_dict

class DualEncoder(UMM):
    def __init__(self, 
                 config, 
                 dt_config: DualTokenizerConfig, 
                quantization: MixVectorQuantization,
                spec_head: Optional[Spec_Head] = None,
                ):

        nn.Module.__init__(self)

        self.config = config
        self.dt_config = dt_config
        self.num_bins = self.dt_config.win // 2 + 1
        self.sample_rate = dt_config.sample_rate
        self.quantization = quantization
        if isinstance(spec_head, partial):
            spec_head = spec_head()
        self.spec_head = spec_head
        self.eps = 1e-5

        self.audio_encoder = AudioEncoder(config)   # without spectrogram input
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config, layer_idx) for layer_idx in range(dt_config.encoder_block)]
        )

        
    def encoder(self, input_dict):
        output_dict = {}
        feature = input_dict['mel']

        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        # only support one insert layer for now
        for i in range(len(self.encoder_layers)):
            hidden_states = self.encoder_layers[i](
                hidden_states, position_embeddings=position_embeddings
            )
        output_dict = {
            "latent": hidden_states,    # [B, 25hz*T, hidden-dim]
        }
        return output_dict



class AE_UMM_Dual(AE_UMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def decoder(self, input_dict):
        # input_dict: latent, position_embeddings
        position_embeddings = input_dict["position_embeddings"]
        hidden_states = input_dict["latent"]
        for i in range(self.insert_layer_nums[0], len(self.encoder_layers)):
            hidden_states = self.encoder_layers[i](
                hidden_states, position_embeddings=position_embeddings
            )
        output_dict = {
            "latent": hidden_states,
        }
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        wav = self.prepare_wav(wav)
        input_dict = self.preprocessing(wav)
        output_dict = {}
        feature = input_dict['mel']

        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.encoder_layers):
            if self.insert_layer_nums is not None and i in self.insert_layer_nums:
                insert_module = self.insert_modules[self.insert_layer_nums.index(i)]
                insert_input_dict = {'latent': hidden_states}
                insert_output_dict = insert_module(insert_input_dict)
                output_dict.update(insert_output_dict)
                if i == self.insert_layer_nums[-1]:
                    break
            
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        output_dict.update({
            "audio": wav,
            "mel": input_dict["mel"],
            "prevq_latent": hidden_states,    # pre-vq latents
            "position_embeddings": position_embeddings,
        })
        return output_dict
    

if __name__ == "__main__":
    import torch
    from recipes.umm2.models.config import UMMConfig
    from recipes.umm2.models.tasks.vq_module import VQ, EMAVectorQuantizerEntropy
    config = UMMConfig(
        vq_stale_count=100,
        vq_type='VQEntropy',
        vq_decay=0.99,
        rvq=4,
        hidden_size=1024,
        vq_codebook_dim=32,
        vq_codebook_size=32768,
        vq_proj_noise=0,
        )

    dt_config = DualTokenizerConfig(
        sample_rate=24000,
        stride=240,
    )
    
    dummy_input = {
        "audio": torch.zeros([1, 24000 * 30]), 
    }
    quantization = MixVectorQuantization(config, dt_config, dist=False)
    # spec_head = Spec_Head(config, dt_config)
    stage4 = DualEncoder(config, dt_config, quantization, spec_head=None)
    stage1 = AE_UMM_Dual(config, takes=['audio'], 
                         provides=['mel', 'audio', 'prevq_latent', 'latent', 'vq_ids', 'vq_entropy', 'vq_embs', 'loss_vq', "prevq_embs"],
                         insert_modules={12: VQ(config, task="vq", provides=['vq_ids', 'vq_embs', 'loss_vq', "vq_embs", "prevq_embs"],
                                                vq_scheme=EMAVectorQuantizerEntropy(config.vq_codebook_size, 
                                                                                    config.vq_codebook_dim,dist=False),)}
                         )

    # stage1_enc_out_dict = stage1._compute_encoder(dummy_input)
    stage1_enc_out_dict = stage1.wav2token(dummy_input["audio"])
    # stage1_enc_out_dict.update({
    #     'vq_embs': torch.rand(1, 30*25, 32, 1),
    #     'vq_ids': torch.randint(0, config.vq_codebook_size, (1, 30*25, 1))
    # })

    stage4_enc_out_dict = stage4.encoder(stage1_enc_out_dict)
    quant_out_dict = stage4.quantization(stage1_enc_out_dict, stage4_enc_out_dict)
    print(quant_out_dict.keys(), stage4_enc_out_dict.keys())
    stage1_enc_out_dict.update(quant_out_dict)
    import pdb; pdb.set_trace()
    out_dict = stage1.decoder(stage1_enc_out_dict)

    # out_dict = stage4.spec_head(stage4_enc_out_dict)
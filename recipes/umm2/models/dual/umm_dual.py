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

from dataclasses import dataclass
from typing import Dict, List, Union, Callable, Optional
from functools import partial
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DualTokenizerConfig:
    sample_rate: int = 44100
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



class AE_UMM_Dual(AE_UMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @torch.no_grad()
    def _compute_encoder(self, batch):
        input_dict = self.get_feature(batch)  
        output_dict = {}
        insert_output_dict = {}
        feature = input_dict['mel']

        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        assert self.insert_layer_nums is not None

        # only support one insert layer for now
        for i in range(self.insert_layer_nums[0]):
            hidden_states = self.encoder_layers[i](
                hidden_states, position_embeddings=position_embeddings
            )
        # TODO: verify if is_training works
        insert_module = self.insert_modules[0]
        insert_input_dict = {'latent': hidden_states}
        insert_output_dict = insert_module(insert_input_dict)
        output_dict.update(insert_output_dict)

        output_dict.update({
            "position_embeddings": position_embeddings,
            "audio": batch['audio'],
            "mel": input_dict["mel"],
            "latent": hidden_states,
        })
        return output_dict
    
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
        # wav = prepare_wav(wav)
        input_dict = self.preprocessing(wav)
        output_dict = {}
        feature = input_dict['mel']

        print('stage1', wav.shape, feature.shape)
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
        
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        # rvq
        self.rvq = dt_config.rvq - 1
        self.eps = 1e-5
        self.RVQ = nn.ModuleList([])
        self.stale_tolerance = dt_config.vq_stale_count
        for r in range(self.rvq):
            if dt_config.vq_type != 'RP_simple':
                raise NotImplementedError
            self.RVQ.append(
                EMAVectorQuantizerRPSimple(r=r+1,    
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

        # only takes out the first layer of pretrained vq    
        vq_output_dict = self.forward_vq(hidden_states, enc_out_dict)

        vq_emb = vq_output_dict["vq_embs"]
        if self.training:
            batch_size = vq_emb.shape[0]
            # here, quantizer dropout is applied to each [sample]
            quantizer_dropout = torch.randint(vq_emb.shape[-1], size=(batch_size,))
            # TODO: dropout of token layers should be sin curve instead uniform
            dropout_rvq_embs = vq_emb[range(batch_size), ..., quantizer_dropout]
        else:
            if inference_R is not None:
                dropout_rvq_embs = vq_emb[..., inference_R]
            else:
                dropout_rvq_embs = vq_emb[..., -1]
                
        hidden_states = self.vq_proj_out(dropout_rvq_embs)

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
    

    def forward_vq(self, z, enc_out_dict):
        z_R1 = enc_out_dict['vq_embs'][..., 0]
        loss_R1 = enc_out_dict["loss_rvq"][..., 0]
        id_R1 = enc_out_dict['vq_ids'][..., 0]

        z_unitnorm = z * torch.rsqrt(z.pow(2).sum(-1, keepdim=True) + self.eps) # [B, T, D]
        
        residual = z_unitnorm - z_R1

        quantized, indices, rvq_loss, ppl = [z_R1], [id_R1], [], []
        for r in range(len(self.RVQ)):
            this_z_q, this_indices, this_vq_loss, this_ppl = self.RVQ[r](residual)
            residual = residual - this_z_q
            # if r == 0:
            #     quantized.append(this_z_q)
            # else:
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
            "loss_R1": loss_R1,
            "ppl": ppl,
        }
        return output_dict


class CustomizeConv2dSubsampling(Conv2dSubsampling):
    def __init__(
        self, input_dim, output_dim, kernel, padding1, padding2, use_bn=True, act_fn=nn.ReLU, stride=None
    ):
        nn.Module.__init__(self)
        if stride is None:
            stride = [2, 2]

        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel, stride[0], padding1),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
            nn.Conv2d(256, 256, kernel, stride[1], padding2),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.linear = nn.Linear(input_dim * int(256 / (stride[0] * stride[1])), output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, config, dt_config):
        super().__init__()
        self.config = config
        self.feature_encoder = Conv2dSubsampling(
            dt_config.emb_dim,
            config.hidden_size,
            config.feature_encoder_kernel,
            config.feature_encoder_padding,
            # [1, 2], [2, 2], # hard code
            use_bn=config.get("use_bn", True),
            stride=config.get("downsample_strides", [2, 2]),
        )
        self.conformer_layer = (
            ConformerEncoderLayer(config)
            if config.get("first_conformer", True)
            else nn.Identity()
        )

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.conformer_layer(x)
        return x


class Spec_Head(nn.Module):
    def __init__(
        self, 
        config, 
        dt_config,
        # takes=["spec", "latent"], 
        # provides=["spec", "spec_out", "loss"],
        # bypasses=[],
        # task="spec",
        # loss_weight=1.0,
        # lr_ratio=1.0,
        # is_frozen: bool = False,
        use_conv=True,
    ):
        super().__init__()

        self.config = config
        self.dt_config = dt_config
        self.use_conv = use_conv
        if use_conv:
            self.spec_head = Conv2dUpsampling(
                config.hidden_size, 
                dt_config.num_bins * 2, 
                use_bn=config.get("use_bn", True),
                stride=config.get("upsample_strides", None),
                pad=config.get("upsample_pads", None),
            )
        else:
            raise NotImplementedError
    
    # def get_metrics(self, spec, recon_spec):
    #     recon_spec = recon_spec.contiguous().float()
    #     spec = torch.cat((spec.real, spec.imag), -1).contiguous().float()
    #     spec_loss = self.spec_loss_fn.float()(recon_spec, spec)
    #     return {"loss": spec_loss["stft_loss"]}
    
    def forward(self, batch):
        # spec = batch["spec"]
        spec_out = self.spec_head(batch['latent']).transpose(-2, -1)
        # metric_dict = self.get_metrics(spec, spec_out)
        output_dict = {
            # "spec": spec,
            "spec_out": spec_out, 
        } 
        return output_dict

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

        self.input_FC = nn.Sequential(
            RMVN(self.num_bins * 3 + 1),
            nn.Linear(self.num_bins * 3 + 1, self.dt_config.emb_dim))
        self.audio_encoder = AudioEncoder(config, dt_config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config, layer_idx) for layer_idx in range(dt_config.encoder_block)]
        )

        # self.discriminator = MultiFreqDiscriminator(sr=dt_config.sample_rate, nch=1, channels=16)

    def wave_encode(self, input):
        # input shape: B, 1, nsample
        # B, F, T
        spec = torch.stft(input.float(), n_fft=self.dt_config.win, hop_length=self.dt_config.stride,
                          window=torch.hann_window(self.dt_config.win).to(input.device), 
                          return_complex=True)
        spec = spec[..., :-1]   # truncate 1 sample for stft

        # [B, 1, T]
        this_power = (spec.abs().pow(2).mean(1, keepdim=True) + self.eps).sqrt()
        feature = torch.cat([spec.abs() / this_power,   # [B, F, T]
                            spec.real / this_power, 
                            spec.imag / this_power,
                            torch.log(this_power)], 1)  # B*nch, F+1, T
        feature = self.input_FC(feature.transpose(-2, -1))    # [B*nch, T, N]
        input_dict = {"spec_feature": feature,  # [B, T, N]
                     'spec': spec}
        return input_dict

    def wave_decode(self, spec_out, nsample=None):
        B, _, T = spec_out.shape
        spec_out = spec_out.reshape(B, self.num_bins, 2, T).float()
        spec_out = torch.complex(spec_out[:, :, 0], spec_out[:, :, 1])
        wav_out = torch.istft(spec_out, n_fft=self.dt_config.win, hop_length=self.dt_config.stride, 
                               window=torch.hann_window(self.dt_config.win).to(spec_out.device),
                               length=nsample)
        return wav_out
        
    def encoder(self, batch):
        audio_key = 'audio_'+str(self.sample_rate)
        # with torch.cuda.amp.autocast(enabled=False) and torch.no_grad():
            # wav = batch[audio_key].squeeze(dim=1).float()
            # wav = self.pad_audio(wav)
        if audio_key not in batch:
            audio_key = 'audio'
        wav = batch[audio_key].float()
        input_dict = self.wave_encode(wav)    
        feature = input_dict['spec_feature']    # [B, T, N]
        print('stage4', wav.shape, feature.shape)
        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.encoder_layers):
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        output_dict = {
            audio_key: batch[audio_key],
            "spec": input_dict["spec"], # [B, 100hz*T, 1025] complex
            "latent": hidden_states,    # [B, 25hz*T, hidden-dim]
        }
        return output_dict

class Encoder(nn.Module):
    def __init__(self, config, num_token, nblock):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(num_token, config.hidden_size))
        self.bert = nn.ModuleList([
            ConformerEncoderLayer(config, layer_idx) for layer_idx in range(nblock)])

    def forward(self, x):
        # x shape: B (batch_size), T (frame), G (group), N (dim)
        B = x.shape[0]
        x = rearrange(x, 'b t g n -> (b t) g n')
        cls_token = self.cls_token.unsqueeze(0).repeat(x.shape[0], 1, 1)  # B*T, L, N
        x = torch.cat([cls_token, x], -2)  # B*T, L+G, N
        x, _ = self.bert(x, mode='noncausal')
        cls_token = x[:,:cls_token.shape[-2]]  # B*T, L, N
        cls_token = rearrange(cls_token, '(b t) l n -> b t l n', b=B)
        return cls_token

class Decoder(nn.Module):
    def __init__(self, config, num_token, nblock):
        super().__init__()

        self.pred_token = nn.Parameter(torch.randn(num_token, config.hidden_size))
        self.bert = nn.ModuleList([
            ConformerEncoderLayer(config, layer_idx) for layer_idx in range(nblock)])
    
    def forward(self, x):
        # x shape: B, T, C, G, N
        B, C = x.shape[0], x.shape[2]
        x = rearrange(x, 'b t c g n -> (b t) c g n')
        pred_token = self.pred_token.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # B*T, C, L, N
        x = torch.cat([x, pred_token], -2)  # B*T, C, G+L, N
        x = rearrange(x, 'b c g n -> (b c) g n')
        x, _ = self.bert(x, mode='noncausal')
        pred_token = x[:,-pred_token.shape[-2]:]  # B*T*C, L, N
        pred_token = rearrange(pred_token, '(b t c) g n -> b t c g n', b=B, c=C)

        return pred_token


def MVN(input, groups=1, eps=1e-5):
    # input shape: (*, N)

    input_shape = input.shape
    assert input_shape[-1] % groups == 0

    input_float = input.reshape(-1, groups, input_shape[-1] // groups).float()
    input_norm = (input_float - input_float.mean(-1, keepdim=True)) * torch.rsqrt(input_float.var(-1, keepdim=True) + eps)

    return input_norm.type_as(input).reshape(input_shape)

class RMVN(nn.Module):
    def __init__(self, dim, groups=1, trainable=True):
        super().__init__()

        self.mean = nn.Parameter(torch.zeros(1, dim), requires_grad=trainable)
        self.std = nn.Parameter(torch.ones(1, dim), requires_grad=trainable)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input shape: (*, N)

        input_shape = input.shape
        assert input_shape[-1] % self.groups == 0

        input_norm = MVN(input, self.groups, self.eps)
        output = input_norm.reshape(-1, input_shape[-1]) * self.std + self.mean

        return output.reshape(input_shape)

class DualTokenizer(nn.Module):
    def __init__(
        self, config,
        dual_tokenizer_config,
    ):
        super().__init__()
        self.eps = 1e-5
        self.config = config
        self.dt_config = dual_tokenizer_config

        self.num_bins = self.dt_config.win // 2 + 1
        self.input_FC = nn.Sequential(
            RMVN(self.num_bins * 3 + 1),
            nn.Linear(self.num_bins * 3 + 1, self.dt_config.emb_dim))
        self.enc2token = nn.Sequential(
            RMVN(self.dt_config.emb_dim), 
            nn.Linear(self.dt_config.emb_dim, self.dt_config.token_dim), 
            nn.Tanh())
        
        self.encoder = Encoder(config, num_token=config.rvq-1, nblock=dual_tokenizer_config.encoder_block)
        self.decoder = Decoder(config, num_token=config.rvq-1, nblock=dual_tokenizer_config.decoder_block)


    def encode(self, input):
        B, nch, nsample = input.shape
        assert nch == 1
        feature = self.wave_encode(input)  # B, T, N
        feature = rearrange(feature, 'b (t g) n -> b t g n', g=self.dt_config.group)

        # encoding
        enc_output = self.encoder(feature)  # B, T, L, N
        feature = self.enc2token(enc_output)  # B, T, L, H
        quantized, indices = self.quantizer(feature)
        return quantized, indices, nsample

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

        frame_rate = self.stages[0].config.frame_rate
        rate1 = int(sample_rate1 / frame_rate)

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

        audio_recon_44100 = stage4.wave_decode(quant_out_dict["spec_out"], 
                                               nsample=batch["audio_44100"].shape[-1])
        quant_out_dict.update(audio_recon_44100=audio_recon_44100)
        quant_out_dict.update(spec=stage4_enc_out_dict["spec"])
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

    dt_config = DualTokenizerConfig()
    
    dummy_input = {
        "audio": torch.zeros([1, 24000 * 30]), 
        "audio_44100": torch.zeros([1, 44100 * 30]),
    }


    # stage1 = AE_UMM_Dual(config, dt_config)

    quantization = MixVectorQuantization(config, dist=False)
    spec_head = Spec_Head(config, dt_config)
    stage4 = DualEncoder(config, dt_config, quantization, spec_head)

    # stage1_enc_out_dict = stage1.wav2token(dummy_input["audio"])
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
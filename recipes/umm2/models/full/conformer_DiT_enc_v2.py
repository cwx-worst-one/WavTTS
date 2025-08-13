import torch
import torch.nn as nn
import torch.nn.functional as F
from recipes.umm2.models.tasks.dit_module import (
    LlamaDiffusion,
)
import math
import numpy as np
from recipes.umm2.models.umm_fm import (
    UMMModified,
    BaseStage,
    AudioEncoder, 
    ConformerRotaryPositionalEmbedding, 
    pad_btd_to, INPUT_SEQ_LEN_ALIGNMENT,    
)


class MelNorm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def denorm_mel(self, mel):
        return (mel * self.std) + self.mean

    def norm_mel(self, mel):
        return (mel - self.mean) / self.std
    

class Stage2DiTEnc(UMMModified):
    def __init__(self, config, 
                        dit_config, 
                        sigma=0.2,   
                        takes=[],
                        provides=[],
                        bypasses=[],
                        lr_ratio=1.0,
                        loss_weight=None,
                        is_frozen=False,):
        BaseStage.__init__(self, 
            takes,
            provides,
            bypasses,
            lr_ratio,
            loss_weight,
            is_frozen,
            config=config,
        )
        self.config = config
        self.dit_config = dit_config
        self.sigma = sigma
        self.prepare_encoder()
        self.prepare_diffusion()
        self.prepare_vq_fc()

    def prepare_encoder(self, ):
        config = self.config
        # ==== original ==== #
        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)

        self._init_audio_transform()
        
        if self.config.get("freeze_umm", True):
            self.audio_encoder.requires_grad_(False)
            self.embed_positions.requires_grad_(False)
            self.audio_transform.requires_grad_(False)
            self.encoder_layers.requires_grad_(False)
        

    def prepare_diffusion(self,):
        self.DiT = LlamaDiffusion(self.dit_config)

    def prepare_vq_fc(self, ):
        config = self.config
        self.vq_proj_in = nn.Linear(
            config.hidden_size, config.vq_codebook_dim, bias=False
        )

        if self.config.add_vq:
            if self.config.vq_type == "UQ":
                from recipes.umm2.models.tasks.uq_module import UniversalQuantizer
                self.uq_norm = nn.Tanh()
                self.uq = UniversalQuantizer(bits=int(math.log2(config.vq_codebook_size)),)
                # no need for vq_proj_out as no tokenizer decoder is defined here.
            elif self.config.vq_type == "RVQ":
                from recipes.umm2.models.tasks.rvq_module import RVQ
                from recipes.umm2.models.tasks.rvq_module import EMAResidualVectorQuantizerRP
                self.rvq = EMAResidualVectorQuantizerRP(
                    vq_type="EMARPSimple",
                    codebook_size=config.vq_codebook_size,
                    codebook_dim=config.vq_codebook_dim,
                    rvq=config.rvq, 
                    dist=False, # for debug 1gpu
                )
            else:
                raise NotImplementedError

    def prepare_inference(self, inference_config):
        self.inference_config = inference_config

        self.bn_chunk_size = self.dit_config.window_size[1]
        self.token_chunk_size = (
            self.bn_chunk_size * self.config.frame_rate // self.dit_config.bn_frame_rate
        )
        self.token_overlap = int(np.prod(self.DiT.hp.token_downscales))

        self.bn_padding = self.inference_config["bn_padding"]
        self.diffusion_nfe = self.inference_config["diffusion_nfe"]
        self.text_cfg_w = self.inference_config["text_cfg_w"]
        self.mem_efficient = self.inference_config["mem_efficient"]
        self.diffusion_sampler = self.inference_config["diffusion_sampler"]
        self.bn_norm = MelNorm(self.inference_config["bn_norm_mean"], self.inference_config["bn_norm_std"])
        self.diffusion_precision = self.inference_config["diffusion_precision"]
        self.embed_padding = self.inference_config["embed_padding"]

        self.token_embed_chunk = self.inference_config["token_embed_chunk"]
        self.token_embed_chunk_size = self.inference_config["token_embed_chunk_size"]

    
    def _compute_fused_kernel(self, batch):
        grad_context = torch.no_grad() if self.config.get("freeze_umm", True) else torch.enable_grad()

        with grad_context:
            # Extract features
            mel, mel_len = self.get_feature(batch)
            # flops = self.audio_encoder.get_flops(*mel.shape)
            
            # Encode features
            feature, feature_mask = self.audio_encoder(mel, mel_len)
            hidden_states = self.encoder_input_dropout(feature)

            seqlen = hidden_states.shape[1]
            hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
            pos_emb = self.pos_enc(hidden_states)[1]
    
            len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
            assert len_diff >= 0, \
                f"conformer_mask should be larger than feature_mask by {len_diff}"
            conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()

            # Encoder
            for layer_index, layer in enumerate(self.encoder_layers):
                hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                    use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
            
            hidden_states = hidden_states[:, 0:seqlen, :]

        # add vq
        vq_loss, vq_ids, ppl = None, None, None
        bn_hidden_states = self.vq_proj_in(hidden_states) 
        # standard normalize
        if not self.config.add_vq:
            # mel sigma VAE
            continuous_embed = (bn_hidden_states - bn_hidden_states.mean(dim=-1, keepdim=True)) / (bn_hidden_states.var(dim=-1, keepdim=True) + 1e-5).sqrt()
            continuous_embed = continuous_embed + torch.randn_like(continuous_embed) * self.sigma
        else:
            if self.config.vq_type == "UQ":
                hidden_states = self.uq_norm(bn_hidden_states)  # values between [-1, 1]
                # 25Hz, 32dim, 3bit = 4.8kbps
                quantized_embed = self.uq(hidden_states, return_indices=False)
                continuous_embed = quantized_embed
            elif self.config.vq_type == "RVQ":
                # no vq_proj_noise
                output_dict = self.rvq({"latent": bn_hidden_states, 
                                        "attn_mask": conformer_mask})   # {embs, ids, loss, ppl}
                vq_emb = output_dict["vq_embs"]    # [B, T, N, R]
                if self.training:
                    batch_size = vq_emb.shape[0]
                    if torch.rand(1) > 0.5: # all R embs
                        continuous_embed = vq_emb[..., -1]
                    else:   # quantizer dropout
                        quantizer_dropout = torch.randint(vq_emb.shape[-1], size=(batch_size,))
                        continuous_embed = vq_emb[range(batch_size), ..., quantizer_dropout]
                else:
                    continuous_embed = vq_emb[..., -1]
                vq_loss = output_dict["loss"]
                vq_ids = output_dict["vq_ids"]
                vq_ids = vq_ids[:, 0:seqlen]

        bs = continuous_embed.shape[0]
        if self.training:
            drop_idx = torch.rand(bs) < self.dit_config.cond_dropout
            if torch.sum(drop_idx) > 0:
                continuous_embed[drop_idx] = self.dit_config.cond_padding
                # print(f"umm_dropout drop_idx={torch.sum(drop_idx)}/{bs}")

        ### diffusion ###
        # continuous feature
        DiT_input = {
            'bn': batch['bn'].to(torch.bfloat16),  # [B, T*50, 128] float
            'embed': continuous_embed.to(torch.bfloat16),  # [B, T*25, 32] float
            'bn_mask': batch['bn_mask'].to(torch.bfloat16),
        }

        pred, target = self.DiT(DiT_input)

        output_dict = {
            'mel': mel, 
            'mel_len': mel_len,
            'dit_pred': pred,
            'dit_target': target,
            'vq_ids': vq_ids,
        }
        
        if vq_loss is not None:
            output_dict["loss_rvq"] = vq_loss
            output_dict["loss"] = output_dict["loss_rvq"].sum()
        if ppl is not None:
            output_dict["ppl"] = output_dict["ppl"]
        return output_dict
    
    @torch.no_grad()
    def inference_tokenizer(self, wav, wav_len=None):
        def forward(wav, wav_len):
            ####################
            ## Tokenizer part ##
            ####################
            mel, mel_len = self.get_feature({'audio': wav, 'audio_length': wav_len})  
            # audio encoder
            feature, feature_mask = self.audio_encoder(mel, mel_len)
            hidden_states = self.encoder_input_dropout(feature)
            
            seqlen = hidden_states.shape[1]
            hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
            pos_emb = self.pos_enc(hidden_states)[1]
    
            len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
            assert len_diff >= 0, \
                f"conformer_mask should be larger than feature_mask by {len_diff}"
            conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()

            # encoder layers
            for i, layer in enumerate(self.encoder_layers):
                hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                    use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
            
            hidden_states = hidden_states[:, 0:seqlen, :]

            # add vq
            bn_hidden_states = self.vq_proj_in(hidden_states) 
            # standard noramlize
            if not self.config.add_vq:
                # mel sigma VAE
                continuous_embed = (bn_hidden_states - bn_hidden_states.mean(dim=-1, keepdim=True)) / (bn_hidden_states.var(dim=-1, keepdim=True) + 1e-5).sqrt()
                continuous_embed = continuous_embed + torch.randn_like(continuous_embed) * self.sigma
            else:
                if self.config.vq_type == "UQ":
                    hidden_states = self.uq_norm(bn_hidden_states)  # values between [-1, 1]
                    # 25Hz, 32dim, 3bit = 4.8kbps
                    quantized_embed = self.uq(hidden_states, return_indices=False)
                    continuous_embed = quantized_embed
                elif self.config.vq_type == "RVQ":
                    # no vq_proj_noise
                    output_dict = self.rvq({"latent": bn_hidden_states, 
                                            "attn_mask": conformer_mask})    # {embs, ids, loss, ppl}
                    vq_emb = output_dict["embs"]    # [B, T, N, R]
                    continuous_embed = vq_emb[..., -1]
                    # vq_loss = output_dict["loss"]
                    # vq_ids = output_dict["ids"]

            return continuous_embed


        if wav_len is None:
            wav_len = torch.tensor(wav.shape[-1]).unsqueeze(0).repeat(wav.shape[0]).long().to(wav.device)

        if self.token_embed_chunk:
            st = 0
            input_wav = []
            input_wav_len = []
            n_samples = wav.shape[-1]
            chunk_size = self.token_embed_chunk_size
            while st < n_samples:
                st_sample, et_sample = int(st*self.config.sample_rate), int((st+chunk_size)*self.config.sample_rate)
                # merge the tail if the remaining chunk is too short (<5s)
                if n_samples - et_sample < self.config.sample_rate * 5 \
                or wav[..., et_sample:].shape[-1] < self.config.sample_rate * 5:
                    et_sample = n_samples
                this_input_wav = wav[..., st_sample:et_sample]
                input_wav.append(this_input_wav)
                this_input_wav_len = torch.tensor(this_input_wav.shape[-1]).unsqueeze(0).repeat(this_input_wav.shape[0]).long().to(this_input_wav.device)
                left_wav_len = torch.clamp(wav_len - st_sample, min=0)
                input_wav_len.append(torch.min(this_input_wav_len, left_wav_len))
                if et_sample >= n_samples:
                    break
                st += chunk_size
            umm_vae = [forward(this_wav, this_wav_len) for this_wav, this_wav_len in zip(input_wav, input_wav_len)]
            umm_vae = torch.cat(umm_vae, dim=1)
        else:
            umm_vae = forward(wav, wav_len)
        return umm_vae


    @torch.no_grad()
    def inference_align(self, wav, wav_len=None): # chunk inference
        dtype = torch.bfloat16
        ##########################
        ## Align feature length ##
        ##########################
        token_len = int(wav.shape[-1] / self.config.sample_rate * self.config.frame_rate)
        n_chunks = math.ceil(token_len / self.token_chunk_size)
        aligned_token_len = (
            math.ceil(token_len / self.token_chunk_size) * self.token_chunk_size
            + self.token_overlap # 100, 25
        )
        token_pad_len = aligned_token_len - token_len
        aligned_wav_len = int(aligned_token_len / self.config.frame_rate * self.config.sample_rate)
        print(
            f"token align from {token_len} to {aligned_token_len} ({token_pad_len=} {n_chunks=})\n",
            f"wav align from {wav.shape[-1]} to {aligned_wav_len} ({aligned_wav_len-wav.shape[-1]=})"
        )
        # pad wav according to aligned token length
        wav = F.pad(wav, (0, int(aligned_wav_len - wav.shape[-1])))

        bs = wav.shape[0]
        aligned_bn_ctx_len = self.bn_chunk_size * n_chunks
        bn_ctx = torch.ones([bs, aligned_bn_ctx_len, self.dit_config.in_channels]) * self.bn_padding
        print(f"bn_ctx align from 0 to {aligned_bn_ctx_len} {n_chunks=})")
        total_frame = math.ceil(token_len * self.dit_config.bn_frame_rate / self.config.frame_rate)
        print(f"{total_frame=}")

        ##########################
        ## Tokenizer ##
        ##########################
        umm_vae = self.inference_tokenizer(wav, wav_len)
        assert umm_vae.shape[1] == aligned_token_len

        ###############################
        ## chunk-inference diffusion ##
        ###############################
        inputs = {
            'all_token_embed': umm_vae,
            'all_bn_ctx': bn_ctx.to(umm_vae.device)}   # [B, T, 32]
        
        # apply text cfg
        if self.text_cfg_w != 1:
            inputs = self.DiT.make_cfg_input(inputs, padding_value=self.dit_config.cond_padding)
            # self.dit_config.cond_padding

        all_token_key = "all_token" if "all_token" in inputs else "all_token_embed"
        token_key = "token" if "token" in inputs else "token_embed" 
        
        full_mel = []
        self.DiT.clear_infer_params(self.diffusion_nfe, 
            total_frame, bs, self.text_cfg_w, self.mem_efficient)
        for chunk_idx in range(n_chunks):
            token_start_index = chunk_idx * self.token_chunk_size - (0 if chunk_idx == 0 else self.token_overlap)
            token_end_index = (chunk_idx + 1) * self.token_chunk_size + self.token_overlap

            bn_start_index = chunk_idx * self.bn_chunk_size
            bn_end_index = min((chunk_idx + 1) * self.bn_chunk_size, total_frame)

            inputs[token_key] = inputs[all_token_key][:, token_start_index:token_end_index]
            inputs["bn_ctx"] = inputs["all_bn_ctx"][:, bn_start_index:bn_end_index]
            print(f"{chunk_idx=} token={(token_start_index,token_end_index, token_end_index-token_start_index)}({inputs[all_token_key].shape[1]}, {inputs[token_key].shape[1]})")
            print(f"{chunk_idx=} bn_ctx={(bn_start_index, bn_end_index, bn_end_index-bn_start_index)}({inputs['all_bn_ctx'].shape[1]} {inputs['bn_ctx'].shape[1]})")

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                chunk_mel = self.DiT.chunk_inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    use_infer_params=True,
                    first=chunk_idx == 0,
                    # n_token_hierarchy_eval=self.n_token_hierarchy_eval,
                )
                print(chunk_mel.shape)
            # print(f"{chunk_idx=} mel={chunk_mel.shape=}")
            self.DiT.update_infer_params(
                bn_end_index - bn_start_index, last=chunk_idx + 1 >= n_chunks - 1
            )
            full_mel.append(chunk_mel)

        self.DiT.clear_infer_params(self.diffusion_nfe)
        full_mel = torch.cat(full_mel, dim=-1)
        full_mel = self.bn_norm.denorm_mel(full_mel)

        return full_mel
    
    @torch.no_grad()
    def inference_prefix(self, wav, wav_len=None):
        dtype = torch.bfloat16
        ##########################
        ## Tokenizer ##
        ##########################
        umm_vae = self.inference_tokenizer(wav, wav_len)

        bs = umm_vae.shape[0]
        ratio = int(self.dit_config.bn_frame_rate // self.config.frame_rate)
        bn_ctx_pad_len = ratio * umm_vae.shape[1]
        bn_ctx = torch.ones([bs, bn_ctx_pad_len, self.dit_config.in_channels]) * self.bn_padding
        print(f"bn_ctx align from 0 to {bn_ctx_pad_len})")
        total_frame = (ratio + 1) * umm_vae.shape[1] + 1  # (add sos_emb)
        print(f"{total_frame=}")

        ###############################
        ## chunk-inference diffusion ##
        ###############################
        inputs = {
            'all_token_embed': umm_vae,
            # 'all_bn_ctx': bn_ctx.to(umm_vae.device)
            }   # [B, T, 32]
        
        # apply text cfg
        if self.text_cfg_w != 1:
            inputs = self.DiT.make_cfg_input(inputs, padding_value=self.dit_config.cond_padding)
            # # self.dit_config.cond_padding

        full_mel = []
        overlap_token_len = 0 #1 * self.config.frame_rate
        input_token_embed = []
        all_token_key = "all_token" if "all_token" in inputs else "all_token_embed"
        if self.token_embed_chunk and umm_vae.shape[1] / self.config.frame_rate > 60:
            st = 0
            n_tokens = umm_vae.shape[1]
            chunk_size = self.token_embed_chunk_size * self.config.frame_rate
            while st < n_tokens:
                input_token_embed.append(inputs[all_token_key][:, st:st+chunk_size])
                st += (chunk_size - overlap_token_len)
        else:
            input_token_embed = [inputs[all_token_key]]

        token_key = "token" if "all_token" in inputs else "token_embed"
        
        for i in range(len(input_token_embed)):
            total_frame = (ratio + 1) * input_token_embed[i].shape[1] + 1  # (add sos_emb)
            self.DiT.clear_infer_params(self.diffusion_nfe, total_frame, bs, self.text_cfg_w, self.mem_efficient)
            inputs[token_key] = input_token_embed[i]

            print(f"{i=} {token_key=} {inputs[token_key].shape=}")
            # inputs["bn_ctx"] = inputs["all_bn_ctx"]
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                chunk_mel = self.DiT.prefix_inference(
                    inputs,
                    self.diffusion_nfe,
                    self.diffusion_sampler,
                    text_cfg_w=self.text_cfg_w,
                    use_infer_params=True,
                )
            chunk_mel = self.bn_norm.denorm_mel(chunk_mel)
            print(chunk_mel.shape)
            full_mel.append(chunk_mel)
        
        return full_mel


    @torch.no_grad()
    def inference(self, wav, wav_len=None):
        if self.dit_config.cond_mode == "align":
            return self.inference_align(wav, wav_len)

        elif self.dit_config.cond_mode == "prefix":
            return self.inference_prefix(wav, wav_len)


from recipes.umm2.modules.stages.umm_DiT import init_sacodec
def init_vocoder():
    vocoder = init_sacodec(checkpoint_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/andrew.shaw/sacodec_v5/sacodec_vB_44k_50h_128d_umm_rope_b2_vocal_h1536_ummdimfix_pretrain/checkpoints/sacodec_checkpoint_epoch=0_step=180000_val_loss=1.1525.ckpt",
                            local_rank=0,
                            cache_dir=".module_cache/vocoder/sacodec",
                            version="umm")
    return vocoder["sacodec"]

@torch.no_grad()
def vae2wav(model, latents):
    # latents = latents.transpose(1, 2) # B L D -> B D L
    latents = model.denormalize_features(latents, 0, 1)
    # latents = latents.transpose(1, 2) # B D L -> B L D
    audio_hat = model.decode_latents(latents)
    return audio_hat # B x CH x L

@torch.no_grad()
def wav2vae(model, wav):
    # wav = B x CH x L
    latents = model.get_latents(wav) # B L D
    latents = model.normalize_features(latents, 0, 1)
    return latents


from recipes.umm2.models.config import UMMConfig
from recipes.umm2.models.tasks.dit_module import ModelArgs as DiTConfig
def test_align_dit(config, vocoder, inference=False):
    dit_config = DiTConfig(
        in_channels=128,
        out_channels=128,
        token_upscales=[2],
        token_downscales=[],
        window_size=[400, 200],
        token_embed_dim=32,
        use_window_mask=True, 
        window_type="blockwise",
        causal=False,
        bn_frame_rate=50,
        target_type="velocity",
        use_unet_style_skip_connect=True,
        cond_mode="align",
    )
    model = Stage2DiTEnc(config, dit_config, sigma=0.2, 
                         takes=['audio','audio_length']).cuda()
    print(model)

    if inference:
        inference_config = {
        "bn_padding": -5,
        "diffusion_nfe": 10,
        "text_cfg_w": 1.4,
        "mem_efficient": True,
        "diffusion_sampler": "ddim",
        "bn_norm_mean": 0,
        "bn_norm_std": 1,
        "diffusion_precision": "bf16",
        "embed_padding": 0,
        "token_embed_chunk": False,
        "token_embed_chunk_size": 45,
    }
        model = model.eval()
        model.prepare_inference(inference_config)
    

    B, sec = 2, 30
    dummy_input = {
        'audio': torch.randn(B, 1, config.sample_rate * sec).to("cuda"),
        'bn': torch.randn(B, dit_config.bn_frame_rate * sec, dit_config.in_channels).to("cuda"),
        'bn_mask': torch.ones(B, dit_config.bn_frame_rate * sec).to("cuda"),
        'text_tokens': torch.randint(0, 105880, size=(B, 10)).to("cuda"),
    }

    with torch.cuda.amp.autocast(enabled=True):
        if not inference:
            result = model(dummy_input)
            for key in result:
                print(key, result[key].shape if isinstance(result[key], torch.Tensor) else result[key])

        else:
            infer_audio = torch.randn(B, 1, 2934823).to("cuda")
            mel_vae = model.inference_tokenizer(infer_audio)
            print(infer_audio.shape, mel_vae.shape)

            wav_vae = model.inference(infer_audio)
            print(infer_audio.shape, wav_vae.shape)

            wav = vae2wav(vocoder, wav_vae.mT)
            print(wav.shape)

def test_prefix_dit(config, vocoder, inference=False):
    dit_config = DiTConfig(
        in_channels=128,
        out_channels=128,
        token_upscales=[2],
        token_downscales=[],
        token_embed_dim=32,
        window_size=[-1, -1],
        use_window_mask=False, 
        causal=False,
        bn_frame_rate=50,
        target_type="velocity",
        use_unet_style_skip_connect=True,
        cond_mode="prefix",
    )
    model = Stage2DiTEnc(config, dit_config, sigma=0.2, 
                         takes=['audio','audio_length']).cuda()
    print(model)

    if inference:
        inference_config = {
        "bn_padding": -5,
        "diffusion_nfe": 10,
        "text_cfg_w": 1.4,
        "mem_efficient": True,
        "diffusion_sampler": "ddim",
        "bn_norm_mean": 0,
        "bn_norm_std": 1,
        "diffusion_precision": "bf16",
        "embed_padding": 0,
        "token_embed_chunk": False,
        "token_embed_chunk_size": 45,
    }
        model = model.eval()
        model.prepare_inference(inference_config)
    
    B, sec = 2, 30
    dummy_input = {
        'audio': torch.randn(B, 1, config.sample_rate * sec).to("cuda"),
        'bn': torch.randn(B, dit_config.bn_frame_rate * sec, dit_config.in_channels).to("cuda"),
        'bn_mask': torch.ones(B, dit_config.bn_frame_rate * sec).to("cuda"),
        'text_tokens': torch.randint(0, 105880, size=(B, 10)).to("cuda"),
    }

    with torch.cuda.amp.autocast(enabled=True):
        if not inference:
            result = model(dummy_input)
            for key in result:
                print(key, result[key].shape if isinstance(result[key], torch.Tensor) else result[key])

        else:
            infer_audio = torch.randn(B, 1, 2934823).to("cuda")
            mel_vae = model.inference_tokenizer(infer_audio)
            print(infer_audio.shape, mel_vae.shape)

            wav_vae = model.inference(infer_audio)
            print(infer_audio.shape, wav_vae.shape)

            wav = vae2wav(vocoder, wav_vae.mT)
            print(wav.shape)

if __name__ == "__main__":
    from recipes.umm2.models.tasks.rvq_module import EMAResidualVectorQuantizerRP
    rvq_scheme = EMAResidualVectorQuantizerRP(
        vq_type="EMARPSimple",
        codebook_size=32768,
        codebook_dim=32,
        rvq=4, 
        dist=False, # for debug 1gpu
    )
    config = UMMConfig(
        frame_rate=25,
        sample_rate=24000,
        n_chroma=12,
        vq_codebook_dim=32,
        add_vq=True,
        vq_codebook_size=8,
        vq_type="RVQ",
        vq_scheme=rvq_scheme,
    )

    vocoder = init_vocoder()

    inference = False
    test_align_dit(config, vocoder, inference)
    # test_prefix_dit(config, vocoder, inference)

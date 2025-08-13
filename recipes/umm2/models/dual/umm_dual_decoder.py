from recipes.umm2.models.umm_fm import UMMModified, ConformerEncoderLayer, pad_btd_to, INPUT_SEQ_LEN_ALIGNMENT

from mariana.models.audio.conformer import ConformerLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from recipes.umm2.models.tasks.vq_module import VQ, EMAVectorQuantizerEntropy
from recipes.umm2.models.tasks.rvq_module import RVQ, EMAResidualVectorQuantizerEntropy, EMAResidualVectorQuantizerRP

class UMMDualDecoder(UMMModified):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        semantic_decoder_n_layer = config.get("semantic_decoder_num_hidden_layers", 4)
        # by default, the last 12 encoder_layers are served as 'acoustic_decoder'
        if not config.use_fused_kernel:
            self.semantic_decoder = nn.ModuleList(
                [ConformerEncoderLayer(config) for _ in range(semantic_decoder_n_layer)]
            )
        else:
            if not config.get("use_causal_conformer", False):
                self.semantic_decoder = nn.Sequential(*[
                    ConformerLayer(self.noncausal_conformer_config, None, i) for i in range(semantic_decoder_n_layer)
                ])
            else:
                raise NotImplementedError("causal conformer is not supported for semantic decoder")

    def _init_vq_module(self, config):
        super()._init_vq_module(config) # init self.vq_layer

        self.semantic_vq_type = config.get("semantic_decoder_vq_type", "separate_EMAEntropy")
        distance_type = getattr(config, 'vq_distance_type', 'euclidean')
        if self.semantic_vq_type.startswith('separate'):
            semantic_vq_type = self.semantic_vq_type.split("_")[1]
            # Initialize VQ components
            quantize = EMAVectorQuantizerEntropy(
                config.get("semantic_vq_codebook_size", 8192),
                config.get("semantic_vq_codebook_dim", 32), 
                decay=config.vq_decay,
                distance_type=distance_type
            )
            self.semantic_vq_layer = VQ(
                config, 
                task='vq', 
                loss_weight=config.w_loss_vq, 
                vq_scheme=quantize
            )
        elif self.semantic_vq_type.startswith('share'):
            self.semantic_vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
            
    def _compute_fused_kernel(self, batch):
        # Extract features
        mel, mel_len = self.get_feature(batch)
        flops = self.audio_encoder.get_flops(*mel.shape)

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

        if self.training:
            audio_input_shape = list(hidden_states.shape)
            audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
            flops += audio_encoder_flops * len(self.encoder_layers)

        # Initialize tracking variables
        vq_output_dict = None

        # acoustic decoder
        for layer_index, layer in enumerate(self.encoder_layers):
            if layer_index == self.vq_layer_idx:
                # separate VQ for acoustic & semantic decoder
                if self.semantic_vq_type.startswith('separate'):  
                    # 1) acoustic quantizer
                    acoustic_vq_output_dict = self.vq_layer({
                                        'latent': hidden_states,
                                        'attn_mask' : conformer_mask
                                        })
                    
                    acoustic_preVQ_embs = acoustic_vq_output_dict['prevq_embs'] # [B, T, vq_codebook_dim]
                    acoustic_hidden_states = acoustic_vq_output_dict['quantized_out']   # [B, T, H]
                    # 2) semantic quantizer
                    semantic_vq_output_dict = self.semantic_vq_layer({
                            'latent': hidden_states,
                            'attn_mask' : conformer_mask})
                    semantic_preVQ_embs = semantic_vq_output_dict['prevq_embs'] # [B, T, vq_codebook_dim]
                    semantic_hidden_states = semantic_vq_output_dict['quantized_out'] # [B, T, H]
                    
                    # orthogonal loss between acoustic_preVQ_embs & semantic_preVQ_embs
                    # The intention is to minimize the similarity of acoustic latent and semantic latent (towards 0, not -1)
                    orthogonal_loss = F.cosine_similarity(acoustic_preVQ_embs, semantic_preVQ_embs, dim=-1).abs()
                    orthogonal_loss = orthogonal_loss[conformer_mask.bool()].mean()

                    vq_output_dict = {
                        "loss": 0,
                        'aux/loss_orthogonal': orthogonal_loss * self.config.get("w_loss_orthogonal", 0)}
                    for key in acoustic_vq_output_dict.keys():
                        if key == 'loss':
                            vq_output_dict["loss"] += acoustic_vq_output_dict[key]
                            vq_output_dict["loss"] += semantic_vq_output_dict[key]
                        else:
                            vq_output_dict["acoustic_"+key] = acoustic_vq_output_dict[key]
                            vq_output_dict["semantic_"+key] = semantic_vq_output_dict[key]
                    vq_output_dict['loss'] += orthogonal_loss
                
                elif self.semantic_vq_type.startswith('shared'):    # shared RVQ
                    # semantic & acoustic quantizer
                    vq_output_dict = self.vq_layer({
                                        'latent': hidden_states,
                                        'attn_mask' : conformer_mask
                                        })
                    # shared RVQ
                    semantic_vq_emb = vq_output_dict['vq_emb'][..., 0] # [B, T, vq_codebook_dim]
                    semantic_hidden_states = self.semantic_vq_proj_out(semantic_vq_emb)

                hidden_states = acoustic_hidden_states  

            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]

        hidden_states = hidden_states[:, 0:seqlen, :]
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        # semantic decoder
        for layer_index, layer in enumerate(self.semantic_decoder):
            semantic_hidden_states = layer(semantic_hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
        semantic_hidden_states = semantic_hidden_states[:, 0:seqlen, :]
        flops += semantic_hidden_states.shape[0] * semantic_hidden_states.shape[1] * 2

        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['semantic_latent'] = semantic_hidden_states
        batch['flops'] = flops * 3  # extra 2x for backward.
        batch['attn_mask'] = feature_mask

        vq_output_dict['acoustic_vq_ids'] = vq_output_dict['acoustic_vq_ids'][:, 0:seqlen]
        vq_output_dict['semantic_vq_ids'] = vq_output_dict['semantic_vq_ids'][:, 0:seqlen]

        if vq_output_dict:
            batch.update(vq_output_dict)
        return batch 

    def _wav2token_fused(self, audio, audio_len=None, **kwargs):
        if audio_len is None:
            audio_len = torch.tensor([audio.shape[-1]] * audio.shape[0], device=audio.device, dtype=torch.long)
        
        mel, mel_len = self.preprocessing(audio, audio_len)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)
        
        output_dict = {}
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]

        len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
        assert len_diff >= 0, \
            f"conformer_mask should be larger than feature_mask by {len_diff}"
        conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()

        for layer_index, layer in enumerate(self.encoder_layers):
            if self.vq_layer is not None and layer_index == self.vq_layer_idx:
                # separate VQ for acoustic & semantic decoder
                if self.semantic_vq_type.startswith('separate'):  
                    acoustic_vq_output_dict = self.vq_layer({
                                        'latent': hidden_states,
                                        'attn_mask' : conformer_mask
                                        })
                    semantic_vq_output_dict = self.semantic_vq_layer({
                                        'latent': hidden_states,
                                        'attn_mask' : conformer_mask
                                        })
                    vq_output_dict = {
                        "loss": 0,}
                    for key in acoustic_vq_output_dict.keys():
                        if key == 'loss':
                            vq_output_dict["loss"] += acoustic_vq_output_dict[key]
                            vq_output_dict["loss"] += semantic_vq_output_dict[key]
                        else:
                            vq_output_dict["acoustic_"+key] = acoustic_vq_output_dict[key]
                            vq_output_dict["semantic_"+key] = semantic_vq_output_dict[key]

                    output_dict["quantized_latent"] = acoustic_vq_output_dict['quantized_out'][:, :seqlen, :]
                    output_dict["quantized_semantic_latent"] = semantic_vq_output_dict['quantized_out'][:, :seqlen, :]
                else:
                    raise NotImplementedError(f"semantic_vq_type {self.semantic_vq_type} is not implemented")
                break

            # Pass the hidden_states through the current Conformer encoder layer
            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
        
        output_dict['mel'] = mel
        output_dict['mel_len'] = mel_len
        output_dict['attn_mask'] = feature_mask
        vq_output_dict['acoustic_vq_ids'] = vq_output_dict['acoustic_vq_ids'][:, 0:seqlen]
        vq_output_dict['semantic_vq_ids'] = vq_output_dict['semantic_vq_ids'][:, 0:seqlen]
        output_dict.update(vq_output_dict)  
        
        return output_dict

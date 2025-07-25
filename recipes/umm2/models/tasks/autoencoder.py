from typing import Dict
from torch import nn
import torch
from recipes.umm2.models.umm_fm import UMM
from recipes.umm2.models.base import BaseStage

INPUT_SEQ_LEN_ALIGNMENT = 32

def pad_btd_to(inputs, align):
    seqlen = inputs.shape[1]
    pad_len = (align - seqlen % align) % align
    inputs = torch.nn.functional.pad(inputs, [0, 0, 0, pad_len])
    mask = torch.ones_like(inputs[:, :, 0]).float()
    mask[:, seqlen:] = 0
    return inputs, mask.float()

class AE_UMM(UMM):
    def __init__(
        self, 
        config,
        takes=["audio"],
        provides=["audio", "mel", "latent", "vq_ids", "vq_entropy"],
        bypasses=[],
        lr_ratio: float = 0.1,
        loss_weight: float = None,
        is_train: bool = True,
        is_frozen: bool = False,  # if is_frozen, loss_weight must be None
        prevq_latent: bool = False,
        insert_modules: Dict [int, BaseStage] = None, 
        # insert modules in encoder_layers with dictionary (i.e., <num_layer>: module)
        # if module is None, skip the layers after num_layer
    ):
        UMM.__init__(self, config, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen)

        self.config = config
        self.is_train = is_train
        self.prevq_latent = prevq_latent

        if insert_modules is not None:
            self.insert_layer_nums = list(insert_modules.keys())
            self.insert_modules = nn.ModuleList([insert_modules[k] for k in self.insert_layer_nums]) 
        else:
            self.insert_layer_nums = None

    def _compute(self, batch):
        input_dict = self.get_feature(batch)  
        output_dict = {}
        insert_output_dict = {}
        feature = input_dict['mel']
        if self.is_train:
            flops = self.audio_encoder.get_flops(*feature.shape)
        else:
            flops = 0
        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        # position_embeddings = self.embed_positions(hidden_states)

        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]

        if self.is_train:
            audio_input_shape = list(hidden_states.shape)
            audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
            flops += audio_encoder_flops * len(self.encoder_layers)

        for i, layer in enumerate(self.encoder_layers):
            if self.insert_layer_nums is not None and i in self.insert_layer_nums:
                insert_module = self.insert_modules[self.insert_layer_nums.index(i)]
                insert_input_dict = {'latent': hidden_states}
                insert_output_dict = insert_module(insert_input_dict)
                output_dict.update(insert_output_dict)
                if not self.prevq_latent:   
                    # use the latent with quantization (vq_emb)
                    hidden_states = insert_output_dict["latent"]

            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                            use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]

            # hidden_states = layer(
            #     hidden_states, position_embeddings=position_embeddings
            # )

            # if self.is_train:
            #     flops += self.encoder_layers[0].calc_flops(hidden_states.shape) 

        hidden_states = hidden_states[:, 0:seqlen, :]

        if self.is_train:
            flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        output_dict.update({
            "audio": batch['audio'],
            "mel": input_dict["mel"],
            "latent": hidden_states,
            "flops": flops * 3,  # extra 2x for backward.
        })
        return output_dict
    


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_wav(self, wav):
        """Check audio dimensions and pad."""
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        padded_wav = self.pad_audio(wav.float())
        return padded_wav
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        wav = self.prepare_wav(wav)
        input_dict = self.preprocessing(wav)
        output_dict = {}
        feature = input_dict['mel']
        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        # position_embeddings = self.embed_positions(hidden_states)

        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]

        for i, layer in enumerate(self.encoder_layers):
            if self.insert_layer_nums is not None and i in self.insert_layer_nums:
                insert_module = self.insert_modules[self.insert_layer_nums.index(i)]
                insert_input_dict = {'latent': hidden_states}
                insert_output_dict = insert_module(insert_input_dict)
                output_dict.update(insert_output_dict)
                if i == self.insert_layer_nums[-1]:
                    break

            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                            use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]

            # hidden_states = layer(
            #     hidden_states, position_embeddings=position_embeddings
            # )

        hidden_states = hidden_states[:, 0:seqlen, :]

        output_dict.update({
            "audio": wav,
            "mel": input_dict["mel"],
            "latent": hidden_states,
        })
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2latent(self, wav, layer_idx=12):
        wav = self.prepare_wav(wav)
        input_dict = self.preprocessing(wav)
        output_dict = {}
        feature = input_dict['mel']
        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.encoder_layers):
            if i == layer_idx:
                output_dict.update(latent=hidden_states)
                break            
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        output_dict.update({
            "audio": wav,
            "mel": input_dict["mel"],
            "latent": hidden_states,
        })
        return output_dict
    
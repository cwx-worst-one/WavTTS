
from typing import Dict
import torch
from torch import nn

from recipes.umm2.models.tasks.autoencoder import AE_UMM
from recipes.umm2.models.base import BaseStage

class WAV2TOKENS(AE_UMM):
    def __init__(
        self, 
        config,
        takes=["audio"],
        provides=["latent", "vq_ids", "prevq_latent"],
        bypasses=[],
        leave_middle=False,
        insert_modules: Dict [int, BaseStage] = None, 
        # insert modules in encoder_layers with dictionary (i.e., <num_layer>: module)
        # if module is None, skip the layers after num_layer
    ):
        AE_UMM.__init__(self, config, takes, provides, bypasses, insert_modules=insert_modules)
        self.leave_middle = leave_middle


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _compute(self, wav):
        """Convert audio file to tokens (after Vector Quantization)."""
        input_dict = self.get_feature(wav)  
        feature = input_dict['mel']
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        output_dict = {}
        insert_output_dict = {}
        #from IPython import embed; embed(using=False);
        for i, layer in enumerate(self.encoder_layers):
            if self.insert_layer_nums is not None and i in self.insert_layer_nums:
                insert_module = self.insert_modules[self.insert_layer_nums.index(i)]
                insert_output_dict = insert_module({'latent': hidden_states})
                output_dict.update(insert_output_dict)
                break

            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        if self.leave_middle:
            emb_len = hidden_states.shape[-2]  # (b, t, d)
            output_len = min(emb_len, int(self.config.sample_len * self.config.frame_rate))  # based on length for training the tagging model (e.g., 30s)
            start_pos = max(emb_len - output_len, 0) // 2
            output_dict["vq_ids"] = output_dict["vq_ids"][:, start_pos : start_pos + output_len]
            hidden_states = hidden_states[:, start_pos : start_pos + output_len, :]

        output_dict.update({
            "mel": input_dict["mel"],
            "prevq_latent": hidden_states,
        })

        return output_dict

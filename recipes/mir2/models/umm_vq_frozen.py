from typing import Dict
import torch
from torch import nn

from recipes.umm2.models.umm_fm import UMM
from recipes.umm2.models.base import BaseStage

class AE_UMM(UMM):
    def __init__(
        self, 
        config,
        takes=[],
        provides=[],
        bypasses=[],
        lr_ratio: float = 1.0,
        loss_weight: float = None,
        is_frozen: bool = False,
        is_train: bool = True,
        token_only: bool = False,
        pre_vq_latent: bool = False,
        insert_modules: Dict [int, BaseStage] = None, 
        # insert modules in encoder_layers with dictionary (i.e., <num_layer>: module)
        # if module is None, skip the layers after num_layer
    ):
        UMM.__init__(self, config, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen)

        self.config = config
        self.pre_vq_latent = pre_vq_latent
        self.is_train = is_train
        self.token_only = token_only

        if insert_modules is not None:
            self.vq_layer_idx = list(insert_modules.keys())[0]    # only take the first vq_layer_idx in insert_modules
            self.insert_modules = nn.ModuleList([insert_modules[self.vq_layer_idx]]) # only use the first vq_module
        else:
            self.vq_layer_idx = None
     
    @torch.no_grad()
    def encoder_vq(self, batch):
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
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.encoder_layers):
            prevq_latent = hidden_states
            if i == self.vq_layer_idx:
                if not self.pre_vq_latent:  
                    # use vector quantized latent for the succeeding encoder
                    insert_input_dict = {'latent': hidden_states}
                    insert_output_dict = self.insert_modules[0](insert_input_dict)
                    output_dict.update({
                        "postvq_latent": insert_output_dict["latent"],
                        "vq_ids": insert_output_dict["vq_ids"]
                    })
                break # end of the module

            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            if self.is_train:
                flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) 
        
        output_dict.update({
            "audio": batch['audio'],
            "latent": prevq_latent,
            "flops": flops,  
            "position_embeddings": position_embeddings,
        })
            
        return output_dict


    def _compute(self, batch):
        output_dict = self.encoder_vq(batch) 
        flops = output_dict["flops"]
        position_embeddings = output_dict["position_embeddings"]
        hidden_states = output_dict["latent"]
        if not self.is_train:
            emb_len = hidden_states.shape[-2]  # (b, t, d)
            output_len = min(emb_len, int(self.config.sample_len * self.config.frame_rate))  # based on length for training the tagging model (e.g., 30s)
            start_pos = max(emb_len - output_len, 0) // 2
            output_dict["vq_ids"] = output_dict["vq_ids"][:, start_pos : start_pos + output_len]
            output_dict["postvq_latent"] = output_dict["postvq_latent"][:, start_pos : start_pos + output_len, :]
            hidden_states = hidden_states[:, start_pos : start_pos + output_len, :]
            if self.token_only:
                return output_dict

        for layer in self.encoder_layers[self.vq_layer_idx :]:  # from layer vq_layer_idx
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            if self.is_train:
                flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) 

        if self.is_train:
            flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        output_dict.update({
            "latent": hidden_states,
            "flops": flops * 3, # extra 2x for backward.
        })

        return output_dict

if __name__ == '__main__':
    from recipes.umm2.models.config import UMMConfig
    from recipes.umm2.models.tasks.vq_module import VQ

    config = UMMConfig(
        frame_rate=25,
        sample_len=30.0,
    )
    vq_module = VQ(
        config,
        is_frozen=True,
    )
    model = AE_UMM(
        config,
        takes=["audio"],
        provides=["latent", "vq_ids", "postvq_latent"],
        loss_weight=None,
        is_frozen=True,
        is_train=False,
        pre_vq_latent= False,
        insert_modules= {12: vq_module}, 
    )

    device = torch.device("cuda:5") 
    model.to(device)

    data = {'audio': torch.randn(1, 24000*60, dtype=torch.float).to(device)}
    output = model.forward(data)
    
    print([(k, output[k].shape) for k in ['vq_ids', 'latent', 'postvq_latent']])
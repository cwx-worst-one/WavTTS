from recipes.umm2.models.umm_fm import UMM

import torch
import torch.nn as nn
import torch.nn.functional as F
from recipes.umm2.models.tasks.mel_head import Mel_Head
from recipes.umm2.models.tasks.dit_module import (
    LlamaDiffusion,
)
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm2.modules.metric.common import STFTLoss
from recipes.umm2.transforms.chroma import ChromaSpectrogram
from recipes.umm.models.voc_modules.pitch_predictor.inference import PerceptualPitchPredictor
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.utils.mel_utils import torch_wav2spec
from recipes.umm2.models.full.conformer_DiT_enc import Stage2DiTEnc
from recipes.umm2.models.tasks.ctc_head import CTC_Head, CTC_Head_advanced
from recipes.umm2.models.tasks.chroma_head import Chroma_Head
from recipes.umm2.models.tasks.mel_head import Mel_Head
from recipes.umm2.models.tasks.f0_vuv_head import F0_VUV_Head_Pitchpdt

class Stage2DiTFull(Stage2DiTEnc):
    def __init__(self, config, dit_config, sigma=0.2):
        super(Stage2DiTFull, self).__init__(config, dit_config, sigma=sigma)
        if self.config.get("use_umm_heads", False):
            self.prepare_heads()

    def prepare_heads(self, ):
        self.heads = nn.ModuleList([])
        
        if self.config.get("add_mel", False):
            self.heads.append(Mel_Head(self.config))

        ## CTC head ##
        if self.config.get("add_ctc", False):
            # self.heads.append(CTC_Head_advanced(self.config))
            self.heads.append(CTC_Head(self.config))
        
        ## chroma head ##
        if self.config.get("add_chroma", False):
            self.heads.append(Chroma_Head(self.config))

        ## f0/vuv head ##
        if self.config.get("add_pitch", False):
            self.heads.append(F0_VUV_Head_Pitchpdt(self.config))
    
    def prepare_vq_fc(self, ):
        self.vq_proj_in = nn.Linear(
            config.hidden_size, config.vq_codebook_dim, bias=False
        )
        self.vq_proj_out = nn.Linear(
            config.vq_codebook_dim, config.hidden_size, bias=False
        )

    def forward(self, batch):
        input_dict = self.get_feature(batch)  
        # audio encoder
        feature = input_dict['mel']
        flops = self.audio_encoder.get_flops(*feature.shape)
        feature = self.audio_encoder(feature)
        # pre-encoder 
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)
        # encoder layers
        conformer_hidden_states = []
        for i, layer in enumerate(self.encoder_layers):
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            if i == self.config.vq_layer_idx - 1:
                # add vq
                downsample_hidden_states = self.vq_proj_in(hidden_states)
                upsample_hidden_states = self.vq_proj_out(downsample_hidden_states)
                conformer_hidden_states.append(downsample_hidden_states)
                hidden_states = upsample_hidden_states
            else:    
                conformer_hidden_states.append(hidden_states)
            flops += self.encoder_layers[i].get_flops(*hidden_states.shape[0:2])    


        ### diffusion ###
        # continuous feature
        continous_embed = conformer_hidden_states[self.config.vq_layer_idx - 1] # middle
        # standard noramlize
        continuous_embed = (continous_embed - continous_embed.mean(dim=-1, keepdim=True)) / (continous_embed.var(dim=-1, keepdim=True) + 1e-5).sqrt()
        continuous_embed = continuous_embed + torch.randn_like(continuous_embed) * self.sigma

        DiT_input = {
            'bn': batch['bn'],          # [B, T*50, 128] float
            'embed': continous_embed,   # [B, T*25, 32] float
            'bn_mask': batch['bn_mask'],
        }

        pred, target = self.DiT(DiT_input)

        output_dict = {
            'dit_pred': pred,
            'dit_target': target,
        }
        
        # loss heads
        head_input_dict = {
            'mel': input_dict["mel"],
            'latent': conformer_hidden_states[-1],
            'token': batch["text_tokens"],
            'audio': batch['audio'],
        }
        head_loss_dict = self.compute_heads(head_input_dict)
        output_dict.update(head_loss_dict)
        output_dict["flops"] = flops + head_loss_dict["flops"]
        return output_dict
        
        

    def compute_heads(self, head_input_dict):
        output_dict = {}
        flops = 0
        for head in self.heads:
            head_output_dict = head(head_input_dict)
            output_dict.update(head_output_dict)
            flops += head_output_dict["flops"] if "flops" in head_output_dict else 0
        output_dict["flops"] = flops
        return output_dict


    
if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    from recipes.umm2.models.tasks.dit_module import ModelArgs as DiTConfig
    config = UMMConfig(
        frame_rate=25,
        sample_rate=24000,
        n_chroma=12,
        vq_codebook_dim=32,
        add_chroma=True,
        add_ctc=True,
        add_pitch=True, 
        add_mel=True,
        use_umm_heads=True,
        ctc_ignore_empty=True,
    )
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
    )
    model = Stage2DiTFull(config, dit_config).cuda()
    print(model)

    B, sec = 2, 30
    dummy_input = {
        'audio': torch.randn(B, 1, config.sample_rate * sec).to("cuda"),
        'bn': torch.randn(B, dit_config.bn_frame_rate * sec, dit_config.in_channels).to("cuda"),
        'bn_mask': torch.ones(B, dit_config.bn_frame_rate * sec).to("cuda"),
        'text_tokens': torch.randint(0, 105880, size=(B, 10)).to("cuda"),
    }

    with torch.cuda.amp.autocast(enabled=True):
        result = model(dummy_input)
    print([(key, result[key].shape if isinstance(result[key], torch.Tensor) else result[key]) for key in result])
    # print(result['dit_pred'].shape, result['dit_target'].shape)
from typing import Dict
import torch
from torch import nn
from einops import rearrange

from recipes.umm2.models.umm_melrof import UMM
from recipes.umm2.models.base import BaseStage

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
        mel = self.get_feature(batch)["mel"]

        output_dict = {}
        insert_output_dict = {}

        x = batch['audio'].squeeze(dim=1).float()
        x = self.pad_audio(x)
        # (batch_size, input_channels, segment_samples)
        
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.stft(x)

        x = x.to(x.dtype) # (b, c, f, t)

        # Separate into subbands
        x = self.multi_band_transform(x)
        # (b, t, k, n)

        for i, layer in enumerate(self.transformer_stack):
            if self.insert_layer_nums is not None and i in self.insert_layer_nums:
                insert_module = self.insert_modules[self.insert_layer_nums.index(i)]
                insert_input_dict = {'latent': x}
                insert_output_dict = insert_module(insert_input_dict)
                output_dict.update(insert_output_dict)
                if not self.prevq_latent:   
                    x = insert_output_dict["latent"]

            x = layer(x, self.rotary_emb_t, self.rotary_emb_k)
        # (b, t, k, n)

        x = self.multi_band_embed(x)
        # (b, t, k, num_out_feature)

        x = rearrange(x, "b t k n -> b t (k n)")
        # batch x time x feature

        output_dict.update({
            "audio": batch['audio'],
            "mel": mel,
            "latent": x,
            "flops": 0, 
        })
       
        return output_dict


if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    from recipes.umm2.models.tasks.vq_multiband import VQ
    from recipes.umm2.models.tasks.vq_module import EMAVectorQuantizer

    config = UMMConfig(
        use_power_stft=True,
        use_avg_pool=False,
        hop_length=240,
        mel_bands=16,
        num_feature=128, # melrof has num_feature,
        use_checkpoint=True,
        use_flash_attn=True,
        enforce_dropout=0.1,
        out_num_feature=64,
        add_vq=True,
        vq_codebook_dim=32,
    )
    dummy_input = {'audio': torch.randn(2, 1, 24000 * 30).to("cuda")}

    quantizer = EMAVectorQuantizer(
        codebook_size=config.vq_codebook_size,
        codebook_dim=config.vq_codebook_dim,
        decay=0.99,
        dist=False
        ).to("cuda")

    vq_model = VQ(
        config,
        single_codebook=False,
        #vq_scheme=quantizer
        ).to("cuda")

    model = AE_UMM(
        config,
        insert_modules= {12: vq_model}
    ).to("cuda")
    
    output = model(dummy_input)

    print(output['mel'].shape, output['latent'].shape, output['vq_ids'].shape, output['vq_entropy'], output["loss"])
    # torch.Size([2, 3000, 128]) torch.Size([2, 750, 1024]) torch.Size([2, 750, 16]) tensor(10.3971, device='cuda:0') tensor(0.5980, device='cuda:0', grad_fn=<MulBackward0>)
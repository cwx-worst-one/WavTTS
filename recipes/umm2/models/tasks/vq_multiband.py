import torch
import inspect
from torch import nn
from einops import rearrange

from recipes.umm2.models.tasks.vq_module import EMAVectorQuantizer
from recipes.umm2.models.umm_melrof import MLP_embed, MLP_proj
from recipes.umm2.models.base import BaseStage


class VQ(BaseStage):
    def __init__(
        self, 
        config,
        takes=["latent"], 
        provides=["latent", "vq_ids", "vq_entropy"],
        bypasses=[],
        task="vq",
        loss_weight=1.0,
        lr_ratio=1.0,
        # vq_scheme=None,  # since vq_codebook_size is divided into num_bands, it's better to hard code
        single_codebook=False,
        dist=True, # for debug
        is_frozen=False,
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio, is_frozen=is_frozen)

        self.config = config
        self.num_bands = config.mel_bands
        self.single_codebook = single_codebook
        if single_codebook:
            self.vq_codebook_size = config.vq_codebook_size
            self.vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=0.99,
                dist=dist, # for debug
                )
        else:
            self.vq_codebook_size = int(config.vq_codebook_size // self.num_bands)
            self.vq = nn.ModuleList([])
            for i in range(self.num_bands):
                self.vq.append(EMAVectorQuantizer(
                    codebook_size=self.vq_codebook_size,
                    codebook_dim=config.vq_codebook_dim,
                    decay=0.99,
                    dist=dist, # for debug
                    ))

        self.vq_proj_in = MLP_embed(
            num_feature=config.num_feature, 
            mel_bands=config.mel_bands,
            out_dim=config.vq_codebook_dim,
            use_checkpoint=config.use_checkpoint
            )
        self.vq_proj_out = MLP_proj(
            in_dim=config.vq_codebook_dim,
            num_feature=config.num_feature,
            mel_bands=config.mel_bands,
            )

        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        if is_frozen:
            self.eval()

    def quantize(self, x):
        vq_output_dict = {
            "embs": [],
            "vq_ids": [],
            "vq_entropy": [],
            "loss": [],       
        }

        for i in range(self.num_bands):
            sub_x = x[:, :, i, :].squeeze(2)  # [batch, time, vq_codebook_size]
            vq_output = self.vq(sub_x)
            vq_output_dict["embs"].append(vq_output["embs"])
            vq_output_dict["vq_ids"].append(vq_output["ids"])
            vq_output_dict["vq_entropy"].append(vq_output["entropy"])
            vq_output_dict["loss"].append(vq_output["loss"])

        vq_output_dict["embs"] = torch.stack(vq_output_dict["embs"], dim=2)  # [batch, time, band, vq_codebook_size]
        vq_output_dict["vq_ids"] = torch.stack(vq_output_dict["vq_ids"], dim=2)  # [batch, time, band],
        vq_output_dict["vq_entropy"] = torch.mean(torch.stack(vq_output_dict["vq_entropy"]))
        vq_output_dict["loss"] = torch.mean(torch.stack(vq_output_dict["loss"]))
        return vq_output_dict
    
    def multiband_quantize(self, x):
        vq_output_dict = {
            "embs": [],
            "vq_ids": [],
            "vq_entropy": [],
            "loss": [],       
        }
        for i, vq_func in enumerate(self.vq):
            sub_x = x[:, :, i, :].squeeze(2)  # [batch, time, vq_codebook_size]
            vq_output = vq_func(sub_x)
            vq_output_dict["embs"].append(vq_output["embs"])
            vq_output_dict["vq_ids"].append(vq_output["ids"] + int(i * self.vq_codebook_size))  
            # shift vq_ids by band*vq_codebook_size
            vq_output_dict["vq_entropy"].append(vq_output["entropy"])
            vq_output_dict["loss"].append(vq_output["loss"])

        vq_output_dict["embs"] = torch.stack(vq_output_dict["embs"], dim=2)  # [batch, time, band, vq_codebook_size]
        vq_output_dict["vq_ids"] = torch.stack(vq_output_dict["vq_ids"], dim=2)  # [batch, time, band],
        vq_output_dict["vq_entropy"] = torch.mean(torch.stack(vq_output_dict["vq_entropy"]))
        vq_output_dict["loss"] = torch.mean(torch.stack(vq_output_dict["loss"]))
        return vq_output_dict

    def _compute(self, data):

        hidden_states = data['latent']  # [batch, time, band, num_feature]
        x = self.vq_proj_in(hidden_states) # [batch, time, band, vq_codebook_size]

        if self.config.get("vq_proj_noise", 0) > 0:
            noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(0) / self.config.vq_proj_noise
            x += torch.randn_like(x) * noise_scale 
            self.cnt.add_(1)          

        if self.single_codebook:
            output_dict = self.quantize(x)
        else:
            output_dict = self.multiband_quantize(x)

        x = output_dict["embs"]  # [batch, time, band, vq_codebook_size]
        hidden_states = self.vq_proj_out(x)   # [batch, time, band, num_feature]

        output_dict["latent"] = hidden_states  # [batch, time, band, num_feature]      
        output_dict["vq_ids"] = rearrange(output_dict["vq_ids"], "b t k -> b (t k)")  # [batch, time*band] for display purpose

        return output_dict


if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    n_secs = 10

    config = UMMConfig(
        add_vq=True,
        mel_bands=16,
        num_feature=128, # melrof has num_feature,
        vq_codebook_dim=32,
        use_checkpoint=False,
    )
    dummy_input = {
        "latent": torch.randn(2, config.frame_rate * n_secs, config.mel_bands, config.num_feature).to("cuda"), 
        #  [batch, time, band, feature]
    }

    model = VQ(
        config, 
        dist=False,
        single_codebook=True,
        ).to("cuda")
    output = model(dummy_input)
    
    #from IPython import embed; embed(using=False)
    print(output['latent'].shape, output['vq_ids'].shape, output['vq_entropy'], output["loss"])
    # torch.Size([2, 250, 16, 128]) torch.Size([2, 4000]) tensor(7.6245, device='cuda:0') tensor(0.0182, device='cuda:0', grad_fn=<MulBackward0>)
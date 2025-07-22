import torch
from torch.nn import functional as F

from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm2.transforms.pesto import load_model

class Pesto_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["audio", "latent"], 
        provides=["f0", "loss"],
        bypasses=[],
        task="pesto_f0",
        loss_weight=1.0,
        lr_ratio=1.0,
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        step_size = config.hop_length / config.sample_rate * 1000  # in milliseconds
        self.pesto_pitch = load_model("mir-1k", step_size=step_size, sampling_rate=config.sample_rate)
        self.f0_pesto = Conv2dUpsampling(config.hidden_size, 1)

    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x
        
    def get_metrics(self, f0, recon_f0):
        # F0
        recon_f0 = recon_f0.contiguous().float()
        f0 = f0.contiguous().float()
        f0_loss = torch.abs(recon_f0 - f0).mean()
        return f0_loss

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, x):
        wav = x.squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        pesto_f0, conf = self.pesto_pitch(wav)
        return  pesto_f0

    def _compute(self, batch):
        f0 = self.get_feature(batch['audio'])
        f0_out = self.f0_pesto(batch["latent"])
        f0_out = f0_out.squeeze(-1)
        f0 = f0[:, :f0_out.shape[-1]]

        loss = self.get_metrics(f0, f0_out)
        
        output_dict = {
            "pf0": f0,
            "pf0_out": f0_out,
            "loss": loss,
        } 
        return output_dict

if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig()
    dummy_input = {
        "audio": torch.randn(4, config.sample_rate * 8),
        "latent": torch.randn(4, config.frame_rate * 8, config.hidden_size), #  [batch, time, hidden]
    }
    f0_head = Pesto_Head(
        config,
        takes=["audio", "latent"],
        provides=["f0", "f0_out", "loss"],
        bypasses=[],
        task="f0_vuv",
        loss_weight=1.0,
        lr_ratio=1.0,
    )
    result = f0_head(dummy_input)
    print(result["f0"].shape, result["f0_out"].shape, result["loss"])

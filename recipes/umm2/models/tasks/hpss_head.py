import torch
from torch import nn
from einops import rearrange

from recipes.umm2.models.base import BaseStage
from recipes.umm2.transforms.chroma import HPSS_Chroma_Onset
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm2.modules.metric.common import STFTLoss

class HPSS_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["audio", "latent"], 
        provides=["loss"],
        bypasses=[],
        task="hpss",
        loss_weight=1.0,
        lr_ratio=1.0,
        use_conv=True,
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        self.use_conv = use_conv
        self.hpss_chroma_onset = HPSS_Chroma_Onset(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )
        self.chroma_loss_fn = STFTLoss()

        if use_conv:
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, 
                config.n_chroma, 
                use_bn=config.get("use_bn", True)
            )
            self.onset_head = Conv2dUpsampling(config.hidden_size, 1)
        else:
            time_pool_length = int(config.sample_rate//config.hop_length//config.frame_rate)
            self.n_chroma = config.n_chroma
            self.chroma_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(config.hidden_size//2, config.n_chroma * time_pool_length),
            )
            self.onset_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(config.hidden_size//2, time_pool_length),
            )        

    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return nn.functional.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    def get_metrics(self, chroma, recon_chroma, onset, recon_onset):
        recon_chroma = recon_chroma.contiguous().float()
        chroma = chroma.contiguous().float()
        chroma_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)
        recon_onset = recon_onset.contiguous().float()
        onset = onset.contiguous().float()
        onset_loss = torch.abs(recon_onset - onset).mean()

        loss = (2/3) * chroma_loss["stft_loss"] + (1/3) * onset_loss
        return {"loss": loss}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, x):
        wav = x.squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        chroma, onset = self.hpss_chroma_onset(wav)
        chroma = nn.functional.normalize(chroma[:, :, :-1].transpose(1, 2), p=2, dim=-1)
        onset = onset[:, :-1]
        return chroma, onset

    def _compute(self, batch):
        chroma, onset = self.get_feature(batch['audio'])
        chroma_out = self.chroma_head(batch['latent'])
        onset_out = self.onset_head(batch['latent'])
        
        #from IPython import embed; embed(using=False)
        if self.use_conv:
            flops = self.chroma_head.get_flops(*batch['latent'].shape)
            flops +=  self.onset_head.get_flops(*batch['latent'].shape)
            onset_out = onset_out.squeeze(-1)
        else: 
            flops = 0  # neglect flops
            chroma_out = rearrange(chroma_out, "b t (c f) -> b (t c) f", f=self.n_chroma)   # c = time_pool_length
            onset_out = rearrange(onset_out, "b t c -> b (t c)")   # c = time_pool_length

        metric_dict = self.get_metrics(chroma, chroma_out, onset, onset_out)

        output_dict = {
            "chroma": chroma,
            "chroma_out": chroma_out, 
            "onset": onset,
            "onset_out": onset_out,
            "flops": flops * 3, 
            "loss": metric_dict["loss"],
        } 
        return output_dict

if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig(n_chroma = 12)

    dummy_input = {
        "audio": torch.randn(4, config.sample_rate * 8),
        "latent": torch.randn(4, config.frame_rate * 8, config.hidden_size), #  [batch, time, hidden]
    }
    hpss_head = HPSS_Head(
        config,
        takes=["audio", "latent"],
        provides=["chroma", "chroma_out", "onset", "onset_out", "loss"],
        bypasses=[],
        task="hpss",
        loss_weight=1.0,
        lr_ratio=1.0,
        use_conv=False,
    )
    result = hpss_head(dummy_input)
    print(result["chroma"].shape, result["chroma_out"].shape, result["onset"].shape, result["onset_out"].shape, result["loss"])
    # torch.Size([4, 800, 12]) torch.Size([4, 800, 12]) torch.Size([4, 800]) torch.Size([4, 800]) tensor(28.8931, grad_fn=<MulBackward0>)
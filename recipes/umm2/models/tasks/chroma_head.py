import torch
from torch import nn
from einops import rearrange

from recipes.umm2.models.base import BaseStage
from recipes.umm2.transforms.chroma import ChromaSpectrogram
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm2.modules.metric.common import STFTLoss
 
class Chroma_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["audio", "latent"], 
        provides=["loss"],
        bypasses=[],
        task="chroma",
        loss_weight=1.0,
        lr_ratio=1.0,
        use_conv=True,
        reduction="none"
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        self.use_conv = use_conv
        self.chroma_transform = ChromaSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )

        norm_flag = config.get("head_output_norm", False)
        self.chroma_norm = None
        if norm_flag:
            self.chroma_norm = nn.LayerNorm(config.hidden_size)

        if use_conv:
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, 
                config.n_chroma, 
                use_bn=config.get("use_bn", True),
                stride=config.get("upsample_strides", None),
                pad=config.get("upsample_pads", None),
            )
        else:
            time_pool_length = int(config.sample_rate//config.hop_length//config.frame_rate)
            self.n_chroma = config.n_chroma
            self.chroma_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(config.hidden_size//2, config.n_chroma * time_pool_length),
            )
        self.chroma_loss_fn = STFTLoss(reduction=reduction)

    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return nn.functional.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    def get_metrics(self, chroma, recon_chroma, chroma_len):
        recon_chroma = recon_chroma.contiguous().float()
        chroma = chroma.contiguous().float()
        B, T, _ = chroma.shape
        mask = torch.arange(T, device=chroma.device)[None, :] < chroma_len[:, None]
        stft_loss = self.chroma_loss_fn.float()(recon_chroma, chroma)['stft_loss']

        if stft_loss.ndim == 3:  # [B, T, D]
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            valid_count = mask.sum() * stft_loss.shape[-1]
            masked_loss = (stft_loss * mask.float()).sum() / valid_count.clamp(min=1)
        else:
            masked_loss = stft_loss  # fallback

        return {"loss": masked_loss}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, x):
        wav = x.squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        chroma = self.chroma_transform(wav)[:, :, :-1].transpose(1, 2)
        chroma = nn.functional.normalize(chroma, p=2, dim=-1)
        return chroma

    def forward(self, batch):
        
        chroma = self.get_feature(batch['audio'])
        chroma_len = batch['mel_len'] # suppose we alread have wav_lenn in batch
        latent = batch['latent']

        if self.chroma_norm is not None:
            latent = self.chroma_norm(latent)
        else:
            latent = latent
        chroma_out = self.chroma_head(latent)
        if self.use_conv:
            flops = self.chroma_head.get_flops(*latent.shape)  
        else: 
            flops = 0  # neglect flops
            chroma_out = rearrange(chroma_out, "b t (c f) -> b (t c) f", f=self.n_chroma)   # c = time_pool_length
        
        metric_dict = self.get_metrics(chroma, chroma_out, chroma_len)
        loss = metric_dict["loss"] * self.loss_weight

        output_dict = {
            "flops": flops * 3, # extra 2x for backward
            "loss": loss,
            "aux/loss_chroma": metric_dict["loss"],
        } 
        return output_dict

if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig(n_chroma=12)
    
    dummy_input = {
        "latent": torch.randn(4, config.frame_rate * 30, config.hidden_size).to("cuda"), #  [batch, time, hidden]
        "audio": torch.randn(4, config.sample_rate * 30).to("cuda"), # [batch, time, n_chroma]
    }
    chroma_head = Chroma_Head(
        config,
        takes=["audio", "latent"],
        provides=["chroma", "chroma_out", "flops", "loss"],
        bypasses=[],
        task="chroma_head",
        loss_weight=1.0,
        lr_ratio=1.0,
        use_conv=True,
    ).to("cuda")

    result = chroma_head(dummy_input)
    print(result["chroma"].shape, result["chroma_out"].shape, result["loss"])
    # torch.Size([4, 3000, 12]) torch.Size([4, 3000, 12]) tensor(1.8230, device='cuda:0', grad_fn=<MulBackward0>)
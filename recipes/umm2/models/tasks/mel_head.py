import torch
from torch import nn
from einops import rearrange

from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm2.modules.metric.common import STFTLoss


class Mel_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["mel", "latent"], 
        provides=["mel", "mel_out", "flops", "loss"],
        bypasses=[],
        task="mel",
        loss_weight=1.0,
        lr_ratio=1.0,
        is_frozen: bool = False,
        use_conv=True,
        reduction="none"
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio, is_frozen)
        
        self.config = config
        self.use_conv = use_conv
        

        norm_flag = config.get("head_output_norm", False)
        self.mel_norm = None
        if norm_flag:
            self.mel_norm = nn.LayerNorm(config.hidden_size)

        if use_conv:
            self.mel_head = Conv2dUpsampling(
                config.hidden_size, 
                config.n_mels, 
                use_bn=config.get("use_bn", True),
                stride=config.get("upsample_strides", None),
                pad=config.get("upsample_pads", None),
            )
        else:
            time_pool_length = int(config.sample_rate//config.hop_length//config.frame_rate)
            self.n_mels = config.n_mels
            self.mel_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(config.hidden_size//2, config.n_mels * time_pool_length),
            )
        self.mel_loss_fn = STFTLoss(reduction=reduction)

    def get_metrics(self, mel, recon_mel, mel_len):
        recon_mel = recon_mel.contiguous().float()
        mel = mel.contiguous().float()

        B, T, _ = mel.shape
        mask = torch.arange(T, device=mel.device)[None, :] < mel_len[:, None]  # [B, T] bool
        
        mel_loss_dict = self.mel_loss_fn(recon_mel, mel)
        stft_loss = mel_loss_dict["stft_loss"]

        if stft_loss.ndim == 3:  # [B, T, D]
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            valid_count = mask.sum() * stft_loss.shape[-1]
            masked_loss = (stft_loss * mask.float()).sum() / valid_count.clamp(min=1)
        else:
            masked_loss = stft_loss  # fallback

        return {"loss": masked_loss}

    def forward(self, batch):
        mel = batch["mel"]
        mel_len= batch["mel_len"]
        latent = batch['latent']
    
        if self.mel_norm is not None:
            latent = self.mel_norm(latent)
        else:
            latent = latent

        mel_out = self.mel_head(latent)

        if self.use_conv:
            flops = self.mel_head.get_flops(*latent.shape)  
        else: 
            flops = 0  # neglect flops
            mel_out = rearrange(mel_out, "b t (c f) -> b (t c) f", f=self.n_mels)   # c = time_pool_length
        
        metric_dict = self.get_metrics(mel, mel_out, mel_len)
        loss = metric_dict["loss"] * self.loss_weight
    
        output_dict = {
            "mel": mel,
            "mel_out": mel_out, 
            "flops": flops * 3, # extra 2x for backward.
            "loss": loss,
            "aux/loss_mel": metric_dict["loss"],
        } 
        return output_dict


# using forward
class MelHeadRevised(nn.Module):

    def __init__(
            self, 
            config,
            use_conv=True,
            reduction="none",
    ):
        super().__init__()
        self.config = config
        self.use_conv = use_conv
        self.reduction = reduction
        norm_flag = config.get("head_output_norm", False)
        self.mel_norm = None
        if norm_flag:
            self.mel_norm = nn.LayerNorm(config.hidden_size)

        if use_conv:
            self.mel_head = Conv2dUpsampling(
                config.hidden_size, 
                config.n_mels, 
                use_bn=config.get("use_bn", True),
                stride=config.get("upsample_strides", None),
                pad=config.get("upsample_pads", None),
            )
        else:
            time_pool_length = int(config.sample_rate//config.hop_length//config.frame_rate)
            self.n_mels = config.n_mels
            self.mel_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size//2),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(config.hidden_size//2, config.n_mels * time_pool_length),
            )
        self.mel_loss_fn = STFTLoss(reduction=reduction)

    def get_metrics(self, mel, recon_mel, mel_len):
        recon_mel = recon_mel.contiguous().float()
        mel = mel.contiguous().float()

        B, T, _ = mel.shape
        mask = torch.arange(T, device=mel.device)[None, :] < mel_len[:, None]  # [B, T] bool
        
        mel_loss_dict = self.mel_loss_fn(recon_mel, mel)
        stft_loss = mel_loss_dict["stft_loss"]

        if stft_loss.ndim == 3:  # [B, T, D]
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            valid_count = mask.sum() * stft_loss.shape[-1]
            masked_loss = (stft_loss * mask.float()).sum() / valid_count.clamp(min=1)
        else:
            masked_loss = stft_loss  # fallback

        return {"loss": masked_loss}
    
    def forward(self, batch):
        mel = batch["mel"]
        mel_len= batch["mel_len"]
    
        if self.mel_norm is not None:
            latent = self.mel_norm(batch['latent'])
        else:
            latent = batch['latent']
        mel_out = self.mel_head(latent)
        if self.use_conv:
            flops = self.mel_head.get_flops(*batch['latent'].shape)  
        else: 
            flops = 0  # neglect flops
            mel_out = rearrange(mel_out, "b t (c f) -> b (t c) f", f=self.n_mels)   # c = time_pool_length
        
        metric_dict = self.get_metrics(mel, mel_out, mel_len)

        output_dict = {
            "mel": mel,
            "mel_out": mel_out, 
            "flops": flops * 3, # extra 2x for backward.
            "loss": metric_dict["loss"], 
        } 
        return output_dict



if __name__ == "__main__":
    import torch
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 4
    max_sequence_length_latent = 30 * 25
    max_sequence_length_mel = max_sequence_length_latent * 4

    latent = torch.randn(batch_size, max_sequence_length_latent, config.hidden_size).to(device)
    mel = torch.randn(batch_size, max_sequence_length_mel, config.n_mels).to(device)

    latent_lengths = torch.randint(low=100, high=max_sequence_length_latent + 1, size=(batch_size,), device=device)
    mel_lengths = latent_lengths * 4

    latent_mask = torch.arange(max_sequence_length_latent, device=device)[None, :] < latent_lengths[:, None]

    dummy_input = {
        "latent": latent,
        "mel": mel,
        "latent_mask": latent_mask,
        "mel_len": mel_lengths,
    }

    mel_head = MelHeadRevised(config, use_conv=False).to(device)

    result = mel_head(dummy_input)
    print(result["mel"].shape, result["mel_out"].shape, result["loss"])

    print(result["loss"])
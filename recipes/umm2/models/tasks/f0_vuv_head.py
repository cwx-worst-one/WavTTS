import torch
from torch.nn import functional as F
import torch.nn as nn
from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import Conv2dUpsampling
from recipes.umm.models.rmvpe import RMVPE

from recipes.umm.models.voc_modules.pitch_predictor.inference import PerceptualPitchPredictor
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.utils.mel_utils import torch_wav2spec


def f0_normalize(f0):
    _f0 = f0.clone()
    f0 = torch.log1p(f0)
    f0_mean = torch.mean(f0[_f0 != 0])
    f0_std = torch.std(f0[_f0 != 0])
    f0_std = torch.where(f0_std == 0, torch.ones_like(f0_std), f0_std)
    f0[_f0 != 0] = (f0[_f0 != 0] - f0_mean) / f0_std
    return f0

def get_vuv(f0):
    vuv = f0.clone()
    vuv[vuv != 0] = 1
    return vuv


class F0_VUV_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["audio", "latent"], 
        provides=["f0", "vuv", "loss"],
        bypasses=[],
        task="f0_vuv",
        loss_weight=1.0,
        lr_ratio=1.0,
        f0_vuv_weights = [1.0, 1.0],
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        self.f0_vuv_weights = f0_vuv_weights

        hop_length = config.hop_length * 16000 // config.sample_rate
        print("RMVPE hop_length (on 16k):", hop_length)
        self.rmvpe = RMVPE(hop_length=hop_length)

        norm_flag = config.get("head_output_norm", False)
        self.f0_vuv_norm = None
        if norm_flag:
            self.f0_vuv_norm = nn.LayerNorm(config.hidden_size)

        self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2,
                                            use_bn=config.get("use_bn", True),
                                            stride=config.get("upsample_strides", None),
                                            pad=config.get("upsample_pads", None),)

    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x
        
    def get_metrics(self, f0, recon_f0, vuv, recon_vuv):
        # F0
        recon_f0 = recon_f0.contiguous().float()
        f0 = f0.contiguous().float()
        f0_loss = (torch.abs(recon_f0 - f0) * vuv).sum() / (torch.sum(vuv) + 1)
        # vuv
        recon_vuv = recon_vuv.contiguous().float()
        vuv = vuv.contiguous().float()
        vuv_loss = F.binary_cross_entropy_with_logits(recon_vuv, vuv)

        loss = self.f0_vuv_weights[0] * f0_loss + self.f0_vuv_weights[1] * vuv_loss 

        return {"loss": loss}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, x):
        wav = x.squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        f0 = self.rmvpe.batch_infer(
                wav, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
        f0 = f0[:, :-1]
        vuv = get_vuv(f0)
        f0 = f0_normalize(f0)
        return f0, vuv

    def _compute(self, batch):
        f0, vuv = self.get_feature(batch['audio'])

        if self.f0_vuv_norm is not None:
            latent = self.f0_vuv_norm(batch['latent'])
        else:
            latent = batch['latent']

        f0_vuv_out = self.f0_vuv_head(latent)
        f0_out = f0_vuv_out[:, :, 0:1].squeeze(-1)
        vuv_out = f0_vuv_out[:, :, 1:].squeeze(-1)

        metric_dict = self.get_metrics(f0, f0_out, vuv, vuv_out)

        output_dict = {
            "f0": f0,
            "f0_out": f0_out,
            "vuv": vuv,
            "vuv_out": vuv_out,
            "loss": metric_dict["loss"],
        } 
        return output_dict



class F0_VUV_Head_Pitchpdt(BaseStage):
    def __init__(
        self, 
        config, 
        takes=["audio", "latent"], 
        provides=["f0", "vuv", "loss"],
        bypasses=[],
        task="f0_vuv",
        loss_weight=1.0,
        lr_ratio=1.0,
        f0_vuv_weights = [1.0, 1.0],
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio)

        self.config = config
        self.f0_vuv_weights = f0_vuv_weights
        # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
        self.pitch_pdt = PerceptualPitchPredictor()
            
        # This head reconstructs the f0_hz and vuv signals from a mel-160 spectrogram
        self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2,
                                            use_bn=config.get("use_bn", True),
                                            stride=config.get("upsample_strides", None),
                                            pad=config.get("upsample_pads", None))

        # This is the mel-160 transform for the pre-trained pitch detector.
        self.mel_160_transform = lambda x: torch_wav2spec(
            x, num_mels=160, sample_rate=config.sample_rate
        )
            
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x
        
    def get_metrics(self, f0, recon_f0, vuv, recon_vuv, lengths):

        B, T, _ = f0.shape
        device = f0.device

        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        f0 = f0.float()
        recon_f0 = recon_f0.float()
        vuv = vuv.float()
        recon_vuv = recon_vuv.float()
        
        # F0
        f0_loss_mask = mask * vuv
        f0_loss = (torch.abs(recon_f0 - f0) * f0_loss_mask).sum() / (f0_loss_mask.sum() + 1)
        
        # vuv
        vuv_loss = F.binary_cross_entropy_with_logits(
                    recon_vuv, 
                    vuv, 
                    weight=mask, 
                    reduction='sum')/ (mask.sum() + 1e-6)

        loss = self.f0_vuv_weights[0] * f0_loss + self.f0_vuv_weights[1] * vuv_loss 

        return {"loss": loss}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, x):
        wav = x.squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        return wav

    def forward(self, batch):
        x = self.get_feature(batch['audio'])
        # This is mel-160 required by the pre-trained pitch detector
        mel_160 = self.mel_160_transform(x)
        mel_len = batch['mel_len']
        latent = batch["latent"]
        
        # This is pitch_detector logic.
        pd_output = self.pitch_pdt.forward(
            mel_160
        )  # use mel_160 signal
        f0 = pd_output[
            :, :, 0
        ]  # this output by default is in a log scale. See `recipes/umm/modules/pitch_predictor_task.compute_ground_truth_pitch()`
        vuv = pd_output[:, :, 1]
        vuv = pitch_utils.post_process_vuv_logits_to_binary_state(
            vuv
        )  # vuv is now a binary state like the output of RVMPE
        f0 = f0.unsqueeze(2)
        vuv = vuv.unsqueeze(2)

        f0_vuv_out = self.f0_vuv_head(latent)
        f0_out = f0_vuv_out[:, :, 0:1]
        vuv_out = f0_vuv_out[:, :, 1:]
        # sometimes f0_gt, vuv_gt is 1 timestep longer than f0_out, vuv_out
        f0_vuv_trim_len = pitch_utils.compute_min_lengths(
            f0_out, f0, axis=1
        )
        f0 = f0[:, :f0_vuv_trim_len, :]
        f0_out = f0_out[:, :f0_vuv_trim_len, :]
        vuv = vuv[:, :f0_vuv_trim_len, :]
        vuv_out = vuv_out[:, :f0_vuv_trim_len, :]
        metric_dict = self.get_metrics(f0, f0_out, vuv, vuv_out, mel_len)

        loss = metric_dict["loss"] * self.loss_weight

        output_dict = {
            "f0": f0,
            "f0_out": f0_out,
            "vuv": vuv,
            "vuv_out": vuv_out,
            "loss": loss,
            f"aux/loss_{self.task}": metric_dict["loss"]
        }         
        return output_dict


if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig()
    dummy_input = {
        "audio": torch.randn(2, config.sample_rate * 8),
        "latent": torch.randn(2, config.frame_rate * 8, config.hidden_size), #  [batch, time, hidden]
    }
    f0_head = F0_VUV_Head_Pitchpdt(
        config,
        takes=["audio", "latent"],
        provides=["f0", "vuv", "loss"],
        bypasses=[],
        task="f0_vuv",
        loss_weight=1.0,
        lr_ratio=1.0,
    )
    result = f0_head._compute(dummy_input)
    print(result["f0"].shape, result["vuv"].shape,
          result["f0_out"].shape, result["vuv_out"].shape)

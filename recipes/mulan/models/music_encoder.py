import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoProcessor

from recipes.mae.models.mut import PretrainedMuTWrapper, RMSNorm


class MusicEncoder(nn.Module):
    def __init__(self, pretrained_model, emb_dim: int = 128, sample_rate: int = 24000):
        super(MusicEncoder, self).__init__()
        processor = AutoProcessor.from_pretrained(pretrained_model)
        music_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_extract = {  # Use a dict to avoid auto convert fp16
            "mel": torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                # 400 (25ms) in AST with 16K.
                # 42.6ms if using 1024 in 24K.
                # 46.43ms in music tagging (1024, 22050Hz)
                n_fft=1024,
                # 10ms in AST with 16K .
                # 21.3ms if using 512 in 24K.
                # 23.2ms in music tagging task (512, 22050Hz)
                hop_length=240,
                f_min=0,
                f_max=sample_rate // 2,
                n_mels=128,
                window_fn=torch.hann_window,
                power=2.0,
                center=True,
                pad_mode="reflect",
            ),
            "db": torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
            "mel_spec_mean": torch.nn.Parameter(
                torch.tensor(-2.0645375), requires_grad=False
            ),
            "mel_spec_std": torch.nn.Parameter(
                torch.tensor(0.94333875), requires_grad=False
            ),
        }
        self.processor = processor
        self.music_model = music_model
        self.music_model.gradient_checkpointing_enable()
        self.music_linear = nn.Linear(768, emb_dim)
        # spec_aug:
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=192)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=48)

    def manually_to_device(self, device):
        for k, v in self.feat_extract.items():
            self.feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
        x = audio
        x = x - torch.mean(x, dim=1, keepdim=True)  # remove DC offset
        mel_spec = self.feat_extract["mel"](x.float())
        mel_spec = self.feat_extract["db"](mel_spec)
        mel_spec = (mel_spec - self.feat_extract["mel_spec_mean"]) / (
            self.feat_extract["mel_spec_std"] * 2
        )  # normalize with AST setting
        if spec_aug:
            mel_spec = self.fm(self.tm(mel_spec))
        mel_spec = mel_spec.to(x.dtype)
        mel_spec = F.pad(mel_spec, ((0, 1024 - mel_spec.shape[-1])))
        mel_spec = mel_spec.permute(0, 2, 1)
        outputs = self.music_model(input_values=mel_spec)
        last_hidden_state = outputs["last_hidden_state"]
        music_output = self.music_linear(last_hidden_state[:, 0, :])
        music_embed = F.normalize(music_output, p=2, dim=1)
        return music_embed


class MuTWrapper(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super(MuTWrapper, self).__init__()

        mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        mut = PretrainedMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type="cls",
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


def get_music_encoder(music_encoder="ast", emb_dim=128):
    sample_rate = 24000
    if music_encoder == "ast":
        return MusicEncoder(
            "MIT/ast-finetuned-audioset-10-10-0.4593", emb_dim, sample_rate
        )
    elif music_encoder == "mut":
        return MuTWrapper(emb_dim)
    else:
        raise NotImplementedError

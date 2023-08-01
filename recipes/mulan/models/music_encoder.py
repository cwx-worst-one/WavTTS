import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoProcessor

from recipes.mae.models.mut import MuT, PretrainedMuTWrapper, PretrainedMuTWrapper25hz,RMSNorm
from samantha.utils.flops_calculator import llama_calculator


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
    def __init__(self, emb_dim: int = 128, output_type="cls", seq_len=500):
        super(MuTWrapper, self).__init__()

        self.emb_dim = emb_dim
        if seq_len == 500:
            mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        elif seq_len == 250:
            mlp_head = nn.Sequential(RMSNorm(1280), nn.AvgPool2d((2, 1)), nn.Linear(1280, emb_dim))
        else:
            raise NotImplementedError(f"Seq len {seq_len} is not supported yet.")
        mut = PretrainedMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def flops_fn(self, batch_size):
        # only mut flops, ignore final projection
        mut = self.mut.mut
        flops = 0
        # add mut flops
        flops += llama_calculator(
            mut.num_layers,
            mut.hidden_size,
            mut.intermediate_size,
            0,  # no embedding layer
            250,  # seq_len after melspec plus [CLS]
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * mut.hidden_size * self.emb_dim
        return flops


class MuTWrapper25hz(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="cls", seq_len=250):
        super(MuTWrapper25hz, self).__init__()

        self.emb_dim = emb_dim
        if seq_len == 250:
            mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        else:
            raise NotImplementedError(f"Seq len {seq_len} is not supported yet.")
        mut = PretrainedMuTWrapper25hz(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=046400-loss_0=10-kaggle.pth",
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def flops_fn(self, batch_size):
        # only mut flops, ignore final projection
        mut = self.mut.mut
        flops = 0
        # add mut flops
        flops += llama_calculator(
            mut.num_layers,
            mut.hidden_size,
            mut.intermediate_size,
            0,  # no embedding layer
            250,  # seq_len after melspec plus [CLS]
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * mut.hidden_size * self.emb_dim
        return flops

class MuTinyWrapper(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="seq"):
        super(MuTinyWrapper, self).__init__()

        self.emb_dim = emb_dim
        mlp_head = nn.Sequential(RMSNorm(1280), nn.AvgPool2d((2, 1)), nn.Linear(1280, emb_dim))
        mut = PretrainedMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
            num_layers=16,
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def flops_fn(self, batch_size):
        # only mut flops, ignore final projection
        mut = self.mut.mut
        flops = 0
        # add mut flops
        flops += llama_calculator(
            mut.num_layers,
            mut.hidden_size,
            mut.intermediate_size,
            0,  # no embedding layer
            250,  # seq_len after melspec
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * mut.hidden_size * self.emb_dim
        return flops

class MuTinyWrapper25hz(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="seq"):
        super(MuTinyWrapper25hz, self).__init__()

        self.emb_dim = emb_dim
        mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        mut = PretrainedMuTWrapper25hz(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=046400-loss_0=10-kaggle.pth",
            num_layers=16,
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def flops_fn(self, batch_size):
        # only mut flops, ignore final projection
        mut = self.mut.mut
        flops = 0
        # add mut flops
        flops += llama_calculator(
            mut.num_layers,
            mut.hidden_size,
            mut.intermediate_size,
            0,  # no embedding layer
            250,  # seq_len after melspec
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * mut.hidden_size * self.emb_dim
        return flops

class MuTFinal25hz(nn.Module):
    def __init__(
        self,
        emb_dim=32,
    ):
        super(MuTFinal25hz, self).__init__()
        mut = MuT(
            spec_shape=(128, 1000),
            patch_shape=(128, 4),
            num_classes=1000,
            sample_rate=24000,
            dim=1280,
            depth=16,
            heads=16,
            dim_head=80,
            channels=1,
            mlp_dim=5120,
            checkpointing=True,
            use_flash_attn=False,
            output_type="seq",
        )
        mut.mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        self.emb_dim = emb_dim
        self.mut = mut

    def manually_to_device(self, device):
        for k, v in self.mut.logmel_frontend["logmel"].feat_extract.items():
            self.mut.logmel_frontend["logmel"].feat_extract[k] = v.to(device)

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    def flops_fn(self, batch_size):
        # only mut flops, ignore final projection
        mut = self.mut
        flops = 0
        # add mut flops
        flops += llama_calculator(
            mut.num_layers,
            mut.hidden_size,
            mut.intermediate_size,
            0,  # no embedding layer
            250,  # seq_len after melspec
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * mut.hidden_size * self.emb_dim
        return flops


def get_music_encoder(music_encoder="ast", emb_dim=128, output_type="cls", seq_len=500):
    sample_rate = 24000
    if music_encoder == "ast":
        print("using ast")
        return MusicEncoder(
            "MIT/ast-finetuned-audioset-10-10-0.4593", emb_dim, sample_rate
        )
    elif music_encoder == "mut":
        print("using mut 50hz")
        return MuTWrapper(emb_dim, output_type, seq_len)

    elif music_encoder == "mut-25HZ":
        print("using mut 25hz")
        return MuTWrapper25hz(emb_dim, output_type, 250)
    elif music_encoder == "mut-tiny":
        print("using mut tiny")
        return MuTinyWrapper(emb_dim)
    elif music_encoder == "mut-tiny-25HZ":
        print("using mut tiny")
        return MuTinyWrapper25hz(emb_dim)
    elif music_encoder == "mut-final-25HZ":
        print("using mut final 25hz")
        return MuTFinal25hz(emb_dim)
    else:
        raise NotImplementedError



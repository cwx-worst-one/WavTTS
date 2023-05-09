import torch
import torch.nn as nn
import yaml
from soundstream.models.vqgan_res import VQGAN
from soundstream.utils.utils import HParams
from transformers import AutoModel


def norm_waveform(waveform):
    return waveform * min(0.99 / waveform.abs().max(), 1)


def apply_along_dim(function, x, dim: int = 0):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=dim)], dim=dim)


class EncoderTransformBase(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        pass


class OldEncoderTransformBase(nn.Module):
    def __init__(self, sample_rate: int, n_output_frames: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_output_frames = n_output_frames

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        pass


def get_config_from_file(file):
    with open(file, "r") as f:
        hp = yaml.safe_load(f)
    hp = HParams(**hp)
    return hp


def remove_ddp_module(ckpt):
    from collections import OrderedDict

    new_dict = OrderedDict()
    for key in ckpt:
        new_key = key.replace("module.", "", 1)
        new_dict[new_key] = ckpt[key]
    return new_dict


class SoundStreamTransform(EncoderTransformBase):
    def __init__(
        self,
        sample_rate: int,
        config: str = (
            "/mnt/bn/audio-diffusion/pretrained_models"
            "/soundstream/config_libri_causal.yaml"
        ),
        ckpt_path: str = (
            "/mnt/bn/audio-diffusion/pretrained_models"
            "/soundstream/checkpoints/990k_ckpt.pyt"
        ),
        return_continuous_latents: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(sample_rate)
        self.sample_rate = sample_rate

        self.num_quantizers = 6
        self.codebook_size = 1024
        self.return_continuous_latents = return_continuous_latents

        self.hp = get_config_from_file(config).hparams

        assert self.hp.source_sample_rate == self.sample_rate, Exception(
            "SoundStream sample rate (24kHz) is incompatible with the configured sample"
            " rate"
        )

        self.model = VQGAN(self.hp)
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(remove_ddp_module(ckpt["G"]))
        self.model.eval()

    def encode(self, waveform):
        if self.model.training:
            print("SoundStream is in training mode, setting to eval!")
            self.model = self.model.eval()
        return self.model.encode(waveform)

    def quantize(self, x):
        if self.model.training:
            print("SoundStream is in training mode, setting to eval!")
            self.model = self.model.eval()
        quant_out, loss, quant_index = self.model.quant(x)
        if isinstance(quant_index, (tuple, list)):
            quant_index = torch.stack(quant_index, dim=1)  # [bs, n_codebook, t]
        else:
            quant_index = quant_index.unsqueeze(1)
        return quant_index

    def decode(self, codes):
        self.model.eval()
        if self.return_continuous_latents:
            quant_output = codes
        else:
            quant_output = self.model.get_quant_output_from_index(codes)
        return self.model.decode(quant_output)

    def forward(self, audio):
        self.model.eval()
        codes = self.quantize(self.encode(audio))
        if self.return_continuous_latents:
            codes = self.model.get_quant_output_from_index(codes)
        return codes

    # NOTE: left here for reference, this is what SoundStream used to preprocess audio:
    # def audio_preprocess(self, audio_path):
    #     wav, sr = librosa.load(audio_path, sr=None)
    #     if sr != self.hp.sample_rate:
    #         new_len = round(wav.shape[-1] * self.hp.sample_rate / sr)
    #         wav = resample(wav, new_len)
    #     abs_wav_max = np.max(np.abs(wav))
    #     if abs_wav_max > 1:
    #         wav = wav / abs_wav_max
    #     wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
    #     return wav


# TODO: For future work
class ArchiSound(EncoderTransformBase):
    def __init__(
        self, sample_rate: int, key: str = "dmae1d-ATC32-v3", latent_dim: int = 32
    ):
        super().__init__(sample_rate)
        self.model = AutoModel.from_pretrained(
            f"/mnt/bn/audio-diffusion/pretrained_models/archinetai/{key}",
            trust_remote_code=True,
        )

    def encode(self, waveform, sample_rate):
        assert waveform.dim() == 2 and waveform.size(0) == 2
        return self.model.encode(waveform.unsqueeze(0)).squeeze(0)

    def decode(self, embedding, steps=20):
        assert embedding.dim() == 3 and embedding.size(1) == self.latent_dim
        if self.key == "dmae1d-ATC32-v3":
            if embedding.size(2) > 1024:
                embedding = embedding[:, :, :1024]
            elif embedding.size(2) < 1024:
                embedding = torch.cat(
                    [embedding, torch.zeros(self.latent_dim, 1024 - embedding.size(2))],
                    dim=2,
                )
        waveform = self.model.decode(embedding, num_steps=steps)
        waveform = apply_along_dim(norm_waveform, waveform, 0)
        return waveform


if __name__ == "__main__":
    ss = SoundStreamTransform(sample_rate=24000)
    dummy_audio = torch.randn((2, 1, 240000))
    dummy_encoded = ss(dummy_audio)
    dummy_decoded = ss.decode(dummy_encoded)

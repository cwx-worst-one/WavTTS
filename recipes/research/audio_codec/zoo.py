from typing import Optional

import torch

from samantha.data.audio_utils import convert_audio, convert_audio_channels
from samantha.dataio.data_bucket import data_bucket
from samantha.models.base import LightningModuleBase
from samantha.utils.hdfs_tools import hdfs_torch_load


class AudioCodec(LightningModuleBase):

    def __init__(self):
        super().__init__()
        ckpt_path = data_bucket("models/nac/44.1k_stereo_128cd_49hz_kl1e-6.pt")
        self.generator = hdfs_torch_load(ckpt_path, jit=True)
        self.sample_rate = 44100
        self.latent_dim = 128
        self.frame_rate = 49
        self.mean = 0.0027362
        self.std = 0.175602

    def get_z(self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int = 120):
        assert audio.ndim == 3

        with torch.cuda.amp.autocast(enabled=False):
            if sample_rate != self.sample_rate:
                audio = convert_audio(
                    audio, sample_rate, self.sample_rate, audio.shape[1]
                )

        # chunked inference
        max_samples = sample_rate * chunk_seconds
        audio = audio.split(max_samples, dim=-1)
        zs = []
        for a in audio:
            z = self.generator.get_z(a, sample_rate)
            zs.append(z)
        zs = torch.cat(zs, dim=-1)

        zs = (zs - self.mean) / self.std
        return zs

    def decode_z(self, z: torch.Tensor, chunk_seconds: int = 120) -> torch.Tensor:
        z = (z * self.std) + self.mean

        max_frames = self.frame_rate * chunk_seconds
        zs = z.split(max_frames, dim=-1)
        pred_audio = []
        for z in zs:
            pred_audio.append(self.generator.decode(z))
        return torch.cat(pred_audio, dim=-1)  # [B, C, T]


class AudioCodecBase(LightningModuleBase):

    def __init__(
        self, ckpt_path: str, mean: Optional[float] = None, std: Optional[float] = None
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.mean = mean
        self.std = std

        if self.mean is None and self.std is None:
            self.normalize = False
        else:
            self.normalize = True

        self.codec = hdfs_torch_load(ckpt_path, jit=True, map_location="cpu")
        self.sample_rate = self.codec.sample_rate
        self.latent_dim = self.codec.latent_dim
        self.frame_rate = self.codec.frame_rate

        if hasattr(self.codec, "vae_beta"):
            self.vae_beta = self.codec.vae_beta

        if hasattr(self.codec, "n_channels"):
            self.n_channels = self.codec.n_channels

        if hasattr(self.codec, "hop_length"):
            self.hop_length = self.codec.hop_length

    def get_z(self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int = 480):
        assert audio.ndim == 3

        with torch.cuda.amp.autocast(enabled=False):
            if sample_rate != self.sample_rate:
                audio = convert_audio(
                    audio, sample_rate, self.sample_rate, self.n_channels
                )

            if audio.shape[1] != self.n_channels:
                audio = convert_audio_channels(audio, self.n_channels)

        zs = self.codec.get_z(audio, self.sample_rate, chunk_seconds=chunk_seconds)

        if self.normalize:
            zs = (zs + self.mean) / self.std
        return zs  # [B, D, T]

    def decode_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            z = (z * self.std) - self.mean
        return self.codec.decode_z(z)  # [B, C, T]

    def forward(
        self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int = 480
    ) -> torch.Tensor:
        return self.get_z(audio, sample_rate, chunk_seconds)


class AudioCodec_68b8441_32l(AudioCodecBase):

    def __init__(self):
        mean = 0.0887217
        std = 2.4461970
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-68b8441-step=1295000.ckpt-latent=32.pt"
        super().__init__(ckpt_path, mean, std)
        self.n_channels = 2


class AudioCodec_f9db856_32l(AudioCodecBase):

    def __init__(self):
        mean = 0.0602417
        std = 2.32578778
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-f9db856-step=1050000.ckpt-latent=32.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_e129ae2_32l(AudioCodecBase):

    def __init__(self):
        mean = 0.0
        std = 1.0
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-e129ae2-step=2000000.ckpt-latent=32.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_527ee6f_128l(AudioCodecBase):

    def __init__(self):
        mean = -0.040663
        std = 5.77683
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-527ee6f-step=1895000.ckpt-latent=128.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_1eee0ba_128l(AudioCodecBase):

    def __init__(self):
        mean = 0.009916
        std = 2.054546
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-1eee0ba-step=1560000.ckpt-latent=128.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_32aee0d_32l(AudioCodecBase):

    def __init__(self):
        mean = 0.0043274
        std = 0.3154755
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-32aee0d-step=335000.ckpt-latent=32.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_66618b7_64l(AudioCodecBase):

    def __init__(self):
        mean = 0.00457702
        std = 0.29215997
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-66618b7-step=690000.ckpt-latent=64.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_d29a86d_64l(AudioCodecBase):

    def __init__(self):
        mean = -0.01444156
        std = 0.29670456
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-d29a86d-step=900000.ckpt-latent=64.pt"
        # ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-d29a86d-step=1785000.ckpt-latent=64.pt" # TODO
        super().__init__(ckpt_path, mean, std)


class AudioCodec_8bfa192_64l(AudioCodecBase):

    def __init__(self):
        mean = 0.0
        std = 1.0
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-8bfa192-step=1580000.ckpt-latent=96.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_0f25752_64l(AudioCodecBase):

    def __init__(self):
        mean = 0.101882257
        std = 1.141777992
        ckpt_path = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/models/uac/uac-0f25752-step=705000.ckpt-latent=64.pt"
        super().__init__(ckpt_path, mean, std)


class AudioCodec_310f946_64l(AudioCodecBase):

    def __init__(self):
        ckpt_path = data_bucket("models/uac/uac-310f946-step=275000.ckpt-latent=64.pt")
        super().__init__(ckpt_path)


class AudioCodec_f81b3fa_64l(AudioCodecBase):

    def __init__(self):
        ckpt_path = data_bucket("models/uac/uac-f81b3fa-step=2000000.ckpt-latent=64.pt")
        super().__init__(ckpt_path)


class AudioCodec_7c355ea_64l(AudioCodecBase):

    # s
    def __init__(self):
        ckpt_path = data_bucket("models/uac/uac-7c355ea-step=2000000.ckpt-latent=64.pt")
        super().__init__(ckpt_path)

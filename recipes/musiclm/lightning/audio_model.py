import torch
#from encodec import EncodecModel as Encodec
from pytorch_lightning import LightningModule


class EncoderTransformBase(LightningModule):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        pass


class SoundStreamModel(EncoderTransformBase):
    def __init__(
        self,
        sample_rate: int,
        encoder_fp="/mnt/bn/audio-diffusion/pretrained_models/soundstream_v2/checkpoints/ss0.pt",  # noqa
        decoder_fp="/mnt/bn/audio-diffusion/pretrained_models/soundstream_v2/checkpoints/ss_decoder_0.pt",  # noqa
    ):
        super().__init__(sample_rate)
        self.sample_rate = sample_rate

        self.encoder = torch.jit.load(encoder_fp, map_location="cpu").eval()
        self.decoder = torch.jit.load(decoder_fp, map_location="cpu").eval()

        self.num_quantizers = 12
        self.codebook_size = 1024
        self.frame_rate = 50

    def encode(self, waveform):
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            _, _, quant = self.encoder(waveform.float())
        return torch.stack(quant, dim=1)

    def decode(self, x):
        if self.decoder.training:
            self.decoder.eval()
        return self.decoder(x)

    def forward(self, audio):
        self.encoder.eval()
        return self.encode(audio)


class EncodecModel(EncoderTransformBase):
    def __init__(self, sample_rate: int, target_bandwidth: float) -> None:
        super().__init__(sample_rate)

        if sample_rate == 24000:
            self.model = Encodec.encodec_model_24khz()
        elif sample_rate == 48000:
            self.model = Encodec.encodec_model_48khz()
        else:
            raise NotImplementedError(
                "This sample rate is not supported for EnCodec (24kHz/48kHz)"
            )
        self.model.set_target_bandwidth(target_bandwidth)
        self.num_quantizers = 16
        self.codebook_size = 1024

    def encode(self, waveform):
        return self.model.encode(waveform)

    def quantize(self, encoded_frames):
        return torch.cat(
            [encoded[0] for encoded in encoded_frames], dim=-1
        )  # [B, n_q, T]

    def decode(self, codes, scale: float = 0.05):
        # TODO: address scale
        if self.model.normalize:
            scale = torch.tensor((scale), device=codes.device)
        else:
            scale = None

        if self.model.training:
            self.model.eval()

        decoded = []
        for c in codes:
            decoded.append(self.model.decode([(c.unsqueeze(dim=0), scale)]))
        return torch.cat(decoded, dim=0)

    def forward(self, audio):
        self.model.eval()
        return self.quantize(self.encode(audio))

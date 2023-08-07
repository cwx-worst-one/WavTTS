import torch
from pytorch_lightning import LightningModule
from dac import DAC
from dac.utils import download




class DACModel(LightningModule):

    num_quantizers = 9
    frame_rate = 86
    codebook_size = 1024

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        assert sample_rate == 44100
        self.sample_rate = sample_rate

        if sample_rate == 16000:
            model_path = download(model_type="16khz")
        elif sample_rate == 24000:
            model_path = download(model_type="22khz")
        elif sample_rate == 44100:
            model_path = download(model_type="44khz")
        else:
            raise NotImplementedError(
                "This sample rate is not supported for DAC (16kHz/24kHz/44.1kHz)"
            )
        
        self.model = DAC.load(model_path).eval()

    def preprocess(self, audio, src_sample_rate: int):
        return self.model.preprocess(audio, src_sample_rate)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, src_sample_rate: int) -> torch.Tensor:
        self.model.eval()
        audio = self.preprocess(audio, src_sample_rate)
        return self.model.encode(audio)
    
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.dac.decode(x)

    def forward(self, audio: torch.Tensor, src_sample_rate: int) -> torch.Tensor:
        return self.encode(audio, src_sample_rate)

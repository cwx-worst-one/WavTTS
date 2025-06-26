import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import (
    AudioEncoder,
    ConformerRotaryPositionalEmbedding,
    ConformerEncoderLayer,
    )
from recipes.umm2.transforms.speech import SpeechTransform



class UMM(BaseStage):
    def __init__(
        self,
        config,       
        takes=["audio"],
        provides=["latent"],
        bypasses=[],
        lr_ratio=1.0,
        loss_weight=None,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight)

        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        self.encoder_layers = nn.ModuleList(
            [ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.audio_transform = SpeechTransform(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=0,
            f_max=config.sample_rate // 2,
            #return_phase=False,
        )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)

        self.config = config

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, batch):
        wav = batch['audio'].float() # squeeze(dim=1).
        if wav.dim() == 2:
            wav = wav.unsqueeze(dim=1)
        wav = self.pad_audio(wav)
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(wav, normalize=normalize)
        return mel

    def _compute(self, batch):
        mel = self.get_feature(batch)  
        feature = self.audio_encoder(mel)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        return {"latent": hidden_states}


if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    dummy_input = {
        'audio': torch.randn(4, 1, 24000 * 30),
        #'mel': torch.randn(2, 400, 128),
        }
    config = UMMConfig()
    output_dict = UMM(config)(dummy_input)
    print([(k, v.shape) for k, v in output_dict.items()])
    print(output_dict['latent'].shape)

    

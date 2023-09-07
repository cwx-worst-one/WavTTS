import torch
from torch import nn

from recipes.mir_benchmark.models.frontend import Frontend
from recipes.mir_benchmark.models.tagging import TaggingInstrumentGRUStage
from recipes.musicfm.models.best_rq import BEST_RQ
from samantha.core import BaseModel


class InstrumentTagging(nn.Module):
    def __init__(self, is_flash=False):
        super(InstrumentTagging, self).__init__()
        self.model = self.load_model(is_flash)
        self.instrument_tags = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ensemble",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Synth Effects",
            "Synth Lead",
            "Synth Pad",
            "Vocal",
        ]

    def load_model(self, is_flash):
        # model
        foundation_model = BEST_RQ(
            codebook_dim=16,
            codebook_size=8192,
            hop_length=240,
            n_fft=2047,
            n_mels=128,
            conv_dim=512,
            encoder_dim=1024,
            encoder_depth=24,
            mask_hop=0.4,
            mask_prob=0.6,
            is_flash=is_flash,
            stat_path="/mnt/bn/audio-diffusion/pretrained_models/musicfm/playlist_stats.json",
            model_path="/mnt/bn/audio-diffusion/pretrained_models/musicfm/bestrq25_330k.pt",
        )
        frontend = Frontend(model=foundation_model, layer_ix=24, is_update=True)
        backend = TaggingInstrumentGRUStage(n_channel=1024)
        model = BaseModel(
            input_names=["audio"],
            output_names=["instrument_pred"],
            stages=[frontend, backend],
        )

        # load checkpoint
        S = torch.load(
            "/mnt/bn/audio-diffusion/pretrained_models/unified_tagging/instrument_only.pt"
        )["state_dict"]
        SS = {k[6:]: v for k, v in S.items()}
        model.load_state_dict(SS)

        return model

    def predict(self, x):
        """
        x (torch.Tensor): input shape with (batch, 1, sample length)
        out (torch.Tensor): output shape with (batch, 17)
        """

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)[0]
        return out

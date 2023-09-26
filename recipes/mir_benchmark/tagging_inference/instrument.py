import torch
from torch import nn

from recipes.mir_benchmark.models.frontend import Frontend
from recipes.mir_benchmark.models.tagging import TaggingInstrumentGRUStage
from recipes.mir_benchmark.utils.utils import get_tags
from recipes.musicfm.models.best_rq import BEST_RQ
from samantha.core import BaseModel

INSTRUMENT_TAGS = [
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

INSTRUMENT_TAGS_THRESHOLDS = [
    0.3,
    0.5,
    0.3,
    0.5,
    0.2,
    0.5,
    0.5,
    0.4,
    0.4,
    0.2,
    0.2,
    0.2,
    0.4,
    0.2,
    0.2,
    0.3,
    0.0,    # will always output "Vocal" -> side effect of training data
]
class InstrumentTagging(nn.Module):
    def __init__(self, is_flash=False):
        super(InstrumentTagging, self).__init__()
        self.model = self.load_model(is_flash)
        self.instrument_tags = INSTRUMENT_TAGS
        # Manually tuned to maximize per-class F1 scores
        thresholds = torch.Tensor(INSTRUMENT_TAGS_THRESHOLDS)
        self.register_buffer('thresholds', thresholds, persistent=False)
        assert len(self.instrument_tags) == len(self.thresholds)

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

    @torch.no_grad()
    def predict(self, x):
        """
        Input:
            x (torch.Tensor): input shape with (batch, 1, sample length)
        Output:
            tags (list[list[str]]): list of tags for each example in batch, always on CPU
            probs (torch.Tensor): output shape with (batch, n_classes)
        """
        self.model.eval()
        probs = self.model(x)[0]
        tags = get_tags(probs, self.instrument_tags, self.thresholds)
        return tags, probs

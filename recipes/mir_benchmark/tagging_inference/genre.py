import torch
from torch import nn

from recipes.mir_benchmark.models.frontend import Frontend
from recipes.mir_benchmark.models.tagging import TaggingGenreGRUStage
from recipes.mir_benchmark.utils.utils import get_tags
from recipes.musicfm.models.best_rq import BEST_RQ
from samantha.core import BaseModel

GENRE_TAGS = [
    "pop",
    "rock",
    "electronic",
    "hiphop_rap",
    "reggae",
    "rnb_soul",
    "Metal",
    "Jazz",
    "Blues",
    "Country",
    "Folk",
    "Indie",
    "K_pop",
    "Indie_Pop",
    "Muslim",
    "Indo_Christian",
    "Bollywood",
    "Bollywood_Retro",
    "Urban_Punjabi_Pop",
    "Tamil_Film_Music",
    "Kannada_Film_Music",
    "Telugu_Film_Music",
    "Indian_Independent",
    "Malayalam_Film_Music",
    "Sertanejo",
    "Baile_Funk",
    "Gospel",
    "Samba",
    "Pagode",
    "MPB",
    "Forro",
    "Axe",
    "Reggaeton",
    "Brazilian_Punk",
]

GENRE_TAGS_THRESHOLDS = [
    0.6,
    0.5,
    0.3,
    0.4,
    0.3,
    0.2,
    0.3,
    0.2,
    0.3,
    0.6,
    0.3,
    0.2,
    0.7,
    0.2,
    0.1,
    0.9,
    0.7,
    0.3,
    0.7,
    0.7,
    0.2,
    0.3,
    0.1,
    0.3,
    0.7,
    0.9,
    0.6,
    0.2,
    0.7,
    0.3,
    0.4,
    0.6,
    0.5,
    0.2,
]
class GenreTagging(nn.Module):
    def __init__(self, is_flash=False):
        super(GenreTagging, self).__init__()
        self.model = self.load_model(is_flash)
        self.genre_tags = GENRE_TAGS
        # Manually tuned to maximize per-class F1 scores
        thresholds = torch.Tensor(GENRE_TAGS_THRESHOLDS)
        self.register_buffer('thresholds', thresholds, persistent=False)
        assert len(self.genre_tags) == len(self.thresholds)

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
        backend = TaggingGenreGRUStage(n_channel=1024)
        model = BaseModel(
            input_names=["audio"],
            output_names=["genre_pred"],
            stages=[frontend, backend],
        )

        # load checkpoint
        S = torch.load(
            "/mnt/bn/audio-diffusion/pretrained_models/unified_tagging/genre_only.pt"
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
        tags = get_tags(probs, self.genre_tags, self.thresholds.to(probs.device))
        return tags, probs

import torch
from torch import nn

from recipes.mir_benchmark.models.frontend import Frontend
from recipes.mir_benchmark.models.tagging import TaggingGenreGRUStage
from recipes.musicfm.models.best_rq import BEST_RQ
from samantha.core import BaseModel


class GenreTagging(nn.Module):
    def __init__(self, is_flash=False):
        super(GenreTagging, self).__init__()
        self.model = self.load_model(is_flash)
        self.genre_tags = [
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

    def predict(self, x):
        """
        x (torch.Tensor): input shape with (batch, 1, sample length)
        out (torch.Tensor): output shape with (batch, 34)
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)[0]
        return out

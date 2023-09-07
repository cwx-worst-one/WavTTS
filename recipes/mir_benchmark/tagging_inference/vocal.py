import torch
from torch import nn

from recipes.mir_benchmark.models.frontend import Frontend
from recipes.mir_benchmark.models.tagging import TaggingVocalGRUStage
from recipes.musicfm.models.best_rq import BEST_RQ
from samantha.core import BaseModel


class VocalTagging(nn.Module):
    def __init__(self, is_flash=False):
        super(VocalTagging, self).__init__()
        self.model = self.load_model(is_flash)
        self.vocal_tags = [
            "age_中老年",  # age_old
            "age_中青年",  # age_middle_aged
            "age_幼年",  # age_child
            "age_青年",  # age_young
            "gender_NO",  # gender_no
            "gender_女",  # gender_female
            "gender_男",  # gender_male
            "style_低沉和蔼",  # style_low_and_warm
            "style_厚实低沉",  # style_thick_and_deep
            "style_嘹亮自信",  # style_loud_and_confident
            "style_成熟明亮",  # style_bright
            "style_成熟磁性",  # style_husky
            "style_明亮细腻",  # style_delicate
            "style_淘气萌娃",  # style_playful
            "style_甜美温柔",  # style_sweet
            "style_磁性慵懒",  # style_relaxed
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
        backend = TaggingVocalGRUStage(n_channel=1024)
        model = BaseModel(
            input_names=["audio"],
            output_names=["vocal_pred"],
            stages=[frontend, backend],
        )

        # load checkpoint
        S = torch.load(
            "/mnt/bn/audio-diffusion/pretrained_models/unified_tagging/vocal_only.pt"
        )["state_dict"]
        SS = {k[6:]: v for k, v in S.items()}
        model.load_state_dict(SS)

        return model

    def predict(self, x):
        """
        x (torch.Tensor): input shape with (batch, 1, sample length)
        out (torch.Tensor): output shape with (batch, 16)
        """

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)[0]
        return out

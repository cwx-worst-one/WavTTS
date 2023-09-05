import torch.nn as nn
import torch.nn.functional as F

from recipes.mae.models.mut import PretrainedMuTWrapper, RMSNorm, FreeMuTWrapper


class MuTWrapper(nn.Module):
    def __init__(self, emb_dim: int = 128, output_type="cls"):
        super(MuTWrapper, self).__init__()

        self.emb_dim = emb_dim
        mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        mut = PretrainedMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class DoubleMuTWrapper(nn.Module):
    def __init__(self, emb_dim: int = 2048, output_type="cls"):
        super(DoubleMuTWrapper, self).__init__()
        self.emb_dim = emb_dim
        mlp_head = nn.Sequential(RMSNorm(1280), nn.Linear(1280, emb_dim))
        mut = FreeMuTWrapper(
            output_layer=mlp_head,
            checkpointing=True,
            use_flash_attn=True,
            output_type=output_type,
            pretained_path="mutmae-step=177600-loss_1=5-sf.pth",
        )
        self.mut = mut

    def forward(self, audio, spec_aug=False):
        emb = self.mut(audio, spec_aug=spec_aug)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


def get_music_encoder(music_encoder="mut", emb_dim=128, output_type="cls"):
    if music_encoder == "mut":
        print("using mut 50hz")
        return MuTWrapper(emb_dim, output_type)
    
    elif music_encoder == "double-mut":
        print("using double audio")
        return DoubleMuTWrapper(emb_dim, output_type)
    else:
        raise NotImplementedError

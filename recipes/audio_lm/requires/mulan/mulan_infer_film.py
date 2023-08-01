import itertools
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from recipes.mulan.modules.pl_module import LitMuLanModule

def create_mulan_model(ckpt_path, device):
    litmodel = LitMuLanModule.load_from_checkpoint(ckpt_path)

    # audio tower
    litmodel.music_encoder.eval()
    litmodel.music_encoder.to(device)
    litmodel.music_encoder.mut.manually_to_device(device)
    # text tower
    litmodel.text_encoder.eval()
    litmodel.text_encoder.to(device)

    return litmodel


@torch.no_grad()
def mulan_inference(
    model, text=None, music=None, device="cpu", avg=True, shift_seconds=1
):
    assert (text is not None) ^ (
        music is not None
    ), "text inputs and music input can only select one"

    if text is not None:
        emb = model.encode_text(text)

    if music is not None:
        # print(f"mulan_inference: wav shape is {music.shape}")
        music_encoder = model.music_encoder
        music = music.unfold(1, 24000 * 10, 24000 * shift_seconds)  # [b, n, t]
        # print(f"mulan_inference: unfolded wav shape is {music.shape}")
        b, n, t = music.shape
        music = music.reshape(b * n, t)
        # print(f"mulan_inference: reshaped wav shape is {music.shape}")
        emb = music_encoder(music.unsqueeze(1))
        # print(f"mulan_inference: embe shape is {emb.shape}, b={b}, n={n}, t={t}")
        emb = emb.reshape(b, n, -1)
        if avg:
            # print("averaging embeds")
            emb = F.normalize(emb.mean(dim=1), p=2, dim=1)
    return emb


def mulan_rvq_indexs(z, centers):
    # z: [b, d]
    # center: [n, d]
    indexs = []
    ds = []
    for center in centers:
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(center**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z, rearrange(center, "n d -> d n"))
        )
        min_index = torch.argmin(d, dim=1)  # [b, ]

        # RVQ
        z = z - torch.nn.functional.embedding(min_index, center)

        indexs.append(min_index)
        ds.append(d.min())
    indexs = torch.stack(indexs, dim=1)  # [b, 12]
    return indexs, ds

ckpt = "/mnt/bn/mm-data/projects/mulan/ckpts/mulan-51/mulan-step=003300-median_rank_0=51-kaggle.ckpt"
film_module = LitMuLanModule.load_from_checkpoint(ckpt).eval()
print()
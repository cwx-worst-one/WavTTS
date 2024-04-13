import os
import sys

import pytorch_lightning as pl
import torch
from einops import rearrange


class Engine(pl.LightningModule):
    def __init__(self, model_cls, args):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt

        print("Initializing model....")
        # Init model here so that we can save hyperparameter normally
        self.model = model_cls(args)

        self.args = args
        self._accumulate_embed_batches = args.accumulate_embed_batches
        self.automatic_optimization = self._accumulate_embed_batches == 0
        self.temperature = args.temperature

    def forward(
        self, input_ids, attention_mask, token_type_ids, music_audio, spec_aug=False
    ):
        # in lightning, forward defines the prediction/inference actions

        text_output, music_output = self.model(
            input_ids, attention_mask, token_type_ids, music_audio, spec_aug
        )

        return text_output, music_output

    def _shared_step(self, batch, spec_aug=False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        music_audio = batch["audio"]

        text_vec, music_vec = self(
            input_ids, attention_mask, token_type_ids, music_audio, spec_aug=spec_aug
        )

        return {"text_vec": text_vec, "music_vec": music_vec}


def create_mulan_model(ckpt_path, device):
    # for load model of model_pl
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    engine = Engine.load_from_checkpoint(ckpt_path)
    for k, v in engine.model.music_encoder.feat_extract.items():
        engine.model.music_encoder.feat_extract[k] = v.to(device)
    engine.to(device)
    engine.eval()

    return engine


@torch.no_grad()
def mulan_inference(
    model, text=None, music=None, device="cpu", avg=True, shift_seconds=5
):
    assert (text is not None) ^ (
        music is not None
    ), "text inputs and music input can only select one"

    if text is not None:
        text_encoder = model.model.text_encoder
        emb = text_encoder.encode(text)

    if music is not None:
        music_encoder = model.model.music_encoder
        music = music.unfold(1, 24000 * 10, 24000 * shift_seconds)  # [b, n, t]
        b, n, t = music.shape
        music = music.reshape(b * n, t)
        emb = music_encoder(music)
        emb = emb.reshape(b, n, -1)
        if avg:
            emb = emb.mean(dim=1)
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

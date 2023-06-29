import itertools
from collections import defaultdict

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from einops import rearrange
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch import nn

from recipes.mulan.models.music_encoder import get_music_encoder
from recipes.mulan.models.text_encoder import get_text_encoder


class LitMuLanModule(pl.LightningModule):
    def __init__(
        self,
        music_encoder,
        text_encoder,
        emb_dim,
        spec_aug,
        lr,
        weight_decay,
        temperature,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        self.music_encoder = get_music_encoder(music_encoder, emb_dim)
        self.text_encoder = get_text_encoder(text_encoder, emb_dim)
        self.spec_aug = spec_aug
        self.lr = lr
        self.weight_decay = weight_decay
        if temperature == "learnable":
            self.temperature = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
            )
        else:
            self.temperature = torch.log(torch.tensor(1 / temperature))

        # Validation outputs
        self.val_outputs = dict()

    def on_fit_start(self):
        self.music_encoder.mut.manually_to_device(self.device)

    def on_predict_start(self):
        self.text_encoder.cpu()  # save gpu memory
        self.music_encoder.manually_to_device(self.device)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            base_params = []
            norm_params = []
            no_decay = [
                "temperature",
                "bn",
                "bias",
                "LayerNorm",
                "embeddings",
                "layernorm",
                "norm.bias",
                "norm.weight",
                "rotary",
            ]
            for name, param in self.named_parameters():  # self.parameters()
                _found = False
                for k in no_decay:
                    if k in name:
                        norm_params.append(param)
                        _found = True
                        break
                if not _found:
                    base_params.append(param)

            optimizer = torch.optim.AdamW(
                [{"params": base_params}, {"params": norm_params, "weight_decay": 0.0}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        return optimizer

    def _multiview_loss(self, text_vec, music_vec, music_id):
        # Size of text_vec: (batch_size * world_size, embed_dim),
        # eg: (8*4, 128)=(32,128)
        # Size of music_vec: (batch_size * world_size, embed_dim),
        # eg: (8*4, 128)
        # Slice text embed based on self.global_rank
        # Then calculate the dot product between text and music
        # Size of dot_product: (batch_size, batch_size * world_size), eg: (8, 32)
        batch_size = text_vec.shape[0] // self.trainer.world_size
        start = self.global_rank * batch_size
        end = start + batch_size
        dot_product = torch.mm(text_vec[start:end, :], music_vec.t())

        # Gather all the dot products from all the GPUs
        # Size of dot_product: (batch_size * world_size, batch_size * world_size),
        # eg: (32, 32)
        dot_product = rearrange(
            self.all_gather(dot_product, sync_grads=True), "w b d -> (w b) d"
        )
        dot_product = dot_product * self.temperature.exp()
        dot_product = dot_product - dot_product.max()
        # Calculate exp(dot_product/temperature)
        loss_mat = torch.exp(dot_product)

        # Collect music id
        info_dict = defaultdict(list)
        for i, item in enumerate(music_id.tolist()):
            info_dict[item].append(i)

        false_negative = list()
        for kk in info_dict:
            false_negative.extend(itertools.product(info_dict[kk], repeat=2))

        select_negative = torch.ones_like(loss_mat)

        for item in false_negative:
            select_negative[item[0], item[1]] = 0

        negative_mat = loss_mat * select_negative
        negative_mat[~select_negative.bool()] = negative_mat[
            ~select_negative.bool()
        ].detach()

        loss = (loss_mat.diagonal()) / (
            loss_mat.diagonal()
            + negative_mat.sum(dim=1)
            + negative_mat.sum(dim=0)
            + torch.finfo(loss_mat.dtype).tiny
        )

        loss = torch.mean(-torch.log(loss + torch.finfo(loss.dtype).tiny))
        self.log("tr_loss", loss, prog_bar=True)
        return loss

    def _hit_score_rank(self, source_embed, target_embed):
        assert source_embed.shape[1] == target_embed.shape[1]
        sample_size = source_embed.shape[0]
        mat = torch.matmul(source_embed, target_embed.T)
        smat, indx = mat.sort(dim=1, descending=True)
        score, ranklst = list(), list()
        for i in range(sample_size):
            rank = (indx[i] == i).nonzero(as_tuple=True)[0]
            ranklst.append(rank)
            score.append((sample_size - rank) / sample_size)

        median_rank = sorted(ranklst)[len(ranklst) // 2]

        return {"hit_score": sum(score) / len(score), "median_rank": median_rank}

    def training_step(self, batch, batch_idx):
        # Combine multiple dataloader batches into one batch

        # print("\n\n")

        # print(batch.keys())

        text_vec, music_vec = self._shared_step(batch, spec_aug=self.spec_aug).values()
        text_vec = rearrange(
            self.all_gather(text_vec, sync_grads=True), "w b d -> (w b) d"
        )
        music_vec = rearrange(
            self.all_gather(music_vec, sync_grads=True), "w b d -> (w b) d"
        )
        music_id = rearrange(
            self.all_gather(batch["music_id"], sync_grads=True), "w b -> (w b) "
        )
        
        loss = self._multiview_loss(text_vec, music_vec, music_id)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # print("\n\n")

        # print(batch.keys())
        result = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(result)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            text_vecs = []
            music_vecs = []
            for output in outputs:
                text_vecs += output["text_vec"]
                music_vecs += output["music_vec"]

            text_vecs = torch.stack(text_vecs)
            music_vecs = torch.stack(music_vecs)
            text_vecs = rearrange(self.all_gather(text_vecs), "w b d -> (w b) d")
            music_vecs = rearrange(self.all_gather(music_vecs), "w b d -> (w b) d")

            rank_zero_info(
                f"text_vecs.shape: {text_vecs.shape}, "
                f"music_vecs.shape: {music_vecs.shape}"
            )

            score = self._hit_score_rank(text_vecs, music_vecs)
            self.log_dict(
                {
                    f"median_rank_{dataloader_idx}": score["median_rank"],
                    f"hit_score_{dataloader_idx}": score["hit_score"],
                },
                prog_bar=True,
            )
            self.val_outputs[dataloader_idx] = []

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        music_embed = self.music_encoder(batch["audio"])
        return {"music_vec": music_embed}

    def _shared_step(self, batch, spec_aug=False):
        music_embed = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=spec_aug)
        text_embed = self.text_encoder(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        return {"text_vec": text_embed, "music_vec": music_embed}

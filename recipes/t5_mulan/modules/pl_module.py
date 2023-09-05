import itertools
from collections import defaultdict

import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_info
import torch.distributed as dist

from recipes.t5_mulan.models.music_encoder import get_music_encoder
from recipes.t5_mulan.models.text_encoder import get_text_encoder
from recipes.mulan.modules.gather import GatherLayer
from recipes.musiclm.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine


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
        top_text_config=None,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        output_type = "cls"
        self.music_encoder = get_music_encoder(music_encoder, emb_dim, output_type)
        self.text_encoder = get_text_encoder(text_encoder, emb_dim, config=top_text_config)
        self.temperature = torch.log(torch.tensor(1 / temperature))
        # Validation outputs
        self.val_outputs = []

    def on_fit_start(self):        
        self.music_encoder.mut.manually_to_device(self.device)
        if self.hparams.text_encoder in ["llama", "t5"]:
            self.text_encoder.manually_to_device(self.device)

    def configure_optimizers(self):
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
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue 
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
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = WarmupCosine(
            optimizer,
            init_lr=self.hparams.lr,
            warmup_steps=500,
            cycle_steps=10000,
            min_lr=self.hparams.lr * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _ssl_constrast_loss(self, logits, music_id):
        loss_mat = torch.exp(logits)

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
        return loss
    
    def _prepare_global_logits(self, text_vec, music_vec):
        r"""Prepare logits for global loss calculation."""
        dot_product = torch.mm(text_vec, music_vec.t())
        dot_product = rearrange(
            self.all_gather(dot_product, sync_grads=True), "w b d -> (w b) d"
        )
        dot_product = dot_product * self.temperature.exp()
        logits = dot_product - dot_product.max()
        return logits

    def training_step(self, batch, batch_idx):
        music_vec = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=self.hparams.spec_aug)
        text_vec = self.text_encoder(
            batch["text_embeds"], batch["text_embeds_mask"]
        )
        music_id = batch["music_id"]
        music_vec = rearrange(
            self.all_gather(music_vec, sync_grads=True), "w b d -> (w b) d"
        )
        music_id = rearrange(
            self.all_gather(music_id, sync_grads=True), "w b -> (w b) "
        )
        logits = self._prepare_global_logits(text_vec, music_vec)

        # Compute loss
        loss = self._ssl_constrast_loss(logits, music_id)

        self.log("tr_loss", loss, prog_bar=True)
        return {"loss": loss}

    def _hit_score_rank(self, source_embed, target_embed):
        r"""Compute hit score and rank."""
        mat = torch.matmul(source_embed, target_embed.T)
        sample_size = mat.shape[0]
        rank_zero_info(f"mat shape: {mat.shape}")
        smat, indx = mat.sort(dim=1, descending=True)
        score, ranklst = list(), list()
        for i in range(sample_size):
            rank = (indx[i] == i).nonzero(as_tuple=True)[0]
            ranklst.append(rank)
            score.append((sample_size - rank) / sample_size)

        median_rank = sorted(ranklst)[len(ranklst) // 2]

        return {"hit_score": sum(score) / len(score), "median_rank": median_rank}

    def validation_step(self, batch, batch_idx):
        music_vec = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=self.hparams.spec_aug)
        text_vec = self.text_encoder(
            batch["text_embeds"], batch["text_embeds_mask"]
        )
        aspect_list_vec = self.text_encoder(
            batch["aspect_list_embeds"], batch["aspect_list_embeds_mask"]
        )
        self.val_outputs.append({
            "music_vec": music_vec,
            "text_vec": text_vec,
            "aspect_list_vec": aspect_list_vec,
        })

    def on_validation_epoch_end(self):
        music_vecs = []
        text_vecs = []
        aspect_list_vecs = []
        for output in self.val_outputs:
            music_vecs += output["music_vec"]
            text_vecs += output["text_vec"]
            aspect_list_vecs += output["aspect_list_vec"]
        music_vecs = torch.stack(music_vecs)
        text_vecs = torch.stack(text_vecs)
        aspect_list_vecs = torch.stack(aspect_list_vecs)

        # Compute hit score and rank
        score_0 = self._hit_score_rank(aspect_list_vecs, music_vecs)
        score_1 = self._hit_score_rank(text_vecs, music_vecs)
        self.log_dict(
            {
                f"median_rank_0": score_0["median_rank"],
                f"median_rank_1": score_1["median_rank"],
                f"hit_score_0": score_0["hit_score"],
                f"hit_score_1": score_1["hit_score"],
            },
            sync_dist=True,
            prog_bar=True,
        )
        self.val_outputs = []
    
    def all_gather(self, tensor, sync_grads=False):
        if sync_grads:
            return GatherLayer.apply(tensor)
        else:
            output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output, tensor)
            return output

import itertools
from collections import defaultdict

import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_info
from torch import nn
import torch.distributed as dist

from recipes.mulan.models.music_encoder import get_music_encoder
from recipes.mulan.models.text_encoder import get_text_encoder
from recipes.mulan.modules.gather import GatherLayer
from recipes.mulan.modules.llama_model import LlamaConfig
from recipes.musiclm.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.utils.model_metric import ModelMetric


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
        music_seq_len: int = 500,
        use_supcontrast: bool = True,
        use_flatclr: bool = True,
        use_fine: bool = False,
        gather_batches: bool = True,
        model_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        output_type = "seq" if use_fine else "cls"
        self.music_encoder = get_music_encoder(music_encoder, emb_dim, output_type, music_seq_len)
        self.text_encoder = get_text_encoder(text_encoder, emb_dim, output_type, model_path)
        self.temperature = torch.log(torch.tensor(1 / temperature))
        # Validation outputs
        self.val_outputs = dict()

    def on_fit_start(self):        
        if self.hparams.music_encoder == "mut-final-25HZ":
            self.music_encoder.manually_to_device(self.device)
        else:
            self.music_encoder.mut.manually_to_device(self.device)
        if self.hparams.text_encoder in ["llama", "t5"]:
            self.text_encoder.manually_to_device(self.device)

        # self.music_encoder.mut.manually_to_device(self.device)

    def on_predict_start(self):
        self.text_encoder.cpu()  # save gpu memory
        self.music_encoder.manually_to_device(self.device)

    def setup(self, stage: str):
        # Variables for MFU calculation
        self.metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs={
                "music_tower": self.music_encoder,
                # "text_tower": self.text_encoder,
            },
        )

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
            cycle_steps=50000,
            min_lr=self.hparams.lr * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
    
    def _sup_constrast_loss(self, logits, music_id):
        music_id = music_id.view(-1, 1)
        mask = torch.eq(music_id, music_id.T)
        if self.hparams.use_flatclr:
            exp_logits = torch.exp(logits) * (~mask)
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        return loss

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

        if self.hparams.use_flatclr:
            loss = (loss_mat.diagonal()) / (
                negative_mat.sum(dim=1)
                + negative_mat.sum(dim=0)
                + torch.finfo(loss_mat.dtype).tiny
            )
        else:
            loss = (loss_mat.diagonal()) / (
                loss_mat.diagonal() 
                + negative_mat.sum(dim=1)
                + negative_mat.sum(dim=0)
                + torch.finfo(loss_mat.dtype).tiny
            )
        loss = torch.mean(-torch.log(loss + torch.finfo(loss.dtype).tiny))
        return loss

    def _prepare_local_logits(self, text_vec, music_vec):
        r"""Prepare logits for local loss calculation."""
        dot_product = torch.mm(text_vec, music_vec.t())
        dot_product = dot_product * self.temperature.exp()
        logits = dot_product - dot_product.max()
        return logits
    
    def _prepare_global_logits(self, text_vec, music_vec):
        r"""Prepare logits for global loss calculation."""
        # Size of text_vec: (batch_size, embed_dim),
        # Size of music_vec: (world_size * batch_size, embed_dim),
        dot_product = torch.mm(text_vec, music_vec.t())

        # Gather all the dot products from all the GPUs
        # Size of dot_product: (batch_size * world_size, batch_size * world_size),
        # eg: (32, 32)
        dot_product = rearrange(
            self.all_gather(dot_product, sync_grads=True), "w b d -> (w b) d"
        )
        dot_product = dot_product * self.temperature.exp()
        logits = dot_product - dot_product.max()
        return logits
    
    def _fine_sim_matrix(self, text_vec, music_vec):
        r"""Prepare logits for fine loss calculation."""
        # Ref: https://arxiv.org/abs/2111.07783
        # Size of text_vec: (batch_size, text_seq_len, embed_dim),
        # Size of music_vec: (world_size * batch_size, music_seq_len, embed_dim),
        music_vec_t = rearrange(music_vec, "b s d -> b d s")

        # For loop, (we cannot use torch.matmul on full batch)
        score1 = []
        score2 = []
        for i in range(len(text_vec)):
            dot_product = torch.matmul(text_vec[i:i+1], music_vec_t)
            s1 = torch.mean(torch.max(dot_product, dim=-1)[0], dim=-1)
            s2 = torch.mean(torch.max(dot_product, dim=-2)[0], dim=-1)
            score1.append(s1)
            score2.append(s2)
        score1 = torch.stack(score1)
        score2 = torch.stack(score2)
        # Gather all the scores from all the GPUs
        score1 = rearrange(
            self.all_gather(score1, sync_grads=True), "b w d -> (b w) d"
        )
        score2 = rearrange(
            self.all_gather(score2, sync_grads=True), "b w d -> (b w) d"
        )

        # Sum up the scores
        score = score1 + score2
        return score

    def _prepare_fine_logits(self, text_vec, music_vec):
        r"""Prepare logits for fine loss calculation."""
        score = self._fine_sim_matrix(text_vec, music_vec)
        score = score * self.temperature.exp()
        logits = score - score.max()
        return logits

    def training_step(self, batch, batch_idx):
        text_vec, music_vec = self._shared_step(batch, spec_aug=self.hparams.spec_aug).values()
        music_id = batch["music_id"]
        if self.hparams.use_fine:
            # fine loss, only gather music_vec and music_id,
            # text_vec will be gathered in _fine_sim_matrix
            music_vec = rearrange(
                self.all_gather(music_vec, sync_grads=True), "w b s d -> (w b) s d"
            )
            music_id = rearrange(
                self.all_gather(music_id), "w b -> (w b) "
            )
            logits = self._prepare_fine_logits(text_vec, music_vec)
        elif self.hparams.gather_batches:
            music_vec = rearrange(
                self.all_gather(music_vec, sync_grads=True), "w b d -> (w b) d"
            )
            music_id = rearrange(
                self.all_gather(music_id, sync_grads=True), "w b -> (w b) "
            )
            logits = self._prepare_global_logits(text_vec, music_vec)
        else:
            logits = self._prepare_local_logits(text_vec, music_vec)    

        # Compute loss
        if self.hparams.use_supcontrast:
            loss = self._sup_constrast_loss(logits, music_id)
        else:
            loss = self._ssl_constrast_loss(logits, music_id)

        self.log("tr_loss", loss, prog_bar=True)
        return {"loss": loss}

    def _hit_score_rank(self, source_embed, target_embed):
        r"""Compute hit score and rank."""
        if self.hparams.use_fine:
            # fine sim, only gather music_vec, 
            # text_vec will be gathered in _fine_sim_matrix
            target_embed = rearrange(self.all_gather(target_embed), "w b s d -> (w b) s d")
            mat = self._fine_sim_matrix(source_embed, target_embed)
        else:
            source_embed = rearrange(self.all_gather(source_embed), "w b d -> (w b) d")
            target_embed = rearrange(self.all_gather(target_embed), "w b d -> (w b) d")
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
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
            batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids", None)
        )

        # mfu calculation
        batch_size, seq_len = batch["input_ids"].size()[:2]
        num_tokens = batch_size * seq_len  # text token only
        self.metric.update(
            num_tokens=num_tokens,
            stage=self.trainer.state.stage,
            model_kwargs={
                "music_tower": {"batch_size": batch_size},
                # "text_tower": {"batch_size": batch_size, "seq_len": seq_len},
            }
        )

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.metric.compute(step=self.trainer.global_step)
            self.log_dict(metric, sync_dist=True)
            # calculate text length (min, max, mean)
            text_len = batch["attention_mask"].sum(dim=1).float()
            self.log_dict(
                {
                    "text_len/min": text_len.min(),
                    "text_len/max": text_len.max(),
                    "text_len/mean": text_len.mean(),
                },
                sync_dist=True,
            )

        return {"text_vec": text_embed, "music_vec": music_embed}
    
    def all_gather(self, tensor, sync_grads=False):
        if sync_grads:
            return GatherLayer.apply(tensor)
        else:
            output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output, tensor)
            return output

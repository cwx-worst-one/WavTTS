import itertools
from collections import defaultdict
import math

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_info

from recipes.mulan.models.music_encoder import get_music_encoder
from recipes.mulan.models.text_encoder import get_text_encoder
from recipes.mulan.modules.gather import GatherLayer
from recipes.musiclm.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.utils.model_metric import ModelMetric
from samantha.utils.hparams import DotDict


class FiLMGenModel(pl.LightningModule):
    def __init__(
        self,
        music_encoder,
        text_encoder,
        decoder_cls,
        criterion_cls,
        required_modules,
        size_params,
        lr,
        gen_lr,
        weight_decay,
        gen_batch_size: int = 4,
        temperature: float = 0.1,
        mulan_loss_weight: float = 0.5,
        use_flatclr: bool = False,
        gather_batches: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        self.size_params = DotDict(size_params)

        output_type = "seq"
        seq_len = 250
        self.music_encoder = get_music_encoder(
            music_encoder, self.size_params.emb_dim, output_type, seq_len
        )
        self.text_encoder = get_text_encoder(
            text_encoder, self.size_params.emb_dim, output_type
        )
        self.decoder = decoder_cls()
        self.criterion = criterion_cls()
        self.requires = {}

        self.temperature = torch.log(torch.tensor(1 / self.hparams.temperature))

        # Validation outputs
        self.val_outputs = dict()

    def on_fit_start(self):
        self.music_encoder.mut.manually_to_device(self.device)

    def on_predict_start(self):
        self.text_encoder.cpu()  # save gpu memory
        self.music_encoder.manually_to_device(self.device)

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def setup(self, stage: str):
        # Variables for MFU calculation
        self.metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs={
                "music_tower": self.music_encoder,
                "text_tower": self.text_encoder,
            },
        )

        if stage == "fit" and not self.requires:
            print("Loading required modules")
            self.load_required_modules()
            print("Required modules loaded")

    def configure_optimizers(self):
        film_params = []
        gen_params = []
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
            _found = False
            for k in no_decay:
                if k in name:
                    norm_params.append(param)
                    _found = True
                    break
            if not _found:
                if name.startswith("decoder."):
                    gen_params.append(param)
                else:
                    film_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": film_params},
                {"params": norm_params, "weight_decay": 0.0},
                {"params": gen_params, "lr": self.hparams.gen_lr},
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = WarmupCosine(
            optimizer,
            init_lr=self.hparams.lr,
            warmup_steps=1000,
            cycle_steps=20000,
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
        ## FILM
        result = self._shared_mulan_step(batch)
        text_vec, music_vec = result["text_vec"], result["music_vec"]
        music_id = batch["music_id"]
        ## FILM loss
        # fine loss, only gather music_vec and music_id,
        # text_vec will be gathered in _fine_sim_matrix
        music_vec = rearrange(
            self.all_gather(music_vec, sync_grads=True), "w b s d -> (w b) s d"
        )
        music_id = rearrange(
            self.all_gather(music_id, sync_grads=True), "w b -> (w b) "
        )
        logits = self._prepare_fine_logits(text_vec, music_vec)
        loss_mulan = self._ssl_constrast_loss(logits, music_id)

        ## GEN loss
        loss_acoustic, acc_acoustic = self._shared_decoder_step(text_vec, music_vec, batch)

        loss = (
            self.hparams.mulan_loss_weight * loss_mulan
            + (1 - self.hparams.mulan_loss_weight) * loss_acoustic
        )

        self.log("tr_loss", loss, prog_bar=True)
        self.log("tr_loss_film", loss_mulan, prog_bar=True)
        self.log("tr_loss_gen", loss_acoustic, prog_bar=True)
        self.log("tr_acc_gen", acc_acoustic, prog_bar=True)

        return {"loss": loss}

    def _hit_score_rank(self, source_embed, target_embed):
        target_embed = rearrange(self.all_gather(target_embed), "w b s d -> (w b) s d")
        mat = self._fine_sim_matrix(source_embed, target_embed)
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
        result = self._shared_mulan_step(batch)
        text_vec, music_vec = result["text_vec"], result["music_vec"]
        loss_acoustic, acc_acoustic = self._shared_decoder_step(text_vec, music_vec, batch)
        result["loss_acoustic"] = loss_acoustic
        result["acc_acoustic"] = acc_acoustic
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []

        self.val_outputs[dataloader_idx].append(result)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            text_vecs = []
            music_vecs = []
            loss_acoustic = 0
            accu_acoustic = 0

            for output in outputs:
                text_vecs += output["text_vec"]
                music_vecs += output["music_vec"]
                loss_acoustic += output["loss_acoustic"]
                accu_acoustic += output["acc_acoustic"]
            accu_acoustic /= len(outputs)
            loss_acoustic /= len(outputs)

            text_vecs = torch.stack(text_vecs)
            music_vecs = torch.stack(music_vecs)

            score = self._hit_score_rank(text_vecs, music_vecs)
            self.log_dict(
                {
                    f"median_rank_{dataloader_idx}": score["median_rank"],
                    f"hit_score_{dataloader_idx}": score["hit_score"],
                    f"loss_acoustic_{dataloader_idx}": loss_acoustic.item(),
                    f"accu_acoustic_{dataloader_idx}": accu_acoustic.item(),
                },
                prog_bar=True,
            )
            self.val_outputs[dataloader_idx] = []

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        music_embed = self.music_encoder(batch["audio"])
        return {"music_vec": music_embed}

    def _shared_mulan_step(self, batch, spec_aug=False):
        music_embed = self.music_encoder(batch["audio"].unsqueeze(1), spec_aug=spec_aug)
        text_embed = self.text_encoder(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )

        # mfu calculation
        batch_size, seq_len = batch["input_ids"].size()[:2]
        num_tokens = batch_size * seq_len  # text token only
        self.metric.update(
            num_tokens=num_tokens,
            stage=self.trainer.state.stage,
            model_kwargs={
                "music_tower": {"batch_size": batch_size},
                "text_tower": {"batch_size": batch_size, "seq_len": seq_len},
            }
        )

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.metric.compute(step=self.trainer.global_step)
            self.log_dict(metric, sync_dist=True)

        return {"text_vec": text_embed, "music_vec": music_embed}

    @torch.no_grad()
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def prepare_decoder_feature(self, wavs):
        device = wavs.device
        num_coarse = self.size_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse, device=device)
            * self.size_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.size_params.soundstream_codebook_size
        )
        input_ids = torch.cat([sos_ids, soundstream_ids[:, :-1]], dim=1)
        return input_ids, soundstream_ids

    def _shared_decoder_step(self, text_embeds, music_embeds, batch):
        gen_bsz = self.hparams.gen_batch_size
        text_ratio = self.size_params.text_beta / self.size_params.text_warmup_steps * min(self.size_params.text_warmup_steps, self.global_step)
        text_amount = math.floor(text_ratio * gen_bsz)

        audio = batch["audio"][:gen_bsz, :]
        text_embeds = text_embeds[:text_amount, :]
        music_embeds = music_embeds[text_amount:gen_bsz, :]
        mulan_embeds = torch.cat([text_embeds, music_embeds], dim=0)
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_decoder_feature(audio.float())
        logits = self.decoder(input_ids=input_ids, encoder_hidden_states=mulan_embeds)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        return loss, accu
    
    def all_gather(self, tensor, sync_grads=False):
        if sync_grads:
            return GatherLayer.apply(tensor)
        else:
            output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output, tensor)
            return output

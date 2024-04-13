import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from apps.bigmusic.umm.ar.datasets.mir.music_sft import MusicSFTDataResult
from apps.bigmusic.umm.ar.datasets.mir.taxonomies.music_sft_en import MusicSFTTokenizerEN
from apps.bigmusic.umm.ar.datasets.mir.taxonomies.music_sft_zh import MusicSFTTokenizerZH
from apps.bigmusic.umm.ar.datasets.tokenizers.tagging import MI1_MusicTaggingConfig, MI1_MusicTaggingResult
from apps.bigmusic.umm.ar.datasets.tokenizers.tokenizers import UMMTokenizer
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.utils.distributed import rank_zero_first
from samantha.utils.hdfs_tools import hdfs_get

logger = logging.getLogger()


@dataclass
class MI1_MusicTaggingMusicSFTResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: Optional[LossDict] = None

class MI1_MusicTaggingMusicSFT(DefaultTrainingBaseModule):

    def __init__(
        self,
        config: MI1_MusicTaggingConfig,
        tag_tokenizer: MusicSFTTokenizerEN,
        audio_tokenizer: Optional[UMMTokenizer] = None,
    ) -> None:
        super().__init__(ignore_hparams=["audio_tokenizer"])
        self.config = config
        self.tag_tokenizer = tag_tokenizer
        self.audio_tokenizer = audio_tokenizer

        if audio_tokenizer:
            self.audio_tokenizer_identifier = self.audio_tokenizer.identifier

        tag_head = nn.Sequential(
            nn.Linear(self.config.n_embd, len(self.tag_tokenizer)),
        )

        self.heads = nn.ModuleDict(
            {
                "tag": tag_head,
            }
        )

        ## Metrics
        self.roc_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=len(self.tag_tokenizer),
            average="macro",
            thresholds=None,
        )
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multilabel",
            num_labels=len(self.tag_tokenizer),
            average="macro",
            thresholds=None,
        )

    def forward(self, hidden_states: torch.Tensor, apply_sigmoid: bool = True) -> MI1_MusicTaggingMusicSFTResult:
        # [b, t, n_embd]
        hidden_states = hidden_states.mean(dim=1)
        logits = self.heads.tag(hidden_states)

        if apply_sigmoid:
            logits = logits.sigmoid()

        return MI1_MusicTaggingMusicSFTResult(logits=logits, hidden_states=hidden_states)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossDict:
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return {"loss": loss}

    def step(self, batch: MusicSFTDataResult, return_loss: bool) -> MI1_MusicTaggingMusicSFTResult:
        result = self.audio_tokenizer.forward(batch.audio)
        result = self.forward(result.hidden_states, apply_sigmoid=False)
        if return_loss:
            result.loss = self.loss(result.logits, batch.tag_ids)
        return result

    def validation_step(self, batch: MusicSFTDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)

        self.roc_auc(result.logits, batch.tag_ids)
        self.pr_auc(result.logits, batch.tag_ids)
        self.log(
            "valid/roc_auc", self.roc_auc, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "valid/pr_auc", self.pr_auc, prog_bar=True, on_step=False, on_epoch=True
        )
        return result.loss["loss"]

    def on_test_start(self) -> None:
        self.roc_auc.reset()
        self.pr_auc.reset()

    def test_step(self, batch: MusicSFTDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.roc_auc(result.logits, batch.tag_ids)
        self.pr_auc(result.logits, batch.tag_ids)
        self.log("test/roc_auc", self.roc_auc, on_step=True, on_epoch=True)
        self.log("test/pr_auc", self.pr_auc, on_step=True, on_epoch=True)
        return result.loss["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),  # TODO optimizers for foundation/head models
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            fused=False,
        )
        return {
            "optimizer": optimizer,
        }

    def logits_to_tag_tokens(self, logits: torch.Tensor, topk: int) -> torch.Tensor:
        _, valid_tags_indices = torch.topk(logits, k=topk)
        return valid_tags_indices

    @torch.no_grad()
    def predict_tags(self, hidden_states: torch.Tensor, topk: int) -> MI1_MusicTaggingResult:
        result = self.forward(hidden_states)
        valid_tags_indices = self.logits_to_tag_tokens(result.logits, topk=topk)
        tag_names = self.tag_tokenizer.decode(valid_tags_indices)
        tag_probs = result.logits.softmax(dim=-1)
        return MI1_MusicTaggingResult(
            logits=result.logits,
            hidden_states=result.hidden_states,
            tag_ids=valid_tags_indices,
            tag_names=tag_names,
            tag_probabilities=tag_probs,
        )


class MI1_MusicClassificationMusicSFT(DefaultTrainingBaseModule):

    def __init__(
        self,
        config: MI1_MusicTaggingConfig,
        tag_tokenizer: MusicSFTTokenizerEN,
        audio_tokenizer: Optional[UMMTokenizer] = None,
    ) -> None:
        super().__init__(ignore_hparams=["audio_tokenizer"])
        self.config = config
        self.tag_tokenizer = tag_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self._n_classes = len(tag_tokenizer)

        if audio_tokenizer:
            self.audio_tokenizer_identifier = self.audio_tokenizer.identifier

        ## Metrics
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
            average="macro",

        )

        head = nn.Sequential(
            nn.Linear(self.config.n_embd, self.n_classes),
        )

        self.heads = nn.ModuleDict(
            {
                self.config.head_name: head,
            }
        )

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def forward(self, hidden_states: torch.Tensor) -> MI1_MusicTaggingMusicSFTResult:
        # [b, t, n_embd]
        hidden_states = hidden_states.mean(dim=1)
        logits = self.heads[self.config.head_name](hidden_states)
        return MI1_MusicTaggingMusicSFTResult(logits=logits, hidden_states=hidden_states)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossDict:
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss}

    def step(self, batch: MusicSFTDataResult, return_loss: bool) -> MI1_MusicTaggingMusicSFTResult:
        result = self.audio_tokenizer.forward(batch.audio)
        result = self.forward(result.hidden_states)

        # result = self.forward(batch.hidden_states)

        if return_loss:
            result.loss = self.loss(result.logits, batch.tag_ids)
        return result

    def training_step(self, batch: MusicSFTDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="train", prog_bar=True, rank_zero_only=True)

        self.accuracy(result.logits, batch.tag_ids)
        self.log(
            "train/accuracy", self.accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        return result.loss["loss"]

    def validation_step(self, batch: MusicSFTDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)

        self.accuracy(result.logits, batch.tag_ids)
        self.log(
            "valid/accuracy", self.accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        return result.loss["loss"]

    def on_test_start(self) -> None:
        self.accuracy.reset()

    def test_step(self, batch: MusicSFTDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.accuracy(result.logits, batch.tag_ids)
        self.log("test/accuracy", self.accuracy, on_step=True, on_epoch=True)
        return result.loss["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),  # TODO optimizers for foundation/head models
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            fused=False,
        )
        return {
            "optimizer": optimizer,
        }

    def logits_to_tag_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


    @torch.no_grad()
    def predict_tags(self, hidden_states: torch.Tensor) -> MI1_MusicTaggingResult:
        result = self.forward(hidden_states)
        valid_tags_indices = self.logits_to_tag_tokens(result.logits)
        tag_names = self.tag_tokenizer.decode_batch(valid_tags_indices)
        tag_probs = result.logits.softmax(dim=-1)
        return MI1_MusicTaggingResult(
            logits=result.logits,
            hidden_states=result.hidden_states,
            tag_ids=valid_tags_indices,
            tag_names=tag_names,
            tag_probabilities=tag_probs,
        )

    @torch.no_grad()
    def predict_tags_full_audio(self, hidden_states: torch.Tensor):
        # hidden_states contains a batch of audios from the same song
        # we take the average of the batch logits in order to get our final prediction
        result = self.forward(hidden_states)
        
        audio_avg_logits = result.logits.mean(dim=0, keepdim=True)

        valid_tags_indices = self.logits_to_tag_tokens(audio_avg_logits)
        tag_names = self.tag_tokenizer.decode_batch(valid_tags_indices)
        tag_probs = audio_avg_logits.softmax(dim=-1)
        return MI1_MusicTaggingResult(
            logits=audio_avg_logits,
            hidden_states=result.hidden_states,
            tag_ids=valid_tags_indices,
            tag_names=tag_names,
            tag_probabilities=tag_probs,
        )

class M1Tagger(nn.Module):

    def __init__(self, load_tokenizer=False, confidence_threshold=0, cache_dir='.m1_cache', tokenizer="En"):
        super().__init__()

        self.confidence_threshold = confidence_threshold
        logger.info("Initialising M1 models")
        if tokenizer == "Zh":
            with rank_zero_first():
                cache_dir = Path(cache_dir)/'zh'
                cache_dir.mkdir(exist_ok=True, parents=True)
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_genre_1_v0_umm_lr5.0e-5_precision32_2layer_mcc30k_ummcn_zh_step=0030000.ckpt", str(cache_dir/'m1_genre.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_mood_v0_umm_lr5.0e-5_precision32_2layer_mcc30k_ummcn_zh_step=0030000.ckpt", str(cache_dir/'m1_mood.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_scene_v0_umm_lr5.0e-5_precision32_2layer_mcc30k_ummcn_zh_step=0030000.ckpt", str(cache_dir/'m1_scene.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_vocal_gender_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_ummcn_step=0030000.ckpt", str(cache_dir/'m1_vocal_gender.ckpt'))
        elif tokenizer == "En":
            with rank_zero_first():
                cache_dir = Path(cache_dir)/'en'
                cache_dir.mkdir(exist_ok=True, parents=True)
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_genre_zh_step=0030000.ckpt", str(cache_dir/'m1_genre.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_mood_zh_step=0030000.ckpt", str(cache_dir/'m1_mood.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_v0_umm_lr5.0e-5_precision32_1layer_scene_zh_step=0030000.ckpt", str(cache_dir/'m1_scene.ckpt'))
                hdfs_get("hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_vocal_v0_umm_1layer_classification_vocalgender_step=0030000.ckpt", str(cache_dir/'m1_vocal_gender.ckpt'))
        else:
            raise ValueError(f"Unknown audio tokenizer: {tokenizer}")

        self.models = nn.ModuleDict({
            "genre": MI1_MusicClassificationMusicSFT.load_from_checkpoint(cache_dir/"m1_genre.ckpt"),
            "mood": MI1_MusicClassificationMusicSFT.load_from_checkpoint(cache_dir/"m1_mood.ckpt"),
            "scene": MI1_MusicClassificationMusicSFT.load_from_checkpoint(cache_dir/"m1_scene.ckpt"),
            "vocal_gender": MI1_MusicClassificationMusicSFT.load_from_checkpoint(cache_dir/"m1_vocal_gender.ckpt"),
        })

        self.load_tokenizer = load_tokenizer
        # TODO (janne) Check if MIR's tokenizer ckpt is different from the one used in AR, load audio tokenizer with no_grad to recompute hidden_states
        # TODO (janne) If same tokenizer, do not create self.audio_tokenizer
        logger.info("Initialising M1-UMM model")

        if self.load_tokenizer:
            if tokenizer == "Zh":
                # self.audio_tokenizer = UMMTokenizerCN().eval()
                raise ValueError(f"Unknown audio tokenizer: {tokenizer}")
            elif tokenizer == "En":
                self.audio_tokenizer = UMMTokenizer().eval()
            else:
                raise ValueError(f"Unknown audio tokenizer: {tokenizer}")
            self.audio_tokenizer.freeze()
                
        logger.info(f"M1 model vocabularies:")
        for k in self.models:
            logger.info(f"{k}: {self.models[k].tag_tokenizer.vocab}")

        logger.info("Completed M1 model initialisation")

    @torch.no_grad()
    def get_predictions(self, hidden_states, audio: Optional[torch.Tensor] = None) -> Dict[str, MI1_MusicTaggingResult]:
        results = {}

        if self.load_tokenizer:
            audio = audio.to(self.audio_tokenizer.device)
            audio = audio.squeeze(dim=1)
            hidden_states = self.audio_tokenizer(audio).hidden_states

        for tag_key in self.models:
            results[tag_key] = self.models[tag_key].predict_tags(hidden_states)
        return results

def init_m1_tagger(hpath, local_rank, cache_dir, tokenizer='En', **kwargs):
    device = torch.device(f"cuda:{local_rank}")
    m1_tagger = M1Tagger(hpath, cache_dir=cache_dir, tokenizer=tokenizer, **kwargs).eval().to(device)
    return { "m1_tagger": m1_tagger }

def get_m1_tags(requires, hidden_states, audio: Optional[torch.Tensor]):
    m1_tagger: M1Tagger = requires["m1_tagger"]
    preds = m1_tagger.get_predictions(hidden_states, audio)
    # postprocess predictions
    batch_size = audio.shape[0]
    m1_tags = [{} for i in range(batch_size)]
    for tag_key in preds:            
        result = preds[tag_key]
        candidate_tag_names = np.array(result.tag_names)
        tag_name_probs = result.tag_probabilities[torch.arange(result.tag_probabilities.size(0)).unsqueeze(0), result.tag_ids].squeeze()
        tag_name_probs = tag_name_probs.cpu()
        for i in range(len(candidate_tag_names)):
            if tag_name_probs[i] > m1_tagger.confidence_threshold:
                tag = candidate_tag_names[i]
            else:
                tag = ""
            m1_tags[i][tag_key] = tag
    return m1_tags
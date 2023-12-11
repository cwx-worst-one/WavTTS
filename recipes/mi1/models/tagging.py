from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from recipes.datasets.mir.tagging import TaggingDataResult, TaggingDataModule
from samantha.models.base import DefaultTrainingBaseModule, LossDict, LightningModuleBase
from samantha.dataio.data_bucket import data_bucket
from recipes.mi1.models.tokenizers import UMMTokenizer, UMMResult, MusicTagTokenizer, MusicTagTokenizerResult

MTAT_TO_MCC = {
    "classical": "Classical Music",
    "techno": "Techno",
    "electronic": "Electronic Music",
    "rock": "Rock",
    "opera": "Opera",
    "pop": "Pop",
    "classic": "Classical Music",
    "new age": "New Age",
    "dance": "Dance Pop",
    "country": "Country",
    "metal": "Metal",
    "guitar": "Guitar",
    "strings": "Strings",
    "drums": "Percussions",
    "piano": "Acoustic Piano",
    "violin": "Violin",
    "synth": "Synthesizers",
    "female": "Female",
    "woman": "Female",
    "male": "Male",
    "flute": "Flute",
    "man": "Male",
    "male voice": "Male",
    "male vocal": "Male",
    "female vocal": "Female",
    "cello": "Cello",
    "female voice": "Female",
    
    # my own mapping:
    "singing": "With vocal",
    "vocals": "With vocal",
    "vocal": "With vocal",
    "voice": "With vocal",
    "no vocal": "No vocal",
    "no vocals": "No vocal",
    "no voice": "No vocal",
    "fast": "Fast",
    "slow": "Slow",
    "loud": "Loud",
    "quiet": "Quiet",
    "soft": "Quiet",
    "ambient": "Ambient",
    "choir": "Choir",
    "choral": "Choir",
    "weird": "Weird",
    "beat": "Beat",
    "beats": "Beat",
    "solo": "Solo",
    "indian": "",
    "harpsichord": "",
    "harp": "",
    "sitar": "",
}


@dataclass
class MI1_MusicTaggingInput:
    hidden_states: torch.Tensor
    audio: Optional[torch.Tensor] = None
    tag: Optional[torch.Tensor] = None


@dataclass
class MI1_MusicTaggingResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: Optional[LossDict] = None
    tag_ids: Optional[List[str]] = None
    tag_names: Optional[List[str]] = None
    tag_probabilities: Optional[List[float]] = None

@dataclass
class MI1_MusicTaggingConfig:
    head_name: str
    sample_rate: int = 24000
    n_embd: int = 1024
    duration: float = 29.1
    n_embd_head: int = 512
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5

class MI1_MusicTagging(DefaultTrainingBaseModule):
    output_dim: int = 50

    _ignore_tags: set = {"techno", "harpsichord", "indian", "harp", "sitar"}

    def __init__(
        self,
        config: MI1_MusicTaggingConfig,
        audio_tokenizer: Optional[UMMTokenizer] = None,
    ) -> None:
        super().__init__(ignore_hparams=["audio_tokenizer"])
        self.config = config
        self.audio_tokenizer = audio_tokenizer

        if audio_tokenizer:
            self.audio_tokenizer_identifier = self.audio_tokenizer.identifier
            
        head_tagging = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_embd_head),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.config.n_embd_head, self.output_dim),
        )

        self.heads = nn.ModuleDict(
            {
                "tagging": head_tagging,
            }
        )

        ## Metrics
        self.roc_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=self.output_dim,
            average="macro",
            thresholds=None,
        )
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multilabel",
            num_labels=self.output_dim,
            average="macro",
            thresholds=None,
        )

        self._tag_names = TaggingDataModule._tag_names

    def state_dict(self, destination, prefix, keep_vars):
        state_dict = super().state_dict()
        heads_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "audio_tokenizer" not in k:
                heads_state_dict[k] = v
        return heads_state_dict

    @property
    def tag_names(self) -> List[str]:
        return self._tag_names

    @property
    def max_seq_len(self):
        return int(self.musicfm.frame_rate * self.config.duration)

    def forward(self, hidden_states: torch.Tensor, apply_sigmoid: bool = True) -> MI1_MusicTaggingResult:
        # [b, t, n_embd]
        hidden_states = hidden_states.mean(dim=1)
        logits = self.heads.tagging(hidden_states)

        if apply_sigmoid:
            logits = logits.sigmoid()
        return MI1_MusicTaggingResult(logits=logits, hidden_states=hidden_states)

    def get_inputs(self, batch: Union[TaggingDataResult, MI1_MusicTaggingInput]) -> MI1_MusicTaggingInput:
        if type(batch) == MI1_MusicTaggingInput:
            return batch
        
        result = self.audio_tokenizer.forward(batch.audio)
        return MI1_MusicTaggingInput(
            hidden_states=result.hidden_states,
            tag=batch.tag,
        )

    @staticmethod
    def loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossDict:
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return {"loss": loss}

    def step(self, batch: Union[TaggingDataResult, MI1_MusicTaggingInput], return_loss: bool) -> MI1_MusicTaggingResult:
        inputs = self.get_inputs(batch)
        result = self.forward(inputs.hidden_states, apply_sigmoid=False)
        if return_loss:
            result.loss = self.loss(result.logits, batch.tag)
        return result

    def validation_step(self, batch: TaggingDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)

        self.roc_auc(result.logits, batch.tag)
        self.pr_auc(result.logits, batch.tag)
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

    def test_step(self, batch: TaggingDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.roc_auc(result.logits, batch.tag)
        self.pr_auc(result.logits, batch.tag)
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

    def tag_tokens_to_names(self, logits: torch.Tensor, valid_tags_indices: torch.Tensor):
        valid_tags_indices = valid_tags_indices.cpu()
        
        tag_names = []
        mcc_tag_names = []
        tag_probabilities = []
        for idx, tag_indices in enumerate(valid_tags_indices):
            names = []
            m_names = []
            probs = []
            for t in tag_indices:
                name = self._tag_names[t]
                mcc_name = MTAT_TO_MCC[name]
                # skip duplicate predictions and ignored tags
                if mcc_name in m_names or name in self._ignore_tags:
                    continue

                m_names.append(mcc_name)
                names.append(name)
                probs.append(logits[idx, t].item())

            tag_names.append(names)
            mcc_tag_names.append(m_names)
            tag_probabilities.append(probs)
        return dict(
            tag_names=tag_names,
            mcc_tag_names=mcc_tag_names,
            tag_probabilities=tag_probabilities,
        )

    @torch.no_grad()
    def predict_tags(self, hidden_states: torch.Tensor, topk: int) -> MI1_MusicTaggingResult:
        result = self.forward(hidden_states)
        valid_tags_indices = self.logits_to_tag_tokens(result.logits, topk=topk)
        tag_result = self.tag_tokens_to_names(result.logits, valid_tags_indices)
        return MI1_MusicTaggingResult(
            logits=result.logits,
            hidden_states=result.hidden_states,
            tag_names=tag_result["tag_names"],
            mcc_tag_names=tag_result["mcc_tag_names"],
            tag_probabilities=tag_result["tag_probabilities"],

        )


class MI1_MusicTaggingTokenizer(LightningModuleBase):

    def __init__(self, topk: int):
        super().__init__()
        self._topk = topk
        self._ckpt_path = data_bucket("mi-1_music_tagging/v0_lr1.0e-4_precision32_preload/checkpoints/step=0030000.ckpt")
        
        self.model = MI1_MusicTagging.load_from_checkpoint(self._ckpt_path)
        self.tokenizer = MusicTagTokenizer()
        self.freeze()
        self.eval()

    @property
    def topk(self) -> int:
        return self._topk

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.tokenizer)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states: Optional[torch.Tensor] = None, tag_names: Optional[Union[List[str], List[List[str]]]] = None, device: Optional[torch.device] = None) -> MusicTagTokenizerResult:
        if self.model.training:
            self.model.eval()

        if tag_names is None:
            result = self.model.predict_tags(hidden_states, topk=self.topk)
            tag_names = result.mcc_tag_names

        result = self.tokenizer(tag_names, device=device)
        return result


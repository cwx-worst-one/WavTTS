import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from byteformers.models import LlamaConfig, LlamaModel
from recipes.research.diff.cond import NumberConditioner
from recipes.research.diff.mulan import Mulan
from recipes.research.diff.text_processor import (
    BPMTextProcessor,
    MulanPretrainMetadataTextProcessor,
    WordProcessor,
)
from recipes.research.zoo.umm.dual_umm import DualUMM
from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class ARConfig:
    sample_rate: int = 44100
    max_duration: int = 30

    n_embd: int = 1536
    n_layer: int = 24
    n_head: int = 24

    cond_n_embd: int = 768

    # Optimizer
    learning_rate: float = 3.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_steps: int = 8000
    cycle_steps: int = 1000000
    min_lr: float = learning_rate


@dataclass
class ARResult:
    logits: torch.Tensor
    token_ids: torch.Tensor
    input_emb: torch.Tensor
    prefix_cond: torch.Tensor
    loss: Optional[LossDict] = None


class AR(DefaultTrainingBaseModule):
    def __init__(self, config: ARConfig):
        super().__init__()
        self.config = config
        logger.info("Loading MelCodec model checkpoint...")

        # self.mel_codec = get_mel_codec("5545212", ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/5545212/2024-05-26/07-47-24/checkpoints/step=273000.ckpt")
        # self.mel_codec = MelMAE(MelMAEConfig(sample_rate=config.sample_rate, patch_size=(320, 2)))  # 50hz

        from recipes.research.mel_codec.mel_rq import get_mel_codec, get_mel_vq

        self.mel_codec = get_mel_vq(
            "e55c641",
            ckpt_path="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/mel_codec/default/e55c641/2024-06-19/14-18-52/checkpoints/step=050000.ckpt",
        )

        self.mel_codec.eval()
        self.mel_codec.freeze()
        assert (
            self.config.sample_rate == self.mel_codec.sample_rate
        ), "Sampling rates do not match"

        self.vocab_size = self.mel_codec.codebook_size * self.mel_codec.n_codebooks
        self.frame_rate = math.ceil(
            self.mel_codec.frame_rate * self.mel_codec.n_codebooks
        )
        self.max_seq_len = math.ceil(self.frame_rate * config.max_duration)

        ###
        ### Conditioning Models
        ###
        self.seconds_start_emb = NumberConditioner(
            config.cond_n_embd, min_val=0, max_val=512
        )
        self.seconds_total_emb = NumberConditioner(
            config.cond_n_embd, min_val=0, max_val=512
        )

        logger.info("Loading Multi-Modal model checkpoint...")
        self.cond_model = Mulan(output_type="seq", text_feature_layer_idx=-2)
        self.cond_model.n_embd = 768  # NOTE: CLAP n_embd is actually 768
        self.cond_latent_dim = self.cond_model.n_embd
        if self.cond_latent_dim == config.cond_n_embd:
            self.proj_cond = nn.Identity()
        else:
            self.proj_cond = nn.Linear(
                self.cond_latent_dim, config.cond_n_embd, bias=False
            )
        self.cond_model.eval()
        self.cond_model.freeze()

        ## Text preprocessors, for CLIP text only
        self.bpm_processor = BPMTextProcessor(min_bpm=40, max_bpm=200, bpm_step=10)
        self.words_processor = WordProcessor(min_word_len=3, seed=42)

        self.text_processor = MulanPretrainMetadataTextProcessor(seed=42)

        self.input_emb = nn.Embedding(
            self.vocab_size, config.n_embd
        )  ## TODO: replace with post-quant?
        self.proj_prefix = nn.Linear(config.cond_n_embd, config.n_embd, bias=False)

        # TODO: aliasing may occur longer duration
        model_config = LlamaConfig(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            use_rotary_embeddings=True,
            is_causal=True,
        )
        self.model = LlamaModel(model_config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        self.initialize_weights()

    def setup(self, stage: Optional[str] = None):
        self.cond_model.cast_to_rank(self.local_rank)

    def initialize_weights(self):
        logger.info("Initializing weights...")

    def get_mel_codes(self, audio: torch.Tensor, sample_rate: int):
        with torch.no_grad():
            if self.mel_codec.training:
                self.mel_codec = self.mel_codec.eval()
            codes = self.mel_codec.get_codes(audio, sample_rate)  # [B, C, T]
            codes = codes.detach()

            flattened_codes = self.mel_codec.flatten_codes(codes)
            # codes2 = self.mel_codec.unflatten_codes(flattened_codes)
            # torch.testing.assert_close(codes, codes2)
            return flattened_codes

    def get_multimodal_embs(self, text: List[str]) -> Dict[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if self.cond_model.training:
                    self.cond_model = self.cond_model.eval()

                cond_result = self.cond_model.forward(text=text)
                cond_emb = cond_result.hidden_states.detach()  # [B, T_multimodal, D]
                cond_attn_mask = cond_result.attention_mask

        cond_emb = self.proj_cond(cond_emb)
        return {"embeds": cond_emb, "attention_mask": cond_attn_mask}

    def get_cond_embs(
        self,
        seconds_start: List[int],
        seconds_total: List[int],
        text: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        ###
        ### Prepare and align conditioning
        ###

        sec_start_result = self.seconds_start_emb.forward(seconds_start, device=device)
        sec_total_result = self.seconds_total_emb.forward(seconds_total, device=device)

        cond_result = self.get_multimodal_embs(text)
        prefix_cond = torch.cat(
            (
                sec_start_result["embeds"],
                sec_total_result["embeds"],
                cond_result["embeds"],
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                sec_start_result["attention_mask"],
                sec_total_result["attention_mask"],
                cond_result["attention_mask"],
            ),
            dim=1,
        )
        return {"prefix_cond": prefix_cond, "attention_mask": attention_mask}

    def preprocess_text(
        self,
        descriptions: List[str],
        keywords: List[str],
        genres: List[str],
        instruments: List[str],
        bpms: List[str],
        seed: int,
    ) -> List[str]:
        bpms = self.bpm_processor(bpms)

        descriptions = self.words_processor(descriptions, shuffle=False)
        keywords = self.words_processor(keywords, shuffle=False)  # TODO
        genres = self.words_processor(
            genres, shuffle=False
        )  # in case of primary/secondary genres, don't shuffle
        instruments = self.words_processor(instruments, shuffle=False)  # TODO

        processed_text = self.text_processor(
            descriptions, keywords, genres, instruments, seed=seed
        )
        return processed_text

    def forward(
        self,
        mel_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_cond: torch.Tensor,
        prefix_attn_mask: torch.Tensor,
    ):

        ## TODO: re-arrange prefix based on attn mask

        ## proj input/cond
        input_emb = self.input_emb.forward(mel_token_ids)
        prefix_cond = self.proj_prefix.forward(prefix_cond)

        input_emb = torch.cat((prefix_cond, input_emb), dim=1)
        attention_mask = torch.cat((prefix_attn_mask, attention_mask), dim=1)

        attention_mask = None
        logits = self.model.forward(input_emb, attention_mask=attention_mask)
        logits = self.lm_head(logits)
        return ARResult(
            logits=logits,
            token_ids=mel_token_ids,
            input_emb=input_emb,
            prefix_cond=prefix_cond,
        )

    def step(self, batch: AudioDataResult, batch_idx: int, return_loss: bool):
        audio = batch.audio
        seconds_start = [math.floor(s.seek_time) for s in batch.segment_info]
        seconds_total = [math.floor(s.meta.duration) for s in batch.segment_info]

        ## shutterstock data
        # descriptions = [i["description"] for i in batch.index]
        # keywords = [i["keywords"] for i in batch.index]
        # genres = [i["genres"] for i in batch.index]
        # instruments = [i["instruments"] for i in batch.index]
        # bpms = [i["bpm"] for i in batch.index]
        # text = self.preprocess_text(
        #     descriptions, keywords, genres, instruments, bpms, seed=batch_idx + self.global_rank
        # )
        # logger.info(text)
        text = ["test"] * audio.shape[0]

        ## get features
        mel_token_ids = self.get_mel_codes(audio, self.config.sample_rate)

        # mel_token_ids = torch.randint_like(audio, high=self.mel_codec.codebook_size, dtype=torch.long, requires_grad=False)
        # mel_token_ids = self.mel_codec.flatten_codes(mel_token_ids)

        embs = self.get_cond_embs(seconds_start, seconds_total, text, audio.device)

        attention_mask = torch.ones_like(
            mel_token_ids
        )  # NOTE: assuming no silence here!
        result = self.forward(
            mel_token_ids, attention_mask, embs["prefix_cond"], embs["attention_mask"]
        )

        cond_len = embs["prefix_cond"].shape[1]
        logits = result.logits[:, cond_len:]

        with torch.cuda.amp.autocast(enabled=False):
            result.loss = self.loss(logits, mel_token_ids)
        return result

    def loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> LossDict:
        batch_size = logits.shape[0]

        logits = logits[:, :-1]
        target_ids = target_ids[:, 1:]

        logits = rearrange(logits, "b s c -> (b s) c")
        target_ids = rearrange(target_ids, "b s -> (b s)")

        loss = self.criterion.forward(logits, target_ids)
        return {"loss": loss, "perplexity": loss.exp(), "batch_size": batch_size}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        scheduler = WarmupCosine(
            optimizer,
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.cycle_steps,
            self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def sample(
        self,
        text: List[str],
        seconds_start: List[int],
        seconds_total: List[int],
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.95,
    ):
        batch_size = len(text)

        embs = self.get_cond_embs(seconds_start, seconds_total, text, self.device)

        sampled_token_ids = torch.empty(
            (batch_size, 0), device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(sampled_token_ids)

        start_frame = 0
        for _ in tqdm(range(start_frame, self.max_seq_len), desc="Sampling"):
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                result = self.forward(
                    sampled_token_ids,
                    attention_mask,
                    embs["prefix_cond"],
                    embs["attention_mask"],
                )
                logits = result.logits[:, -1]

            logits = logits / temperature

            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float("Inf")

            probs = logits.softmax(dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = probs.argmax(dim=-1, keepdim=True)
                # _, idx_next = torch.topk(probs, k=1, dim=-1)

            sampled_token_ids = torch.cat((sampled_token_ids, idx_next), dim=1)
        return sampled_token_ids


if __name__ == "__main__":
    config = ARConfig(
        sample_rate=44100, max_duration=30, n_layer=8, n_embd=512, n_head=8
    )
    model = AR(config).to("cuda")
    model.setup()
    print(model.summarize())

    batch_size = 2
    x = torch.randn(
        batch_size, 2, config.max_duration * config.sample_rate, device=model.device
    )

    segment_info = SegmentInfo(
        meta=AudioMeta(path=None, duration=30, sample_rate=config.sample_rate),
        seek_time=0.0,
        n_frames=x.shape[2],
        total_frames=x.shape[2],
        sample_rate=config.sample_rate,
        channels=x.shape[1],
        data_type="music_vocal",
        lyrics=None,
    )
    index = {
        "keywords": "upbeat, technology, technological, pulsing, lively, kawaii, japan, cute, bubblegum, bouncy",
        "genres": "country",
        "instruments": "drums, guitar, piano",
        "bpm": "121",
        "description": "Bouncy and cute",
    }

    batch = AudioDataResult(
        audio=x,
        shard=None,
        key=None,
        segment_info=[segment_info] * batch_size,
        index=[index] * batch_size,
    )
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        result = model.step(batch, batch_idx=0, return_loss=True)

    with torch.no_grad():
        pred_noise = model.sample(["test"], [0], [10], temperature=1.0)

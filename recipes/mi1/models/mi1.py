from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2Config

from recipes.datasets.billboard import BillboardDataResult
from recipes.mi1.models.tagging import MI1_MusicTaggingTokenizer
from recipes.mi1.models.tokenizers import UMMTokenizer

# TODO: move
from recipes.musiclm.inference.utils import sample
from samantha.models.base import GenerativeBaseModule, LossDict
from samantha.models.ctiga.gpt import GPTLMHeadModel
from samantha.models.llama import Llama, LlamaConfig
from samantha.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.transforms.tokenizers.phoneme import LyricPhonemeTokenizer
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class MI1Config:
    sample_rate: int
    max_seconds: int
    n_layer: int
    n_head: int
    n_embd: int
    n_inner: int
    learning_rate: float
    warmup_steps: float
    cycle_steps: float
    weight_decay: float
    betas: Tuple[float, float]
    lyric_tokenizer: LyricPhonemeTokenizer
    top_k_music_tags: int


@dataclass
class MI1Input:
    lyrics: Optional[List[str]] = None
    lyrics_tokens: Optional[torch.Tensor] = None
    tag_names: Optional[Union[List[str], List[List[str]]]] = None
    audio: Optional[torch.Tensor] = None
    mel: Optional[torch.Tensor] = None


@dataclass
class MI1Result:
    audio_logits: torch.Tensor
    cond_logits: torch.Tensor
    loss: Optional[LossDict] = None


class LyricConditioningModelResult(NamedTuple):
    embeds: torch.Tensor
    token_ids: torch.Tensor
    text: Optional[str] = None
    normalized_text: Optional[str] = None


class LyricPhonemeConditioningModel(nn.Module):
    def __init__(self, tokenizer: LyricPhonemeTokenizer, n_embd: int):
        super().__init__()
        # TODO: evaluate newline character: <n> or  <n>  ?
        self.tokenizer = tokenizer
        self.n_embd = n_embd
        self._pad_token_id = self.tokenizer.pad_token_id

        self.lyric_emb = nn.Embedding(
            len(self.tokenizer), n_embd, padding_idx=self.pad_token_id
        )

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def _verify_inputs(self, t: torch.Tensor) -> None:
        assert t.ndim == 2

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_text = None
        if token_ids is None:
            result = self.tokenizer(text, device=device)
            token_ids = result["token_ids"]
            normalized_text = result["normalized_text"]

        self._verify_inputs(token_ids)
        return LyricConditioningModelResult(
            embeds=self.lyric_emb(token_ids),
            token_ids=token_ids,
            text=text,
            normalized_text=normalized_text,
        )


class MusicTaggingConditioningModelResult(NamedTuple):
    embeds: torch.Tensor
    token_ids: torch.Tensor
    normalized_text: Optional[str] = None
    tag_names: Optional[str] = None


class MusicTaggingConditioningModel(nn.Module):
    # NOTE: BERT's vocab size is pretty big... we can reduce significantly with
    # our own tagging vocabulary.

    def __init__(self, n_embd: int, topk_tags: int):
        super().__init__()
        self._topk_tags = topk_tags

        # we can add more tokenizers here, for chords/beats/etc.
        self.tokenizer = MI1_MusicTaggingTokenizer(topk_tags)
        self._pad_token_id = self.tokenizer.pad_token_id

        self.tag_emb = nn.Embedding(
            len(self.tokenizer), n_embd, padding_idx=self.pad_token_id
        )

    @property
    def topk_tags(self) -> int:
        return self._topk_tags

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        tag_names: Optional[Union[List[str], List[List[str]]]] = None,
        device: Optional[torch.device] = None,
    ) -> MusicTaggingConditioningModelResult:
        result = self.tokenizer(
            hidden_states=hidden_states, tag_names=tag_names, device=device
        )
        embeds = self.tag_emb(result.token_ids)
        return MusicTaggingConditioningModelResult(
            embeds=embeds,
            token_ids=result.token_ids,
            normalized_text=result.normalized_text,
            tag_names=result.tag_names,
        )


class MI1(GenerativeBaseModule):
    _pad_token_id: int = -100

    def __init__(self, config: MI1Config):
        super().__init__()
        self.config = config

        # TODO send to data bucket
        self.audio_tokenizer = UMMTokenizer()
        self._vocab_size = self.audio_tokenizer.vocab_size

        # TODO padding
        self.tokenizer_emb = nn.Embedding(
            len(self),
            self.config.n_embd,
            # padding_idx=self._pad_token_id
        )

        self.model_config = GPT2Config(
            vocab_size=1,  # placeholder. Embedding table gets removed on load
            max_position_embeddings=0,
            use_cache=False,
            n_positions=0,  # No absolute position embedding
            num_logits=len(self),
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=self.config.n_inner,
            use_rms_norm=True,
            activation_function="swiglu",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-6,
            initializer_range=0.02,
            rescale_prenorm_residual=False,
            rms_norm=True,
            rotary_emb_fraction=1.0,
            rotary_emb_interleaved=True,
            rotary_emb_compat="default",  # "byteformer"
            tie_word_embeddings=False,
            qkv_proj_bias=False,
            out_proj_bias=False,
            mlp_fc1_bias=False,
            mlp_fc2_bias=False,
            use_flash_attn=True,
            fused_bias_fc=True,
            fused_mlp=False,
            fused_dropout_add_ln=True,
            residual_in_fp32=True,
        )

        self.model = GPTLMHeadModel(self.model_config)
        # init_weights_fn = partial(_init_weights, n_layer=self.model.config.num_hidden_layers)
        # self.input_embedders.apply(init_weights_fn)
        # self.target_embedder.apply(init_weights_fn)
        del self.model.transformer.embeddings.word_embeddings

        self.criterion = CrossEntropyLoss(ignore_index=self._pad_token_id)
        self._max_seq_len = self.audio_tokenizer.frame_rate * self.config.max_seconds

        self.init_conditioning_models()

    def init_conditioning_models(self):
        # conditioning models
        # TODO Idea: lyrics have semantic meaning. We now only condition on phonemes / pronounciation
        # let's also condition on semantic meaning of the lyric.

        logger.info("Initializing LyricPhonemeConditioningModel")
        self.cond_lyrics = LyricPhonemeConditioningModel(
            self.config.lyric_tokenizer, self.model_config.n_embd
        )
        if self.cond_lyrics.pad_token_id != self._pad_token_id:
            logger.warning(
                "WARNING: self.cond_lyrics.pad_token_id != self._pad_token_id"
            )
 
        # logger.info("Initializing MusicTaggingConditioningModel")
        # self.cond_music_tags = MusicTaggingConditioningModel(
        #     self.config.n_embd, self.config.top_k_music_tags
        # )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def sos_token_id(self) -> int:
        return self.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.vocab_size + 1

    def __len__(self):
        # Base vocabulary plus extra tokens
        return self.vocab_size + 2

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, p in self.named_parameters():
    #         if p.requires_grad and p.grad is None:
    #             print(name)

    def cond_inputs_to_embeds(
        self,
        inputs: MI1Input,
        device: torch.tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        # music tags
        # music_tagging_result = self.cond_music_tags.forward(
        #     hidden_states, tag_names=inputs.tag_names, device=device
        # )
        # music_tag_embeds = music_tagging_result.embeds

        # lyrics
        lyric_result = self.cond_lyrics.forward(
            text=inputs.lyrics, token_ids=inputs.lyrics_tokens, device=device
        )
        lyric_embeds = lyric_result.embeds

        # create final conditioning embeddings:
        # cond_embeds = torch.cat((music_tag_embeds, lyric_embeds), dim=1)
        cond_embeds = lyric_embeds
        return cond_embeds, cond_embeds.shape[1]

    def add_sos_token(self, audio_tokens: torch.Tensor):
        sos_token_ids = torch.full(
            (audio_tokens.shape[0], 1),
            self.sos_token_id,
            dtype=torch.long,
            device=audio_tokens.device,
        )
        return torch.cat((sos_token_ids, audio_tokens), dim=1)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        result = self.model.forward(
            inputs_embeds=inputs_embeds, inference_params=inference_params
        )
        return result.logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> LossDict:
        # NOTE: this assumes SOS is prepended already to the targets
        
        # logits: <SOS_EMBED, AUDIO_EMBEDS>
        # targets: <SOS_TOKEN, AUDIO_TOKENS>

        # prepare targets, so in/out becomes:
        # in: [SOS, ..., N]
        # out [n, ... EOS]
        batch_size = targets.shape[0]

        # logits: <SOS_EMBED, AUDIO_EMBEDS>
        # targets: <AUDIO_TOKENS, EOS_TOKEN>
        targets = targets[:, 1:].clone()
        eos_token_ids = torch.full(
            (batch_size, 1),
            self.eos_token_id,
            dtype=torch.long,
            device=targets.device,
        )
        targets = torch.cat((targets, eos_token_ids), dim=1)

        logits = rearrange(logits, "b t c -> (b t) c")
        targets = rearrange(targets, "b t -> (b t)")

        nll_loss = self.criterion(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        return dict(loss=nll_loss, accuracy=accuracy, ppl=nll_loss.exp())

    def step(self, batch: BillboardDataResult, return_loss: bool) -> MI1Result:
        mel = batch.get("mel")
        audio = batch.get("target_audio")  # audio
        lyrics_tokens = batch.get("lyrics_tokens")

        inputs = MI1Input(audio=audio, lyrics_tokens=lyrics_tokens, mel=mel)

        tokenizer_res = self.audio_tokenizer.forward(audio=inputs.audio, mel=inputs.mel)
        hidden_states = tokenizer_res.hidden_states

        # <AUDIO_TOKENS>
        audio_tokens = tokenizer_res.vq_ids

        # <SOS_TOKEN, AUDIO_TOKENS>
        audio_tokens = self.add_sos_token(audio_tokens)

        # <SOS_EMBED, AUDIO_EMBEDS>
        audio_embeds = self.tokenizer_emb(audio_tokens)

        # <MUSIC_TAG_EMBEDS, LYRIC_EMBEDS>
        cond_embeds, _ = self.cond_inputs_to_embeds(
            inputs, device=audio_embeds.device, hidden_states=hidden_states
        )

        # <MUSIC_TAG_EMBEDS, LYRIC_EMBEDS, SOS_EMBED, AUDIO_EMBEDS>
        inputs_embeds = torch.cat((cond_embeds, audio_embeds), dim=1)
        cond_len = cond_embeds.shape[1]

        logits = self.forward(inputs_embeds)

        cond_logits, audio_logits = torch.split(logits, [cond_len, logits.shape[1]-cond_len], dim=1)

        result = MI1Result(
            audio_logits=audio_logits,
            cond_logits=cond_logits
        )
        if return_loss:
            # <SOS_EMBED, AUDIO_EMBEDS>
            result.loss = self.loss(audio_logits, audio_tokens)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=1.0e-8,
        )

        scheduler = WarmupCosine(
            optimizer,
            init_lr=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            cycle_steps=self.config.cycle_steps,
            min_lr=self.config.learning_rate * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def generate(
        self,
        inputs: MI1Input,
        precision,
        temperature: float = 1.0,
        sampling_threshold: float = 0.95,
        sampling_mode: str = "top_p",
    ) -> torch.Tensor:
        cond_embeds, _ = self.cond_inputs_to_embeds(inputs, device=self.device)
        batch_size = cond_embeds.shape[0]

        sampled_audio_tokens = torch.empty(
            (batch_size, 0), dtype=torch.long, device=self.device
        )

        audio_inputs = self.add_sos_token(sampled_audio_tokens)
        audio_embeds = self.tokenizer_emb(audio_inputs)

        ## cast to training precision:
        inputs_embeds = torch.cat((cond_embeds, audio_embeds), dim=1)

        for _ in tqdm(range(self.max_seq_len)):

            with torch.cuda.amp.autocast(enabled=True, dtype=precision):
                logits = self.forward(inputs_embeds)

            ## cast to full precision to avoid simplex errors
            logits = logits[:, -1].float()

            idx_next = sample(
                logits,
                temp=temperature,
                thresh=sampling_threshold,
                mode=sampling_mode,
                return_probs=False,
            )

            predict_token = idx_next[:, None]
            predict_embeds = self.tokenizer_emb(predict_token)

            inputs_embeds = torch.cat((inputs_embeds, predict_embeds), dim=1)

            sampled_audio_tokens = torch.cat(
                (sampled_audio_tokens, predict_token), dim=1
            )

        # TODO: remove SOS(?), EOS, now done in inference
        return sampled_audio_tokens

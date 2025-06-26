import math
import random
from abc import abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from mariana.utils.audio.audio_logger import AudioLogger
from samantha.utils.ctiga.inference_params import InferenceParams
from recipes.bigmusic.datasets.transforms.lyrics_segment import (
    crop_pad_to_seq_length,
    random_crop_pad_to_seq_length,
)
from recipes.bigmusic.datasets.utils.zh_vocab_dev import get_tag_map, prob_to_tag


logger = AudioLogger()


class BaseEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 0,
        add_sos: bool = False,
        add_eos: bool = False,
        add_none: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
    ):
        """
        add_sos, add_eos: Add SOS or EOS to the vocab
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_id = None
        self.eos_id = None
        self.none_id = None
        if add_sos:
            self.sos_id = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        if add_eos:
            self.eos_id = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        if add_none:
            self.none_id = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        if self.sos_id is None and with_sos:
            raise ValueError(
                "Must initialize embedder with add_sos=True for with_sos=True"
            )
        self.with_sos = with_sos
        if self.eos_id is None and with_eos:
            raise ValueError(
                "Must initialize embedder with add_eos=True for with_eos=True"
            )
        self.with_eos = with_eos

    @abstractmethod
    def embed(
        self,
        requires: Optional[dict] = None,
        batch: Optional[dict] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def prepare_embed(
        self, requires: Optional[dict] = None, batch: Optional[dict] = None, **kwargs
    ) -> dict:
        """Prepare embedding for SemanticModule"""
        pass

    def get_device(self):
        return next(self.parameters()).device

    def get_dummy_token_ids_and_length(
        self, embeds: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ):
        device = self.get_device()
        batch_size, seq_len, _ = embeds.shape
        if token_ids is None:
            token_ids = torch.zeros((batch_size, seq_len)).long().to(device)
        token_length = torch.zeros((batch_size), device=device) + seq_len
        return {
            "token_embeds": embeds,
            "token_ids": token_ids,
            "token_length": token_length,
        }


class ContinuousEmbedder(BaseEmbedder):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        add_sos: bool = False,
        add_eos: bool = False,
        add_none: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=0,
            add_sos=add_sos,
            add_eos=add_eos,
            add_none=add_none,
            with_sos=with_sos,
            with_eos=with_eos,
        )
        self.embedder = nn.Embedding(self.vocab_size, input_dim, **kwargs)

        if input_dim != embedding_dim:
            self.projection = nn.Linear(input_dim, embedding_dim, bias=False)
        else:
            self.projection = nn.Identity()

    @abstractmethod
    def get_embeds(self, requires, batch) -> torch.Tensor:
        # override to return embedding function
        raise NotImplementedError()

    def get_sos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.sos_id is not None
        ), "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_id,
            dtype=torch.long,
            device=device,
        )
        return sos_ids

    def get_sos_embed(self, batch_size: int) -> torch.Tensor:
        sos_ids = self.get_sos_token(batch_size)
        return self.projection(self.embedder(sos_ids))

    def get_eos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.eos_id is not None
        ), "Error getting eos id. Must initialize embedder with add_eos=True"
        device = self.get_device()
        eos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.eos_id,
            dtype=torch.long,
            device=device,
        )
        return eos_ids

    def get_eos_embed(self, batch_size: int) -> torch.Tensor:
        eos_ids = self.get_eos_token(batch_size)
        return self.projection(self.embedder(eos_ids))

    def embed(self, requires: dict, batch: dict, **kwargs) -> torch.Tensor:
        embeds = self.get_embeds(requires, batch, **kwargs)
        embeds = self.projection(embeds)
        if self.with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        if self.with_eos:
            eos_embed = self.get_eos_embed(embeds.size(0))
            embeds = torch.cat([embeds, eos_embed], dim=1)
        return embeds


class MulanEmbedder(ContinuousEmbedder):
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 1024,
        add_sos: bool = False,
        add_eos: bool = False,
        add_none: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        data_type: str = "text",  # default data type
        min_audio_length: int = 10 * 24000,
        mulan_crop: bool = True,
        # mulan_average=True,  # unused
        dropout: float = 0.0,
        text_emb_max_len: int = 100,
        return_hidden_state: bool = False,
        text_key: str = "freeform_text",
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            add_none=add_none,
            with_sos=with_sos,
            with_eos=with_eos,
            **kwargs,
        )

        self.data_type = data_type
        self.min_audio_length = min_audio_length  # 10s * 24k sample rate
        self.mulan_crop = mulan_crop
        self.dropout = dropout
        self.text_emb_max_len = text_emb_max_len
        self.return_hidden_state = return_hidden_state
        self.text_key = text_key

    def get_embeds(
        self, requires, input_audio_or_text, data_type: Optional[str] = None
    ) -> torch.Tensor:
        if data_type is None:
            data_type = self.data_type  # set default data type
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(
                requires,
                input_audio_or_text,
                data_type,
                return_hidden_state=self.return_hidden_state,
            )
            if len(mulan_embeds.shape) == 2:
                mulan_embeds = mulan_embeds[:, None, :]  # bs x d -> bs x seq_len x d
            mulan_embeds = mulan_embeds[:, : self.text_emb_max_len, :]
            return mulan_embeds
        ## CFG
        if data_type == "none":
            bs = len(input_audio_or_text)
            device = self.get_device()
            mulan_embeds = self.embedder(
                torch.tensor([self.none_id] * bs, device=device)
            )
            return mulan_embeds.unsqueeze(1)
        if data_type == "embed":
            if input_audio_or_text.dim() == 2:
                return input_audio_or_text.unsqueeze(1)
            return input_audio_or_text
        # Audio
        if data_type == "music":
            if input_audio_or_text.dim() == 1:
                input_audio_or_text = input_audio_or_text.unsqueeze(0)
            if input_audio_or_text.shape[-1] < self.min_audio_length:
                input_audio_or_text = crop_pad_to_seq_length(
                    input_audio_or_text, self.min_audio_length
                )
            else:
                if self.mulan_crop and self.training:
                    # Always use moving average of mulan emb for validation and inference.
                    input_audio_or_text = random_crop_pad_to_seq_length(
                        input_audio_or_text, self.min_audio_length
                    )
            bs = input_audio_or_text.shape[0]
            device = self.get_device()
            if self.training:
                if random.random() < self.dropout:
                    mulan_embeds = self.embedder(
                        torch.tensor([self.none_id] * bs, device=device)
                    ).unsqueeze(1)
                else:
                    mulan_embeds = get_mulan_embeds(
                        requires, input_audio_or_text, data_type
                    )
            else:
                if torch.sum(input_audio_or_text) == 0:
                    # This is the CFG path
                    mulan_embeds = self.embedder(
                        torch.tensor([self.none_id] * bs, device=device)
                    ).unsqueeze(1)
                else:
                    # This is the conditioned path
                    mulan_embeds = get_mulan_embeds(
                        requires, input_audio_or_text, data_type
                    )
            if mulan_embeds.dim() == 2:
                mulan_embeds = mulan_embeds[:, None, :]  # bs x d -> bs x seq_len x d
            return mulan_embeds

    def prepare_embed(self, requires: dict, batch: dict, **kwargs) -> dict:
        batch_size = _infer_batch_size(batch)
        emb = self.embed(
            requires, batch.get(self.text_key, [None] * batch_size), **kwargs
        )
        return self.get_dummy_token_ids_and_length(emb)


class XValEmbedder(BaseEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        max_value: float = 120.0,
        norm_max_value: float = 10.0,
        item_key: Optional[str] = None,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        **kwargs,
    ) -> None:
        # token #0: xval, none_id: pad
        super().__init__(
            vocab_size=1,
            add_sos=add_sos,
            add_eos=add_eos,
            add_none=True,
            with_sos=with_sos,
            with_eos=with_eos,
        )  # single_token + <pad> (no eos)
        self._token_id = 0  # do not change
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)
        self.max_value = max_value
        self.norm_max_value = norm_max_value
        self.logged = 0
        self.item_key = item_key

    def normalize(self, x):
        x = torch.clamp(x, max=self.max_value)
        norm = x / self.max_value * self.norm_max_value
        norm[x <= 0] = (
            1  # always use the pad embedding where x <= 0, no need to scale the embedding in this case
        )
        return norm

    # input can be either 2d tensor (section-level) or 1d tensor (sample-level)
    def embed(
        self, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO (Yilin): add SOS and EOS

        if input.ndim == 1:
            input = input.unsqueeze(1)

        device = self.get_device()
        token_ids = (
            torch.where(
                input > 0, torch.tensor(self._token_id), torch.tensor(self.none_id)
            )
            .long()
            .to(device)
        )
        # normalize the input to range
        normalized_input = self.normalize(input).float()
        embeds = self.embedder(token_ids) * normalized_input.unsqueeze(2)
        if self.logged < 5:
            logger.info(f"Xval input: {input}")
            logger.info(f"token_ids ({token_ids.shape}): {token_ids}")
            logger.info(f"normalized input ({normalized_input.shape}): {normalized_input}")
            self.logged += 1

        return embeds, token_ids, normalized_input

    def prepare_embed(self, batch: dict) -> dict:
        item = batch[self.item_key]
        embeds, _, _ = self.embed(item)
        return self.get_dummy_token_ids_and_length(embeds)


class TokenEmbedder(BaseEmbedder):
    # Token embedder - takes in tokens, and then embeds
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        id_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
        )
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)
        self.id_key = id_key

    @abstractmethod
    def get_tokens(
        self, requires: Optional[dict], batch: Optional[dict], **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_sos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.sos_id is not None
        ), "Error getting sos id. Must initialize embedder with add_sos=True"
        device = self.get_device()
        sos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_id,
            dtype=torch.long,
            device=device,
        )
        return sos_ids

    def get_eos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.eos_id is not None
        ), "Error getting eos id. Must initialize embedder with add_eos=True"
        device = self.get_device()
        eos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.eos_id,
            dtype=torch.long,
            device=device,
        )
        return eos_ids

    def get_sos_embed(self, batch_size: int) -> torch.Tensor:
        return self.embedder(self.get_sos_token(batch_size))

    def get_eos_embed(self, batch_size: int) -> torch.Tensor:
        return self.embedder(self.get_eos_token(batch_size))

    def tokenize(
        self, requires=None, batch=None, token_ids=None, **kwargs
    ) -> torch.Tensor:
        if token_ids is None:
            token_ids = self.get_tokens(requires, batch, **kwargs)
        if self.with_sos:
            token_ids = torch.cat(
                [self.get_sos_token(token_ids.size(0)), token_ids], dim=1
            )
        if self.with_eos:
            token_ids = torch.cat(
                [token_ids, self.get_eos_token(token_ids.size(0))], dim=1
            )
        return token_ids

    def embed(self, requires=None, batch=None, token_ids=None) -> dict[str, torch.Tensor]:
        """Return a dict because token ids might be updated with SOS/EOS"""
        token_ids = self.tokenize(requires, batch, token_ids)
        embeds = self.embedder(token_ids)
        return {
            "token_ids": token_ids,
            "embeds": embeds,
        }

    def prepare_embed(self, batch: dict, **kwargs) -> dict:
        item = batch[self.id_key]
        if len(item.shape) == 1:
            item = item.reshape(-1, 1)
        result = self.embed(token_ids=item, **kwargs)
        return self.get_dummy_token_ids_and_length(result["embeds"], token_ids=result["token_ids"])


class BestRQTokenEmbedder(TokenEmbedder):
    def __init__(
        self,
        vocab_size: int = 32_768,
        embedding_dim: int = 1024,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        chunk_size: Optional[int] = None,
        store_hidden_states: bool = False,
        sample_rate: int = 24000,
        slice_method: Optional[str] = "even",
        chunk_dur: float = 60.0,
        item_key: str = "target_audio",
        length_key: str = "target_audio_length",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
            *kwargs,
        )
        if chunk_size is not None and store_hidden_states:
            raise ValueError(
                "Tokenizer currently does not support both chunking and saving last hidden state"
            )
        self.store_hidden_states = (
            store_hidden_states  # save hidden states for m1 classifier
        )
        self.last_hidden_state = None
        self.chunk_size = chunk_size
        self.tag_map = get_tag_map()
        self.sample_rate = sample_rate
        if not slice_method:
            self.slice_method = None
            self.chunk_dur = None
        else:
            if slice_method not in ["even", "max"]:
                raise ValueError(f"Invalid slice method {slice_method}")
            self.slice_method = slice_method
            self.chunk_dur = chunk_dur
        self.item_key = item_key
        self.length_key = length_key

        # NOTE (Yilin): Hacky implementation...
        if self.with_sos ^ self.with_eos:
            raise ValueError("Both with_sos and with_eos should both be True or False")

    def get_tokens(self, requires, input_audio, **kwargs) -> torch.Tensor:
        if self.store_hidden_states:
            results = get_bestrq_umm_outputs(requires, input_audio, **kwargs)
            token_ids = results["vq_ids"]
            self.hidden_states = results["hidden_states"]
            return token_ids
        else:
            results = get_bestrq_umm_tokens(
                requires, input_audio, self.chunk_size, **kwargs
            )
            if isinstance(results, dict) and "vq_ids" in results:
                results = results["vq_ids"]
            return results

    def tokenize_and_tag(self, requires, input_audio, **kwargs):
        lit_module = requires["Stage3"]
        umm2_out = lit_module.wav2tokentag(input_audio)
        tag_logits = umm2_out["tags_logits"]
        return umm2_out["vq_ids"], tag_logits

    def _key_with_max_value(d, topk=1, threshold=0):
        return max(d, key=d.get)

    def convert_tag_logits_to_tags(self, tag_logits):
        return prob_to_tag(tag_logits, self.tag_map)

    def tokenize(
        self, requires=None, batch=None, token_ids=None, **kwargs
    ) -> torch.Tensor:
        """Override: Do not add SOS and EOS here"""
        if token_ids is None:
            token_ids = self.get_tokens(requires, batch, **kwargs)
        return token_ids
    
    def _get_target_ids_and_token_lengths(self, requires: dict, batch: dict, **kwargs) -> dict:
        device = self.get_device()
        # Prepare target ids
        assert self.item_key in ["target_audio", "target_token_ids"]
        if self.item_key == "target_token_ids":
            target_ids = batch[self.item_key]
        else:
            target_audio_length = float(batch[self.item_key].shape[-1]) / self.sample_rate
            if target_audio_length <= self.chunk_dur or self.slice_method is None:
                wav_length = batch['audio_length'].view(-1).to(batch['target_audio'].device)
                target_ids = self.tokenize(
                    requires, 
                    batch[self.item_key],
                    wav_length=wav_length,
                    ).to(device)
            else:
                target_ids = []
                n_samples = batch[self.item_key].shape[-1]
                # evenly slice the audio in a way that the `chunk_size` is as close as possible to `chunk_dur`
                if self.slice_method == "even":
                    chunk_num = math.ceil(target_audio_length / self.chunk_dur)
                    chunk_size = math.ceil(target_audio_length / chunk_num)
                # always slice the audio with the maximum `chunk_dur`, combine the tail audio < 1s
                elif self.slice_method == "max":
                    chunk_size = self.chunk_dur
                else:
                    # Should be impossible to reach this branch
                    raise NotImplementedError(
                        f"{self.slice_method} is not implemented as audio slice method"
                    )
                # TODO (qinxin): implement section-level audio slice method
                # elif self.slice_method == 'section':
                #     raise NotImplementedError(f"(WIP) {self.slice_method} implementation")
                # else:
                #     raise NotImplementedError(f"{self.slice_method} is not implemented as audio slice method")

                st = 0
                while st < n_samples:
                    _st, _et = int(st * self.sample_rate), int(
                        (st + chunk_size) * self.sample_rate
                    )
                    # merge the tail if the remaining chunk is too short (< 1s)
                    if (
                        n_samples - _et < self.sample_rate * 1
                        or batch[self.item_key][..., _et:].shape[-1] < self.sample_rate * 1
                    ):
                        _et = n_samples
                    target_audio = batch[self.item_key][...,_st:_et]
                    wav_length = []
                    for audio_length in batch['audio_length']:
                        if _st > audio_length:
                            wav_length.append(0)
                        else:
                            valid_et = min(_et, audio_length)
                            wav_length.append(valid_et - _st)
                    wav_length = torch.LongTensor(wav_length).to(target_audio.device)
                    _target_id = self.tokenize(
                        requires, 
                        target_audio,
                        wav_length=wav_length,
                        ).to(device)
                    target_ids.append(_target_id)
                    if _et >= n_samples:
                        break
                    st += chunk_size
                target_ids = torch.cat(target_ids, dim=-1)

        target_lengths = batch[self.length_key].to(device)

        # add sos and eos id
        if self.with_sos and self.with_eos:
            target_ids = F.pad(target_ids, (1, 1))
            target_ids[:, 0] = self.sos_id
            eos_indices = (target_lengths + 1).unsqueeze(1)  # set last index to EOS
            target_ids.scatter_(1, eos_indices, self.eos_id)
            target_lengths = target_lengths + 2  # +2 for eos and sos

        return target_ids, target_lengths

    def prepare_embed(self, requires: dict, batch: dict, **kwargs) -> dict:
        target_ids, target_lengths = self._get_target_ids_and_token_lengths(requires, batch, **kwargs)
        result = self.embed(token_ids=target_ids)
        target_embeds = result["embeds"]
        target_ids = result["token_ids"]
        target_ids_unpad = unpad_sequence(target_ids, target_lengths, batch_first=True)
        target_embeds = unpad_sequence(target_embeds, target_lengths, batch_first=True)
        # TODO (qq) return a list. placeholder for intermediate leadsheet tokens.
        return [
            {
                "token_embeds": target_embeds,
                "token_ids": target_ids_unpad,
                "token_ids_batched": target_ids,
                "token_length": target_lengths,
            }
        ]


class BestRQMultiTokenEmbedder(BestRQTokenEmbedder):
    def __init__(
        self,
        vocab_size: int = 32_768,
        embedding_dim: int = 1024,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        chunk_size: Optional[int] = None,
        store_hidden_states: bool = False,
        sample_rate: int = 24000,
        slice_method: Optional[str] = "even",
        chunk_dur: float = 60.0,
        item_key: str = "target_audio",
        length_key: str = "target_audio_length",
        pattern: Optional[str] = "delay",
        group: int = 2,
        null_placeholder: bool = True,
        encoder: str = 'fc',
        decoder: Union[str, dict] = 'fc',
        semantic_codebook_depth: int = 1,
        **kwargs,
    ):
        """
        Embedder for Multi-token prediction
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
            chunk_size=chunk_size,
            store_hidden_states=store_hidden_states,
            sample_rate=sample_rate,
            slice_method=slice_method,
            chunk_dur=chunk_dur,
            item_key=item_key,
            length_key=length_key,
            **kwargs,
        )

        self.group = group
        self.pattern = pattern
        self.null_placeholder = null_placeholder
        self.embedding_dim = embedding_dim
        self.R = semantic_codebook_depth

        if isinstance(decoder, str):
            decoder = {"model_type": decoder}

        if self.pattern is None:
            self.pattern = 'parallel'
            self.group = 1
            encoder = 'skip'
            decoder = {'model_type': 'fc'}

        self.null_id = None
        if self.null_placeholder and self.pattern:
            self.vocab_size += 1
            self.null_id = self.vocab_size - 1
            self.embedder = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.null_id)

        if self.R > 1:
            self.embedder = nn.ModuleList([])
            for r in range(self.R):
                self.embedder.append(nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.null_id))

        self.encoder_type = encoder
        if self.encoder_type == 'fc':    # concat -> fc
            self.encoder = nn.Sequential(
                nn.Linear(self.group * embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        elif self.encoder_type in ['skip', 'sum']:
            self.encoder = None # operation instead of layers
        else:
            raise NotImplementedError(f"Unsupported multi-token encoder {self.encoder_type}")
        
        self.decoder_type = decoder["model_type"]
        if self.decoder_type in ['fc']:    # concat -> fc
            self.decoder = None
        elif self.decoder_type in ['transformer', 'RQ_transformer', 'ARQ_transformer']:
            self.decoder = decoder["model_cls"]()
        elif self.decoder_type in ['context_fc']:
            self.decoder = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),    # context + cond
                nn.SiLU(),
                nn.Linear(embedding_dim, decoder["output_dim"]),    # num_logits
            )
        else:
            raise NotImplementedError(f"Unsupported multi-token decoder {self.decoder_type}")

    def get_sos_embed(self, batch_size):
        if self.R == 1:
            return self.embedder(self.get_sos_token(batch_size))
        else:
            return self.embedder[0](self.get_sos_token(batch_size))

    def embed(self, requires=None, batch=None, token_ids=None) -> dict[str, torch.Tensor]:
        token_ids = self.tokenize(requires, batch, token_ids)
        # (qinxin) Here, to align with super class func embed(), target_token_length is not given,
        # so interpret target token length with added sos_id and eos_id 
        token_length = torch.where(token_ids == self.eos_id)[1]  # [BS]
        batch_size = token_ids.shape[0]
        batch_token_embeds, batch_token_lengths, batch_group_token_ids = [], [], []
        for b in range(batch_size):
            if self.R == 1: # group on time frames
                this_token_ids = token_ids[b, 1:token_length[b]]    # [T_b]
                if len(this_token_ids) % self.group != 0:
                    this_token_ids = F.pad(this_token_ids, 
                                    (0, self.group - len(this_token_ids) % self.group), 
                                    'constant', self.eos_id)
                this_token_ids = this_token_ids.reshape(-1, self.group).mT  # [T_b//G, G], [G, T_b//G]
            elif self.R == self.group:  # group on hierarchy
                this_token_ids = token_ids[b, 1:token_length[b]].mT    # [T_b, R (G)] => [R (G), T_b]
            else:
                raise NotImplementedError(f"Unsupported semantic codebook depth {self.R} and group {self.group}")

            if self.pattern == 'delay':
                this_token_ids = F.pad(this_token_ids, (self.group - 1, 0), 'constant', self.null_id)
                this_token_ids = torch.stack([torch.roll(this_token_ids[d], d + 1 - self.group) for d in range(self.group)])
                # add EOS tokens, this_token_ids: [G, T]
                if self.null_id in this_token_ids[:, -self.group:]:
                    null_mask = this_token_ids[:, -self.group:] == self.null_id
                    this_token_ids[:, -self.group:][null_mask] = self.eos_id

            if this_token_ids.shape[1] == 0: 
                this_token_ids = torch.ones(self.group, 1, dtype=torch.long, device=this_token_ids.device) * self.sos_id
            if sum(this_token_ids[:, -1] == self.eos_id) != self.group:
                this_token_ids = F.pad(this_token_ids, (0, 1), 'constant', self.eos_id)
            
            # [G, T] => [G, T, D] --- multi-token encoder ---> [T, D]
            this_token_embeds = self.embed_token_id(this_token_ids)
            this_sos_embed = self.get_sos_embed(1).squeeze(0)
            this_token_embeds = torch.cat([
                this_sos_embed,    # SOS token, [1, D]
                this_token_embeds,             # grouped tokens + EOS tokens, [T, D]
            ], dim=0)
            
            batch_token_embeds.append(this_token_embeds)    # [T, D]
            batch_group_token_ids.append(this_token_ids) # [G, T-1] (no SOS token, with EOS tokens, serve as targets to be predicted by LM)
            batch_token_lengths.append(this_token_embeds.shape[0])

        token_embeds = pad_sequence(batch_token_embeds, batch_first=True, padding_value=0.)

        return {
            "embeds": token_embeds,
            "token_ids": token_ids,
            "group_token_ids": batch_group_token_ids,
            "batch_token_lengths": torch.LongTensor(batch_token_lengths),
        }

    def embed_token_id(self, token_ids, frame_idx=None):
        """token_ids: [G, T] / [B, G, T]"""
        if frame_idx is not None and frame_idx + 1 < self.group and self.pattern == 'delay':
            assert token_ids.shape[-1] == 1
            token_ids[..., self.group-(frame_idx+1):, 0] = self.null_id

        if self.R == 1:
            token_embeds = self.embedder(token_ids) # [*, G, T] => [*, G, T, D]
        else:
            token_embeds = torch.stack([self.embedder[r](token_ids[..., r, :]) for r in range(self.R)], dim=-3) # [*, G, T] => [*, G, T, D]

        if token_ids.ndim == 2:   # [G, T, D] => [T, D]
            if self.encoder_type == 'skip':   # [G, T, D] => [T, D]
                token_embeds = token_embeds[0]
            elif self.encoder_type == 'sum': # => [G, T, D] => [T, D]
                token_embeds = token_embeds.sum(dim=0)
            else:    # [G, T, D] => [T, G, D] => [T, G*D] => [T, D]
                token_embeds = self.encoder(token_embeds.transpose(1, 0).reshape(-1, self.group*self.embedding_dim))
        else:   # [B, G, T, D] => [B, T, D]
            if self.encoder_type =='skip':   # [B, T]
                token_embeds = token_embeds[:, 0]
            elif self.encoder_type =='sum':  # [B, T]
                token_embeds = token_embeds.sum(dim=1)
            else:   # [B, G, T, D] => [B, T, G, D] => [B, T, G*D] => [B, T, D]
                token_embeds = self.encoder(token_embeds.transpose(2, 1).reshape(token_ids.shape[0], -1, self.group*self.embedding_dim))
        return token_embeds


    def decode_target_id(self, target_id, delete_null=False, add_sos=True, delay_back=False):
        """target_id: [G, T]"""
        # [G, T] -- reorganize --> [T, G]
        if self.pattern == 'delay' and delay_back:
            target_id = torch.stack([torch.roll(target_id[d], -d) for d in range(self.group)])

        target_id = target_id.mT
        if self.R == 1: # flatten --> [T*G]
            target_id = target_id.reshape(-1)
            if add_sos: # add on time axis
                target_id = F.pad(target_id, (1, 0), 'constant', self.sos_id)
        elif self.R == self.group: # [T, G]
            if add_sos: # add on time axis
                target_id = F.pad(target_id, (0, 0, 1, 0), 'constant', self.sos_id)
            if (target_id[-1] == self.eos_id).sum() != self.R:
                target_id[-1] = self.eos_id
        
        if delete_null:
            target_id = target_id[target_id != self.null_id]
        return target_id

    def init_decoder(self, batch_size, max_seq_len=8000):
        model_input = dict()
        model_input["inputs_embeds"] = self.get_sos_embed(batch_size).to(self.decoder.lm_head.weight.dtype)
        self.inference_params = InferenceParams(max_sequence_len=max_seq_len, max_batch_size=batch_size)
        # pre allocate rotary cos/sin to reduce recompute them per step
        for layer in self.decoder.transformer.layers:
            if hasattr(layer.mixer, "rotary_emb"):
                layer.mixer.rotary_emb._update_cos_sin_cache(model_input["inputs_embeds"], max_seq_len)
                
    def decode_hidden_state_inference(self, hidden_state, last_predict_token_emb=None, 
                                      frame_idx=None, decoder_embeds=None, cfg_path=None):
        """hidden_state: [B, T=1, D], last_predict_token_emb: [B, T=1, D]"""
        if self.decoder is None or self.decoder_type not in ['transformer', 'RQ_transformer', 'context_fc', 'ARQ_transformer']:
            return hidden_state
        
        model_input = {"inputs_embeds": None}
        batch_size = hidden_state.shape[0]

        def AR_decoder(model_input, hidden_state, last_predict_token_emb=None):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is not None:
                model_input["inputs_embeds"] = torch.cat((last_predict_token_emb[:batch_size], hidden_state), 
                                                        dim=1)
            else:
                model_input["inputs_embeds"] = torch.cat((self.get_sos_embed(batch_size).to(self.decoder.lm_head.weight.dtype),
                                                        hidden_state), dim=1)
            decoder_output = self.decoder(**model_input, inference_params=self.inference_params,
                                        position_ids=None, last_token_only=False,)
            self.inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            # [B, T=2, G*D]
            target_logits = decoder_output.logits
            target_logits = target_logits[:, -1:]   # [B, T=1, G*D]
            return target_logits
        
        def context_fc(hidden_state, last_predict_token_emb):
            batch_size = hidden_state.shape[0]
            if last_predict_token_emb is None:
                decoder_input = torch.cat((hidden_state, self.get_sos_embed(batch_size)), dim=-1)
            else:
                decoder_input = torch.cat((hidden_state, last_predict_token_emb[:batch_size]), dim=-1)
            decoder_output = self.decoder(decoder_input)
            return decoder_output
        
        def RQ_transformer(model_input, hidden_state):
            logger.info("RQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        def ARQ_transformer(model_input, hidden_state):
            logger.info("ARQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        if frame_idx == 0:
            if self.decoder_type in ['transformer', 'RQ_transformer']:
                self.init_decoder(batch_size, decoder_embeds=decoder_embeds, cfg_path=cfg_path)
        
        if self.decoder_type == 'transformer':
            target_logits = AR_decoder(model_input, hidden_state, last_predict_token_emb)
        elif self.decoder_type == 'context_fc':
            target_logits = context_fc(hidden_state, last_predict_token_emb)

        return target_logits


    def decode_hidden_state(self, hidden_state, token_embeds, prefix_length, target_length=None,
                            token_ids=None):
        """
        This function is called during training (teacher-forcing) for AR-based decoder,
        for simplicity, the prefix length will be truncated.
        hidden_state: [B, T//G, D] serves as condition to generate upcoming N=group tokens
        target_ids: [B, T] 
        """
        def AR_decoder(hidden_state, token_embeds, prefix_length, target_length):
            B, T, D = hidden_state.size()
            hidden_state_expanded = hidden_state.unsqueeze(2)  # [B, T, 1, D]
            token_embeds_expanded = token_embeds.unsqueeze(2)  # [B, T, 1, D]
            # prefix_emb...; emb_sos; cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            group_embeds = torch.cat((token_embeds_expanded, hidden_state_expanded), dim=2) # [B, T, 2, D]

            # remove LM prefix (keep the sos_emb)
            group_embeds = [group_embeds[b][prefix_length[b]:].reshape(-1, D) for b in range(B)] # B x [2*T_b (interleaved style), D]
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   

            # [B, 2T, G*D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits
            # [B, G, 2T, D]
            target_logits = torch.stack(torch.chunk(target_logits, chunks=self.group, dim=-1), dim=1)
            # outputs corresponding to cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # shape: [G, 2(Tb), D], index: -1 to remove input shift (target_length - 1)
                this_target_logits = target_logits[b, :, :(target_length[b]-1)*2]
                this_target_logits = this_target_logits[:,1::2]  # [G, Tb, D]
                # G, 2T, n_logits => 2T, G, n_logits => (2T*G), n_logits
                _target_logits = this_target_logits.transpose(1, 0).reshape(-1, n_logits)
                batch_target_logits.append(torch.cat(
                    (torch.zeros(prefix_length[b], n_logits).to(_target_logits.device),
                    _target_logits,
                    ), dim=0))
            # [B, pT+T, n_logits]
            return batch_target_logits

        def RQ_transformer(hidden_state, token_ids, prefix_length, target_length):
            """
            hidden_state (with prefix): [B, pT + T//G, D]
            token_ids (with prefix): [B, pt + T]
            """
            B, _, D = hidden_state.size()
            group_embeds = []
            # input: concatenation of [LM_cond, single_token_emb] (B*(T//G), time_steps=G, 2D)
            # output: token logits (B*(T//G), time_steps=G, n_logits)
            for b in range(B):
                # 1) prepare group token embeddings
                # w/o sos_id; with all eos_id
                this_token_ids = token_ids[b][prefix_length[b]:prefix_length[b] + (target_length[b]-1)*self.group]
                if len(this_token_ids) % self.group != 0:
                    this_token_ids = F.pad(this_token_ids, (0, self.group - len(this_token_ids) % self.group), 
                                            'constant', self.eos_id)
                this_token_embeds = self.embedder(this_token_ids)  # [T, D] (exclude sos_embed, include all eos_embed)
                this_group_token_embeds = this_token_embeds.reshape(-1, self.group, this_token_embeds.shape[-1])    # [T//G, G, D]
                # shift group_token_embeds by 1: group_1,...,group_G => sos_embed, group_1,...,group_G-1
                this_group_token_embeds = torch.cat((self.get_sos_embed(this_group_token_embeds.shape[0]),
                                                     this_group_token_embeds[:, :-1]), dim=1)
                # 2) prepare LM condition embeddings
                this_hidden_state = hidden_state[b][prefix_length[b]:prefix_length[b] + target_length[b] - 1]  # [T//G, D]
                this_group_hidden_state = this_hidden_state.unsqueeze(1).repeat(1, self.group, 1)   # [T//G, G, D]

                this_group_embeds = torch.cat((this_group_hidden_state, this_group_token_embeds), -1)   # [T//G, G, 2D]
                group_embeds.append(this_group_embeds)

            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   # [B, T//G, G, 2D]
            group_embeds = group_embeds.reshape(-1, self.group, 2*D)    # [B*(T//G), G, 2D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits     # [B*(T//G), G, N]

            target_logits = target_logits.reshape(B, -1, self.group, target_logits.shape[-1])   # [B, T//G, G, N]
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # [G, T//G, N] => [T//G, G, N] => [T, N]
                this_target_logits = target_logits[b, :target_length[b]-1].reshape(-1, n_logits)   
                batch_target_logits.append(torch.cat(
                    (torch.zeros(prefix_length[b], n_logits).to(this_target_logits.device),
                    this_target_logits,
                    ), dim=0))
            return batch_target_logits
        
        def ARQ_transformer(hidden_state, token_ids, prefix_length, target_length):
            """
            hidden_state (with prefix): [B, pT + T//G, D]
            token_embeds (with prefix): [B, pt + T, D]
            """
            B, T, D = hidden_state.size()
            group_embeds = []
            # prefix_emb...; emb_sos; cond_{1~G},emb_1,...,emb_G; 
            #                         cond_{G+1~2G},emb_{G+1}...,emb_{2G}; ...; 
            #                         cond_{T*G~(T+1)*G},emb_{T*G}...,emb_{(T+1)*G}
            for b in range(B):
                # 1) prepare group token embeddings
                this_token_ids = token_ids[b][prefix_length[b]:prefix_length[b] + (target_length[b]-1)*self.group]
                if len(this_token_ids) % self.group != 0:
                    this_token_ids = F.pad(this_token_ids, (0, self.group - len(this_token_ids) % self.group), 
                                    'constant', self.eos_id)
                this_token_embeds = self.embedder(this_token_ids)  # [T, D] (exclude sos_embed, include all eos_embed)
                this_group_token_embeds = this_token_embeds.reshape(-1, self.group, this_token_embeds.shape[-1])    # [T//G, G, D]
                
                # 2) prepare LM condition embeddings
                this_hidden_state = hidden_state[b][prefix_length[b]:prefix_length[b] + target_length[b] - 1]  # [T//G, D]
                this_group_hidden_state = this_hidden_state.unsqueeze(1)   # [T//G, 1, D] (exclude sos_embed)

                # 3) insert LM into group token embeddings
                assert this_group_token_embeds.shape[0] == this_group_hidden_state.shape[0]
                this_group_embeds = torch.cat((this_group_hidden_state, this_group_token_embeds), dim=1)    # [T//G, G+1, D]
                this_group_embeds = torch.cat((self.get_sos_embed(1)[0], this_group_embeds.reshape(-1, D)), dim=0) # 1+T//G*(G+1), D

                group_embeds.append(this_group_embeds)
            
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   # [B, 1+T//G*(G+1), D]
            target_logits = self.decoder(inputs_embeds=group_embeds).logits     # [B,1+(T//G)*(G+1), N]

            # exclude sos_embed
            target_logits = target_logits[:, 1:].reshape(B, -1, self.group+1, target_logits.shape[-1])   # [B, T//G, G+1, N]
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # [T//G, G+1, N] => [T//G, G, N] => [T, N]
                this_target_logits = target_logits[b, :target_length[b]-1, :self.group].reshape(-1, n_logits)  # 
                batch_target_logits.append(torch.cat(
                    (torch.zeros(prefix_length[b], n_logits).to(this_target_logits.device),
                    this_target_logits,
                    ), dim=0))
            return batch_target_logits

        def context_fc_decoder(hidden_state, token_embeds, prefix_length):
            # remove LM prefix (keep the sos_emb)
            B, T, D = hidden_state.size()
            group_embeds = torch.cat((hidden_state, token_embeds), -1)  # [B, T, D*2]
            group_embeds = [group_embeds[b][prefix_length[b]:] for b in range(B)] # B x [T_b, 2D (concat style)]
            group_embeds = pad_sequence(group_embeds, batch_first=True, padding_value=0.)   
            
            target_logits = self.decoder(group_embeds)  # [B, T, N]
            target_logits = torch.stack(torch.chunk(target_logits, chunks=self.group, dim=-1), dim=1)   # [B, G, T//G, N]
            # outputs corresponding to cond_{1~G},emb_{1~G}; cond_{G+1~2G},emb_{G+1~2G};...; cond_{T*G~(T+1)*G}
            n_logits = target_logits.shape[-1]
            batch_target_logits = []
            for b in range(B):
                # [G, T//G, N] => [T//G, G, N] => [T, N]
                this_target_logits = target_logits[b, :, :target_length[b]-1].transpose(1, 0).reshape(-1, n_logits)   
                batch_target_logits.append(torch.cat(
                    (torch.zeros(prefix_length[b], n_logits).to(this_target_logits.device),
                    this_target_logits,
                    ), dim=0))
            return batch_target_logits

        assert token_embeds is not None
        if self.pattern != 'parallel':
            raise NotImplementedError()

        if self.decoder_type == 'transformer':
            assert target_length is not None
            batch_target_logits = AR_decoder(hidden_state, token_embeds, prefix_length, target_length)
        elif self.decoder_type == 'context_fc':
            batch_target_logits = context_fc_decoder(hidden_state, token_embeds, prefix_length)
        elif self.decoder_type == 'RQ_transformer':
            batch_target_logits = RQ_transformer(hidden_state, token_ids, prefix_length, target_length)
        elif self.decoder_type == 'ARQ_transformer':
            batch_target_logits = ARQ_transformer(hidden_state, token_ids, prefix_length, target_length)
        
        batch_target_logits = pad_sequence(batch_target_logits, batch_first=True, padding_value=0.)
        return batch_target_logits

    def decode_target_logits(self, target_logits, prefix_length, target_length,
                            token_embeds=None, target_ids=None):
        """
        target_logits: [B, T//G, G*D] => [B, G, T, D]
        prefix_length / target_length: [B]
        """
        prefix_length = prefix_length.long()
        if self.decoder_type in ['transformer', 'context_fc', 'RQ_transformer', 'ARQ_transformer']: # input: [B, T//G, D]
            return self.decode_hidden_state(target_logits, token_embeds, prefix_length, target_length, target_ids)
        
        if target_logits.ndim == 3: # [B, G, T, D]
            target_logits = torch.stack(torch.chunk(target_logits, chunks=self.group, dim=-1), dim=1)

        n_logits = target_logits.shape[-1]
        batch_size = target_logits.shape[0]
        batch_target_logits = []
        for b in range(batch_size):
            this_target_logits = target_logits[b, :, prefix_length[b]:prefix_length[b]+target_length[b]-1] # here, -1 to remove input shift (target_length - 1)
            if self.pattern == 'delay': # [G, T, n_logits] == reorganize ==> [T, G, n_logits]
                _target_logits = torch.stack([torch.roll(this_target_logits[d], -d) for d in range(self.group)]).transpose(1, 0)
            elif self.pattern == 'parallel':
                _target_logits = this_target_logits.transpose(1, 0)
            this_prefix_target_logits = target_logits[b, :, :prefix_length[b]].transpose(1, 0)  # [pT, G, n_logits]

            if self.R == 1:    #  [T, G, n_logits] => [(T*G), n_logits]
                _target_logits = _target_logits.reshape(-1, n_logits)
                this_prefix_target_logits = this_prefix_target_logits[:, 0] # [pT, n_logits]

            this_target_logits = torch.cat([
                this_prefix_target_logits,    #   no loss will be computed within prefix_length (including SOS token), so choose group[0]
                _target_logits,              
            ], dim=0)    
            batch_target_logits.append(this_target_logits)

        batch_target_logits = pad_sequence(batch_target_logits, batch_first=True, padding_value=0)
        
        return batch_target_logits
    
    def prepare_embed(self, requires: dict, batch: dict, **kwargs) -> dict:
        device = self.get_device()
        target_ids, target_lengths = self._get_target_ids_and_token_lengths(requires, batch, **kwargs)
        new_target_id = []
        for b in range(target_ids.shape[0]):
            eos_index = torch.where(target_ids[b] == self.eos_id)[0]
            if target_lengths[b] < 10 and eos_index < 10:
                logger.info(f"Warning: target length is too short: sample{b}: {target_lengths[b]}/{eos_index}, skip it")
                logger.info(batch['target_tokens_length'][b], batch['target_audio'][b].shape)
            else:
                new_target_id.append(target_ids[b])
        target_ids = torch.stack(new_target_id, dim=0)
        result = self.embed(token_ids=target_ids)
        target_embeds = result["embeds"]
        target_group_token_ids = result["group_token_ids"]
        target_ids_unpad = [self.decode_target_id(target_group_token_ids[b], delete_null=False, add_sos=True) for b in range(target_embeds.shape[0])]
        target_lengths = torch.LongTensor([target_ids_unpad[b].shape[0] for b in range(target_embeds.shape[0])]).to(device)
        # TODO (qq) return a list. placeholder for intermediate leadsheet tokens.
        return [
            {
                "token_embeds": target_embeds,
                "token_ids": target_ids_unpad,
                "token_ids_batched": target_ids,
                "token_length": target_lengths,
            }
        ]


class TokenXvalEmbedder(TokenEmbedder):
    """Xval-style embedding by multiplying lyrics token embedding by scaling factors"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        is_varlen: bool = False,
        id_key: str = "lyrics_tokens",
        length_key: str = "lyrics_tokens_length",
        coff_key: str = "lyrics_coffs",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
            *kwargs,
        )
        self.is_varlen = is_varlen
        self.id_key = id_key
        self.length_key = length_key
        self.coff_key = coff_key

    def embed(
        self,
        requires: Optional[Any] = None,  # not used
        batch: Optional[Any] = None,  # not used
        token_ids: Optional[torch.Tensor] = None,
        token_wise_multiplication: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        assert token_ids is not None, "token_ids must be provided"
        assert (
            token_wise_multiplication is not None
        ), "token_wise_multiplication must be provided"

        # token_wise_multiplication is the scaling factor to each embedding of token, 1.0 means no change, 0.0 means disable
        # Here we set (scaling factor) == (float time in second) as the representaion of time tokens
        token_ids = self.tokenize(requires, batch, token_ids)  # SOS and EOS added here
        embedding = self.embedder(token_ids)
        batch_size = token_wise_multiplication.shape[0]
        if self.with_sos:
            sos_token = self.get_sos_token(batch_size)
            sos_coff = torch.ones_like(sos_token).float()
            token_wise_multiplication = torch.cat(
                [sos_coff, token_wise_multiplication], dim=1
            )
        if self.with_eos:
            eos_token = self.get_eos_token(batch_size)
            eos_coff = torch.ones_like(eos_token).float()
            token_wise_multiplication = torch.cat(
                [token_wise_multiplication, eos_coff], dim=1
            )
        token_wise_multiplication = token_wise_multiplication.unsqueeze(-1)
        embedding = embedding * token_wise_multiplication
        return {
            "token_ids": token_ids,
            "embeds": embedding,
        }

    def prepare_embed(self, batch: dict) -> dict:
        lyrics_tokens = batch[self.id_key]
        batch_size = lyrics_tokens.shape[0]
        lyrics_token_length = batch[self.length_key]
        lyrics_coffs = batch.get(self.coff_key, torch.ones_like(lyrics_tokens))
        result = self.embed(
            token_ids=lyrics_tokens, token_wise_multiplication=lyrics_coffs
        )
        lyrics_embeds = result["embeds"]  # with SOS emb and EOS emb
        lyrics_tokens = result["token_ids"]  # with SOS token and EOS token
        if self.is_varlen:
            # NOTE: Do not use self-incrementing (e.g. length += 1)
            # otherwise it also modifies the tensor in the batch
            if self.with_sos:
                lyrics_token_length = lyrics_token_length + 1
            if self.with_eos:
                lyrics_token_length = lyrics_token_length + 1
            lyrics_tokens = unpad_sequence(
                lyrics_tokens, lyrics_token_length, batch_first=True
            )
            lyrics_embeds = unpad_sequence(
                lyrics_embeds, lyrics_token_length, batch_first=True
            )
        else:
            lyrics_token_length = lyrics_tokens.shape[1] + torch.zeros(
                (batch_size), device=self.get_device()
            )
        return {
            "token_embeds": lyrics_embeds,
            "token_ids": lyrics_tokens,
            "token_length": lyrics_token_length,
        }


class RotaryEmbedding2D(nn.Module):
    def __init__(self, h, w, dim, freq_scale=1.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer('inv_freq', inv_freq)
        self.freq_scale = freq_scale

        self.h, self.w = h, w
        grid = self._create_grid(h, w)
        rotation_matrix = self.get_rotation_matrix(grid)
        self.register_buffer('rotation_matrix', rotation_matrix)

    def _create_grid(self, h, w):
        rows = torch.arange(h, dtype=torch.float32)
        cols = torch.arange(w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1)
        return grid  # [H, W, 2]
    
    def get_rotation_matrix(self, grid):
        pos = grid * self.freq_scale
        sin_row = torch.sin(pos[..., 0:1] * self.inv_freq)
        cos_row = torch.cos(pos[..., 0:1] * self.inv_freq)
        sin_col = torch.sin(pos[..., 1:2] * self.inv_freq)
        cos_col = torch.cos(pos[..., 1:2] * self.inv_freq)

        rotation_matrix = torch.cat([cos_row, -sin_row, cos_col, -sin_col], dim=-1)
        return rotation_matrix.view(self.h, self.w, self.dim)


class TokenXvalPosEmbedder(TokenXvalEmbedder):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        add_sos: bool = False,
        add_eos: bool = False,
        with_sos: bool = False,
        with_eos: bool = False,
        is_varlen: bool = False,
        id_key: str = "lyrics_tokens",
        length_key: str = "lyrics_tokens_length",
        coff_key: str = "lyrics_coffs",
        pos_key: str = "lyrics_pos",
        mode: str = "concat",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
            is_varlen=is_varlen,
            id_key=id_key,
            length_key=length_key,
            coff_key=coff_key,
            *kwargs,
        )
        self.pos_key = pos_key
        if mode not in ["add", "concat"]:
            raise ValueError(f"Invalid mode {mode}")
        self.mode = mode

        self.blank_id = -1

        pos_emb_dim = embedding_dim
        if self.mode == "concat":
            pos_emb_dim = 256

        # @qinxin: here assume maximum of 200 lines and maximum 2000 phonemes per line
        # ![future warning]!
        # cfg definition for lyrics position can be quite tricky
        # currently ill position works well (ill position without section tag and linebreak: [0,1,-1],[0,2,-1],[0,3,-1],...,[0,n_phoneme,-1])
        self.rotary_emb = RotaryEmbedding2D(h=200, w=2000, dim=pos_emb_dim)
        self.pos_emb_fc = nn.Sequential(
            nn.Linear(pos_emb_dim, pos_emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim, bias=False),
        )
        nn.init.constant_(self.pos_emb_fc[0].weight, 0)
        nn.init.constant_(self.pos_emb_fc[-1].weight, 0)

        if self.mode == "concat":
            self.concat_fc = nn.Linear(embedding_dim * 2 + pos_emb_dim, embedding_dim, bias=False)

    def encode_2dpos(self, token_wise_position: torch.Tensor) -> torch.Tensor:
        # token_wise_position: [B, T, 2]
        # return: [B, T, embedding_dim]
        mask = (token_wise_position != self.blank_id).sum(-1).bool().unsqueeze(-1)  # [B, T, 1]
        token_wise_position[token_wise_position == self.blank_id] = 0
        position_embedding = torch.stack([
            self.rotary_emb.rotation_matrix[token_wise_position[b, :, 0], token_wise_position[b, :, 1]] 
            for b in range(token_wise_position.shape[0])], dim=0)   # [B, T, embedding_dim]
        masked_pos_emb = self.pos_emb_fc(position_embedding) * mask
        return masked_pos_emb
        
    def encode_section(self, section_ids: torch.Tensor) -> torch.Tensor:
        # section_ids: [B, T]
        # return: [B, T, embedding_dim]
        section_mask = (section_ids != self.blank_id).unsqueeze(-1) # [B, T, 1]
        section_ids[section_ids == self.blank_id] = 0
        return self.embedder(section_ids) * section_mask

    def embed(
        self,
        requires: Optional[Any] = None,  # not used
        batch: Optional[Any] = None,  # not used
        token_ids: Optional[torch.Tensor] = None,
        token_wise_multiplication: Optional[torch.Tensor] = None,
        token_wise_position: Optional[torch.Tensor] = None,
    ) -> dict:
        # token_wise_multiplication is the scaling factor to each embedding of token, 1.0 means no change, 0.0 means disable
        # Here we set (scaling factor) == (float time in second) as the representaion of time tokens
        token_ids = self.tokenize(requires, batch, token_ids)  # SOS and EOS added here
        embedding = self.embedder(token_ids)
        batch_size = token_wise_multiplication.shape[0]
        pos_size = token_wise_position.shape[2]
        if self.with_sos or self.with_eos:
            blank_pad = torch.ones((batch_size, 1, pos_size), dtype=torch.int32, device=self.get_device()) * self.blank_id
        else:
            blank_pad = None
        if self.with_sos:
            sos_token = self.get_sos_token(batch_size)
            sos_coff = torch.ones_like(sos_token).float()
            token_wise_multiplication = torch.cat([sos_coff, token_wise_multiplication], dim=1)
            token_wise_position = torch.cat([blank_pad, token_wise_position], dim=1)
        if self.with_eos:
            eos_token = self.get_eos_token(batch_size)
            eos_coff = torch.ones_like(eos_token).float()
            token_wise_multiplication = torch.cat([token_wise_multiplication, eos_coff], dim=1)
            token_wise_position = torch.cat([token_wise_position, blank_pad], dim=1)
        position_embedding = self.encode_2dpos(token_wise_position[..., :2])
        section_embedding = self.encode_section(token_wise_position[..., 2])
        token_wise_multiplication = token_wise_multiplication.unsqueeze(-1)
        if self.mode == "add":
            embedding = embedding * token_wise_multiplication + position_embedding + section_embedding
        elif self.mode == "concat":
            embedding = self.concat_fc(torch.cat((
                embedding * token_wise_multiplication,
                section_embedding,
                position_embedding,
            ), dim=-1))
        else:
            # should not reach here since mode has been checked in __init__
            raise NotImplementedError(f"Invalid mode {self.mode}")
        return {
            "token_ids": token_ids,
            "embeds": embedding,
        }

    def prepare_embed(self, batch: dict) -> dict:
        lyrics_tokens = batch[self.id_key]
        batch_size = lyrics_tokens.shape[0]
        lyrics_token_length = batch[self.length_key]
        lyrics_coffs = batch.get(self.coff_key, torch.ones_like(lyrics_tokens))
        lyrics_pos = batch.get(self.pos_key, torch.ones_like(lyrics_tokens).unsqueeze(-1) * self.blank_id)
        result = self.embed(
            token_ids=lyrics_tokens,
            token_wise_multiplication=lyrics_coffs, 
            token_wise_position=lyrics_pos,
        )
        lyrics_embeds = result["embeds"]  # with SOS emb and EOS emb
        lyrics_tokens = result["token_ids"]  # with SOS token and EOS token
        if self.is_varlen:
            # NOTE: Do not use self-incrementing (e.g. length += 1)
            # otherwise it also modifies the tensor in the batch
            if self.with_sos:
                lyrics_token_length = lyrics_token_length + 1
            if self.with_eos:
                lyrics_token_length = lyrics_token_length + 1
            lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
            lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
        else:
            lyrics_token_length = lyrics_tokens.shape[1] + torch.zeros((batch_size), device=self.get_device())
        return {
            "token_embeds": lyrics_embeds,
            "token_ids": lyrics_tokens,
            "token_length": lyrics_token_length,
        }


class MultiTagsEmbedder(BaseEmbedder):
    # MultiTags embedder - takes in list, and then embeds
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        add_sos=False,
        add_eos=False,
        with_sos=False,
        with_eos=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            add_sos=add_sos,
            add_eos=add_eos,
            with_sos=with_sos,
            with_eos=with_eos,
        )
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)

    @abstractmethod
    def get_tokens(self, requires, batch, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def get_sos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.sos_id is not None
        ), "Error getting sos id. Must initialize embedder with add_sos=True"
        device = self.get_device()
        sos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_id,
            dtype=torch.long,
            device=device,
        )
        return sos_ids

    def get_eos_token(self, batch_size: int) -> torch.Tensor:
        assert (
            self.eos_id is not None
        ), "Error getting eos id. Must initialize embedder with add_eos=True"
        device = self.get_device()
        eos_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.eos_id,
            dtype=torch.long,
            device=device,
        )
        return eos_ids

    def get_sos_embed(self, batch_size: int) -> torch.Tensor:
        return self.embedder(self.get_sos_token(batch_size))

    def get_eos_embed(self, batch_size: int) -> torch.Tensor:
        return self.embedder(self.get_eos_token(batch_size))

    def tokenize(
        self, requires=None, batch=None, token_ids=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()

    def embed(
        self, requires=None, batch=None, token_ids=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()


class MultiTagsCategoricalEmbedder(MultiTagsEmbedder):
    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 1024,
        add_sos: bool = False,
        add_eos: bool = False,
        id_key: str = "style_input_ids",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            **kwargs,
        )
        self.id_key = id_key

    def get_tokens(
        self, requires=None, batch: Optional[dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None, "Error getting tokens. Must provide batch"

        batch_tag_ids = []
        masks = []
        max_tags_num = 0
        style_input_ids = batch[self.id_key]
        for style_tags in style_input_ids:
            for tags in style_tags:
                tag_ids = tags
                batch_tag_ids.append(torch.tensor(tag_ids))
                masks.append(torch.tensor([1 for tag in tags]))
                max_tags_num = max(max_tags_num, len(tags))
        num_categories = max(
            len(x) for x in style_input_ids
        )  # they should all be the same, just in case
        batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        batch_tag_ids = torch.reshape(batch_tag_ids, [-1, num_categories, max_tags_num])
        masks = torch.reshape(masks, [-1, num_categories, max_tags_num])
        device = self.get_device()
        return torch.as_tensor(batch_tag_ids).to(device), torch.as_tensor(masks).to(
            device
        )

    def tokenize(
        self, requires=None, batch=None, token_ids=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, masks = self.get_tokens(requires, batch, **kwargs)
        return token_ids, masks

    def embed(self, requires=None, batch=None, token_ids=None) -> torch.Tensor:
        token_ids, masks = self.tokenize(requires, batch, token_ids)
        token_ids_shape = token_ids.shape
        _token_ids = torch.reshape(
            token_ids, [token_ids_shape[0] * token_ids_shape[1], token_ids_shape[2]]
        )
        _token_ids = _token_ids.to(dtype=torch.long)
        masks_shape = masks.shape
        _masks = torch.reshape(
            masks, [masks_shape[0] * masks_shape[1], masks_shape[2], 1]
        )
        _masks = _masks.to(dtype=torch.long)
        _embedding = self.embedder(_token_ids)
        _masks_sum = torch.sum(_masks, dim=1)
        _masks_sum = torch.where(
            _masks_sum > 0, _masks_sum, torch.ones_like(_masks_sum)
        )
        embedding = torch.sum(_embedding * _masks, dim=1) / _masks_sum
        embedding = torch.reshape(
            embedding, [token_ids_shape[0], token_ids_shape[1], -1]
        )
        if self.with_sos:
            sos_embedding = self.get_sos_embed(embedding.size(0))
            embedding = torch.cat([sos_embedding, embedding], dim=1)
        if self.with_eos:
            eos_embedding = self.get_eos_embed(embedding.size(0))
            embedding = torch.cat([embedding, eos_embedding], dim=1)
        return embedding

    def prepare_embed(self, batch: dict) -> dict:
        embeds = self.embed(batch=batch)
        return self.get_dummy_token_ids_and_length(embeds)


@torch.no_grad()
def get_mulan_embeds(
    requires, x, data_type="music", average=True, return_hidden_state=False
):
    if data_type == "music":
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"], music=x.float(), device=x.device, avg=average
        )
    elif data_type == "text":
        # x should be a list of strings
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"],
            text=x,
            device=requires["mulan"].device,
            return_hidden_state=return_hidden_state,
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return mulan_embeds


@torch.no_grad()
def get_bestrq_umm_tokens(requires, batch, chunk_size=None, **kwargs):
    if "Stage3" in requires:
        lit_module = requires["Stage3"]
    elif "Stage3Conv1D" in requires:
        lit_module = requires["Stage3Conv1D"]
    elif "convumm_gan_model" in requires:
        lit_module = requires["convumm_gan_model"]
    elif "UMM2_30s" in requires:
        lit_module = requires["UMM2_30s"]
    else:
        raise ValueError(f"Can't find UMM in requires")
    if chunk_size is None or batch.shape[-1] <= chunk_size:
        vq_ids = lit_module.wav2token(batch, **kwargs)
    else:
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        batch_size = batch.shape[0]
        batch_chunked = batch.unfold(-1, chunk_size, chunk_size)
        num_chunks = batch_chunked.shape[1]
        vq_ids = lit_module.wav2token(
            batch_chunked.reshape(batch_size * num_chunks, 1, chunk_size), **kwargs
        )
        vq_ids = vq_ids.reshape(batch_size, -1)
        # Compute leftover
        if num_chunks * chunk_size < batch.shape[-1]:
            samples_per_token = num_chunks * chunk_size // vq_ids.shape[-1]
            leftover_tokens = (
                batch.shape[-1] - num_chunks * chunk_size
            ) // samples_per_token
            vq_ids_leftover = lit_module.wav2token(
                batch[..., -chunk_size:].unsqueeze(1), **kwargs
            )
            vq_ids = torch.cat(
                [vq_ids, vq_ids_leftover[..., -leftover_tokens:]], dim=-1
            )
    return vq_ids


@torch.no_grad()
def get_bestrq_umm_outputs(requires, batch):
    lit_module = requires["Stage3"]
    assert hasattr(
        lit_module.model, "wav2token_alloutputs"
    ), f"For M1 training, UMM model ({type(lit_module.model)}) version must support wav2token_alloutputs function"
    return lit_module.model.wav2token_alloutputs(batch)


def _infer_batch_size(batch):
    batch_size = [
        len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)
    ][0]
    return batch_size

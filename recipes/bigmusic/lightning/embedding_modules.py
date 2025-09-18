import json
import random
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Dict, Any, Union, List
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch import distributed, Tensor
from collections import defaultdict
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length
from recipes.bigmusic.datasets.mir_data_util import NONE_LABEL, get_categorical_vocab
from recipes.bigmusic.datasets.utils.zh_vocab_dev import get_tag_map, prob_to_tag

import samantha.utils.hdfs_helper as hh
from samantha.utils.ctiga.inference_params import InferenceParams

import samantha
from mariana.utils.audio.audio_logger import AudioLogger
logger = AudioLogger()
import os


# Functions
@torch.no_grad()
def get_wav2vec_embeds(requires, x):
    b, t = x.size()
    feats, feat_mask = requires["ssl_frontend"](
        x, torch.LongTensor([t]).repeat([b]).to(x.device)
    )
    wav2vec_embeds, _ = requires["semantic"](feats, feat_mask)
    return wav2vec_embeds

@torch.no_grad()
def get_wav2vec_tokens(requires, x):
    wav2vec_tokens = w2v_bert_tokenization(
        frontend=requires["ssl_frontend"],
        w2v_model=requires["semantic"],
        wavs=x.float(),
        centers=requires["semantic_centers"],
        device=x.device,
    )
    return wav2vec_tokens

@torch.no_grad()
def get_soundstream_tokens(requires, x):
    output = requires["ss"](x.float())[2]
    output = torch.stack(output, dim=2)
    return output

@torch.no_grad()
def get_mulan_embeds(
    requires, x, data_type="music", average=True, return_hidden_state=False
):
    if data_type == "music":
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"],
            music=x.float(),
            device=x.device,
            avg=average,
        )
    elif data_type == "text":
        # x should be a list of strings        
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"],
            text=x,
            device=requires["mulan"].device,
            return_hidden_state=return_hidden_state
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return mulan_embeds

@torch.no_grad()
def get_mulan_embeds_2(
    requires: Dict,
    x: Union[Tensor, List[Tensor], List[str]],
    data_type: str = "music",
    average_audio_embd: bool = False,  # If true, average chunk audio embeddings
    normalize_audio_embd: bool = False,  # If true, normalize audio embedding to norm=1
    shift_seconds: float = 5.,  # Hop length in seconds
    return_hidden_state: bool = False
):
    """
    Get mulan embeddings for music or text data via the mulan_inference_2() function,
    which handles audio batching differently. @Yatong Bai

    Args:
        requires (Dict):
            A dictionary containing the required models and functions.
        x (Tensor or List[Tensor] or List[str]): 
            Input data for music or text.
        data_type (str, optional):
            Type of input data, either "music" or "text". Defaults to "music".
        average (bool, optional):
            Whether to average the embeddings. Defaults to True.
        return_hidden_state (bool, optional):
            Whether to return the hidden state. Defaults to False.

    Returns:
        torch.Tensor: The mulan embeddings.
    """
    if data_type == "music":
        mulan_embeds = requires["mulan_infer_fn_2"](
            model=requires["mulan"],
            music=x, text=None,
            average_audio_embd=average_audio_embd,
            normalize_audio_embd=normalize_audio_embd,
            shift_seconds=shift_seconds,
        )
    elif data_type == "text":
        # x should be a list of strings        
        mulan_embeds = requires["mulan_infer_fn_2"](
            model=requires["mulan"],
            text=x, music=None,
            return_hidden_state=return_hidden_state
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return mulan_embeds

@torch.no_grad()
def get_mulan_tokens(requires, x, data_type="music"):
    mulan_embeds = get_mulan_embeds(x, data_type=data_type)
    mulan_tokens, _ = requires["mulan_rvq_fn"](
        mulan_embeds, requires["mulan_centers"]
    )
    return mulan_tokens

@torch.no_grad()
def get_t5_embeds(requires, x):
    return requires['t5'](input_ids=x)['last_hidden_state']

@torch.no_grad()
def get_bestrq_umm_tokens(requires, batch, chunk_size=None, **kwargs):
    if 'Stage3' in requires:
        lit_module = requires['Stage3']
    elif 'Stage3Conv1D' in requires:
        lit_module = requires['Stage3Conv1D']
    elif 'convumm_gan_model' in requires:
        lit_module = requires['convumm_gan_model']
    elif 'UMM2_30s' in requires:
        lit_module = requires['UMM2_30s']
    else:
        raise ValueError(f"Can't find UMM in requires")
    if chunk_size is None or batch.shape[-1] <= chunk_size:
        vq_ids = lit_module.wav2token(batch, **kwargs)
        if isinstance(vq_ids, dict) and "vq_ids" in vq_ids:
            vq_ids = vq_ids["vq_ids"]
    else:
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        batch_size = batch.shape[0]
        batch_chunked = batch.unfold(-1, chunk_size, chunk_size)
        num_chunks = batch_chunked.shape[1]
        vq_ids = lit_module.wav2token(batch_chunked.reshape(batch_size * num_chunks, 1, chunk_size), **kwargs)
        vq_ids = vq_ids.reshape(batch_size, -1)
        # Compute leftover
        if num_chunks * chunk_size < batch.shape[-1]:
            samples_per_token = num_chunks * chunk_size // vq_ids.shape[-1]
            leftover_tokens = (batch.shape[-1] - num_chunks * chunk_size) // samples_per_token
            vq_ids_leftover = lit_module.wav2token(batch[..., -chunk_size:].unsqueeze(1), **kwargs)
            vq_ids = torch.cat([vq_ids, vq_ids_leftover[..., -leftover_tokens:]], dim=-1)
    return vq_ids

@torch.no_grad()
def get_bestrq_umm_outputs(requires, batch):
    lit_module = requires['Stage3']
    assert hasattr(lit_module.model, 'wav2token_alloutputs'), \
        f"For M1 training, UMM model ({type(lit_module.model)}) version must support wav2token_alloutputs function"
    return lit_module.model.wav2token_alloutputs(batch)

@torch.no_grad()
def get_bestrq_mkii_tokens(requires, batch):
    lit_module = requires['mkii']
    embeds, tokens = lit_module.tokenize(batch)
    return tokens

@torch.no_grad()
def get_bestrq_umm_embeds(requires, batch):
    lit_module = requires['Stage3']
    return lit_module.wav2embed(batch)

@torch.no_grad()
def get_umm2_outputs(requires, batch):
    lit_module = requires['umm2']
    return None


class BaseEmbedder(nn.Module):
    @abstractmethod
    def embed(self, requires, batch, token_ids=None):
        pass

    @abstractmethod
    def get_sos_embed(self, batch_size):
        pass


class ContinuousEmbedder(BaseEmbedder):
    def __init__(self, input_dim, embedding_dim, add_sos=False, add_eos=False, add_none=False):
        super().__init__()
        self.vocab_size = 0
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
        self.embedder = nn.Embedding(self.vocab_size, input_dim)

        if input_dim != embedding_dim:
            self.projection = nn.Linear(input_dim, embedding_dim, bias=False)
        else:
            self.projection = nn.Identity()

    @abstractmethod
    def get_embeds(self, requires, batch):
        # override to return embedding function
        raise NotImplementedError()

    def get_sos_token(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return sos_ids

    def get_sos_embed(self, batch_size):
        sos_ids = self.get_sos_token(batch_size)
        return self.projection(self.embedder(sos_ids))

    def get_eos_token(self, batch_size):
        assert self.eos_id is not None, "Error getting eos id. Must initialize embedder with add_eos=True"
        device = next(self.parameters()).device
        eos_ids = torch.full(size=(batch_size, 1), fill_value=self.eos_id, dtype=torch.long, device=device)
        return eos_ids

    def get_eos_embed(self, batch_size):
        eos_ids = self.get_eos_token(batch_size)
        return self.projection(self.embedder(eos_ids))

    def embed(self, requires, batch, with_sos=False, with_eos=False, **kwargs):
        embeds = self.get_embeds(requires, batch, **kwargs)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        if with_eos:
            eos_embed = self.get_eos_embed(embeds.size(0))
            embeds = torch.cat([embeds, eos_embed], dim=1)
        return embeds


class TokenEmbedder(BaseEmbedder):
    # Token embedder - takes in tokens, and then embeds
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_id = None
        self.eos_id = None
        if add_sos:
            self.vocab_size = self.vocab_size + 1
            self.sos_id = self.vocab_size - 1
        if add_eos:
            self.vocab_size = self.vocab_size + 1
            self.eos_id = self.vocab_size - 1
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)

    @abstractmethod
    def get_tokens(self, requires, batch, **kwargs):
        raise NotImplementedError()

    def get_sos_token(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return sos_ids

    def get_eos_token(self, batch_size):
        assert self.eos_id is not None, "Error getting eos id. Must initialize embedder with add_eos=True"
        device = next(self.parameters()).device
        eos_ids = torch.full(size=(batch_size, 1), fill_value=self.eos_id, dtype=torch.long, device=device)
        return eos_ids

    def get_sos_embed(self, batch_size):
        return self.embedder(self.get_sos_token(batch_size))

    def get_eos_embed(self, batch_size):
        return self.embedder(self.get_eos_token(batch_size))

    def tokenize(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False, **kwargs):
        if token_ids is None:
            token_ids = self.get_tokens(requires, batch, **kwargs)
        if with_sos:
            token_ids = torch.cat([self.get_sos_token(token_ids.size(0)), token_ids], dim=1)
        if with_eos:
            token_ids = torch.cat([token_ids, self.get_eos_token(token_ids.size(0))], dim=1)
        return token_ids

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        token_ids = self.tokenize(requires, batch, token_ids, with_sos, with_eos)
        return self.embedder(token_ids)


class LyricsTokenEmbedder(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, is_varlen=False):
        super().__init__(vocab_size, embedding_dim, add_sos=add_sos, add_eos=add_eos)
        self.is_varlen = is_varlen

    # TODO: (AS) add padding_idx
    def get_tokens(self, requires, input):
        return input
    
    @property
    def device(self):
        return next(self.parameters()).device

    def prepare_embed(self, batch, batch_size, conditions):
        lyrics_tokens, lyrics_token_length = self._prepare_embed(batch, batch_size, conditions)
        lyrics_embeds = self.embed(token_ids=lyrics_tokens, with_sos=False, with_eos=False).to(self.device)
        if self.is_varlen:
            lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
            lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
        return lyrics_embeds, lyrics_tokens, lyrics_token_length

    def _prepare_embed(self, batch, batch_size, conditions):
        """
        Return:
        - lyrics_tokens: Padded lyrics token with optionally sos_id and eos_id
        - lyrics_token_length: Original lyrics_token_length for varlen or padded lyrics_token_length for regular
        """
        if 'lyrics_tokens' in conditions:
            assert 'lyrics_tokens_length' in batch
            lyrics_token_length = batch['lyrics_tokens_length'].clone().to(self.device)
            lyrics_tokens = batch['lyrics_tokens'].clone().to(self.device)
            lyrics_tokens = self.tokenize(token_ids=lyrics_tokens, with_sos=False)
        else:
            lyrics_token_length = torch.zeros((batch_size), device=self.device)
            lyrics_tokens = torch.zeros((batch_size, 0)).long().to(self.device)
        
        padded_lyrics_token_length = lyrics_tokens.shape[1]
        
        # Pad lyrics_tokens to lyrics_token_length. This is for cfg lyrics tokens with tag dropout.
        if max(lyrics_token_length) > lyrics_tokens.shape[1]:
            lyrics_tokens_pad = torch.full((lyrics_tokens.shape[0], max(lyrics_token_length)), 0)
            lyrics_tokens_pad[:, :lyrics_tokens.shape[1]] = lyrics_tokens
            lyrics_tokens = lyrics_tokens_pad.to(self.device)

        if self.sos_id and self.eos_id:        
            lyrics_tokens = F.pad(lyrics_tokens, (1, 1), 'constant', 0)
            lyrics_tokens[:, 0] = self.sos_id
            eos_indices = (lyrics_token_length+1).unsqueeze(1)              # set last index to EOS
            lyrics_tokens.scatter_(1, eos_indices, self.eos_id)
            lyrics_token_length += 2                                        # +2 for eos and sos
            padded_lyrics_token_length += 2
        
        if not self.is_varlen:
            lyrics_token_length = padded_lyrics_token_length

        return lyrics_tokens, lyrics_token_length


class LyricsTokenSectionEmbedder(LyricsTokenEmbedder):
    """
    Treat the positions of all -1 and -2 values in batch.lyrics_tokens as the positions of XVal duration embeddings
    (Multiply the embedding by batch.section_durations).
    -1 indicates section duration, -2 indicates slice duration.
    """
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, is_varlen=False):
        BaseEmbedder.__init__(self)         # call grandparent
        self.is_varlen = is_varlen
        self.vocab_size = vocab_size
        self.dur_id = None
        self.global_dur_id = None
        self.sos_id = None
        self.eos_id = None

        # add_dur
        self.vocab_size = self.vocab_size + 1
        self.dur_id = self.vocab_size - 1

        # add global_dur
        self.vocab_size = self.vocab_size + 1
        self.global_dur_id = self.vocab_size - 1

        if add_sos:
            self.vocab_size = self.vocab_size + 1
            self.sos_id = self.vocab_size - 1
        if add_eos:
            self.vocab_size = self.vocab_size + 1
            self.eos_id = self.vocab_size - 1

        self.embedder = nn.Embedding(self.vocab_size, embedding_dim)

    def normalize_timestamp(self, x, max_duration=240, max_timestamp=10):
        # normalize the timestamp between [0, 5]
        x = torch.clamp(x, max=max_duration)
        return x / max_duration * max_timestamp

    # def insert_token_at_positions(self, tokens, positions, new_token):
    #     """
    #     example usage:
    #     - input:
    #         tokens = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
    #         positions = torch.tensor([[0, 2, -1]])
    #         new_token = 100
    #     - output:
    #         out_tokens = torch.tensor([[1, 100, 2, 3, 100, 0, 0, 0, 0, 0, 0, 0, 0]])
    #     """
    #     out_tokens = torch.zeros(len(tokens) + len(positions), dtype=tokens.dtype)
    #     orig_index = 0
    #     new_index = 0
    #     for pos in positions[positions != -1]:
    #         pos = pos + 1
    #         out_tokens[new_index:new_index + pos - orig_index] = tokens[orig_index:pos]
    #         new_index += pos - orig_index
    #         orig_index = pos
    #         out_tokens[new_index] = new_token
    #         new_index += 1
    #     out_tokens[new_index:new_index + len(tokens) - orig_index] = tokens[orig_index:]
    #     return out_tokens
    
    # def multiply_embeddings_by_duration(self, lyrics_embeds, section_positions, section_durations):
    #     """
    #     example usage:
    #     - input:
    #         lyrics_embeds = torch.tensor([
    #             [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], 
    #             [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    #         ])
    #         section_positions = torch.tensor([[1, 4, -1], [1, 3, 5]])
    #         section_durations = torch.tensor([[5, 10, 0], [5, 20, 10]])
    #     - output:
    #         lyrics_embeds = torch.tensor([
    #             [[1, 1], [5, 5], [1, 1], [1, 1], [10, 10], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
    #             [[1, 1], [5, 5], [1, 1], [20, 20], [1, 1], [10, 10], [1, 1], [1, 1], [1, 1], [1, 1]],
    #         ])
    #     """
    #     mask = section_positions != -1
    #     batch_indices = torch.arange(lyrics_embeds.size(0)).unsqueeze(1).expand_as(section_positions).to(lyrics_embeds.device)
    #     flat_positions = section_positions[mask]
    #     flat_durations = section_durations[mask]
    #     flat_batch_indices = batch_indices[mask]
    #     flat_durations = flat_durations.unsqueeze(-1).expand(-1, lyrics_embeds.size(2))
    #     lyrics_embeds[flat_batch_indices, flat_positions] *= flat_durations
        
    #     return lyrics_embeds
    
    def multiple_embedding_at_indices(self, lyrics_embeds, duration_values, indices):

        durations_expanded = torch.ones_like(lyrics_embeds)
        batch_indices, token_indices = indices.nonzero(as_tuple=True)
        duration_indices = indices.cumsum(dim=1)[indices] - 1
        durations_expanded[batch_indices, token_indices] = duration_values[batch_indices, duration_indices].unsqueeze(-1)
        lyrics_embeds *= durations_expanded
        return lyrics_embeds

    def prepare_embed(self, batch, batch_size, conditions):

        lyrics_tokens, lyrics_token_length = self._prepare_embed(batch, batch_size, conditions)
        lyrics_tokens[lyrics_tokens == -1] = self.dur_id
        lyrics_tokens[lyrics_tokens == -2] = self.global_dur_id

        lyrics_embeds = self.embed(token_ids=lyrics_tokens, with_sos=False, with_eos=False).to(self.device)

        sec_dur_indices = (lyrics_tokens == self.dur_id)
        if self.dur_id in lyrics_tokens:
            section_durations = self.normalize_timestamp(batch['section_durations']).to(self.device)
            lyrics_embeds = self.multiple_embedding_at_indices(lyrics_embeds, section_durations, sec_dur_indices)
        
        slice_dur_indices = (lyrics_tokens == self.global_dur_id)
        if self.global_dur_id in lyrics_tokens:
            slice_durations = self.normalize_timestamp(batch['slice_duration']).unsqueeze(-1).to(self.device)
            lyrics_embeds = self.multiple_embedding_at_indices(lyrics_embeds, slice_durations, slice_dur_indices)

        if self.is_varlen:
            lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
            lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
        else:
            lyrics_token_length = lyrics_tokens.shape[1] + torch.zeros((batch_size), device=self.device)

        return lyrics_embeds, lyrics_tokens, lyrics_token_length
        

    # def prepare_embed(self, batch, batch_size, conditions):
    #     lyrics_tokens, lyrics_token_length = self._prepare_embed(batch, batch_size, conditions)

    #     section_positions = batch['section_tag_token_pos'].to(self.device)              # section_positions: [[0 -1 -1 -1 -1], [1 2 3 4 5], [-1 -1 -1 -1 -1], [1 2 -1 -1 -1]]   (all -1 means vocal type)
    #     section_durations = self.normalize_timestamp(batch['section_durations']).to(self.device)
    #     global_duration = batch['slice_duration'].to(self.device)

    #     # handle empty (e.g., after section_tag cfg)
    #     if section_positions.numel() == 0:
    #         section_positions = torch.tensor([[-1]] * batch_size).to(self.device)
    #         section_durations = torch.tensor([[0]] * batch_size).to(self.device)
        
    #     # shift position by 1 after adding sos token
    #     if self.sos_id:         
    #         mask = section_positions != -1
    #         section_positions[mask] += 1                                # section_positions: [[1 -1 -1 -1 -1], [2 3 4 5 6], [-1 -1 -1 -1 -1], [2 3 -1 -1 -1]
    #     valid_section_positions = [tensor[tensor!=-1] for tensor in section_positions]
    #     section_counts = torch.tensor( [len(x) for x in valid_section_positions] ).to(self.device)      # section_counts: [1, 5, 0, 2]

    #     lyrics_tokens = torch.stack([                                   # lyrics_tokens[0]: [2000 1620 2 428 442 1595 422 593 1595 1593 431 502 1595 1593 1401 ...]
    #         self.insert_token_at_positions(tokens, positions, self.dur_id) for
    #         tokens, positions in zip(lyrics_tokens, section_positions)
    #     ]).to(self.device)                                              # lyrics_tokens[0]: [2000 1620 2002 2 428 442 1595 422 593 1595 1593 431 502 1595 1593 1401 ...]

    #     # update the token positions
    #     offsets = torch.arange(1, section_positions.size(1) + 1).unsqueeze(0).to(self.device)
    #     mask = section_positions != -1
    #     section_positions = torch.where(mask, section_positions + offsets, section_positions)   # section_positions: [[ 1, -1, -1, -1, -1], [ 2,  4,  6,  8, 10], [ -1, -1, -1, -1, -1], [ 2,  4, -1, -1, -1]]
    #     lyrics_token_length = lyrics_token_length + section_counts      # lyrics_token_length: [272, 12, 360, 6] (previous: [271, 7, 359, 4])

    #     # add global duration token when global_duration>0
    #     global_duration_indices = (global_duration > 0).nonzero(as_tuple=True)[0]
    #     global_dur_pos = 1 if self.sos_id else 0        # make sure the global duration token is after sos token (if exists)
    #     lyrics_tokens[global_duration_indices, 1+global_dur_pos:] = lyrics_tokens[global_duration_indices, global_dur_pos:-1]
    #     lyrics_tokens[global_duration_indices, global_dur_pos] = self.global_dur_id
    #     lyrics_token_length[global_duration_indices] += 1
    #     lyrics_embeds = self.embed(token_ids=lyrics_tokens, with_sos=False, with_eos=False).to(self.device)

    #     # multiply the duration embedding by normalized durations
    #     lyrics_embeds = self.multiply_embeddings_by_duration(lyrics_embeds, section_positions, section_durations)

    #     # multiply the global duration embedding by the global normalized durations (if applied)
    #     if global_duration_indices.numel() > 0:
    #         lyrics_embeds[global_duration_indices, global_dur_pos, :] *= self.normalize_timestamp(global_duration)[global_duration_indices].unsqueeze(1).to(self.device)

    #     if self.is_varlen:
    #         lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
    #         lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
    #     else:
    #         lyrics_token_length = lyrics_tokens.shape[1]

    #     return lyrics_embeds, lyrics_tokens, lyrics_token_length


class XValEmbedder(nn.Module):
    def __init__(self, embedding_dim, max_value=120, norm_max_value=10) -> None:
        super().__init__()

        self.token_id = 1   # there is only one token, which is set to 1
        self.pad_id = 0
        self.vocab_size = 2    # single_token + <pad> (no eos)
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim)
        self.max_value = max_value  
        self.norm_max_value = norm_max_value  
        self.logged = 0

    def normalize(self, x):
        x = torch.clamp(x, max=self.max_value)
        norm = x / self.max_value * self.norm_max_value
        norm[x <= 0] = 1  # always use the pad embedding where x <= 0, no need to scale the embedding in this case
        return norm

    # input can be either 2d tensor (section-level) or 1d tensor (sample-level)
    def embed(self, input):
        if input.ndim == 1: 
            input = input.unsqueeze(1)

        device = next(self.parameters()).device
        token_ids = torch.where(input > 0, torch.tensor(self.token_id), torch.tensor(self.pad_id)).long().to(device)
        # normalize the input to range
        normalized_input = self.normalize(input).float()
        embeds = self.embedder(token_ids) * normalized_input.unsqueeze(2)
        if self.logged < 5:
            print(f"Xval input: {input}")
            print(f"token_ids ({token_ids.shape}): {token_ids}")
            print(f"normalized input ({normalized_input.shape}): {normalized_input}")
            self.logged += 1

        return embeds, token_ids, normalized_input


class InstrumentEmbedder(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, is_varlen=False):
        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)
        self.is_varlen = is_varlen

    def add_sos_eos_id(self, padded_tokens, ori_token_length, with_sos=False, with_eos=False):
        """
        padded_tokens: [B, T_max]
        token_length: [B]
        """
        padded_token_length = padded_tokens.shape[1]
        token_length = ori_token_length.clone()

        # Pad tokens to token_length. This is for cfg inst tokens with inst dropout.
        if max(token_length) > padded_token_length:
            padded_tokens = F.pad(padded_tokens, (0, max(token_length) - padded_token_length), 'constant', 0)

        if self.sos_id and with_sos:
            padded_tokens = F.pad(padded_tokens, (1, 0), 'constant', 0)
            padded_tokens[:, 0] = self.sos_id
            token_length += 1                                        # +1 for sos
            padded_token_length += 1

        if self.eos_id and with_eos:        
            padded_tokens = F.pad(padded_tokens, (0, 1), 'constant', 0)
            eos_indices = token_length.unsqueeze(1)              # set last index to EOS
            padded_tokens.scatter_(1, eos_indices, self.eos_id)
            token_length += 1                                           # +1 for eos
            padded_token_length += 1

        if not self.is_varlen:
            token_length = padded_token_length

        return padded_tokens, token_length

    def prepare_embed(self, requires, input: torch.Tensor, token_length: torch.Tensor, with_sos=False, with_eos=False):
        """
        input: [B, N_max_inst]
        token_length: [B]
        """
        batch_size = input.shape[0]
        # padded_tokens: [B, T_max+(with_sos)+(with_eos)], token_length: [B]
        padded_token_ids, token_length = self.add_sos_eos_id(input, token_length, with_sos, with_eos)
        padded_token_embed = self.embedder(padded_token_ids)

        if self.is_varlen:
            token_ids = unpad_sequence(padded_token_ids, token_length, batch_first=True)
            token_embed = unpad_sequence(padded_token_embed, token_length, batch_first=True)
        else:
            token_length = token_embed.shape[1] + torch.zeros((batch_size), device=self.device)
            token_embed = padded_token_embed
            token_ids = padded_token_ids

        return token_ids, token_embed, token_length


class XvalDurationEmbedder(nn.Module):

    def __init__(self, embedding_dim, max_duration=120, max_timestamp=10):
        super().__init__()

        self.token_id = 1   # there is only one token, which is set to 1
        self.pad_id = 0
        self.vocab_size = 2    # single_token + <pad> (no eos)
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim)
        self.max_duration = max_duration    # max of possible section duration
        self.max_timestamp = max_timestamp
        self.logged = 0

    def normalize_timestamp(self, x):
        # normalize the timestamp between [0, 5]
        x = torch.clamp(x, max=self.max_duration)
        return x / self.max_duration * self.max_timestamp

    # durations can be either 2d tensor (section_durations) or 1d tensor (global duration)
    def embed(self, durations):

        if durations.ndim == 1:     # in case of global duration, durations
            durations = durations.unsqueeze(1)

        device = next(self.parameters()).device
        token_ids = torch.where(durations != 0, torch.tensor(self.token_id), torch.tensor(self.pad_id)).long().to(device)
        # normalize the duration to range
        normalized_timestamps = self.normalize_timestamp(durations).float()
        embeds = self.embedder(token_ids) * normalized_timestamps.unsqueeze(2)
        if self.logged < 5:
            print(f"durations: {durations}")
            print(f"token_ids ({token_ids.shape}): {token_ids}")
            print(f"normalized_timestamps ({normalized_timestamps.shape}): {normalized_timestamps}")
            self.logged += 1

        return embeds, token_ids, normalized_timestamps


class BpeTokenEmbedder(BaseEmbedder):
    def __init__(self, tokenizer_path, device, input_key="raw_lyrics"):

        from transformers import AutoTokenizer

        super().__init__()
        assert os.path.exists(tokenizer_path), f"{tokenizer_path} not found in local"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.input_key = input_key
        self.device = device
        self.eos_id = self.tokenizer.eos_token_id
        self.sos_id = self.tokenizer.bos_token_id

    def get_tokens(self, batch=None):
        # print(f"BpeTokenEmbedder {self.input_key} {len(batch[self.input_key])}")
        # for idx, lyrics in enumerate(batch[self.input_key]):
        #     print(f"[{idx}]:{lyrics}")

        tokens = self.tokenizer(batch[self.input_key])['input_ids']
        token_length = torch.tensor([len(token) for token in tokens], dtype=torch.long, device=self.device)
        tokens = [torch.tensor(token, dtype=torch.long, device=self.device) for token in tokens]
        
        return tokens, token_length
        


class LyricsTokenEmbedderV2(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, is_varlen=False, **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            *kwargs,
        )
        self.is_varlen = is_varlen

    def embed(
        self,
        requires=None,
        batch=None,
        token_ids=None,
        with_sos=False,
        with_eos=False,
        token_wise_multiplication=None,
    ):
        # token_wise_multiplication is the scaling factor to each embedding of token, 1.0 means no change, 0.0 means disable
        # Here we set (scaling factor) == (float time in second) as the representaion of time tokens
        token_ids = self.tokenize(requires, batch, token_ids, with_sos, with_eos)
        embedding = self.embedder(token_ids)
        if with_sos:
            sos_token = self.get_sos_token(token_wise_multiplication.size(0))
            sos_coff = torch.ones_like(sos_token).float() # for sos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([sos_coff, token_wise_multiplication], dim=1)
        if with_eos:
            eos_coff = self.get_sos_token(token_wise_multiplication.size(0))
            eos_coff = torch.ones_like(eos_coff).float() # for eos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([token_wise_multiplication, eos_coff], dim=1)
        token_wise_multiplication = token_wise_multiplication.unsqueeze(-1)
        embedding = embedding * token_wise_multiplication
        return embedding

    def prepare_embed(self, batch, batch_size, conditions):
        assert "lyrics_tokens" in (conditions.split(",") if isinstance(conditions, str) else conditions)
        lyrics_tokens = self.get_tokens(batch=batch)
        lyrics_token_length = batch["lyrics_tokens_length"]
        lyrics_coffs = batch.get("lyrics_coffs", torch.ones_like(lyrics_tokens))
        lyrics_embeds = self.embed(token_ids=lyrics_tokens, token_wise_multiplication=lyrics_coffs)
        if self.is_varlen:
            lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
            lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
        else:
            lyrics_token_length = lyrics_tokens.shape[1] + torch.zeros((batch_size), device=self.device)
        return lyrics_embeds, lyrics_tokens, lyrics_token_length

    def get_tokens(self, requires=None, batch=None, **kwargs):
        return batch["lyrics_tokens"]


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


class LyricsTokenPosEmbedderV2(LyricsTokenEmbedderV2):
    def __init__(
        self, vocab_size, embedding_dim, add_sos=False, add_eos=False, is_varlen=False, mode="concat", **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            add_sos=add_sos,
            add_eos=add_eos,
            *kwargs,
        )
        self.is_varlen = is_varlen
        self.blank_id = -1

        self.mode = mode
        pos_emb_dim = embedding_dim
        if self.mode == "concat":
            pos_emb_dim = 256

        # @qinxin: here assume maximum of 200 lines and maximum 2000 phonemes per line
        # ![future warning]!
        # cfg definition for lyrics position can be quite tricky
        # currently ill position works well (ill position without section tag and linebreak: [0,1,-1],[0,2,-1],[0,3,-1],...,[0,n_phoneme,-1])
        self.rotary_emb = RotaryEmbedding2D(h=200, w=2000, dim=pos_emb_dim)
        self.rotary_emb_bak = RotaryEmbedding2D(h=200, w=4000, dim=pos_emb_dim)
        self.pos_emb_fc = nn.Sequential(
            nn.Linear(pos_emb_dim, pos_emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim, bias=False),
        )
        nn.init.constant_(self.pos_emb_fc[0].weight, 0)
        nn.init.constant_(self.pos_emb_fc[-1].weight, 0)

        if self.mode == "concat":
            self.concat_fc = nn.Linear(embedding_dim * 2 + pos_emb_dim, embedding_dim, bias=False)

    def encode_2dpos(self, token_wise_position):
        # token_wise_position: [B, T, 2]
        # return: [B, T, embedding_dim]
        mask = (token_wise_position != self.blank_id).sum(-1).bool().unsqueeze(-1)  # [B, T, 1]
        token_wise_position[token_wise_position == self.blank_id] = 0
        position_embedding = torch.stack([
            self.rotary_emb_bak.rotation_matrix[token_wise_position[b, :, 0], token_wise_position[b, :, 1]] 
            # self.rotary_emb.rotation_matrix[token_wise_position[b, :, 0], token_wise_position[b, :, 1]] 
            for b in range(token_wise_position.shape[0])], dim=0)   # [B, T, embedding_dim]
        masked_pos_emb = self.pos_emb_fc(position_embedding) * mask
        return masked_pos_emb
        
    def encode_section(self, section_ids):
        # section_ids: [B, T]
        # return: [B, T, embedding_dim]
        section_mask = (section_ids != self.blank_id) # [B, T]
        # section_ids[section_ids == self.blank_id] = 0
        section_ids = section_ids * section_mask
        return self.embedder(section_ids) * section_mask.unsqueeze(-1)

    def embed(
        self,
        requires=None,
        batch=None,
        token_ids=None,
        with_sos=False,
        with_eos=False,
        token_wise_multiplication=None,
        token_wise_position=None,
    ):
        # token_wise_multiplication is the scaling factor to each embedding of token, 1.0 means no change, 0.0 means disable
        # Here we set (scaling factor) == (float time in second) as the representaion of time tokens
        token_ids = self.tokenize(requires, batch, token_ids, with_sos, with_eos)
        embedding = self.embedder(token_ids)
        position_embedding = self.encode_2dpos(token_wise_position[..., :2])
        section_embedding = self.encode_section(token_wise_position[..., 2])

        if with_sos:
            sos_token = self.get_sos_token(token_wise_multiplication.size(0))
            sos_coff = torch.ones_like(sos_token).float() # for sos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([sos_coff, token_wise_multiplication], dim=1)
        if with_eos:
            eos_coff = self.get_sos_token(token_wise_multiplication.size(0))
            eos_coff = torch.ones_like(eos_coff).float() # for eos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([token_wise_multiplication, eos_coff], dim=1)
        token_wise_multiplication = token_wise_multiplication.unsqueeze(-1)

        if self.mode == "add":
            embedding = embedding * token_wise_multiplication + position_embedding + section_embedding
        elif self.mode == "concat":
            embedding = self.concat_fc(torch.cat((
                embedding * token_wise_multiplication,
                section_embedding,
                position_embedding,
            ), dim=-1))

        return embedding

    def prepare_embed(self, batch, batch_size, conditions):
        assert "lyrics_tokens" in (conditions.split(",") if isinstance(conditions, str) else conditions)
        lyrics_tokens = self.get_tokens(batch=batch)
        lyrics_token_length = batch["lyrics_tokens_length"]
        lyrics_coffs = batch.get("lyrics_coffs", torch.ones_like(lyrics_tokens))
        lyrics_pos = batch.get("lyrics_pos", torch.ones_like(lyrics_tokens).unsqueeze(-1) * -1)

        lyrics_embeds = self.embed(
            token_ids=lyrics_tokens, token_wise_multiplication=lyrics_coffs, 
            token_wise_position=lyrics_pos
        )

        if self.is_varlen:
            lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_token_length, batch_first=True)
            lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_token_length, batch_first=True)
        else:
            lyrics_token_length = lyrics_tokens.shape[1] + torch.zeros((batch_size), device=self.device)
        return lyrics_embeds, lyrics_tokens, lyrics_token_length


class TagCategoricalEmbedder(TokenEmbedder):
    def __init__(
            self,
            vocab_type='auto',
            max_vocab_size=1024,
            embedding_dim=1024,
            add_sos=False,
            dropout=0.0,
            category_separator="|",
        ):
        if vocab_type == 'auto':
            assert max_vocab_size
            vocab2id = { NONE_LABEL: 0 }
            vocab_size = max_vocab_size
            _num_categories = 5  # (the previous default value)
        else:
            vocab2id, _num_categories = get_categorical_vocab(vocab_type)  # infer num_categories from vocab_type
            vocab_size = max(vocab2id.values()) + 1
        super().__init__(vocab_size, embedding_dim, add_sos)
        self.vocab2id = vocab2id
        self.vocab2count = defaultdict(int)
        self.vocab_type = vocab_type
        self.dropout = dropout
        self.category_separator = category_separator
        self.num_categories = _num_categories

    def sync_tags(self, batch_style_tags):
        "For auto tags, must sync new tags across all workers first for consistent vocab2id"
        if not self.training: return # do not run on validation set - to prevent deadlocks
        all_tags = [tag for style_tags in batch_style_tags for tag in style_tags]
        if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1:
            world_size = distributed.get_world_size()
            gathered_tags = [None for _ in range(world_size)]
            distributed.all_gather_object(gathered_tags, all_tags)
            all_tags = [tag for tags in gathered_tags for tag in tags if tag is not None]

        # update tag counts
        for tag in all_tags: self.vocab2count[tag] += 1

        new_tags = sorted(set(all_tags) - set(self.vocab2id.keys()))
        for new_tag in new_tags:
            if new_tag in self.vocab2id:
                print('Warning: tag already exists. Error.', self.vocab2id)
            if new_tag not in self.vocab2id and len(self.vocab2id) < self.vocab_size:
                self.vocab2id[new_tag] = len(self.vocab2id)

    def get_tag_id(self, tag, dropout=0.0):
        if tag not in self.vocab2id:
            if not self.training and len(tag.strip()) > 0:  # use NONE_LABEL for empty label
                logger.warning(f"Inference Error: Tag {tag} not found in vocab {self.vocab2id}. Please check vocab")
            tag = NONE_LABEL
        if self.training and random.random() < dropout:
            tag = NONE_LABEL
        return self.vocab2id[tag]

    def get_tokens(self, requires, style_texts):
        batch_style_tags = []
        for style_text in style_texts:
            # Strictly separate style_text by the separator
            if isinstance(style_text, str):
                separator = self.category_separator
                # normalized_style_text = style_text.replace("，", separator).replace(",", separator)
                style_tags = [t.strip() for t in style_text.split(separator)]
            elif isinstance(style_text, list):
                style_tags = style_text
            elif isinstance(style_text, dict):
                # TODO: handle use case where dictionary is not sorted
                # style_tag_list = [v for k,v in sorted(style_text.items())]
                style_tags = style_text.values()
            batch_style_tags.append(style_tags)

        # for auto vocab, sync across multiple workers
        self.sync_tags(batch_style_tags)

        batch_tag_ids = []
        for style_tags in batch_style_tags:            
            if self.num_categories and len(style_tags) != self.num_categories:
                tag_ids = [self.get_tag_id(NONE_LABEL)] * self.num_categories
            else:
                tag_ids = [self.get_tag_id(tag, self.dropout) for tag in style_tags]
            batch_tag_ids.append(tag_ids)

        device = next(self.parameters()).device
        return torch.as_tensor(batch_tag_ids).to(device)

    # save auto-growing vocab for inference
    def set_extra_state(self, state: Any): 
        self.vocab2id = state['vocab'] 
        self.vocab2count.update(state.get('counts', {}))
    def get_extra_state(self) -> Any: return { 'vocab': self.vocab2id, 'counts': dict(self.vocab2count) }


class MulanCategoricalEmbedder(BaseEmbedder):
    def __init__(
            self,
            vocab_size=1024,
            input_dim=512,
            embedding_dim=1024,
            min_audio_length=10*24000,
            add_sos=False,
            dropout=0.0,
            category_separator="|",
            vocab_path=None,
        ):
        super().__init__()
        self.none_id = 0
        self.vocab_path = vocab_path
        if vocab_path is not None:
            with hh.hopen(vocab_path) as f:
                self.vocab2id = json.load(f)
            vocab_size = len(self.vocab2id)
            self.vocab_size = vocab_size
        else:
            self.vocab2id = { NONE_LABEL: self.none_id }
            self.vocab_size = vocab_size
            
        self.sos_id = None
        self.eos_id = None
        self.dropout = dropout
        if add_sos:
            self.vocab_size = self.vocab_size + 1
            self.sos_id = self.vocab_size - 1
        self.embedder = nn.Embedding(self.vocab_size, input_dim)
        self.category_separator = category_separator

        if input_dim != embedding_dim:
            self.projection = nn.Linear(input_dim, embedding_dim, bias=False)
        else:
            self.projection = nn.Identity()

        self.vocab2count = defaultdict(int)
        self.dropout = dropout
        self.min_audio_length = min_audio_length # 10s * 24k sample rate
        self.warning_count = 0

    # TokenEmbedder
    def get_sos_token(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return sos_ids

    def get_sos_embed(self, batch_size):
        return self.projection(self.embedder(self.get_sos_token(batch_size)))

    def sync_tags(self, batch_style_tags):
        "For auto tags, must sync new tags across all workers first for consistent vocab2id"
        if not self.training: return # do not run on validation set - to prevent deadlocks
        all_tags = [tag for style_tags in batch_style_tags for tag in style_tags]
        if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1:
            world_size = distributed.get_world_size()
            gathered_tags = [None for _ in range(world_size)]
            distributed.all_gather_object(gathered_tags, all_tags)
            all_tags = [tag for tags in gathered_tags for tag in tags if tag is not None]

        # update tag counts
        for tag in all_tags: self.vocab2count[tag] += 1

        new_tags = sorted(set(all_tags) - set(self.vocab2id.keys()))
        for new_tag in new_tags:
            if new_tag in self.vocab2id:
                logger.warn(f'Warning: tag already exists. Error. {self.vocab2id}')
            if new_tag not in self.vocab2id and len(self.vocab2id) < self.vocab_size:
                self.vocab2id[new_tag] = len(self.vocab2id)

    def get_tag_id(self, tag, dropout=0.0):
        if tag == '': tag = NONE_LABEL
        if tag not in self.vocab2id:
            if not self.training:
                logger.warn(f"Inference Error: Tag {tag} not found in vocab. Please check vocab")
                if self.warning_count < 0: logger.warn(f"Vocab IDs: {self.vocab2id}")
                self.warning_count += 1
                # raise Exception('Tag not found error')
            tag = NONE_LABEL
        if self.training and random.random() < dropout:
            tag = NONE_LABEL
        return self.vocab2id[tag]

    def get_tokens(self, requires, style_texts):
        # style_text: [['Acoustic'], ['Trap'], ['EDM'], ['EDM']]
        batch_style_tags = []
        for style_text in style_texts:
            # accepts comma separated string or ordered list/dict of category values
            if isinstance(style_text, str):
                separator = self.category_separator
                normalized_style_text = style_text.replace("，", separator).replace(",", separator)
                style_tags = normalized_style_text.split(separator)
            elif isinstance(style_text, list):
                style_tags = style_text
            elif isinstance(style_text, dict):
                style_tags = style_text.values()
            batch_style_tags.append(style_tags)

        self.sync_tags(batch_style_tags)

        batch_tag_ids = []
        for style_tags in batch_style_tags:
            tag_ids = [self.get_tag_id(tag, self.dropout) for tag in style_tags]
            batch_tag_ids.append(torch.tensor(tag_ids))
        # batch_tag_ids: [tensor([3]), tensor([4]), tensor([2]), tensor([2])]

        batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True, padding_value=self.none_id)
        # batch_tag_ids: tensor([[3], [4], [2], [2]])
        device = next(self.parameters()).device
        return torch.as_tensor(batch_tag_ids).to(device)

    # save auto-growing vocab for inference
    def set_extra_state(self, state: Any): 
        if self.vocab_path is not None:
            print('Re-using existing vocab', self.vocab2id)
            return
        self.vocab2id = state['vocab'] 
        self.vocab2count.update(state.get('counts', {}))
        print('Loading Vocab counts', self.vocab2count)
    def get_extra_state(self) -> Any: return { 'vocab': self.vocab2id, 'counts': dict(self.vocab2count) }

    def embed(self, requires, batch, with_sos=False, **kwargs):
        embeds = self.get_embeds(requires, batch, **kwargs)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        return embeds
    
    @staticmethod
    def masked_mean(emb, tokens, padding_idx):
        mask = (tokens != padding_idx)
        denom = torch.sum(mask, -1, keepdim=True).clamp(min=1)
        feat = torch.sum(emb * mask.unsqueeze(-1), dim=1) / denom
        return feat

    def get_embeds(
        self,
        requires,
        input_audio_or_text,
        data_type=None,
    ):
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type)
            mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            self.sync_tags([]) # must call sync tags for distributed training
            return mulan_embeds
        if data_type == "category":
            categorical_tokens = self.get_tokens(requires, input_audio_or_text)     # tensor([[3], [4], [2], [2]])
            cat_embeds = self.embedder(categorical_tokens)
            cat_embeds = MulanCategoricalEmbedder.masked_mean(cat_embeds, categorical_tokens, self.none_id)[:, None, :] # bs x cat x emb  e.g., torch.Size([4, 1, 512])
            return cat_embeds
        ## Audio embed
        if data_type == "music":
            if self.training:
                input_audio_or_text = random_crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)
            if input_audio_or_text.shape[-1] < self.min_audio_length:
                input_audio_or_text = crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)
            mulan_embeds = get_mulan_embeds(
                requires, input_audio_or_text, data_type
            )

            # TODO: add dropout per item not batch
            if self.training and random.random() < self.dropout:
                none_embeds = self.embedder(torch.tensor([self.none_id], device=mulan_embeds.device))
                mulan_embeds[..., :] = none_embeds.squeeze(0)
            if mulan_embeds.dim() == 2:
                mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            self.sync_tags([]) # must call sync tags for distributed training
            return mulan_embeds


class MulanEmbedder(ContinuousEmbedder):
    def __init__(
            self,
            data_type='music',
            input_dim=512,
            embedding_dim=1024,
            min_audio_length=10*24000,
            add_sos=False,
            add_none=False,
            mulan_crop=True,
            mulan_average=True,
            dropout=0.0,
            text_emb_max_len=100,
            return_hidden_state=False
        ):
        super().__init__(input_dim, embedding_dim, add_sos, add_none=add_none)

        self.data_type = data_type
        self.min_audio_length = min_audio_length # 10s * 24k sample rate
        self.mulan_crop = mulan_crop
        self.dropout = dropout
        self.text_emb_max_len = text_emb_max_len
        self.return_hidden_state = return_hidden_state

    def get_embeds(
        self,
        requires,
        input_audio_or_text,
        data_type=None,
    ):
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type, return_hidden_state=self.return_hidden_state)
            if len(mulan_embeds.shape) == 2:
                mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            mulan_embeds = mulan_embeds[:, :self.text_emb_max_len, :]
            return mulan_embeds
        ## CFG
        if data_type == "none":
            bs = len(input_audio_or_text)
            device = next(self.parameters()).device
            mulan_embeds = self.embedder(torch.tensor([self.none_id] * bs, device=device))
            return mulan_embeds.unsqueeze(1)
        if data_type == "embed":
            if input_audio_or_text.dim() == 2:
                return input_audio_or_text.unsqueeze(1)
            return input_audio_or_text
        # Audio
        if data_type == "music":
            if input_audio_or_text.dim() == 1: input_audio_or_text = input_audio_or_text.unsqueeze(0)
            if input_audio_or_text.shape[-1] < self.min_audio_length:
                input_audio_or_text = crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)
            else:
                if self.mulan_crop and self.training:
                    # Always use moving average of mulan emb for validation and inference.
                    input_audio_or_text = random_crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)                            
            bs = input_audio_or_text.shape[0]
            device = next(self.parameters()).device            
            if self.training:
                if random.random() < self.dropout:
                    mulan_embeds = self.embedder(torch.tensor([self.none_id] * bs, device=device)).unsqueeze(1)
                else:
                    mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type)
            else:
                if torch.sum(input_audio_or_text) == 0:
                    # This is the CFG path
                    mulan_embeds = self.embedder(torch.tensor([self.none_id] * bs, device=device)).unsqueeze(1)
                else:
                    # This is the conditioned path
                    mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type)
            if mulan_embeds.dim() == 2:
                mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            return mulan_embeds


class MulanTokenEmbedder(TokenEmbedder):
    def __init__(self, data_type='music', num_rvq=12, codebook_size=1024, embedding_dim=1024, add_sos=False):
        vocab_size = num_rvq * codebook_size
        super().__init__(vocab_size, embedding_dim, add_sos)
        self.data_type = data_type
        self.num_rvq = num_rvq
        self.codebook_size = codebook_size

    def get_tokens(self, requires, input_audio):
        mulan_tokens = get_mulan_tokens(requires, input_audio, data_type=self.data_type)
        mulan_tokens = (
            mulan_tokens
            + torch.arange(self.num_rvq, device=self.device) * self.codebook_size
        )
        return mulan_tokens


class LeadsheetTokenEmbedderV2(TokenEmbedder):
    def embed(self, requires=None, input_tokens=None, token_wise_multiplication=None, token_ids=None, with_sos=False, with_eos=False):
        # token_wise_multiplication is the scaling factor to each embedding of token, 1.0 means no change, 0.0 means disable
        # Here we set (scaling factor) == (float time in second) as the representaion of time tokens
        token_ids = self.tokenize(requires, input_tokens, token_ids, with_sos, with_eos)
        embedding = self.embedder(token_ids)
        if with_sos:
            sos_token = self.get_sos_token(token_wise_multiplication.size(0))
            sos_coff = torch.ones_like(sos_token).float() # for sos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([sos_coff, token_wise_multiplication], dim=1)
        if with_eos:
            eos_coff = self.get_sos_token(token_wise_multiplication.size(0))
            eos_coff = torch.ones_like(eos_coff).float() # for eos toke, we simply set token_wise_multiplication=1.0 (not scaling)
            token_wise_multiplication = torch.cat([token_wise_multiplication, eos_coff], dim=1)
        token_wise_multiplication = token_wise_multiplication.unsqueeze(-1)
        embedding = embedding * token_wise_multiplication
        return embedding

    def get_tokens(self, requires, input):
        return input


class REMILeadsheetTokenEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        return input


class IntEmbedder(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos=add_sos, add_eos=add_eos)
    def get_tokens(self, requires, input):
        return input
    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        batch = batch.clamp_min_(0).clamp_max_(self.vocab_size-1).long()
        return super().embed(requires, batch, token_ids, with_sos, with_eos)


class AudioKeyEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        return input


class ChordSeqEmbedder(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos=add_sos, add_eos=add_eos)
    def get_tokens(self, requires, input):
        return input


class T5Embedder(ContinuousEmbedder):
    def __init__(self, input_dim=768, embedding_dim=1024, add_sos=False, model: str = "t5-base", max_length: int = 512):
        super().__init__(input_dim, embedding_dim, add_sos)
        from transformers import AutoTokenizer, T5EncoderModel

        self.required_modules = {
            "tokenizer": AutoTokenizer.from_pretrained(model),
            "encoder": T5EncoderModel.from_pretrained(model).eval()
        }
        self.max_length = max_length
        # self.embedding_features = self.required_modules["encoder"].config.d_model

    @torch.no_grad()
    def get_embeds(self, requires, texts):
        encoded = self.required_modules["tokenizer"](
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        # attention_mask = encoded["attention_mask"].to(device)
        self.required_modules["encoder"].to(device)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32):
            embedding = self.required_modules["encoder"](
                input_ids=input_ids, attention_mask=None
            )["last_hidden_state"]
        return embedding


class MetadataT5TokenEmbedder(ContinuousEmbedder):
    def __init__(self, input_dim=512, embedding_dim=1024, add_sos=False):
        super().__init__(input_dim, embedding_dim, add_sos)

    def get_embeds(self, requires, input):
        return get_t5_embeds(requires, input)


class VocalChromaEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        return input


class SpeakerEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        input = torch.clamp(input, 0, self.vocab_size-2) # subtract sos + 1
        return input


class KeyEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        return input


class TempoLabelEmbedder(TokenEmbedder):
    def get_tokens(self, requires, input):
        return input


class WavToVecTokenEmbedder(TokenEmbedder):
    def __init__(self, vocab_size=1024, embedding_dim=1024, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)

    def get_tokens(self, requires, input_audio):
        return get_wav2vec_tokens(requires, input_audio)
    

class BestRQTokenEmbedder(TokenEmbedder):
    def __init__(
            self,
            vocab_size=32_768,
            embedding_dim=1024,
            add_sos=False,
            add_eos=False,
            chunk_size=None,
            store_hidden_states=False
        ):
        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)
        if chunk_size is not None and store_hidden_states:
            raise Exception("Tokenizer currently does not support both chunking and saving last hidden state")
        self.store_hidden_states = store_hidden_states # save hidden states for m1 classifier
        self.last_hidden_state = None
        self.chunk_size = chunk_size
        self.tag_map = get_tag_map()

    def get_tokens(self, requires, input_audio, **kwargs):
        if self.store_hidden_states:
            results = get_bestrq_umm_outputs(requires, input_audio, **kwargs)
            token_ids = results['vq_ids']
            self.hidden_states = results['hidden_states']
            return token_ids
        else:
            return get_bestrq_umm_tokens(requires, input_audio, self.chunk_size, **kwargs)

    def tokenize_and_tag(self, requires, input_audio, **kwargs):
        lit_module = requires['Stage3']
        umm2_out = lit_module.wav2tokentag(input_audio)
        tag_logits = umm2_out['tags_logits']
        return umm2_out['vq_ids'], tag_logits
    
    def _key_with_max_value(d, topk=1, threshold=0): 
        return max(d, key=d.get)

    def convert_tag_logits_to_tags(self, tag_logits):
        return prob_to_tag(tag_logits, self.tag_map)


class BestRQMultiTokenEmbedder(BestRQTokenEmbedder):
    def __init__(
        self,
        vocab_size=32_768,
        embedding_dim=1024,
        add_sos=False,
        add_eos=False,
        chunk_size=None,
        store_hidden_states=False,
        pattern='delay',
        group=2,
        null_placeholder=True,
        encoder='fc',
        decoder='fc',
        semantic_codebook_depth=1,
    ):
        """
        Embedder for Multi-token prediction
        """
        if chunk_size is not None and store_hidden_states:
            raise Exception("Tokenizer currently does not support both chunking and saving last hidden state")
        self.store_hidden_states = store_hidden_states # save hidden states for m1 classifier
        self.last_hidden_state = None
        self.chunk_size = chunk_size
        self.group = group
        self.pattern = pattern
        self.null_placeholder = null_placeholder
        self.embedding_dim = embedding_dim
        self.R = semantic_codebook_depth

        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)

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

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        token_ids = self.tokenize(requires, batch, token_ids, with_sos, with_eos)
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
        return token_embeds, batch_group_token_ids, torch.LongTensor(batch_token_lengths)
    
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
            print("RQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
                  token sampling and cfg is defined there.")
        
        def ARQ_transformer(model_input, hidden_state):
            print("ARQ transformer inference will be done in `base_modules.py` or `semantic_modules_v4.py` because the \
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
    

class BestRQMKIITokenEmbedder(TokenEmbedder):
    def __init__(self, vocab_size=65_536, embedding_dim=1024, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)

    def get_tokens(self, requires, input_audio):
        tokens = get_bestrq_mkii_tokens(requires, input_audio)
        bs, seq_len, num_vq = tokens.shape
        codebook_size = self.vocab_size // num_vq
        tokens = tokens + torch.arange(num_vq, device=tokens.device) * codebook_size
        return tokens.reshape(bs, -1)


class BestRQEmbedder(ContinuousEmbedder):
    def __init__(self, input_dim=1024, embedding_dim=1024, add_sos=False):
        super().__init__(input_dim, embedding_dim, add_sos)

    def get_embeds(self, requires, input_audio):
        return get_bestrq_umm_embeds(requires, input_audio)


class SoundstreamTokenEmbedder(TokenEmbedder):
    # needs both embeddings (input) and tokens (output)
    def __init__(self, layer_range, codebook_size=1024, embedding_dim=1024, add_sos=False):
        self.layer_range = layer_range
        self.num_layers = layer_range[1] - layer_range[0]
        vocab_size = self.num_layers * codebook_size
        super().__init__(vocab_size, embedding_dim, add_sos)
        self.codebook_size = codebook_size

    def get_tokens(self, requires, input_audio):
        num_layers = self.num_layers
        layer_range = self.layer_range
        codebook_size = self.codebook_size
        b = input_audio.shape[0]
        device = input_audio.device
        soundstream_ids = get_soundstream_tokens(requires, input_audio)

        soundstream_ids = (
            soundstream_ids[:,:,layer_range[0]:layer_range[1]]
            + torch.arange(num_layers, device=device) * codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])
        return soundstream_ids


class DurationEmbedder(nn.Module):
    def __init__(self, durations, embedding_dim):
        super().__init__()
        durations = durations if isinstance(durations, (list, tuple)) else [durations]
        durations = [int(d) for d in durations]
        self.duration2id = {durations[i]: i for i in range(len(durations))}
        self.embedding_dim = embedding_dim
        self.embedder = nn.Embedding(len(durations) + 1, embedding_dim)

    def embed(self, duration, batch_size):
        duration = int(duration)
        duration_id = self.duration2id[duration]
        device = next(self.parameters()).device
        duration_ids = torch.LongTensor([duration_id] * batch_size).to(device)
        return self.embedder(duration_ids).unsqueeze(1)

    def empty_embed(self, batch_size):
        empty_id = len(self.duration2id)
        device = next(self.parameters()).device
        empty_ids = torch.LongTensor([empty_id] * batch_size).to(device)
        return self.embedder(empty_ids).unsqueeze(1)


class StructureEmbedder(nn.Module):
    def __init__(
        self,
        durations,
        embedding_dim,
        structure_labels,
        granularity_in_secs=0.5,
    ):
        super().__init__()
        durations = durations if isinstance(durations, (list, tuple)) else [durations]
        self.max_duration = int(durations[-1])
        self.embedding_dim = embedding_dim
        self.granularity_in_secs = granularity_in_secs
        self.max_seq_len = round(self.max_duration / self.granularity_in_secs)
        self.pad_id = 0
        self.random_id = 1
        self.default_id = 2
        self.label_ids = {}
        for i, label in enumerate(structure_labels):
            assert label not in self.label_ids, f"Duplicate structure: {label}"
            self.label_ids[label] = 3 + i
        self.embedder = nn.Embedding(3 + len(self.label_ids), embedding_dim)
        self.logged = 0

    def time_to_index(self, time):
        return round(time / self.granularity_in_secs)

    def embed(self, batch_structure_labels, target_duration):
        device = next(self.parameters()).device
        structure_ids = torch.full(
            (len(batch_structure_labels), self.max_seq_len), self.pad_id
        ).long().to(device)
        target_seq_len = round(target_duration / self.granularity_in_secs)
        for i, structure_labels in enumerate(batch_structure_labels):
            if structure_labels is None:    # This means no specification, just do whatever
                structure_ids[i, :target_seq_len] = self.random_id
                continue
            structure_labels = [x for x in structure_labels if x[0] in self.label_ids]
            structure_ids[i, :target_seq_len] = self.default_id
            for name, start, end in structure_labels:
                section_id = self.label_ids[name]
                start = min(target_seq_len, max(0, self.time_to_index(start)))
                end = min(target_seq_len, max(0, self.time_to_index(end)))
                structure_ids[i, start:end] = section_id
        if self.logged < 5:
            print(f"target_duration: {target_duration}, target_seq_len: {target_seq_len}")
            print(f"structure_labels: {batch_structure_labels}, structure_ids: {structure_ids}")
            self.logged += 1
        return self.embedder(structure_ids)


class IntensityEmbedder(nn.Module):
    def __init__(self, decimals, embedding_dim, intensity_hz=1):
        super().__init__()
        self.decimals = decimals
        self.multiplier = 10 ** self.decimals
        self.intensity_vocab_size = self.multiplier + 1
        self.embedder = nn.Embedding(self.intensity_vocab_size, embedding_dim)
        self.intensity_hz = intensity_hz
        self.logged = 0

    def quantize(self, batch_intensity_labels):
        device = next(self.parameters()).device
        intensity_ids = torch.clamp(batch_intensity_labels, min=0.0, max=1.0)
        intensity_ids = torch.round(intensity_ids * self.multiplier).long().to(device)
        return intensity_ids

    def unquantize(self, batch_intensity_ids):
        device = next(self.parameters()).device
        intensity_labels = (batch_intensity_ids / self.multiplier).float().to(device)
        return intensity_labels

    def embed(self, batch_intensity_labels, target_duration):
        target_length = int(target_duration * self.intensity_hz)
        device = next(self.parameters()).device
        if batch_intensity_labels.shape[1] > target_length:
            # Just use the first segment
            batch_intensity_labels = batch_intensity_labels[..., :target_length]
        elif batch_intensity_labels.shape[1] < target_length:
            # Just repeat the intensity curve
            tmp = torch.zeros(len(batch_intensity_labels), target_length).to(device)
            st = 0
            while st < target_length:
                length = min(batch_intensity_labels.shape[1], target_length - st)
                tmp[..., st:st + length] = batch_intensity_labels[..., :length]
                st += length
            batch_intensity_labels = tmp
        intensity_ids = self.quantize(batch_intensity_labels)
        if self.logged < 5:
            print(f"target_duration: {target_duration}, intensity_labels: {batch_intensity_labels.shape}, intensity_ids: {intensity_ids.shape}")
            print(f"intensity_labels: {batch_intensity_labels}, intensity_ids: {intensity_ids}")
            self.logged += 1
        return self.embedder(intensity_ids)


class BeatEmbedder(nn.Module):
    def __init__(self, beat_labels, embedding_dim, max_duration, max_timestamp=5):
        super().__init__()
        self.beat2id = {}
        for l in beat_labels:
            self.beat2id[l] = len(self.beat2id)
        self.eos_id = len(self.beat2id)
        self.pad_id = self.eos_id + 1
        # beats + <eos> + <pad>
        self.beat_vocab_size = len(self.beat2id) + 2
        self.embedder = nn.Embedding(self.beat_vocab_size, embedding_dim)
        self.mean_duration = max_duration / 2
        self.max_timestamp = max_timestamp
        self.logged = 0

    def normalize_timestamp(self, x):
        # Per xval paper, normalize the timestamp between [-5, 5]
        return (x - self.mean_duration) / self.mean_duration * self.max_timestamp

    def embed(self, batch_beat_labels, target_duration):
        batch_size = len(batch_beat_labels)
        device = next(self.parameters()).device
        # First, trim all beat labels beyond the target duration
        # and only keep the ones we want to model
        batch_beat_labels = [
            [(x[0], int(x[1])) for x in y if x[0] <= target_duration and int(x[1]) in self.beat2id]
            for y in batch_beat_labels
        ]
        # Find the max beat length to do padding properly
        max_beat_length = max([len(y) for y in batch_beat_labels])
        beat_ids = torch.full((batch_size, max_beat_length + 1), self.pad_id).long().to(device)
        beat_timestamps = torch.zeros((batch_size, max_beat_length + 1)).float().to(device)
        # Fill in the values, the lengths are different so need to do one-by-one
        for i, beat_labels in enumerate(batch_beat_labels):
            beat_ids[i, :len(beat_labels) + 1] = torch.Tensor(
                [self.beat2id[x[1]] for x in beat_labels] + [self.eos_id]
            ).long()
            # For <eos>, always multiply by 1
            beat_timestamps[i, :len(beat_labels) + 1] = torch.Tensor(
                [self.normalize_timestamp(x[0]) for x in beat_labels] + [1.0]
            ).float()
        if self.logged < 5:
            print(f"beat_labels: {batch_beat_labels}")
            print(f"beat_ids ({beat_ids.shape}): {beat_ids}")
            print(f"beat_timestamps ({beat_timestamps}): {beat_timestamps}")
            self.logged += 1
        return self.embedder(beat_ids) * beat_timestamps.unsqueeze(2), beat_ids, beat_timestamps


class MultiTagsEmbedder(BaseEmbedder):
    # MultiTags embedder - takes in list, and then embeds
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_id = None
        self.eos_id = None
        if add_sos:
            self.vocab_size = self.vocab_size + 1
            self.sos_id = self.vocab_size - 1
        if add_eos:
            self.vocab_size = self.vocab_size + 1
            self.eos_id = self.vocab_size - 1
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)

    @abstractmethod
    def get_tokens(self, requires, batch, **kwargs):
        raise NotImplementedError()

    def get_sos_token(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return sos_ids

    def get_eos_token(self, batch_size):
        assert self.eos_id is not None, "Error getting eos id. Must initialize embedder with add_eos=True"
        device = next(self.parameters()).device
        eos_ids = torch.full(size=(batch_size, 1), fill_value=self.eos_id, dtype=torch.long, device=device)
        return eos_ids

    def get_sos_embed(self, batch_size):
        return self.embedder(self.get_sos_token(batch_size))

    def get_eos_embed(self, batch_size):
        return self.embedder(self.get_eos_token(batch_size))    

    def tokenize(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False, **kwargs):
        raise NotImplementedError()

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        raise NotImplementedError()


# class MultiTagsCategoricalEmbedderV2(BaseEmbedder):
#     def __init__(
#             self, 
#             max_vocab_size=1024,
#             embedding_dim=1024,
#             add_sos=False,
#             add_eos=False,
#         ):
#         super().__init__()
        
#         # Initialize special token IDs
#         self.sos_id = None
#         self.eos_id = None
        
#         # Calculate the total vocabulary size considering special tokens
#         total_vocab_size = vocab_size
#         if add_sos:
#             total_vocab_size += 1
#             self.sos_id = total_vocab_size - 1
#         if add_eos:
#             total_vocab_size += 1
#             self.eos_id = total_vocab_size - 1
        
#         # Initialize the embedding layer
#         self.embedder = nn.Embedding(total_vocab_size, embedding_dim, **kwargs)

    
#     def embed(self, token_ids, len_token_ids):



class MultiTagsCategoricalEmbedder(MultiTagsEmbedder):
    def __init__(
            self,
            vocab_type='auto',
            max_vocab_size=1024,
            embedding_dim=1024,
            add_sos=False,
            dropout=0.0,
            category_separator="|",
        ):
        if vocab_type == 'auto':
            assert max_vocab_size
            vocab2id = { NONE_LABEL: 0 }
            vocab_size = max_vocab_size
            #_num_categories = 5  # (the previous default value)
            #_num_categories = 9
            _num_categories = 13
        else:
            vocab2id, _num_categories = get_categorical_vocab(vocab_type)  # infer num_categories from vocab_type
            vocab_size = max(vocab2id.values()) + 1
        super().__init__(vocab_size, embedding_dim, add_sos)
        self.vocab2id = vocab2id
        self.vocab2count = defaultdict(int)
        self.vocab_type = vocab_type
        self.dropout = dropout
        self.category_separator = category_separator
        self.num_categories = _num_categories


        self.vocab2id['Non-Sinking'] = self.vocab2id['non-Sinking']
        self.vocab2id['Cute'] = self.vocab2id['Cute_AUDIO_TIMBRE']


    def get_tag_id(self, tag, dropout=0.0):
        if tag not in self.vocab2id:
            if not self.training and len(tag.strip()) > 0:  # use NONE_LABEL for empty label
                logger.warning(f"Inference Error: Tag {tag} not found in vocab {self.vocab2id}. Please check vocab")
            tag = NONE_LABEL
        if self.training and random.random() < dropout:
            tag = NONE_LABEL
        return self.vocab2id[tag]

    def get_tokens(self, requires, style_texts):
        batch_style_tags = []
        for style_text in style_texts:
            # Strictly separate style_text by the separator
            if isinstance(style_text, str):
                separator = self.category_separator
                style_tags = []
                for t in style_text.split(separator):
                    normalized_style_text = t.replace("，", separator).replace(",", separator)
                    style_tags.append(normalized_style_text.split(separator))
            elif isinstance(style_text, list):
                style_tags = style_text
            elif isinstance(style_text, dict):
                # TODO: handle use case where dictionary is not sorted
                # style_tag_list = [v for k,v in sorted(style_text.items())]
                style_tags = style_text.values()
            else:
                style_tags = [NONE_LABEL] * self.num_categories
            batch_style_tags.append(style_tags)

        batch_tag_ids = []
        masks = []
        max_tags_num = 0
        for style_tags in batch_style_tags:
            if isinstance(style_tags, str):
                style_tags = [style_tags]
            for tags in style_tags:
                if isinstance(tags, str):  # Make it compatible with single tag
                    tags = [tags]
                tag_ids = [self.get_tag_id(tag, self.dropout) for tag in tags]
                batch_tag_ids.append(torch.tensor(tag_ids))
                masks.append(torch.tensor([1 for tag in tags]))
                #_masks = []
                #for tag in tags:
                #    if tag == NONE_LABEL:
                #        _masks.append(self.get_tag_id(NONE_LABEL))
                #    else:
                #        _masks.append(1)
                #masks.append(torch.tensor(_masks))
                max_tags_num = max(max_tags_num, len(tags))
        batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True, padding_value=self.get_tag_id(NONE_LABEL))
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        batch_tag_ids = torch.reshape(batch_tag_ids, [-1, self.num_categories, max_tags_num])
        masks = torch.reshape(masks, [-1, self.num_categories, max_tags_num])
        device = next(self.parameters()).device
        return torch.as_tensor(batch_tag_ids).to(device), torch.as_tensor(masks).to(device)

    def tokenize(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False, **kwargs):
        token_ids, masks = self.get_tokens(requires, batch, **kwargs)
        #print('batch: ', batch)
        #print('token_ids: ', token_ids)
        #print('masks: ', masks)
        return token_ids, masks

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        token_ids, masks = self.tokenize(requires, batch, token_ids, with_sos, with_eos)

        token_ids_shape = token_ids.shape
        _token_ids = torch.reshape(
            token_ids, [token_ids_shape[0]*token_ids_shape[1], token_ids_shape[2]],
        ).to(torch.long)

        masks_shape = masks.shape
        _masks = torch.reshape(
            masks, [masks_shape[0]*masks_shape[1], masks_shape[2], 1],
        ).to(torch.long)

        _embedding = self.embedder(_token_ids)
        #print('_embedding shape: ', _embedding.shape)
        #print('_masks shape: ', _masks.shape)
        #print('_embedding sum shape: ', torch.sum(_embedding, dim=1).shape)
        #print('_masks sum shape: ', torch.sum(_masks, dim=1).shape)
        _masks_sum = torch.sum(_masks, dim=1)
        #print('_masks sum: ', _masks_sum.T)
        _masks_sum = torch.where(_masks_sum>0, _masks_sum, torch.ones_like(_masks_sum))
        #print('_masks sum format: ', _masks_sum.T)
        embedding = torch.sum(_embedding*_masks, dim=1) / _masks_sum
        #print('embedding shape: ', embedding.shape)
        embedding = torch.reshape(embedding, [token_ids_shape[0], token_ids_shape[1], -1])
        #print('embedding reshape: ', embedding.shape)

        if with_sos:
            sos_embedding = self.get_sos_embed(embedding.size(0))
            #print('sos embedding shape: ', sos_embedding.shape)
            embedding = torch.cat([sos_embedding, embedding], dim=1)
        return embedding

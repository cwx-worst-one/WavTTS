import logging
import random
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from abc import abstractmethod
from typing import Any
from torch.nn.utils.rnn import pad_sequence
from torch import distributed
from collections import defaultdict
from recipes.musiclm.transforms.audio import RandomResizedCrop
from recipes.bigmusic.utils.mulan_tag import MulanTagger
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length
from recipes.bigmusic.datasets.mir_data_util import NONE_LABEL, get_categorical_vocab

logger = logging.getLogger()

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
def get_mulan_embeds(requires, x, data_type="music", average=True):
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
            model=requires["mulan"], text=x, device=requires["mulan"].device
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
    assert hasattr(lit_module.model, 'wav2token_alloutputs'), f"For M1 training, UMM model ({type(lit_module.model)}) version must support wav2token_alloutputs function"
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

class BaseEmbedder(nn.Module):
    @abstractmethod
    def embed(self, requires, batch, token_ids=None):
        pass

    @abstractmethod
    def get_sos_embed(self, batch_size):
        pass

class ContinuousEmbedder(BaseEmbedder):
    def __init__(self, input_dim, embedding_dim, add_sos=False):
        super().__init__()
        self.sos_id = 0 if add_sos else None
        if add_sos:
            self.sos_id = 0
            self.embedder = nn.Embedding(1, input_dim)
        else:
            self.sos_id = None

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

    def embed(self, requires, batch, with_sos=False, **kwargs):
        embeds = self.get_embeds(requires, batch, **kwargs)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
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


class TagCategoricalEmbedder(TokenEmbedder):
    def __init__(
            self, vocab_type='auto', max_vocab_size=1024, embedding_dim=1024, add_sos=False, dropout=0.0, num_categories=5, category_separator="|"
        ):
        if vocab_type == 'auto':
            assert max_vocab_size
            vocab2id = { NONE_LABEL: 0 }
            vocab_size = max_vocab_size
        else:
            vocab2id = get_categorical_vocab(vocab_type)
            vocab_size = len(vocab2id)
        super().__init__(vocab_size, embedding_dim, add_sos)
        self.vocab2id = vocab2id
        self.vocab2count = defaultdict(int)
        self.vocab_type = vocab_type
        self.dropout = dropout
        self.category_separator = category_separator
        self.num_categories = num_categories

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
                raise Exception(f"Inference Error: Tag {tag} not found in vocab {self.vocab2id}. Please check vocab")
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
            category_separator="|"
        ):
        super().__init__()
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

        self.vocab2id = { NONE_LABEL: 0 }
        self.vocab2count = defaultdict(int)
        self.dropout = dropout
        self.min_audio_length = min_audio_length # 10s * 24k sample rate

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
                print('Warning: tag already exists. Error.', self.vocab2id)
            if new_tag not in self.vocab2id and len(self.vocab2id) < self.vocab_size:
                self.vocab2id[new_tag] = len(self.vocab2id)

    def get_tag_id(self, tag, dropout=0.0):
        if tag not in self.vocab2id:
            if not self.training:
                raise Exception(f"Inference Error: Tag {tag} not found in vocab {self.vocab2id}. Please check vocab")
            tag = NONE_LABEL
        if self.training and random.random() < dropout:
            tag = NONE_LABEL
        return self.vocab2id[tag]

    def get_tokens(self, requires, style_texts):
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

        batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True, padding_value=self.get_tag_id(NONE_LABEL))
        device = next(self.parameters()).device
        return torch.as_tensor(batch_tag_ids).to(device)

    # save auto-growing vocab for inference
    def set_extra_state(self, state: Any): 
        self.vocab2id = state['vocab'] 
        self.vocab2count.update(state.get('counts', {}))
    def get_extra_state(self) -> Any: return { 'vocab': self.vocab2id, 'counts': dict(self.vocab2count) }

    def embed(self, requires, batch, with_sos=False, **kwargs):
        embeds = self.get_embeds(requires, batch, **kwargs)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        return embeds

    def get_embeds(
        self,
        requires,
        input_audio_or_text,
        data_type=None,
        target_samples_length=None,
    ):
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type)
            mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            self.sync_tags([]) # must call sync tags for distributed training
            return mulan_embeds
        if data_type == "category":
            categorical_tokens = self.get_tokens(requires, input_audio_or_text)
            cat_embeds = self.embedder(categorical_tokens)
            cat_embeds = cat_embeds.mean(1)[:, None, :] # bs x cat x emb
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
            mulan_crop=True,
            mulan_average=True,
        ):
        super().__init__(input_dim, embedding_dim, add_sos)

        self.data_type = data_type
        self.min_audio_length = min_audio_length # 10s * 24k sample rate
        self.mulan_crop = mulan_crop
        self.mulan_average = mulan_average

    def get_embeds(
        self,
        requires,
        input_audio_or_text,
        data_type=None,
        target_samples_length=None,
    ):
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(requires, input_audio_or_text, data_type)
            mulan_embeds = mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
            if not self.mulan_crop and not self.mulan_average:
                assert target_samples_length is not None
                # TODO: don't hardcode shift length
                shift_length = self.min_audio_length // 2
                prefix_length = 1 + (target_samples_length - self.min_audio_length) // shift_length
                mulan_embeds = mulan_embeds.expand(-1, prefix_length, -1)
            return mulan_embeds
        # Audio
        if self.training and self.mulan_crop:
            input_audio_or_text = random_crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)
        if not self.training and not self.mulan_crop and not self.mulan_average:
            assert target_samples_length is not None
            input_audio_or_text = random_crop_pad_to_seq_length(input_audio_or_text, target_samples_length)
        if input_audio_or_text.shape[-1] < self.min_audio_length:
            input_audio_or_text = crop_pad_to_seq_length(input_audio_or_text, self.min_audio_length)
        ## Audio embed
        if data_type == "music":
            mulan_embeds = get_mulan_embeds(
                requires, input_audio_or_text, data_type, average=self.mulan_average
            )
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

class LyricsTokenEmbedder(TokenEmbedder):
    def __init__(self, vocab_size, embedding_dim, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos=add_sos, add_eos=add_eos)
    # TODO: (AS) add padding_idx
    def get_tokens(self, requires, input):
        return input
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

    def get_tokens(self, requires, input_audio, **kwargs):
        if self.store_hidden_states:
            results = get_bestrq_umm_outputs(requires, input_audio, **kwargs)
            token_ids = results['vq_ids']
            self.hidden_states = results['hidden_states']
            return token_ids
        else:
            return get_bestrq_umm_tokens(requires, input_audio, self.chunk_size, **kwargs)

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

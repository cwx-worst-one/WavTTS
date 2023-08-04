import torch
import torch.nn as nn
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from abc import abstractmethod

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
def get_mulan_embeds(requires, x, data_type="music"):
    if data_type == "music":
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"], music=x.float(), device=x.device
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
            self.projection = nn.Linear(input_dim, embedding_dim)
        else:
            self.projection = nn.Identity

    @abstractmethod
    def get_embeds(self, requires, batch):
        # override to return embedding function
        pass

    def get_sos_embed(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        # sos_ids = [[self.sos_id] for _ in  range(batch_size)]
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return self.projection(self.embedder(sos_ids))

    def embed(self, requires, batch, with_sos=False):
        embeds = self.get_embeds(requires, batch)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        return embeds

class TokenEmbedder(BaseEmbedder):
    # Token embedder - takes in tokens, and then embeds
    def __init__(self, vocab_size, embedding_dim, add_sos=False, **kwargs):
        super().__init__()
        if add_sos:
            self.vocab_size = vocab_size + 1
            self.sos_id = vocab_size
        else:
            self.vocab_size = vocab_size
            self.sos_id = None
        self.embedder = nn.Embedding(self.vocab_size, embedding_dim, **kwargs)

    @abstractmethod
    def get_tokens(self, requires, batch):
        pass

    def get_sos_token(self, batch_size):
        assert self.sos_id is not None, "Error getting sos id. Must initialize embedder with add_sos=True"
        device = next(self.parameters()).device
        sos_ids = torch.full(size=(batch_size, 1), fill_value=self.sos_id, dtype=torch.long, device=device)
        return sos_ids
    
    def get_sos_embed(self, batch_size):
        return self.embedder(self.get_sos_token(batch_size))

    def tokenize(self, requires=None, batch=None, token_ids=None, with_sos=False):
        if token_ids is None:
            token_ids = self.get_tokens(requires, batch)
        if with_sos:
            return torch.cat([self.get_sos_token(token_ids.size(0)), token_ids], dim=1)
        return token_ids

    # def empty_tensor(self, batch_size):
    #     return torch.zeros((batch_size, 0), dtype=torch.long, device=self.device)

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False):
        token_ids = self.tokenize(requires, batch, token_ids, with_sos)
        return self.embedder(token_ids)

class MulanEmbedder(ContinuousEmbedder):
    def __init__(self, data_type='music', input_dim=512, embedding_dim=1024, add_sos=False):
        super().__init__(input_dim, embedding_dim, add_sos)
        self.data_type = data_type

    def get_embeds(self, requires, input_audio, data_type=None):
        if data_type is None:
            data_type = self.data_type
        mulan_embeds = get_mulan_embeds(requires, input_audio, data_type)
        return mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
    
    def embed(self, requires, batch, with_sos=False, data_type=None):
        embeds = self.get_embeds(requires, batch, data_type)
        embeds = self.projection(embeds)
        if with_sos:
            sos_embed = self.get_sos_embed(embeds.size(0))
            embeds = torch.cat([sos_embed, embeds], dim=1)
        return embeds

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
    def __init__(self, vocab_size, embedding_dim, add_sos=False):
        super().__init__(vocab_size, embedding_dim, add_sos=add_sos, padding_idx=70)
    # TODO: add padding_idx
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

class WavToVecTokenEmbedder(TokenEmbedder):
    def __init__(self, vocab_size=1024, embedding_dim=1024, add_sos=False):
        super().__init__(vocab_size, embedding_dim, add_sos)

    def get_tokens(self, requires, input_audio):
        return get_wav2vec_tokens(requires, input_audio)
    

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

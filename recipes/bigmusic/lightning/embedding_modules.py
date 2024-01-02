import torch
import torch.nn as nn
from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from abc import abstractmethod
from recipes.musiclm.transforms.audio import RandomResizedCrop
from recipes.bigmusic.utils.mulan_tag import MulanTagger
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length

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

@torch.no_grad()
def get_bestrq_umm_tokens(requires, batch):
    lit_module = requires['Stage3']
    vq_ids = lit_module.wav2token(batch)
    return vq_ids

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
    def get_tokens(self, requires, batch):
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

    def tokenize(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        if token_ids is None:
            token_ids = self.get_tokens(requires, batch)
        if with_sos:
            token_ids = torch.cat([self.get_sos_token(token_ids.size(0)), token_ids], dim=1)
        if with_eos:
            token_ids = torch.cat([token_ids, self.get_eos_token(token_ids.size(0))], dim=1)
        return token_ids

    def embed(self, requires=None, batch=None, token_ids=None, with_sos=False, with_eos=False):
        token_ids = self.tokenize(requires, batch, token_ids, with_sos, with_eos)
        return self.embedder(token_ids)

class MulanEmbedder(ContinuousEmbedder):
    def __init__(self, data_type='music', input_dim=512, embedding_dim=1024, max_audio_length=10*24000, add_sos=False):
        super().__init__(input_dim, embedding_dim, add_sos)
        self.data_type = data_type
        self.max_audio_length = max_audio_length # 10s * 24k sample rate
        self.resize_transform = RandomResizedCrop(max_audio_length) 

    def get_embeds(self, requires, input_audio, data_type=None):
        if data_type is None:
            data_type = self.data_type
        if data_type == 'music':
            if self.training:
                input_audio = self.resize_transform(input_audio)
            else:
                input_audio = input_audio[..., :self.max_audio_length]
        mulan_embeds = get_mulan_embeds(requires, input_audio, data_type)
        return mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d

class MulanTagEmbedder(ContinuousEmbedder):
    def __init__(
            self, data_type='music', input_dim=512, embedding_dim=1024, min_audio_length=10*24000, add_sos=False, 
            mulan_tag_type="mulan_genres", use_mcc_gender=True
        ):
        super().__init__(input_dim, embedding_dim, add_sos)

        self.data_type = data_type
        self.min_audio_length = min_audio_length # 10s * 24k sample rate
        self.mulan_tagger = MulanTagger(mulan_tag_type)
        self.use_mcc_gender = use_mcc_gender

    def get_embeds(self, requires, input_audio, mcc_style_text=None, data_type=None):
        # Text
        if data_type == "text":
            mulan_embeds = get_mulan_embeds(requires, input_audio, data_type)
            return mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
        # Audio
        if self.training:
            input_audio = random_crop_pad_to_seq_length(input_audio, self.min_audio_length)
        if input_audio.shape[-1] < self.min_audio_length:
            input_audio = crop_pad_to_seq_length(input_audio, self.min_audio_length)
        if data_type == "music":
            mulan_embeds = get_mulan_embeds(requires, input_audio, data_type)
            return mulan_embeds[:, None, :] # bs x d -> bs x seq_len x d
        elif data_type == "tag":
            mulan_audio_embeds = get_mulan_embeds(requires, input_audio, "music")
            metadata = self.mulan_tagger.get_tags(requires, audio_embeds=mulan_audio_embeds)
            if self.use_mcc_gender and mcc_style_text:
                for m, mcc_style in zip(metadata, mcc_style_text):
                    if ' female' in mcc_style.lower():
                        m['gender'] = 'Female'
                    elif ' male' in mcc_style.lower():
                        m['gender'] = 'Male'
                    else:
                        m['gender'] = None
            style_text = [self.mulan_tagger.tag_to_style_text(m) for m in metadata]
            mulan_text_embeds = get_mulan_embeds(requires, style_text, "text")
            return mulan_text_embeds[:, None, :] # bs x d -> bs x seq_len x d

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
    def __init__(self, vocab_size=32_768, embedding_dim=1024, add_sos=False, add_eos=False):
        super().__init__(vocab_size, embedding_dim, add_sos, add_eos)

    def get_tokens(self, requires, input_audio):
        return get_bestrq_umm_tokens(requires, input_audio)

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

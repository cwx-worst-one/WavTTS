from copy import deepcopy
from dataclasses import dataclass
from tokenize import Token
from typing import List, NamedTuple, Optional, Union, Tuple, no_type_check

import torch
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence

from recipes.umm.models.umm import FineTunedModel
from recipes.umm.requires.model_initializer import init_stage3
from samantha.models.base import LightningModuleBase

from samantha.utils.hdfs_tools import ARNOLD_REGION
from recipes.mi1.utils import hash_trick, length_to_mask

class Tokenizer:
    """Base tokenizer implementation
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @no_type_check
    def __call__(self, texts: List[Optional[str]],
                 return_text: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


# class NoopTokenizer(Tokenizer):
#     """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
#     The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
#     strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
#     split it to ["Jeff", "Buckley"] and return an index per word.

#     For example:
#     ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
#     ["Metal", "Rock", "Classical"] => [0, 223, 51]
#     """
#     def __init__(self, n_bins: int, pad_idx: int = 0):
#         self.n_bins = n_bins
#         self.pad_idx = pad_idx

#     def __call__(self, texts: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
#         output, lengths = [], []
#         for text in texts:
#             # if current sample doesn't have a certain attribute, replace with pad token
#             if text is None:
#                 output.append(self.pad_idx)
#                 lengths.append(0)
#             else:
#                 output.append(hash_trick(text, self.n_bins))
#                 lengths.append(1)

#         tokens = torch.LongTensor(output).unsqueeze(1)
#         mask = length_to_mask(torch.IntTensor(lengths)).int()
#         return tokens, mask

@dataclass
class UMMResult:
    vq_ids: torch.Tensor
    hidden_states: torch.Tensor
    vq_hidden_states: torch.Tensor
    vq_loss: torch.Tensor
    layer_idx: Optional[int] = None


class UMMTokenizer(LightningModuleBase):
    _frame_rate: int = 25

    _hdfs_path = {
        "US": "hdfs://harunava/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",
        "CN": "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt",
    }

    def __init__(self, finetune: bool = False, cache_dir: str = ".cache"):
        super().__init__()
        self.finetune = finetune

        self.model: FineTunedModel = self.get_model(self._hdfs_path[ARNOLD_REGION], cache_dir)

        # override audio_transform with floating point version:
        # self.model.audio_transform = self.model.audio_transform.float()

        self._layer_idx = self.model.config.vq_layer_idx
        self._n_embd = self.model.config.hidden_size

        if not finetune:
            self.freeze()
            self.eval()

    @property
    def n_embd(self) -> int:
        return self._n_embd

    @property
    def vocab_size(self) -> int:
        # Base vocabulary size
        return self.model.config.vq_codebook_size

    @property
    def frame_rate(self) -> int:
        return self._frame_rate

    @property
    def layer_idx(self):
        return self._layer_idx

    @staticmethod
    def get_model(hpath: str, cache_dir: str) -> FineTunedModel:
        model = init_stage3(hpath, local_rank=0, cache_dir=cache_dir)
        return model["Stage3"].model

    @torch.cuda.amp.autocast(enabled=False)  # TODO: Original UMM was trained with fp32
    def forward(self, audio, mel: Optional[torch.Tensor] = None) -> UMMResult:
        if not self.finetune or self.model.training:
            self.model.eval()

        with torch.set_grad_enabled(self.finetune):
            result = self.model.wav2token_alloutputs(audio, mel=mel)
            assert (
                result["hidden_states"].dtype == torch.float32
            ), "TODO: Original UMM was trained with fp32"

            return UMMResult(
                vq_ids=result.vq_ids,
                hidden_states=result.hidden_states,
                vq_hidden_states=result.vq_states,
                vq_loss=result.vq_loss,
                layer_idx=self.layer_idx,
            )


class MusicTagTokenizerResult(NamedTuple):
    token_ids: torch.Tensor
    normalized_text: str
    tag_names: List[str]


class MusicTagTokenizer:
    _music_tag_cls_token = "<MUSIC_TAG>"

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"cls_token": self._music_tag_cls_token})

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.tokenizer)

    def preprocess_tags(self, tags: List[str]) -> str:
        tags = ", ".join(tags)
        return tags

    def __call__(
        self, tag_names: Union[List[str], List[List[str]]], device: torch.tensor
    ) -> MusicTagTokenizerResult:
        if type(tag_names[0]) != list:
            tag_names = [tag_names]

        assert type(tag_names) == list, "`tag_names` must be a list of tag names"

        normalized_text = []
        for t in tag_names:
            normalized_text.append(self.preprocess_tags(t))

        token_ids = self.tokenizer(
            normalized_text, return_tensors="pt", padding=True, add_special_tokens=True
        )["input_ids"]

        token_ids = token_ids.to(device)
        return MusicTagTokenizerResult(
            token_ids=token_ids, normalized_text=normalized_text, tag_names=tag_names
        )


class NoOpTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.
    """
    def __init__(self, vocab: List[int]):
        self._vocab = sorted(set(vocab))
        self._vocab_size = len(self._vocab)
        self._name2ix = {name: idx for idx, name in enumerate(self._vocab)}
        self._ix2name = {idx: name for idx, name in enumerate(self._vocab)}

    @property
    def vocab(self) -> list:
        return self._vocab
        
    def __len__(self) -> int:
        return self._vocab_size

    def preprocess(self, name: str) -> str:
        return name

    def encode(self, name: str, device: torch.device) -> torch.Tensor:
        idx = self._name2ix[self.preprocess(name)]
        return torch.tensor(idx, dtype=torch.long, device=device)

    def decode(self, tag_id: torch.Tensor) -> str:
        return self._ix2name[tag_id.item()]

    def decode_batch(self, tag_ids: torch.Tensor) -> List[str]:
        tag_names = []
        for tag_id in tag_ids:
            tag_names.append(self.decode(tag_id))
        return tag_names

    def __call__(self, batch_names: List[str], device: torch.device) -> torch.Tensor:
        return self.encode(batch_names, device=device)

class NoOpOneHotTokenizer(NoOpTokenizer):

    def __init__(self, vocab: List[str]):
        super().__init__(vocab)


    def encode(self, names: List[str], device: torch.device) -> torch.Tensor:
        one_hot = torch.zeros(len(self), device=device, dtype=torch.long)
        indices = [self._name2ix[self.preprocess(name)] for name in names]
        one_hot[indices] = 1
        return one_hot

    def encode_batch(self, batch_names: List[List[str]], device: torch.device) -> torch.Tensor:
        batch = []
        for names in batch_names:
            one_hot = self.encode(names, device=device)
            batch.append(one_hot)
        return torch.stack(batch)

    def decode(self, batch_indices: Union[List[List[int]], torch.Tensor]) -> str:
        if type(batch_indices) == torch.Tensor:
            batch_indices = batch_indices.tolist()
            
        batch = []
        for indices in batch_indices:
            batch.append([self._ix2name[idx] for idx in indices])
        return batch



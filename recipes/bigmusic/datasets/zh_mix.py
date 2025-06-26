import copy
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Generator, Iterable, Optional, Union, Set

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from webdataset.pipeline import DataPipeline

from recipes.bigmusic.datasets.mir_data_util import (
    ARTIST_ID_MAP_V2,
    KEY_ID_MAP,
    MODE_ID_MAP,
    ROOT_ID_MAP,
    TEMPO_LABEL_ID_MAP,
    TIME_SIGNATURE_ID_MAP,
)
from recipes.bigmusic.datasets.tokenizers.sami_phoneme import (
    SamiPhonemeSeqTokenizer,
    SamiPhonemeTokenizerError,
)
from recipes.bigmusic.datasets.transforms.lyrics import pad_crop
from recipes.bigmusic.datasets.tokenizers.sami_phoneme import SamiPhonemeSeqTokenizer, SamiPhonemeTokenizerError
from recipes.bigmusic.datasets.utils.zh_datasets import ZhMetaTransform
from samantha.dataio.bigmusic.tokenizers.sami_phoneme_tokenizer import SamiPhonemeSeqTokenizer as SamiPhonemeSeqTokenizerRefactor
from recipes.bigmusic.datasets.utils.zh_meta import (
    SongSlice,
    ZhMetaParseError,
    ZhMetaTransformError,
)
from recipes.datasets.mcc.mix import (
    BaseTransforms,
    WebDatasetBufferPreprocessor,
)
from recipes.datasets.mcc.sami_tokenizer import SamiOfflineTokenizer
from recipes.musiclm.transforms.audio import AbsNormalizeAudio
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    Pad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self

logger = logging.getLogger(__file__)


PHONEME_TOKENIZERS = {
    "sami_phoneme_legacy": lambda: SamiPhonemeSeqTokenizer.init_cached("legacy"),
    "sami_phoneme_v2": lambda: SamiPhonemeSeqTokenizer.init_cached("v2"),
    "sami_phoneme_v3": lambda: SamiPhonemeSeqTokenizer.init_cached("v3"),
    "sami_phoneme_v4": lambda: SamiPhonemeSeqTokenizer.init_cached("v4"),
    "sami_phoneme_refactor": lambda: SamiPhonemeSeqTokenizerRefactor.init_cached("v0"),

    # deprecated, do not use
    "tts_chinese_frontend_model": lambda: SamiOfflineTokenizer(vocab_type="phoneme"),
    "tts_chinese_frontend_model_phonetone": lambda: SamiOfflineTokenizer(vocab_type="phoneme+tone"),
}


PHONE_PAD_ID = 0
# NOTE: This map's keys have to be consistent with `APP_EMB_MAP` in the constructor of `SemanticModule` (v4).
APP_COND_MAP = {
    "zh_vocal": "style_category,speaker_id,lyrics_tokens",
    # "zh_vocal_audio_emb": "style_category,speaker_id,lyrics_tokens,audio_emb",
    # "zh_vocal_audio_trans": "style_category,speaker_id,lyrics_tokens,audio_trans",
}
SUPPORTED_APPS = list(APP_COND_MAP.keys())
# Token placeholders used in LyricsTokenSectionEmbedder
TOKEN_PLACEHOLDER_SECTION_DUR = -1
TOKEN_PLACEHOLDER_SLICE_DUR = -2


########################## Data ######################


class DataError(Exception):
    pass


class CollateError(DataError):
    pass


@dataclass
class DataItem:
    @classmethod
    def collate(cls, items: list["DataItem"]) -> dict[str, Any]:
        """The default collate method that collates all DataItem attributes into a dict.
        If there is any non-DataItem attribute you want to include into the result dict,
        use collate_custom to return a result dict for these attributes, which will then
        be merged into the collate result of DataItem.
        """
        if len(items) == 0:
            raise CollateError("Can not collate when len(items) == 0")
        if not all(isinstance(item, cls) for item in items):
            raise CollateError(f"Inconsistent batch items of class {cls}")

        dicts = []
        for attr in vars(items[0]):
            attr_values = [getattr(item, attr) for item in items]
            if not any(isinstance(value, DataItem) for value in attr_values):
                continue  # either other type or all None
            if not all(isinstance(value, DataItem) for value in attr_values):
                raise CollateError(f"Inconsistent batch items of attribute {attr}")
            dicts.append(attr_values[0].__class__.collate(attr_values))  # attr_values here must be DataItems
        dict_custom = cls.collate_custom(items)  # collate other attributes that are not DataItems
        return safe_merge_dicts(dicts + [dict_custom])
    
    @classmethod
    def collate_custom(cls, items: list["DataItem"]) -> dict[str, Any]:
        """Collate attributes that are not DataItem. This method will be called inside collate."""
        return {}


@dataclass
class DataWithLength:
    """An interface that provides a `length` property for length_fn in the data module to call."""
    @property
    def length(self) -> int:
        raise NotImplementedError()


@dataclass
class DataItemCFGable(DataItem):
    """
    Any attribute that ends with "_cfg" will be treated as a CFG attribute, which will be used to 
    replace its corresponding non-CFG attribute (without the "_cfg" suffix) in the `to_cfg_item` method.

    Whenever `callate_cfg` is called, the returned dict will contain both regular key/value pairs and
    CFG key/value pairs whose keys end with "_cfg" suffix, which will be automatically identified and
    split into a separate batch by SemanticModule (v4).
    """
    def to_cfg_item(self) -> "DataItemCFGable":
        """Return a new DataItemCFGable item with value dropout"""
        _self = copy.deepcopy(self)
        _self.copy_fields_to_cfg()
        for attr in vars(_self):
            value = getattr(_self, attr)
            if isinstance(value, DataItemCFGable):
                setattr(_self, attr, value.to_cfg_item())
        return _self

    @classmethod
    def collate_cfg(cls, items: list["DataItemCFGable"]) -> dict[str, Any]:
        d = cls.collate(items)
        d_cfg = cls.collate([item.to_cfg_item() for item in items])
        d_cfg = {k + "_cfg": v for k, v in d_cfg.items()}
        return safe_merge_dicts([d, d_cfg])

    def copy_fields_to_cfg(self):
        """Copy attributes that end with _cfg to the ones without _cfg suffix, reset _cfg attributes to None"""
        for cfg_attr in vars(self):
            if cfg_attr.endswith("_cfg"):
                attr = cfg_attr[:-4]
                cfg_value = getattr(self, cfg_attr)
                if cfg_value is not None:
                    setattr(self, attr, cfg_value)
                    setattr(self, cfg_attr, None)


@dataclass
class DataAudio(DataItem):
    """
    Attributes:
        audio_tensor: Waveform tensor
        token_length: Target audio token length (the original length without padding)
        use_pad: Whether to use padding for audio_tensor in the collate method
        audio_type: Used as the key in the collated batch dict
    """
    audio_tensor: torch.Tensor
    token_length: int
    use_pad: bool = True
    audio_type: Optional[str] = None
    sample_rate: Optional[int] = None

    @classmethod
    def collate_custom(cls, items: list["DataAudio"]) -> dict[str, Any]:
        def reshape(audio):
            if audio.ndim == 1:
                audio = audio[None, :]
            return audio

        def apply_pad(audio):
            if pad is None:
                return audio
            return pad(audio)
        
        use_pad = _check_value_consistency_return_value([item.use_pad for item in items])
        pad = Pad(n_samples=max([x.audio_tensor.shape[-1] for x in items])) if use_pad else None

        audio_type = _check_value_consistency_return_value([item.audio_type for item in items])
        if audio_type == "prompt_audio":
            audio_type = "audio_prompt"  # naming inconsistency in SemanticModule

        sample_rate = _check_value_consistency_return_value([item.sample_rate for item in items])

        return {
            audio_type: torch.stack([apply_pad(reshape(d.audio_tensor)) for d in items], dim=0),
            "target_tokens_length": torch.LongTensor([d.token_length for d in items]),
            "audio_sample_rate": sample_rate,
        }


@dataclass
class DataLyrics(DataItemCFGable):
    """
    Attributes:
        text: Parsed lyrics (with section tags and singer tags), phrase separated by '\\n'
        phonemes: Parsed phoneme string (with section tags and singer tags), phrase separated by '\\n'
        max_phone_len: Maximum phoneme token length
        symbols: Human readable rokens
        tokens: Integer tokens
        coffs: Embedding coefficients
        token_length: The number of tokens before padding
    """
    text: str
    max_phone_len: int
    phone_pad_id: int

    symbols: list[Union[int, str]]
    tokens: torch.Tensor
    coffs: torch.Tensor
    token_length: int

    encoding_mode: str

    # Inference CFG
    symbols_cfg: list[Union[int, str]] = None
    tokens_cfg: Optional[torch.Tensor] = None
    coffs_cfg: Optional[torch.Tensor] = None
    token_length_cfg: Optional[int] = None

    @classmethod
    def from_song_slice(
        cls,
        song_slice: SongSlice,
        tokenizer: SamiPhonemeSeqTokenizer,
        max_phone_len: int,
        enable_cfg: bool = False,
        enable_dropout_all: bool = False, 
        enable_crop: bool = False,
        drop_slice_duration: Union[bool, float] = False,
        drop_notes: Union[bool, float] = True,
        drop_section_instruments: Union[bool, float] = True,
    ):
        """
        Args
            song_slice: A reformatted SongSlice object
            tokenizer: A SamiPhonemeSeqTokenizer that tokenizes a list of dicts
            max_phone_len: Maximum phoneme token length
            enable_cfg: If CFG items should be initialized
            enable_dropout_all: If CFG items should be initialized in a way that all lyrics tokens and components will be dropped out
            enable_crop: When it is True, automatically crop the tokens if the token length exceeds max_phone_len, otherwise raise error
            drop_slice_duration: 
                - True: Automatically drop the slice_duration embedding.
                - False: Keep a slice_duration embedding
                - float: Drop out probability. Will be converted into a bool value internally.
            drop_notes: Act the same as `drop_slice_duration`, but for notes.
            drop_section_instruments: Act the same as `drop_slice_duration`, but for section instruments.
        """
        def prob_to_bool(prob: Union[bool, float]) -> bool:
            if not isinstance(prob, bool):
                return random.random() < prob
            return prob

        def crop_tokens(tokens: list) -> list:
            if len(tokens) > max_phone_len:
                if enable_crop:
                    logger.info(f"len(tokens) {len(tokens)} exceed max_phone_len")
                    tokens = tokens[:max_phone_len]
                else:
                    raise DataError("Can not tokenize sequence that exceeds max_phone_len")
            return tokens

        if not song_slice.is_reformatted:
            raise DataError("DataLyrics can only parse a reformatted SongSlice")

        drop_slice_duration = prob_to_bool(drop_slice_duration)  # rate -> toggle
        drop_notes = prob_to_bool(drop_notes)  # rate -> toggle
        drop_section_instruments = prob_to_bool(drop_section_instruments)  # rate -> toggle
        encoding_mode = "phoneme" if song_slice.notes is None or drop_notes else "phoneme_note_concat"
        is_vocal = song_slice.has_utterance
        if is_vocal:
            drop_items = ["section_duration", "section_instruments"]  # always remove section durations and section instruments for vocal music
            if drop_slice_duration:
                drop_items.append("slice_duration")
        else:
            drop_items = []
            if drop_slice_duration:
                drop_items.append("slice_duration")
            if drop_section_instruments:
                drop_items.append("section_instruments")
            # always keep section durations for instrumental music
            # section tags have been dropped out outside
        try:
            dicts = song_slice.to_dicts(
                mode=encoding_mode,
                drop_target=drop_items,
            )
        except ZhMetaTransformError as e:
            raise DataError(str(e))
        try:
            output = tokenizer(dicts)
            if isinstance(tokenizer, SamiPhonemeSeqTokenizerRefactor):
                tokens = output['input_ids']
                coffs = output['coffs']
            else:
                tokens, coffs = output
        except SamiPhonemeTokenizerError as e:
            raise DataError(str(e))
        tokens = crop_tokens(tokens)
        coffs = crop_tokens(coffs)
        if isinstance(tokenizer, SamiPhonemeSeqTokenizerRefactor):
            symbols = [tokenizer._id_to_token[t] for t in tokens]
        else:
            symbols = [tokenizer.reversed_vocab[t] for t in tokens]
        token_length = len(tokens)

        if enable_cfg:
            if enable_dropout_all:
                dicts_cfg = song_slice.to_dicts(mode=encoding_mode, drop_target="lyrics_all")
            else:
                dicts_cfg = song_slice.to_dicts(mode=encoding_mode, drop_target="all")
            try:
                output = tokenizer(dicts_cfg)
                if isinstance(tokenizer, SamiPhonemeSeqTokenizerRefactor):
                    tokens_cfg = output['input_ids']
                    coffs_cfg = output['coffs']
                else:
                    tokens_cfg, coffs_cfg = output
            except SamiPhonemeTokenizerError as e:
                raise DataError(str(e))
            tokens_cfg = crop_tokens(tokens_cfg)
            coffs_cfg = crop_tokens(coffs_cfg)
            if isinstance(tokenizer, SamiPhonemeSeqTokenizerRefactor):
                symbols_cfg = [tokenizer._id_to_token[t] for t in tokens_cfg]
            else:
                symbols_cfg = [tokenizer.reversed_vocab[t] for t in tokens_cfg]
            token_length_cfg = len(tokens_cfg)
        else:
            dicts_cfg = None
            tokens_cfg, coffs_cfg = None, None
            symbols_cfg = None
            token_length_cfg = None

        return cls(
            text="\n".join([p.format_text() for p in song_slice.phrases]),
            max_phone_len=max_phone_len,
            # fix to the right value
            phone_pad_id=PHONE_PAD_ID if isinstance(tokenizer, SamiPhonemeSeqTokenizerRefactor) else tokenizer.vocab[tokenizer.SpecialSymbols.PHONE_PAD],
            symbols=symbols,
            tokens=torch.LongTensor(tokens),
            coffs=torch.FloatTensor(coffs),
            token_length=token_length,
            encoding_mode=encoding_mode,
            # CFG
            symbols_cfg=symbols_cfg,
            tokens_cfg=torch.LongTensor(tokens_cfg) if tokens_cfg is not None else None,
            coffs_cfg=torch.FloatTensor(coffs_cfg) if coffs_cfg is not None else None,
            token_length_cfg=token_length_cfg,
        )

    @classmethod
    def collate_custom(cls, items: list["DataLyrics"]) -> dict[str, Any]:
        _check_value_consistency_return_value([item.max_phone_len for item in items])
        _check_value_consistency_return_value([item.phone_pad_id for item in items])

        # Pad or crop the token tensor to max_phone_len
        max_phone_len = items[0].max_phone_len

        tokens = [pad_crop(item.tokens.clone().detach(), max_phone_len, torch.int, items[0].phone_pad_id) for item in items]
        coffs = [pad_crop(item.coffs.clone().detach(), max_phone_len, torch.float, 1.0) for item in items]

        return {
            "lyrics": [d.text for d in items],
            "symbols": [d.symbols for d in items],
            "lyrics_tokens": torch.stack(tokens),
            "lyrics_coffs": torch.stack(coffs),
            "lyrics_tokens_length": torch.LongTensor([d.token_length for d in items]),
            "max_phone_len": torch.LongTensor([d.max_phone_len for d in items]),
            "lyrics_encoding_mode": [d.encoding_mode for d in items],
        }


@dataclass
class DataFreeFormText(DataItemCFGable):
    """
    Attributes:
        text: freeform text
    """
    text: Optional[str] = ''

    # Inference CFG
    text_cfg: Optional[str] = ''

    @classmethod
    def new(
        cls,
        text: Optional[str] = '',
        enable_cfg: bool = False,
    ):  
        return cls(
            text=text if text else '',
            text_cfg="" if (enable_cfg and (text is not None)) else None,
        )

    @classmethod
    def collate_custom(cls, items: list["DataFreeFormText"]) -> dict[str, Any]:
        return {
            "freeform_text": [d.text for d in items],
        }


@dataclass
class DataStyleText(DataItemCFGable):
    """
    Attributes:
        categorical_text: A list of categorical tags.
        categorical_format: "sa_tag" or "multi_tag". It is only useful for intializing the cfg attribute in the new method.
    """
    categorical_text: Optional[list[Union[str, list[str]]]]
    categorical_format: Optional[str] = None  # sa_tag | multi_tag | 9_cat
    # CFG
    categorical_text_cfg: Optional[list[Union[str, list[str]]]] = None

    # Constant
    TAG_INDEX: ClassVar[dict[str, dict[str, int]]] = {
        "sa_tag": {tag: idx for idx, tag in enumerate(["genre", "mood", "scene", "sinking", "lang"])},
        "multi_tag": {tag: idx for idx, tag in enumerate(["genre", "mood", "scene", "speaker", "voice"])},
        "9_cat": {tag: idx for idx, tag in enumerate(["genre", "genre_extra", "extra", "mood", "scene", "speaker", "voice", "lang", "sinking"])},  # unified categories for Chinese Vocal V4
        "10_cat": {tag: idx for idx, tag in enumerate(["genre", "genre_extra", "extra", "mood", "scene", "speaker", "voice", "lang", "sinking", "instrument"])},
        "13_cat": {tag: idx for idx, tag in enumerate(["genre", "genre_extra", "extra", "mood", "scene", "speaker", "voice", "lang", "sinking", "instrument", "tempo", "key", "mode"])},
    }

    def __post_init__(self):
        if self.categorical_format is not None and self.categorical_format not in ["sa_tag", "multi_tag", "9_cat", "10_cat", "13_cat"]:
            raise DataError(f"Invalid categorical_format {self.categorical_format}")
        if isinstance(self.categorical_text, list):
            available_n_categories = [len(self.TAG_INDEX[self.categorical_format])] if self.categorical_format else self.available_n_categories
            if len(self.categorical_text) not in available_n_categories:
                raise DataError(f"Invalid categorical_text length {len(self.categorical_text)}")
    
    @classmethod
    def available_cfg_targets(cls, categorical_format: str) -> list[str]:
        return list(cls.TAG_INDEX[categorical_format].keys())
    
    @property
    def available_n_categories(self) -> list[int]:
        return list(set(len(d) for d in self.TAG_INDEX.values()))

    @classmethod
    def new(
        cls,
        categorical_text: Optional[list[Union[str, list[str]]]],
        categorical_format: Optional[str] = None,
        enable_cfg: bool = False,
        cfg_targets: Optional[Union[str, list[str]]] = None,
    ):
        """Construct a StyleText object with categorical_text_cfg deducted from vocab and cfg_targets"""
        if not categorical_text:  # empty list should be reset to None
            categorical_text = None
        if enable_cfg and cfg_targets is not None:
            categorical_text_cfg = categorical_text[:]  # shallow copy
            if isinstance(cfg_targets, str):
                cfg_targets = [cfg_targets]
            # print(categorical_text_cfg, ' apply cfg to : ')
            for cfg_target in cfg_targets:
                idx = cls.TAG_INDEX[categorical_format][cfg_target]
                categorical_text_cfg[idx] = "" if categorical_format == "sa_tag" else [""]
            # print(categorical_text_cfg)
        else:
            categorical_text_cfg = None 
        return cls(
            categorical_text=categorical_text,
            categorical_format=categorical_format,
            categorical_text_cfg=categorical_text_cfg,
        )

    @classmethod
    def collate_custom(cls, items: list["DataStyleText"]) -> dict[str, Any]:
        return {
            "style_text": [item.categorical_text for item in items],
        }


@dataclass
class DataCategoricalTag(DataItemCFGable):
    """
    Attributes:
        style_text: See DataStyleText
        artist_id: Also known as speaker_id
    """
    style_text: DataStyleText
    artist_id: int
    key: int
    key_root: int
    key_mode: int
    tempo: int 
    tempo_label: int
    time_signature: int
    instruments: list[int]

    # CFG
    artist_id_cfg: Optional[int] = None
    key_cfg: Optional[int] = None
    key_root_cfg: Optional[int] = None
    key_mode_cfg: Optional[int] = None
    tempo_cfg: Optional[int] = None
    tempo_label_cfg: Optional[int] = None
    time_signature_cfg: Optional[int] = None
    instruments_cfg: Optional[list[int]] = None

    DEFAULT_CFG_TARGETS: ClassVar[list[str]] = ["genre", "mood", "speaker"]

    @classmethod
    def new(
        cls,
        style_text: Optional[Union[DataStyleText, list[Union[str, list[str]]]]],
        artist_id: int,
        key: Optional[int]=None,
        key_root: Optional[int]=None,
        key_mode: Optional[int]=None,
        tempo: Optional[int]=None,
        tempo_label: Optional[int]=None,
        time_signature: Optional[int]=None,
        instruments: Optional[list[str]] = None,
        style_text_format: Optional[str] = None,
        enable_cfg: bool = False,
        cfg_targets: Optional[Union[str, list[str]]] = None,
    ):
        """
        Construct a DataCategoricalTag object, whose StyleText object and its categorical_text_cfg
        are deducted from vocab and cfg_targets. If cfg_targets is None, use DEFAULT_CFG_TARGETS as
        the default CFG targets.
        """
        if cfg_targets is None:
            cfg_targets = cls.DEFAULT_CFG_TARGETS
        if not (style_text_format and enable_cfg):
            style_text_format = None
            enable_cfg = False
            cfg_targets_for_style_text = None
            logger.debug("StyleText CFG disabled")
        else: 
            cfg_targets_for_style_text = [
                tag for tag in DataStyleText.available_cfg_targets(style_text_format) if tag in cfg_targets
            ]
            logger.debug(f"StyleText CFG targets: {cfg_targets_for_style_text}")
        if not isinstance(style_text, DataStyleText):
            style_text = DataStyleText.new(
                categorical_text=style_text,
                categorical_format=style_text_format,
                enable_cfg=enable_cfg,
                cfg_targets=cfg_targets_for_style_text,
            )
        key = KEY_ID_MAP["X"] if key is None else key
        key_root = ROOT_ID_MAP["X"] if key_root is None else key_root
        key_mode = MODE_ID_MAP["X"] if key_mode is None else key_mode
        tempo = -1 if tempo is None else tempo
        tempo_label = TEMPO_LABEL_ID_MAP[""] if tempo_label is None else tempo_label
        time_signature = TIME_SIGNATURE_ID_MAP["X"] if time_signature is None else time_signature
        instruments = [] if instruments is None else instruments

        artist_id_cfg = ARTIST_ID_MAP_V2["zh_empty"] if cfg_targets and ("speaker" in cfg_targets) else None
        key_cfg = KEY_ID_MAP["X"] if cfg_targets and ("key" in cfg_targets) else None
        key_root_cfg = ROOT_ID_MAP["X"] if cfg_targets and ("key_root" in cfg_targets) else None
        key_mode_cfg = MODE_ID_MAP["X"] if cfg_targets and ("key_mode" in cfg_targets) else None
        tempo_cfg = -1 if cfg_targets and ("tempo" in cfg_targets) else None
        tempo_label_cfg = TEMPO_LABEL_ID_MAP[""] if cfg_targets and ("tempo_label" in cfg_targets) else None
        time_signature_cfg = TIME_SIGNATURE_ID_MAP["X"] if cfg_targets and ("time_signature" in cfg_targets) else None
        instruments_cfg = [] if cfg_targets and ("instrument" in cfg_targets) else None

        return cls(
            style_text=style_text,
            artist_id=artist_id,
            key=key,
            key_root=key_root,
            key_mode=key_mode,
            tempo=tempo,
            tempo_label=tempo_label,
            time_signature=time_signature,
            instruments=instruments,
            # cfg
            artist_id_cfg=artist_id_cfg,
            key_cfg=key_cfg,
            key_root_cfg=key_root_cfg,
            key_mode_cfg=key_mode_cfg,
            tempo_cfg=tempo_cfg,
            tempo_label_cfg=tempo_label_cfg,
            time_signature_cfg=time_signature_cfg,
            instruments_cfg=instruments_cfg,
        )


    @classmethod
    def collate_custom(cls, items: list["DataCategoricalTag"]) -> dict[str, Any]:
        max_inst_num = 50   # (TODO): here heuristicly hard-code maximum instrument number as 50
        INST_PAD_ID = 0
        instrument_ids, instrument_ids_length = [], []
        for item in items:
            item_insts = [] if item.instruments is None else item.instruments
            instrument_id = pad_crop(
                torch.LongTensor(item_insts), max_inst_num, torch.int, INST_PAD_ID
            )
            instrument_ids.append(instrument_id)
            instrument_ids_length.append(min(len(item_insts), max_inst_num))
        
        return {
            "speaker_id": torch.LongTensor([item.artist_id for item in items]),
            "key": torch.LongTensor([item.key for item in items]),
            "key_root": torch.LongTensor([item.key_root for item in items]),
            "key_mode": torch.LongTensor([item.key_mode for item in items]),
            "tempo": torch.LongTensor([item.tempo for item in items]),
            "tempo_label": torch.LongTensor([item.tempo_label for item in items]),
            "time_signature": torch.LongTensor([item.time_signature for item in items]),
            "instrument": torch.stack(instrument_ids),
            "instrument_length": torch.LongTensor(instrument_ids_length),
        }


@dataclass
class DataSliceDuration(DataItemCFGable):
    """
    Arributes:
        duration: The duration of the audio slice in seconds.
    """
    duration: Optional[float]
    # CFG
    duration_cfg: Optional[float] = None

    @classmethod
    def collate_custom(cls, items: list["DataSliceDuration"]) -> dict[str, Any]:
        return {"slice_duration": torch.tensor([
            -1 if item.duration is None else item.duration for item in items
        ])}


@dataclass
class DataModAudioAutoType(DataItem):
    """Automatically set audio_type for each DataAudio attribute"""
    def __post_init__(self):
        self._update_audio_types()
    
    def _update_audio_types(self):
        for var in vars(self):
            value = getattr(self, var)
            if isinstance(value, DataAudio):
                audio_type = var + "_audio"
                if value.audio_type is not None and value.audio_type != audio_type:
                    DataError("audio_type is already non-empty")
                value.audio_type = var + "_audio"


@dataclass
class DataAudioZhVocal(DataModAudioAutoType):
    target: Optional[DataAudio] = None
    cond: Optional[DataAudio] = None


@dataclass
class DataTextZhVocal(DataItemCFGable):
    lyrics: DataLyrics
    categorical_tag: DataCategoricalTag
    freeform_text: DataFreeFormText
    # slice_duration: DataSliceDuration
    slice_type: str  # vocal | inst  (not actually used)

    @classmethod
    def collate_custom(cls, items: list["DataTextZhVocal"]) -> dict[str, Any]:
        return {"slice_type": [item.slice_type for item in items]}


@dataclass
class DataTextRLZhVocal(DataItemCFGable):
    lyrics: DataLyrics
    categorical_tag: DataCategoricalTag
    freeform_text: DataFreeFormText
    slice_duration: DataSliceDuration
    slice_type: str  # vocal | inst

    @classmethod
    def collate_custom(cls, items: list["DataTextRLZhVocal"]) -> dict[str, Any]:
        return {"slice_type": [item.slice_type for item in items]}


@dataclass
class DataDebug(DataItem):
    data_id: Optional[int] = None
    worker_id: Optional[int] = None
    song_id: Optional[int] = None
    shard: Optional[Any] = None  # which type?

    any: Optional[Any] = None  # put anything here for debugging purpose

    @classmethod
    def collate_custom(cls, items: list["DataDebug"]) -> dict[str, list]:
        any_field = [item.any for item in items]
        return safe_merge_dicts([{
                "data_id": [item.data_id for item in items],
                # required for callbacks.data_logger to work
                "worker_id": [item.worker_id for item in items],
                "song_id": [item.song_id for item in items],
                "shard": [item.shard for item in items],
            },
            {} if all(i is None for i in any_field) else {"any": any_field},
        ])


@dataclass
class DataInferZhVocal(DataModAudioAutoType):
    """Data that is used during inference stage only"""
    index: str
    text_category: str
    # style_text_display: str
    lyrics_display: str  # for frontend display
    lyrics_eval: str  # for WER calculation and video demo
    slice_duration: float  # for force-stop at given time
    # section_durations: list[float] = None  # for frontend display
    extra_video_text: str = None

    # audio prompting
    prompt: Optional[DataAudio] = None
    style: Optional[DataAudio] = None
    crossfade_secs: Optional[float] = None  # crossfade_secs / (1 / frame_rate) should be an integer

    @classmethod
    def collate_custom(cls, items: list["DataInferZhVocal"]) -> dict[str, Any]:
        crossfade_secs = _check_value_consistency_return_value([item.crossfade_secs for item in items])
        return {
            "index": [item.index for item in items],
            "text_category": [item.text_category for item in items],
            # "style_text_display": [item.style_text_display for item in items],
            "lyrics_display": [item.lyrics_display for item in items],
            "lyrics_eval": [item.lyrics_eval for item in items],
            "slice_duration": torch.Tensor([item.slice_duration for item in items]),
            # "section_durations": [item.section_durations for item in items],
            "extra_video_text": [item.extra_video_text for item in items],
            "crossfade_secs": crossfade_secs,
        }


@dataclass
class DataSampleZhVocal(DataItemCFGable, DataWithLength):
    """Main data sample class for Chinese vocal"""
    audio: Optional[DataAudioZhVocal] = None
    text: Optional[DataTextZhVocal] = None
    infer: Optional[DataInferZhVocal] = None
    debug: Optional[DataDebug] = None

    @property
    def length(self) -> int:
        """Number of audio samples"""
        return self.audio.target.audio_tensor.shape[-1]
    
    def __len__(self) -> int:
        return self.length


@dataclass
class DataSampleZhVocalVarlen(DataSampleZhVocal):
    @property
    def length(self) -> int:
        """Number of audio + lyrics tokens"""
        return self.audio.target.token_length + self.text.lyrics.token_length
    
    def __len__(self) -> int:
        return self.length


def _check_existence_consistency_return_type(lst: list) -> Optional[Any]:
    """All the items in the list can only be all None or all non-None"""
    if not (all(item is None for item in lst) or all(item is not None for item in lst)):
        raise CollateError("Item types are not consistent, can not mix None with other types.")
    if None in lst:
        return None
    return type(lst[0])


def _check_value_consistency_return_value(lst: list) -> Any:
    """All items in the list must be identical"""
    if not (len(lst) <= 1 or all(lst[0] == item for item in lst[1:])):
        raise CollateError("Item values are not consistent.")
    return None if len(lst) == 0 else lst[0]


def safe_merge_dicts(dicts: list[dict]):
    """Merge dicts with key duplication checking"""
    x = {}
    for d in dicts:
        x.update(d)
    if len(x) != sum(len(d) for d in dicts):
        raise CollateError("Found duplicated keys.")
    return x


########################## collate function ######################


def collate_fn_zh(
    batch: list[DataItemCFGable],
    app_type: Optional[str]=None,
    conditions: str = "style_text,lyrics_tokens",
    enable_cfg: bool = False,
) -> dict[str, Any]:
    cls = _check_existence_consistency_return_type(batch)
    return safe_merge_dicts([
        {
            "app_type": app_type,
            "conditions": conditions,
        },
        cls.collate_cfg(batch) if enable_cfg else cls.collate(batch)
    ])


########################## Transforms ######################


class VocalTransforms(BaseTransforms):
    name = "VocalTransforms"
    ALL_SR_TO_HANDLE = (16000, 24000, 44100)

    def __init__(
        self,
        app_type: Optional[str] = None,
        data_id: Optional[int] = None,
        meta_transform: Optional[ZhMetaTransform] = None,
        data_sample_rate = 24000,
        sample_rate: int = 24000,  # target sample_rate
        audio_key: str = "audio.npy",
        index_key: str = "__index_data__",
        frame_range: Optional[tuple[int, int]] = None,
        normalize_audio: bool = False,
        tokenizer: Optional[SamiPhonemeSeqTokenizer] = None,
        frame_rate: int = 25,
        segment_max_phone_len: int = 400,
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        slice_duration_dropout_rate: float = 0.0,
        note_dropout_rate: float = 0.0,
        section_instruments_dropout_rate: float = 0.0,
        extra_audio_keys: Optional[list] = None,
        mir_dropout_rate: float = 0.0,
        rewrite_style_text_with_slice_mir: bool = True,
        is_varlen: bool = False,
        max_skip_rate: Optional[float] = None,
        group_dropout_rate: Optional[float] = 0,
        style_text_categorical_format: Optional[str] = None,
        style_text_cfg_targets: Optional[list[str]] = None,
        is_rl: bool = False,
        enable_cfg: bool = False,
        debug: bool = False,
        cfg_targets: Optional[list[str]] = None,
    ):
        super().__init__()
        self.app_type = app_type
        self.data_id = data_id
        self.data_sample_rate = data_sample_rate
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.index_key = index_key
        self.frame_rate = frame_rate
        self.frame_range = frame_range
        self.segment_max_phone_len = segment_max_phone_len
        self.line_break_dropout_rate = line_break_dropout_rate
        self.section_tag_dropout_rate = section_tag_dropout_rate
        self.slice_duration_dropout_rate = slice_duration_dropout_rate
        self.note_dropout_rate = note_dropout_rate
        self.section_instruments_dropout_rate = section_instruments_dropout_rate
        self.extra_audio_keys = [] if extra_audio_keys is None else extra_audio_keys
        self.is_varlen = is_varlen
        self.max_skip_rate = max_skip_rate
        self.tokenizer = tokenizer
        assert meta_transform is not None
        self.meta_transform = meta_transform
        self.mir_dropout_rate = mir_dropout_rate
        self.enable_cfg = enable_cfg
        self.cfg_tagets = cfg_targets
        self.rewrite_style_text_with_slice_mir = rewrite_style_text_with_slice_mir
        self.group_dropout_rate = group_dropout_rate
        self.style_text_categorical_format = style_text_categorical_format if self.enable_cfg else None
        self.style_text_cfg_targets = style_text_cfg_targets if self.enable_cfg else []
        self.is_rl = is_rl
        self.debug = debug

        if self.debug:
            logger.debug(f"[{self.__class__.__name__}] debug mode is enabled.")

        self.base_transforms = {
            sr: self._compose_base_transforms(sr, normalize_audio) \
                for sr in self._supported_sample_rates()
        }

        # An internal state that records the distribution of slice types
        self.slice_type_record = defaultdict(lambda: 0)
        self.required_fields = [
            "style_text",
            "freeform_text",
            "artist_id",
            "lyrics_confidence",
        ]
        self.debug_fields = [
            "utterances",
            "structure_tags",
        ]

    def _compose_base_transforms(self, data_sample_rate, normalize_audio):
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if data_sample_rate!= self.sample_rate:
            base_transforms.append(Resample(data_sample_rate, self.sample_rate))
        if normalize_audio:
            base_transforms.append(AbsNormalizeAudio())
        return Compose(base_transforms)
    
    def _update_slice_type_stats(self, slice_type: str) -> None:
        self.slice_type_record[slice_type] += 1

    def _update_stats(self, skipped: bool, message: Optional[str] = None):
        super()._update_stats(skipped, message)
        # Print slice_type distribution
        if self.count > 0 and self.count % self.log_interval == 0: 
            n_slices = sum(self.slice_type_record.values())
            prob_record = {slice_type: round(count / n_slices, 2) for slice_type, count in self.slice_type_record.items()}
            worker_id = torch.utils.data.get_worker_info()
            if worker_id is not None:
                worker_id = worker_id.id
            else:
                worker_id = "Undefined"
            print(
                f"[{worker_id}] "
                f"Data ID: {self.data_id}, "
                f"{self.count} items, "
                f"slice_type distribution: {prob_record}",
                file=sys.stderr,
                flush=True,
            )

    def _supported_sample_rates(self) -> Set:
        return {*VocalTransforms.ALL_SR_TO_HANDLE, self.data_sample_rate}
    
    def __call__(self, item: dict[str, Any]) -> Generator:
        self.check_skip_rate()

        meta = item[self.index_key]
        if isinstance(meta, str):
            meta = json.loads(meta)

        # Parse and transform meta
        try:
            trans_meta = self.meta_transform(meta)
            required_fields = self.required_fields if not self.debug else self.required_fields + self.debug_fields
            meta_dict = {rf: trans_meta[rf] for rf in required_fields}  # song-level meta
            song_slices = trans_meta["song_slices"]
        except ZhMetaParseError as pe:
            self._update_stats(skipped=True, message=f"Data ID: {self.data_id}, ParseError: {pe}")
            return
        except ZhMetaTransformError as te:
            self._update_stats(skipped=True, message=f"Data ID: {self.data_id}, TransformError: {te}")
            return
    
        # Verify sample rate
        src_sr = item["src_sample_rate"]
        if src_sr not in self._supported_sample_rates():
            self._update_stats(skipped=True, message= \
                               f"Data ID: {self.data_id}. "
                               f"Sample rate mismatch. "
                               f"Expected {self.data_sample_rate}, "
                               f"got {item['src_sample_rate']}"
            )
            return

        try:
            # Handle different sample rates
            transform = self.base_transforms[src_sr]
            audio = transform(item[self.audio_key])
            extra_audio = [transform(item[k]) for k in self.extra_audio_keys]
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Yield one example per segment
        n_skipped_slices = 0
        error_msgs = []
        for song_slice in song_slices:
            try:
                data_sample = self.song_slice_to_data_sample(
                    song_slice=song_slice,
                    meta_dict=meta_dict,
                    audio=audio,
                    extra_audio=extra_audio,
                    debug=self.debug,
                    meta_before_transform=meta,
                    shard=item["__index_url__"],
                )
            except DataError as e:
                error_msgs.append(str(e))
                n_skipped_slices += 1
                continue
            self._update_slice_type_stats(data_sample.text.slice_type)
            yield data_sample

        if n_skipped_slices == len(song_slices):
            self._update_stats(
                skipped=True,
                message=f"Data ID: {self.data_id}, DataError: None of the song slices can be used, reasons: {', '.join(sorted(list(set(error_msgs))))}"
            )
        else:
            self._update_stats(skipped=False)

    def song_slice_to_data_sample(
        self,
        song_slice: SongSlice,
        meta_dict: dict[str, Any],
        audio: Any,
        extra_audio: Any,
        debug: bool = False,
        meta_before_transform: dict[str, Any] = None,
        shard: Optional[str] = None,
    ) -> DataSampleZhVocal:
        """Transform SongSlice to DataSample for Chinese Vocal."""

        # TODO (Yilin): Consider moving the dropout to the `to_dicts` method.
        reformatted_song_slice = song_slice.reformat_and_dropout_(self.line_break_dropout_rate, self.section_tag_dropout_rate)

        debug_any = {
            # Use reformatted_song_slice because it's closer to what the model sees while being trained.
            "song_slice": copy.deepcopy(reformatted_song_slice),
            "meta_dict": copy.deepcopy(meta_dict),
            "meta_before_transform": copy.deepcopy(meta_before_transform),
        } if debug else None

        # rewrite style text from song-level to slice-level according mir info
        this_style_text = copy.deepcopy(meta_dict['style_text'])
        this_freeform_text = copy.deepcopy(meta_dict['freeform_text'])
        
        tempo = -1
        if song_slice.mir_info is not None and self.rewrite_style_text_with_slice_mir and this_style_text is not None:
            this_style_text, _ = song_slice.mir_info.rewrite_style_text(this_style_text, mir_dropout_rate=self.mir_dropout_rate) 
            tempo = song_slice.mir_info.tempo if song_slice.mir_info.tempo and torch.rand(1) > self.mir_dropout_rate else -1

        style_text_list = []
        if this_style_text is not None:
            for st in this_style_text:
                if st is not None:
                    style_text_list.extend(st)
        if this_freeform_text is not None:
            for ft in this_freeform_text:
                if ft is not None:
                    style_text_list.extend(ft)

        enable_cfg_lyrics = self.enable_cfg and self.cfg_tagets and ("lyrics" in self.cfg_tagets)
        enable_cfg_freeform_text = self.enable_cfg and self.cfg_tagets and ("freeform_text" in self.cfg_tagets)
        enable_dropout_all = self.enable_cfg and (self.group_dropout_rate > 0) and self.style_text_cfg_targets  and ("lyrics_all" in self.style_text_cfg_targets )

        data_lyrics = DataLyrics.from_song_slice(
            song_slice=reformatted_song_slice,
            tokenizer=self.tokenizer,
            max_phone_len=self.segment_max_phone_len,
            drop_slice_duration=self.slice_duration_dropout_rate,
            drop_notes=self.note_dropout_rate,
            drop_section_instruments = self.section_instruments_dropout_rate,
            enable_dropout_all=enable_dropout_all,
            enable_cfg=enable_cfg_lyrics,
        )
        if reformatted_song_slice.has_utterance:
            slice_type = "vocal" if data_lyrics.encoding_mode == "phoneme" else "vocal_notes"
        else:
            slice_type = "inst"
        clip, _ = song_slice.slice_audio(audio, self.sample_rate, extra_audio)
        data_sample_cls = DataSampleZhVocalVarlen if self.is_varlen else DataSampleZhVocal
        if self.is_rl:
            data_sample = data_sample_cls(
                audio=DataAudioZhVocal(
                    target=DataAudio(
                        audio_tensor=clip,
                        token_length=int(clip.shape[-1] / self.sample_rate * self.frame_rate),
                        sample_rate=self.sample_rate,
                    ),
                ),
                text=DataTextRLZhVocal(
                    lyrics=data_lyrics,
                    freeform_text=DataFreeFormText.new(
                        text=this_freeform_text if this_freeform_text else '',
                        enable_cfg=enable_cfg_freeform_text,
                    ),
                    slice_duration=DataSliceDuration(
                        duration=song_slice.duration,
                    ),
                    categorical_tag=DataCategoricalTag.new(
                        style_text=this_style_text, #DataStyleText.new(this_style_text),
                        artist_id=meta_dict["artist_id"],
                        key=song_slice.mir_info.key_id if song_slice.mir_info else None,
                        key_root=song_slice.mir_info.key_root_id if song_slice.mir_info else None,
                        key_mode=song_slice.mir_info.key_mode_id if song_slice.mir_info else None,
                        tempo=tempo,
                        tempo_label=song_slice.mir_info.tempo_id if song_slice.mir_info else None,
                        time_signature=song_slice.mir_info.time_signature if song_slice.mir_info else None,
                        instruments=song_slice.mir_info.insts_id if song_slice.mir_info else None,
                        enable_cfg=self.enable_cfg,
                        style_text_format=self.style_text_categorical_format,
                        cfg_targets=self.cfg_tagets,
                    ),
                    slice_type=slice_type,
                ),
                debug=DataDebug(data_id=self.data_id, any=debug_any, shard=shard)
            )
        else:
            data_sample = data_sample_cls(
                audio=DataAudioZhVocal(
                    target=DataAudio(
                        audio_tensor=clip,
                        token_length=int(clip.shape[-1] / self.sample_rate * self.frame_rate),
                        sample_rate=self.sample_rate,
                    ),
                ),
                text=DataTextZhVocal(
                    lyrics=data_lyrics,
                    freeform_text=DataFreeFormText.new(
                        text=this_freeform_text if this_freeform_text else '',
                        enable_cfg=enable_cfg_freeform_text,
                    ),
                    categorical_tag=DataCategoricalTag.new(
                        style_text=this_style_text, #DataStyleText.new(this_style_text),
                        artist_id=meta_dict["artist_id"],
                        key=song_slice.mir_info.key_id if song_slice.mir_info else None,
                        key_root=song_slice.mir_info.key_root_id if song_slice.mir_info else None,
                        key_mode=song_slice.mir_info.key_mode_id if song_slice.mir_info else None,
                        tempo=tempo,
                        tempo_label=song_slice.mir_info.tempo_id if song_slice.mir_info else None,
                        time_signature=song_slice.mir_info.time_signature if song_slice.mir_info else None,
                        instruments=song_slice.mir_info.insts_id if song_slice.mir_info else None,
                        enable_cfg=self.enable_cfg,
                        style_text_format=self.style_text_categorical_format,
                        cfg_targets=self.style_text_cfg_targets,
                    ),
                    slice_type=slice_type,
                ),
                debug=DataDebug(data_id=self.data_id, any=debug_any, shard=shard)
            )
        validate_sample_in_frame_range_optional(data_sample, self.frame_range)
        return data_sample

    def check_skip_rate(self) -> None:
        if not (self.count > 0 and self.count % self.log_interval == 0) or self.max_skip_rate is None:
            return
        worker = torch.utils.data.get_worker_info()
        worker_id = "0"
        if worker is not None:
            worker_id = worker.id
        skip_rate = self.skipped / self.count
        if skip_rate > self.max_skip_rate:
            raise RuntimeError(f"Worker {worker_id} skipped {skip_rate:.2%} of the data, exceeding the max_skip_rate {self.max_skip_rate:.2%}")


def validate_sample_in_frame_range_optional(
    data_sample: DataSampleZhVocal,
    frame_range: Optional[tuple[int, int]]
) -> None:
    if frame_range is None:
        return
    n_frames = data_sample.audio.target.token_length + data_sample.text.lyrics.token_length
    if n_frames < frame_range[0]:
        raise DataError("Out of frame_range, too short")
    if n_frames > frame_range[1]:
        raise DataError("Out of frame_range, too long")


########################## Dataset ######################


class VocalParquetDataset(WebPipeline):
    name = "VocalParqueDataset"

    def __init__(
        self,
        transforms: VocalTransforms,
        data_id: int = 365,  # QQ music: 365 WYY_music: TODO
        url_pattern: str = None,
        extra_audio_keys=None,
        **kwargs,
    ):
        extra_audio_keys = [] if extra_audio_keys is None else extra_audio_keys
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern,
                                 extra_fields_in_data=extra_audio_keys, **kwargs)
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


########################## Data Modules ##########################


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_buffer_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        collate_fn: Optional[Callable] = None,
        do_shuffle: bool = True,    # set to False if shuffling is already done at dataset level
        prefetch_factor: Optional[int] = None,     # set to None to disable prefetching
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.do_shuffle = do_shuffle
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        if self.do_shuffle:
            train_dataset = DataPipeline(
                self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
            )
        else:
            train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        if isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )
                for val in self.validation_dataset
            ]
        else:
            return DataLoader(
                self.validation_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

def expand_item(item, n):
    if not isinstance(item, list):
        return [item] * n
    return item

def length_fn(data_sample: DataWithLength) -> int:
    return data_sample.length

class MixVocalWebDataModule(DataModule):
    def __init__(
        self,
        # basic
        app_type: Optional[str] = None,
        conditions: str = "style_text,lyrics_tokens",
        frame_rate: int = 25,
        sample_rate: int = 24000,
        tokenizer: str = "sami_phoneme_v2",
        # advanced
        pin_memory: bool = True,
        use_pipe: bool = True,
        shuffle_buffer_size: int = 10,
        prefetch_factor: Optional[int] = None,
        num_workers: int = 4,
        max_skip_rate: Optional[Union[float, list[float]]] = None,
        # batch, bucket, length
        batch_size: int = 2,
        val_batch_size: int = 0,
        use_dynamic_batch: bool = False,
        buckets_in_sec: Optional[list[int]] = None,
        buckets_in_frames: Optional[list[int]] = None,
        duration_range: Optional[tuple[int, int]] = None,
        frame_range: Optional[tuple[int, int]] = None,
        segment_max_phone_len: int = 400,
        # slicing
        max_seg_per_track: int = -1,
        segment_method: str = "random",
        slice_mode: str = "section",
        rewrite_style_text_with_slice_mir: bool = True,
        # filtering
        lyrics_confidence: Union[Optional[float], list[Optional[float]]] = None,
        lyrics_confidence_phrase: Union[Optional[float], list[Optional[float]]] = 0.6,
        deepchorus_confidence: Optional[float] = 0.45,
        artist_tagging_confidence: Union[Optional[float], list[Optional[float]]] = 0.5,
        sinking_threshold: float = 0.51,
        mir_filters: Optional[list[str]] = None,
        # dataset
        parquet_dataset_ids: Optional[list[int]] = None,
        parquet_dataset_weights: Optional[list[int]] = None,
        validation_parquet_dataset_ids: int = 3520,
        # dataset sample rate
        data_sample_rate: Union[int, list[int]] = 24000,
        validation_data_sample_rate: int = 24000,
        # dropout
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        slice_duration_dropout_rate: float = 0.0,
        note_dropout_rate: float = 0.0,
        section_instruments_dropout_rate: float = 0.0,
        mir_dropout_rate: float = 0.0,
        # others
        normalize_audio: bool = False,
        extra_audio_keys: Optional[list[str]] = None,
        multi_tasks: Optional[list[str]] = None,
        group_dropout_rate: Optional[float] = 0.0,
        style_text_categorical_format: Optional[str] = None,
        style_text_cfg_targets: Optional[list[str]] = None,
        # rl
        is_rl: bool = False,
        enable_cfg: bool = False,
        cfg_targets: Optional[list[str]] = None,
        # debug
        debug: bool = False,
    ):
        """
        Args:
            app_type: Placeholder (not used for now)
            data_sample_rate: The sample rate of dataset(s). If it's a list, each one corresponds to the data_id in order.
            sample_rate: The sample rate of target_audio
            buckets_in_frames: When it is set, enable varlen prefix training
            duration_range: Duration range in (min, max) format. The maximum value can not exceed max(buckets_in_sec)
            frame_range: Frame range in (min, max) format. The maximum value can not exceed max(buckets_in_frames)
            slice_mode: "section" (slicing by sections) or "full" (taking full songs only)
        """

        buckets_in_sec = [] if buckets_in_sec is None else sorted(buckets_in_sec)
        buckets_in_frames = [] if buckets_in_frames is None else sorted(buckets_in_frames)
        parquet_dataset_ids = [] if parquet_dataset_ids is None else parquet_dataset_ids
        parquet_dataset_weights = [] if parquet_dataset_weights is None else parquet_dataset_weights
        extra_audio_keys = [] if extra_audio_keys is None else extra_audio_keys

        self.is_varlen = bool(buckets_in_frames)
        self.is_rl = is_rl
        self.enable_cfg = enable_cfg
        self.cfg_targets = cfg_targets
        self.debug = debug

        # NOTE: If duration_range or frame_range is inferred from buckets, the lowest bucket will rarely gets used.
        if not duration_range:
            duration_range = (
                buckets_in_sec[0] if buckets_in_sec else 0, 
                buckets_in_sec[-1] if buckets_in_sec else math.floor(buckets_in_frames[-1] / frame_rate)
            )
        if duration_range and buckets_in_sec:
            if duration_range[-1] > buckets_in_sec[-1]:
                raise ValueError("The upper bound of duration_range should not exceed the largest bucket")

        if not frame_range:
            frame_range = (buckets_in_frames[0], buckets_in_frames[-1]) if buckets_in_frames else None
        if frame_range and buckets_in_frames:
            if frame_range[-1] > buckets_in_frames[-1]:
                raise ValueError("The upper bound of frame_range should not exceed the largest bucket")

        def get_vocal_transforms(
            data_id: int,
            audio_key: str,
            index_key: str,
            lyrics_confidence: Optional[float],
            lyrics_confidence_phrase: Optional[float],
            data_sample_rate: int,
            max_skip_rate: Optional[float],
        ) -> VocalTransforms:
            meta_transform = ZhMetaTransform.from_data_id(
                # dataset
                data_id,
                # length
                duration_range=duration_range,
                # slicing
                max_seg_per_track=max_seg_per_track,
                segment_method=segment_method,
                slice_mode=slice_mode,
                # filtering
                lyrics_confidence=lyrics_confidence,
                lyrics_confidence_phrase=lyrics_confidence_phrase,  # phrase (utterance) level confidence
                deepchorus_confidence=deepchorus_confidence,
                artist_tagging_confidence=artist_tagging_confidence,
                sinking_threshold=sinking_threshold,
                multi_tasks=multi_tasks,    # parse targeted info according to specified tasks
                mir_filters=mir_filters,
            )
            return VocalTransforms(
                # basic
                app_type=app_type,
                tokenizer=self.tokenizer,
                sample_rate=sample_rate,
                frame_rate=frame_rate,
                # length
                frame_range=frame_range,
                segment_max_phone_len=segment_max_phone_len,
                # dataset
                data_id=data_id,
                data_sample_rate=data_sample_rate,
                meta_transform=meta_transform,
                # keys
                audio_key=audio_key,
                index_key=index_key,
                extra_audio_keys=extra_audio_keys,
                # dropout
                line_break_dropout_rate=line_break_dropout_rate,
                section_tag_dropout_rate=section_tag_dropout_rate,
                slice_duration_dropout_rate=slice_duration_dropout_rate,
                note_dropout_rate=note_dropout_rate,
                section_instruments_dropout_rate=section_instruments_dropout_rate,
                mir_dropout_rate=mir_dropout_rate,
                # group-wise dropout
                group_dropout_rate=group_dropout_rate,
                style_text_categorical_format=style_text_categorical_format,
                style_text_cfg_targets=style_text_cfg_targets,
                # others
                is_varlen=self.is_varlen,
                max_skip_rate=max_skip_rate,
                normalize_audio=normalize_audio,
                rewrite_style_text_with_slice_mir=rewrite_style_text_with_slice_mir,
                # rl
                is_rl=self.is_rl,
                enable_cfg=self.enable_cfg,
                cfg_targets=self.cfg_targets,
                debug=self.debug,
        )

        def get_parquet_vocal_transforms(
            parquet_id: int,
            lyrics_confidence: Optional[float],
            lyrics_confidence_phrase: Optional[float],
            data_sample_rate: int,
            max_skip_rate: Optional[float],
        ) -> VocalTransforms:
            return get_vocal_transforms(
                data_id=parquet_id,
                audio_key="wav",
                index_key="meta",
                lyrics_confidence=lyrics_confidence,
                lyrics_confidence_phrase=lyrics_confidence_phrase,
                data_sample_rate=data_sample_rate,
                max_skip_rate=max_skip_rate
            )

        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory

        # If conditions are not set, auto-infer conditions based on app
        if app_type is not None and not conditions:
            conditions = APP_COND_MAP[app_type]

        self.collate_fn = partial(collate_fn_zh, app_type=app_type, conditions=conditions, 
                                  enable_cfg=self.enable_cfg)

        self._init_tokenizer(tokenizer)
        self._init_batcher(
            buckets_in_sec=buckets_in_sec,
            buckets_in_frames=buckets_in_frames,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            use_dynamic_batch=use_dynamic_batch,
            sample_rate=sample_rate,
        )

        self.parquet_vocal_datasets = []
        if parquet_dataset_ids:
            # expand confidence parameters to lists
            n_datasets = len(parquet_dataset_ids)

            lyrics_confidence = expand_item(lyrics_confidence, n_datasets)
            lyrics_confidence_phrase = expand_item(lyrics_confidence_phrase, n_datasets)
            data_sample_rate = expand_item(data_sample_rate, n_datasets)
            max_skip_rate = expand_item(max_skip_rate, n_datasets)

            for (
                parquet_id,
                _lyrics_confidence,
                _lyrics_confidence_phrase,
                _data_sample_rate,
                _max_skip_rate,
            ) in zip(
                parquet_dataset_ids,
                lyrics_confidence,
                lyrics_confidence_phrase,
                data_sample_rate,
                max_skip_rate,
            ):
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        transforms=get_parquet_vocal_transforms(
                            parquet_id,
                            lyrics_confidence=_lyrics_confidence,
                            lyrics_confidence_phrase=_lyrics_confidence_phrase,
                            data_sample_rate=_data_sample_rate,
                            max_skip_rate=_max_skip_rate,
                        ),
                        data_id=parquet_id,
                        resampled=True,
                        shardshuffle=True,
                        detshuffle=self.debug,
                        detresampled=self.debug,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                        extra_audio_keys=extra_audio_keys
                    ))
        else:
            NotImplementedError("Non-parquet datasets are not supported.")

        train_dataset = WebPipeline(
            MultiIterableDataset(
                datasets=self.parquet_vocal_datasets,
                weights=parquet_dataset_weights
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalParquetDataset(
                # Completely skip the lyrics confidence filtering and skip rate checking for the validation set
                # The user can choose to pre-filter the dataset
                # This is a quick fix to allow list type to work for validation set
                # Ideally we should set its confidence values separately
                transforms=get_parquet_vocal_transforms(
                    validation_parquet_dataset_ids,
                    lyrics_confidence=None,
                    lyrics_confidence_phrase=None,
                    data_sample_rate=validation_data_sample_rate,
                    max_skip_rate=None,
                ),
                data_id=validation_parquet_dataset_ids,
                resampled=False,
                shardshuffle=True,
                detshuffle=self.debug,
                detresampled=self.debug,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                extra_audio_keys=[]
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )]

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=self.collate_fn,
            prefetch_factor=prefetch_factor,
            do_shuffle=not self.debug,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

    def _init_tokenizer(self, tokenizer):
        if tokenizer not in PHONEME_TOKENIZERS:
            raise ValueError(f"Unsupported tokenizer {tokenizer}.")
        self.tokenizer = PHONEME_TOKENIZERS[tokenizer]()
        # DEPRECATED:
        # if tokenizer == "wordpiece":
        #     self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # elif tokenizer in SUPPORTED_SAMI_TOKENIZERS:
        #     self.tokenizer = tokenizer
        # elif tokenizer == "phoneme":
        #     with local_zero_first():
        #         self.tokenizer = Wav1Vec2PhonemeCTCTokenizer.from_pretrained(
        #             "facebook/wav1vec2-xlsr-53-espeak-cv-ft"
        #         )
        #         self.tokenizer._add_tokens(["<n>"])
        #     phonemizer.logger.get_logger().setLevel(logging.ERROR)
        # else:
        #     self.tokenizer = None

    def _init_batcher(
        self,
        buckets_in_sec,
        buckets_in_frames,
        batch_size,
        val_batch_size,
        use_dynamic_batch,
        sample_rate
    ):


        if val_batch_size == 0:
            val_batch_size = batch_size
        assert (len(buckets_in_sec) > 0) != (len(buckets_in_frames) > 0)
        print(f"dataloader initialized with buckets_in_frames: {buckets_in_frames}")
        print(f"dataloader initialized with buckets_in_sec: {buckets_in_sec}")
        if len(buckets_in_sec) == 0 and (len(buckets_in_frames) == 0):
            raise ValueError(f"Set buckets_in_sec or buckets_in_frames.")
        if len(buckets_in_sec) > 0:
            buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
            maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
            if use_dynamic_batch:
                self.batcher = BucketBatcher(
                    buckets=buckets_samples,
                    dynamic_batch=True,
                    maximum_bucket_size=maximum_bucket_size,
                    length_fn=length_fn,
                )
            else:
                self.batcher = BucketBatcher(
                    buckets=buckets_samples,
                    dynamic_batch=False,
                    batch_size=batch_size,
                    length_fn=length_fn,
                )
            self.val_batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=val_batch_size,
                length_fn=length_fn,
            )
        if len(buckets_in_frames) > 0:
            if use_dynamic_batch:
                maximum_bucket_size = batch_size * buckets_in_frames[-1]
                self.batcher = BucketBatcher(
                    buckets=buckets_in_frames,
                    dynamic_batch=True,
                    maximum_bucket_size=maximum_bucket_size,
                    length_fn=length_fn,
                )
            else:
                self.batcher = BucketBatcher(
                    buckets=buckets_in_frames,
                    dynamic_batch=False,
                    batch_size=batch_size,
                    length_fn=length_fn,
                )
            self.val_batcher = BucketBatcher(
                buckets=buckets_in_frames,
                dynamic_batch=False,
                batch_size=val_batch_size,
                length_fn=length_fn,
            )

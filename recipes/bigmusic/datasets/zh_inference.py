"""
Inference entrance for Chinese Vocal >= V4
"""

import datetime
import json
import logging
import shutil
import urllib.request
import re, random
from dataclasses import dataclass, asdict
from functools import partial, reduce
import operator
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union
from urllib.parse import urlparse
import hashlib
import copy
import pandas as pd
from torchaudio_augmentations import Compose
from webdataset.pipeline import DataPipeline

import samantha.utils.hdfs_helper as hh
from recipes.bigmusic.datasets.lyrics import default_batch_fn, transform_dataset
from recipes.bigmusic.datasets.tokenizers.sami_phoneme import SamiPhonemeSeqTokenizer
from recipes.bigmusic.datasets.transforms.sami_phoneme import SamiTextToPhonemeTransform, transform_phonemes_by_language
from recipes.bigmusic.datasets.transforms.preprocess import PreprocessModule
from recipes.bigmusic.datasets.transforms.remix_mir import RemixMIRTransform, transform_remix_mir_to_song_slice
from recipes.bigmusic.datasets.utils.zh_meta import SongSlice, Note
from recipes.bigmusic.datasets.mir_data_util import (
    tempo_to_label,
    AUDIO_TAGS_KEY_SPECIAL_MAP, 
    ID_ROOT_MAP,
    KEY_ID_MAP,
    MODE_ID_MAP,
    ROOT_ID_MAP,
    TEMPO_LABELS, COARSE_TEMPO_LABELS, COARSE_TEMPO_LABEL2TEMPO_LABEL,
    TEMPO_LABEL_ID_MAP,
    VOCAB2ID_INST_V1,
)
from recipes.bigmusic.datasets.utils.zh_meta import SongSlice, MIRInfo
from recipes.bigmusic.datasets.zh_mix import (
    DataAudio,
    DataCategoricalTag,
    DataDebug,
    DataFreeFormText,
    DataInferZhVocal,
    DataLyrics,
    DataSampleZhVocal,
    DataTextZhVocal,
    DataSliceDuration,
    DataStyleText,
    collate_fn_zh,
    PHONEME_TOKENIZERS,
)
from recipes.bigmusic.datasets.transforms.inst import tagging_inst_to_38
from recipes.datasets.mcc.mix import BaseTransforms
from recipes.datasets.mcc.sami_tokenizer import Phrase
from recipes.musiclm.inference.utils import load_wav
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import SetAudioDimensions, ToTensor

logger = logging.getLogger(__file__)


DEFAULT_AUDIO_PROMPT_CACHE_DIR = ".module_cache/audio_prompt_cache"
DEFAULT_PROMPT_CACHE_DIR = ".module_cache/prompt_cache"


AVAILABLE_CFG_TARGETS = list({tag for index_map in DataStyleText.TAG_INDEX.values() for tag in index_map}) + [
    "freeform_text",
    "lyrics",  # will apply to section_tag, singer_tag, section_durations, and slice_duration (total_duration)
    "line_break",
    "lyrics_all", # will apply to all lyrics tokens and "lyrics" components above
]


STR_TAG_CATEGORY_ORDER = [
    "genre",
    "mood",
    "gender",
    "timbre",
    "scene",
    "genre_extra",
    "extra",
    "lang",
    "sinking",
    "instrument",
    "tempo",
    "key",
    "mode",
]

TAG_CATEGORIES_FOR_PREPROCESS = STR_TAG_CATEGORY_ORDER

PROMPT_SAMPLE_RATE = 44100


AUDIO_PROMPT_CROSSFADE_SECS = 0.2


class ZhInferError(Exception):
    pass


@dataclass
class PromptItemBase:
    """PromptItem for regular lyrics-to-song"""
    index: str
    text_category: str
    style_text: list[list[str]]
    lyrics: str
    speaker_id: Optional[int]
    freeform_text: Optional[str]
    total_duration: Optional[float]

    key: Optional[str]
    tempo: Optional[int]
    tempo_label: Optional[int]
    main_inst: Optional[str]
    main_inst_id: Optional[int]
    insts_id: Optional[list[int]]

    # Fields that will be generated during transform
    lyrics_eval: Optional[str]
    lyrics_display: Optional[str]

    # For engineering-side inference
    frontend_results: Optional[list[str]]

    is_transformed: bool

    # Extra text that can be saved in video display (e.g., user's original prompt before tag rewrite)
    extra_video_text: str

    @property
    def prompt_type(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> "PromptItemBase":
        prompt_item = cls(**cls.transform_prompt_dict(
            prompt_dict,
            load_audio,
            audio_prompt_cache_dir
        ))._post_init_hook()
        return prompt_item

    @classmethod
    def transform_prompt_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            prompt_dict: {column: value}. Supported columns:
            - index: Prompt index.
            - text_category: Prompt category.
            - style_text or text_prompt: Categorical style tags, 3 slots for genre/mood/gender, seaparated by "|".
            - lyrics: Lyrics. If the lyrics only contain section tags or empty, it will be treated as an instrumental music prompt.
            - freeform_text: Any free-form text (Long text will be truncated internally).
            - total_duration: The total duration (in seconds, float). Do not set section_durations and total_duration at the same time. For instrumental music, must use empty lyrics with section_duration.
            - frontend_results: A list of strings of TTS frontend results in the format of "text#phoneme", or a list of dict [{"text": "<text>", "phoneme": "<phoneme>"}, ...]
            audio_prompt_cache_dir: If prompt audios in the CSV file are provided as URLs, the downloaded audio will be placed under this directory.
        """
        style_text = parse_style_text_or_individual_category_keys(prompt_dict)
        total_duration = parse_total_duration(prompt_dict.get("total_duration"))
        style_text_mir, key, tempo, main_inst, insts_id, main_inst_id = \
            parse_mir_info(prompt_dict.get("instrument"), prompt_dict.get("BPM"), prompt_dict.get("key"), prompt_dict.get("mode"), 
                            prompt_dict.get("mir_info"), mir_filters=None) #'root_convert_to_maj') 
        if len(style_text) < 13 and len(style_text_mir) > 0 and \
            any([len(text_mir) > 0 and text_mir[0] != '' for text_mir in style_text_mir]):
            style_text.extend([['']] * (13 - len(style_text))) 
            style_text[9:] = style_text_mir

        return {
            "index": prompt_dict["index"],
            "text_category": (prompt_dict.get("category") or prompt_dict.get("text_category", "default")).replace(" ", "_"),  # set a default category to avoid a flat directory structure
            "style_text": style_text,
            "lyrics": prompt_dict.get("lyrics", "").strip(),
            "speaker_id": parse_speaker_id(prompt_dict.get("speaker_id")),
            "freeform_text": parse_freeform_text(prompt_dict.get("freeform_text")),
            "extra_video_text": prompt_dict.get("extra_video_text"),
            "total_duration": total_duration,

            "key": key,
            "tempo": tempo,
            "main_inst": main_inst,
            "insts_id": insts_id,
            "main_inst_id": main_inst_id,
            "tempo_label": TEMPO_LABEL_ID_MAP[tempo_to_label(tempo)],

            "frontend_results": parse_frontend_results(prompt_dict.get("frontend_results")),
            "lyrics_eval": prompt_dict.get("lyrics_eval"),
            "lyrics_display": prompt_dict.get("lyrics_display"),
            "is_transformed": prompt_dict.get("is_transformed", False),
        }

    def _post_init_hook(self) -> "PromptItemBase":
        lyrics_display = self.lyrics_display if self.lyrics_display else self.lyrics
        lyrics_eval = lyrics_display
        self = copy.deepcopy(self)
        self.lyrics_display = lyrics_display
        self.lyrics_eval = lyrics_eval
        return self

    def load_audio(self, audio_prompt_cache_dir: str) -> "PromptItemBase":
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def language(self) -> str:
        """Get the prompt's language of based on style_text and freeform_text. \"Cantonese\" or empty string \"\"
        If the language is Cantonese, phoneme shift will be triggered.
        """
        style_text_list = []
        if self.style_text:
            style_text_list.extend(reduce(operator.add, self.style_text))
        if self.freeform_text:
            style_text_list.extend([ft.strip() for ft in self.freeform_text.split(",")])
        style_text_list = [tag.lower() for tag in style_text_list if tag]
        lang = ""
        if "Cantonese" in style_text_list or "cantonese" in style_text_list:
            lang = "Cantonese"
        if "Japanese" in style_text_list or "japanese" in style_text_list:
            lang = "Japanese"
        return lang
    
    def get_preprocess_tags(self, random_seed: Optional[int] = None) -> dict:
        def get_tags_by_category(tag_order: list[str], category: str) -> list[str]:
            try:
                tags = self.style_text[tag_order.index(category)]
            except IndexError:
                tags = []
            if isinstance(tags, str):
                return [tags]
            return tags

        if random_seed is not None:
            logger.info(f"preprocess: set random_seed {random_seed}")

        return {
            category: {
                "user_value": get_tags_by_category(STR_TAG_CATEGORY_ORDER, category),
                "predicted_value": [],
                "is_extracted": False
            } for category in TAG_CATEGORIES_FOR_PREPROCESS
        }

    def run_process_tags(
        self,
        preprocess_module: PreprocessModule,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_tags(
            tags=self.get_preprocess_tags(random_seed),
            topic="topic_placeholder",
            duration=self.total_duration,
            config=preprocess_module.config,
            random_seed=random_seed,
        )
    
    def run_process_lyrics(
        self,
        preprocess_module: PreprocessModule,
        process_tags_result: dict,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_lyrics(
            lyrics=self.lyrics,
            duration=process_tags_result["duration"],
            tags_music=process_tags_result.get("tags_music_cached", process_tags_result["tags_music"]),
            config=preprocess_module.config,
            random_seed=random_seed,
        )

    def run_preprocess_module(
        self,
        preprocess_module: PreprocessModule,
        random_seed: Optional[int] = None,
    ) -> dict:
        tags = self.run_process_tags(
            preprocess_module=preprocess_module,
            random_seed=random_seed,
        )
        result = self.run_process_lyrics(
            preprocess_module=preprocess_module,
            process_tags_result=tags,
            random_seed=random_seed,
        )
        return {
            **tags,
            **result,
        }

    def run_additional_transforms(
        self,
        request: dict,
        text_to_phoneme_transform: Optional[SamiTextToPhonemeTransform] = None,
        remix_mir_transform: Optional[RemixMIRTransform] = None,
    ) -> dict:
        """Additional processing for the request."""
        assert text_to_phoneme_transform is not None, "SamiTextToPhonemeTransform is required"
        return run_additional_transforms(
            lyrics=request["lyrics"],
            text_to_phoneme_transform=text_to_phoneme_transform,
            language=self.language,
        )

    def get_additional_extra(self) -> dict:
        """Additional extra items that are required for model inference"""
        # Necessary meta info
        extra = {
            "index": self.index,
            "text_category": self.text_category,
            "extra_video_text": self.extra_video_text,
            "prompt_type": self.prompt_type,
        }
        # Add mir_info only if any of the mir related items is not None
        # mir_info is currently not a part of preprocess
        mir_info = {
            "key": self.key,
            "tempo": self.tempo, 
            "insts_id": self.insts_id,
            "main_inst": self.main_inst,
            "main_inst_id": self.main_inst_id,
        }
        if any(v for v in mir_info.values()):
            extra = {
                **extra,
                "mir_info": mir_info,
            }
        return extra

    def preprocess_to_request(
        self,
        preprocess_module: PreprocessModule,
        text_to_phoneme_transform: Optional[Callable[[str], str]],
        remix_mir_transform: Optional[Callable],
        random_seed: Optional[int] = None,
    ) -> dict:
        """End-to-end generation: use preprocess_py to process the prompt content and get the request"""
        request = self.run_preprocess_module(
            preprocess_module=preprocess_module,
            random_seed=random_seed,
        )
        extra = {
            **self.run_additional_transforms(
                request=request,
                text_to_phoneme_transform=text_to_phoneme_transform,
                remix_mir_transform=remix_mir_transform,
            ),
            **self.get_additional_extra(),
        }
        extra = {k: v for k, v in extra.items() if v is not None}   # remove None values
        request["extra"] = extra
        return request

    def to_data_lyrics(
        self,
        tokenizer: SamiPhonemeSeqTokenizer,
        max_phone_len: int,
        enable_cfg: bool,
        enable_dropout_all: bool, 
    ) -> DataLyrics:
        language = self.language
        duration_at_inference = self.total_duration
        phrases = []
        if self.lyrics and not self.frontend_results:
            raise ZhInferError("No frontend_results found")
        if self.frontend_results:
            for idx, item in enumerate(self.frontend_results):
                text, phonemes = item["text"], item["phonemes"]
                phrase = Phrase.parse(text=text, phonemes=transform_phonemes_by_language(phonemes, language))   # text should already contain section_durations information (if any)
                if not phrase.is_empty:
                    phrases.append(phrase)
        song_slice = SongSlice(
            phrases=phrases,
            is_reformatted=True,
            duration_at_inference=duration_at_inference,
        )
        return DataLyrics.from_song_slice(
            song_slice=song_slice,
            tokenizer=tokenizer,
            max_phone_len=max_phone_len,
            enable_cfg=enable_cfg,
            enable_dropout_all=enable_dropout_all,
            enable_crop=True,  # automatically crop long phoneme sequence
            drop_slice_duration=(self.total_duration is None),
            drop_notes=True,  # SongSlice do not have notes
            drop_section_instruments=False,
        )
    
    def _to_data_infer_dict(self, frame_rate: int) -> dict:
        # WORKAROUND:
        # This method ensures DataModAudioAutoType's post-init hook runs correctly
        # while allowing subclasses to add arguments by overriding it.
        return {
            "index": self.index,
            "text_category": self.text_category,
            "extra_video_text": self.extra_video_text,
            "lyrics_display": self.lyrics_display,
            "lyrics_eval": self.lyrics_eval,
            "slice_duration": self.total_duration,
        }

    def to_data_infer(self, frame_rate: int) -> DataInferZhVocal:
        return DataInferZhVocal(**self._to_data_infer_dict(frame_rate))

    def to_data_sample(
        self,
        tokenizer: SamiPhonemeSeqTokenizer,
        max_phone_len: int,
        frame_rate: int,
        style_text_format: str,
        enable_cfg: bool,
        cfg_targets: Optional[list[str]],
    ) -> DataSampleZhVocal:
        if cfg_targets is not None and not set(cfg_targets).issubset(AVAILABLE_CFG_TARGETS):
            raise ValueError(f"Invalid cfg_targets: {cfg_targets}")
        enable_cfg_lyrics = enable_cfg and cfg_targets and ("lyrics" in cfg_targets)
        enable_cfg_freeform_text = enable_cfg and cfg_targets and ("freeform_text" in cfg_targets)
        enable_dropout_all = enable_cfg and cfg_targets and ("lyrics_all" in cfg_targets)

        return DataSampleZhVocal(
            text=DataTextZhVocal(
                lyrics=self.to_data_lyrics(tokenizer, max_phone_len, enable_cfg_lyrics, enable_dropout_all),
                freeform_text=DataFreeFormText.new(
                    text=self.freeform_text,
                    enable_cfg=enable_cfg_freeform_text,
                ),
                categorical_tag=DataCategoricalTag.new(
                    style_text=self.style_text,
                    artist_id=self.speaker_id,
                    key=KEY_ID_MAP[self.key] if self.key in KEY_ID_MAP else 0, # key_root, key_mode not given
                    tempo=self.tempo,   
                    tempo_label=self.tempo_label,
                    instruments=self.insts_id,
                    style_text_format=style_text_format,
                    enable_cfg=enable_cfg,
                    cfg_targets=cfg_targets,
                ),
                slice_type=self.prompt_type,
            ),
            infer=self.to_data_infer(frame_rate),
            debug=DataDebug(),
        )


@dataclass
class PromptItemVocal(PromptItemBase):
    @property
    def prompt_type(self) -> str:
        return "vocal"


@dataclass
class PromptItemCont(PromptItemBase):
    """PromptItem for audio continuation"""
    lyrics_prompt: str
    audio_prompt: Any
    audio_prompt_path: Any
    audio_prompt_sample_rate: int

    @property
    def prompt_type(self) -> str:
        return "audio_cont"

    @classmethod
    def transform_prompt_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> dict:
        return {
            **super().transform_prompt_dict(prompt_dict, load_audio, audio_prompt_cache_dir),
            "lyrics_prompt": parse_lyrics_prompt(prompt_dict.get("lyrics_prompt")),
            "audio_prompt": load_and_normalize_wav_optional(
                prompt_dict.get("audio_prompt") if load_audio else None,
                audio_prompt_cache_dir=audio_prompt_cache_dir,
            ),
            "audio_prompt_path": prompt_dict.get("audio_prompt"),
            "audio_prompt_sample_rate": PROMPT_SAMPLE_RATE,
        }

    def load_audio(self, audio_prompt_cache_dir: str) -> "PromptItemCont":
        self = copy.deepcopy(self)
        self.audio_prompt = load_and_normalize_wav_optional(
            self.audio_prompt_path,
            audio_prompt_cache_dir=audio_prompt_cache_dir,
        )
        return self

    @property
    def prompt_duration(self) -> float:
        return self.audio_prompt.shape[-1] / self.audio_prompt_sample_rate
    
    def get_additional_extra(self) -> dict:
        return {
            **super().get_additional_extra(),
            "audio_prompt": self.audio_prompt_path,
        }

    def run_process_tags(
        self,
        preprocess_module: PreprocessModule,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_tags(
            tags=self.get_preprocess_tags(random_seed),
            topic="topic_placeholder",
            duration=self.total_duration,
            prompt_duration=self.prompt_duration,
            prompt_lyrics=self.lyrics_prompt,
            config=preprocess_module.config,
            random_seed=random_seed,
        )
    
    def run_process_lyrics(
        self,
        preprocess_module: PreprocessModule,
        process_tags_result: dict,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_lyrics(
            lyrics=self.lyrics,
            duration=process_tags_result["duration"],
            tags_music=process_tags_result.get("tags_music_cached", process_tags_result["tags_music"]),
            prompt_duration=self.prompt_duration,
            prompt_lyrics=process_tags_result["prompt_lyrics"],
            config=preprocess_module.config,
            random_seed=random_seed,
        )
    
    def _to_data_infer_dict(self, frame_rate: int) -> dict:
        return {
            **super()._to_data_infer_dict(frame_rate),
            "prompt": DataAudio(
                audio_tensor=self.audio_prompt,
                token_length=int(self.prompt_duration * frame_rate),
                audio_type="audio",
                sample_rate=self.audio_prompt_sample_rate,
            ),
            "crossfade_secs": AUDIO_PROMPT_CROSSFADE_SECS,
        }


@dataclass
class PromptItemCover(PromptItemBase):
    """PromptItem for cover song"""
    remix_prompt: Any
    remix_prompt_path: Any
    remix_prompt_sample_rate: int
    remix_mir: dict

    @property
    def prompt_type(self) -> str:
        return "cover"

    @classmethod
    def transform_prompt_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> dict:
        return {
            **super().transform_prompt_dict(prompt_dict, load_audio, audio_prompt_cache_dir),
            # It supports either "lyrics" or "lyrics_prompt" as the original lyrics.
            # Override "lyrics_prompt" with "lyrics" if "lyrics" is provided.
            "lyrics": parse_lyrics_prompt(prompt_dict.get("lyrics", prompt_dict.get("lyrics_prompt"))),
            "remix_prompt": load_and_normalize_wav_optional(
                prompt_dict.get("remix_prompt") if load_audio else None,
                audio_prompt_cache_dir=audio_prompt_cache_dir,
            ),
            "remix_prompt_path": prompt_dict.get("remix_prompt"),
            "remix_prompt_sample_rate": PROMPT_SAMPLE_RATE,
            "remix_mir": prompt_dict.get("remix_mir"),
        }

    def load_audio(self, audio_prompt_cache_dir: str) -> "PromptItemCover":
        self = copy.deepcopy(self)
        self.remix_prompt = load_and_normalize_wav_optional(
            self.remix_prompt_path,
            audio_prompt_cache_dir=audio_prompt_cache_dir,
        )
        return self

    @property
    def prompt_duration(self) -> float:
        return self.remix_prompt.shape[-1] / self.remix_prompt_sample_rate
    
    def get_duration_scaling_factor(self, request: dict) -> float:
        # Use a hard-coded ratio to avoid stretching or squeezing the length too heavily
        ratio = 0.25
        required_duration = request["duration"]
        duration_scaling_factor = required_duration / self.prompt_duration
        if duration_scaling_factor > 1:
            duration_scaling_factor = (duration_scaling_factor - 1) * ratio + 1
        else:
            duration_scaling_factor = 1 - (1 - duration_scaling_factor) * ratio
        return duration_scaling_factor

    def run_additional_transforms(
        self,
        request: dict,
        text_to_phoneme_transform: Optional[SamiTextToPhonemeTransform] = None,
        remix_mir_transform: Optional[RemixMIRTransform] = None,
    ) -> dict:
        """Additional processing for the request."""
        assert remix_mir_transform is not None, "RemixMIRTransform is required for cover song"
        return run_additional_transforms(
            lyrics=request["lyrics"],
            remix_prompt=self.remix_prompt,
            remix_mir_transform=remix_mir_transform,
            duration_scaling_factor=self.get_duration_scaling_factor(request),
            language=self.language,
        )

    def to_data_lyrics(
        self,
        tokenizer: SamiPhonemeSeqTokenizer,
        max_phone_len: int,
        enable_cfg: bool,
        enable_dropout_all: bool, 
    ) -> DataLyrics:
        if not self.remix_mir:
            raise ZhInferError("No remix_mir found")
        utterances = self.remix_mir["utterances"]
        structure = self.remix_mir["structure"]
        song_slice = transform_remix_mir_to_song_slice(utterances, structure, self.language)
        song_slice.notes = [Note.from_dict(d) for d in self.remix_mir["notes"]]
        song_slice = song_slice.reformat_and_dropout_(0, 0)
        return DataLyrics.from_song_slice(
            song_slice=song_slice,
            tokenizer=tokenizer,
            max_phone_len=max_phone_len,
            enable_cfg=enable_cfg,
            enable_dropout_all=enable_dropout_all,
            enable_crop=True,  # automatically crop long phoneme sequence
            drop_slice_duration=(self.total_duration is None),
            drop_notes=False,  # use notes in SongSlice
        )


@dataclass
class PromptItemCoverEdit(PromptItemCover):
    """PromptItem for cover song with different lyrics
    Since there's no dedicated preprocess API for this use case,
    the implementation skips the processing for lyrics_prompt and lyrics.
    The user is responsible for their validity.

    Attributes:
        lyrics_prompt: The original lyrics in remix_audio
        lyrics: The lyrics to replace the original lyrics
    """
    lyrics_prompt: Optional[str]

    @property
    def prompt_type(self) -> str:
        return "cover_edit"

    @classmethod
    def transform_prompt_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> dict:
        kwargs = {
            **super().transform_prompt_dict(prompt_dict, load_audio, audio_prompt_cache_dir),
            "lyrics_prompt": parse_lyrics_prompt(prompt_dict.get("lyrics_prompt")),
        }
        if kwargs["lyrics"] == kwargs["lyrics_prompt"]:
            raise ZhInferError("lyrics and lyrics_prompt cannot be the same")
        return kwargs

    def run_process_lyrics(
        self,
        preprocess_module: PreprocessModule,
        process_tags_result: dict,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        result = preprocess_module.process_lyrics(
            lyrics=self.lyrics,
            duration=process_tags_result["duration"],
            tags_music=process_tags_result.get("tags_music_cached", process_tags_result["tags_music"]),
            random_seed=random_seed,
        )
        # We assume the provided lyrics and lyrics_prompt are valid.
        # Skip the processing by overwriting the result with the provided lyrics.
        result["lyrics"] = self.lyrics
        return result

    def run_additional_transforms(
        self,
        request: dict,
        text_to_phoneme_transform: Optional[SamiTextToPhonemeTransform] = None,
        remix_mir_transform: Optional[RemixMIRTransform] = None,
    ) -> dict:
        """Additional processing for the request."""
        assert remix_mir_transform is not None, "RemixMIRTransform is required for cover song"
        return run_additional_transforms(
            lyrics=self.lyrics_prompt,
            lyrics_replace=request["lyrics"],
            remix_prompt=self.remix_prompt,
            remix_mir_transform=remix_mir_transform,
            duration_scaling_factor=self.get_duration_scaling_factor(request),
            language=self.language,
        )


@dataclass
class PromptItemCoverCont(PromptItemCover):
    """PromptItem for cover song and audio continuation.
    Since there's no dedicated preprocess API for this use case,
    the implementation skips the processing for lyrics_prompt and lyrics.
    The user is responsible for their validity.

    Attributes:
        lyrics_prompt: The original lyrics in remix_audio
        lyrics: The lyrics of the continued audio with the same number of lines as those in lyrics_prompt
    """
    lyrics_prompt: Optional[str]

    @property
    def prompt_type(self) -> str:
        return "cover_cont"

    @classmethod
    def transform_prompt_dict(
        cls,
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> dict:
        return {
            **super().transform_prompt_dict(prompt_dict, load_audio, audio_prompt_cache_dir),
            "lyrics_prompt": parse_lyrics_prompt(prompt_dict.get("lyrics_prompt")),
        }

    def run_process_tags(
        self,
        preprocess_module: PreprocessModule,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_tags(
            tags=self.get_preprocess_tags(random_seed),
            topic="topic_placeholder",
            # This is a workaround to avoid the error caused by a long prompt duration
            prompt_duration=60 if self.prompt_duration > 60 else self.prompt_duration,
            prompt_lyrics=self.lyrics_prompt,
            duration=self.total_duration,
            random_seed=random_seed,
        )

    def run_process_lyrics(
        self,
        preprocess_module: PreprocessModule,
        process_tags_result: dict,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return {
            "lyrics": self.lyrics_prompt + "\n" + self.lyrics,
            "duration": self.prompt_duration * 2,
            "lyrics_display": self.lyrics_prompt + "\n" + self.lyrics,
        }

    def run_additional_transforms(
        self,
        request: dict,
        text_to_phoneme_transform: Optional[SamiTextToPhonemeTransform] = None,
        remix_mir_transform: Optional[RemixMIRTransform] = None,
    ) -> dict: 
        assert remix_mir_transform is not None, "RemixMIRTransform is required for cover song"
        # We totally disregard the preprocessed lyrics here because the current preprocess
        # module does not support this use case.
        extra = run_additional_transforms(
            lyrics=self.lyrics_prompt,
            lyrics_replace=self.lyrics,
            remix_prompt=self.remix_prompt,
            remix_mir_transform=remix_mir_transform,
            offset=self.prompt_duration,
            remix_mir_mode="dup_var",
            language=self.language,
        )
        extra["remix_prompt"] = self.remix_prompt_path  # put it back as audio prompt
        return extra

    def _to_data_infer_dict(self, frame_rate: int) -> dict:
        return {
            **super()._to_data_infer_dict(frame_rate),
            "prompt": DataAudio(
                audio_tensor=self.remix_prompt,
                token_length=int(self.prompt_duration * frame_rate),
                audio_type="audio",
                sample_rate=self.remix_prompt_sample_rate,
            ),
            "crossfade_secs": AUDIO_PROMPT_CROSSFADE_SECS,
        }


@dataclass
class PromptItemInst(PromptItemBase):
    """PromptItem for instrumental music"""
    @property
    def prompt_type(self) -> str:
        return "inst"

    def run_process_tags(
        self,
        preprocess_module: PreprocessModule,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        return preprocess_module.process_tags_inst(
            tags=self.get_preprocess_tags(random_seed),
            random_seed=random_seed,
        ) 

    def run_process_lyrics(
        self,
        preprocess_module: PreprocessModule,
        process_tags_result: dict,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Run preprocess_py to process the prompt content and get the request"""
        enable_section_duration_rewrite = True      # hardcode here since this may be only useful for debug
        if enable_section_duration_rewrite:
            duration, lyrics = preprocess_module.transform_duration_inst(
                duration=self.total_duration,
                lyrics=self.lyrics,
                genre=process_tags_result["tags_music"]["genre"],
            )
            result = {
                "lyrics": lyrics,
                "lyrics_display": lyrics,
                "duration": duration,
            }
        else:
            result = {
                "lyrics": self.lyrics,
                "lyrics_display": self.lyrics,
                "duration": self.total_duration,
            }
        return result

    def run_additional_transforms(
        self,
        request: dict,
        text_to_phoneme_transform: Optional[SamiTextToPhonemeTransform] = None,
        remix_mir_transform: Optional[RemixMIRTransform] = None,
    ) -> dict:
        """Additional processing for the request."""
        return run_additional_transforms_inst(request["lyrics"])

    def get_additional_extra(self) -> dict:
        return {
            **super().get_additional_extra(),
            "extra_video_text": self.extra_video_text,
        }


class PromptItemDispatcher:
    @staticmethod
    def from_type(prompt_type: Optional[str]):
        prompt_type_map = {
            "vocal": PromptItemVocal,
            "audio_cont": PromptItemCont,
            "cover": PromptItemCover,
            "cover_edit": PromptItemCoverEdit,
            "cover_cont": PromptItemCoverCont,
            "inst": PromptItemInst,
        }
        return prompt_type_map.get(prompt_type)

    @staticmethod
    def from_dict(
        prompt_dict: dict[str, Any],
        load_audio: bool = True,
        audio_prompt_cache_dir: Optional[str] = None,
    ) -> "PromptItemBase":
        def infer_prompt_cls(prompt_dict: dict[str, Any]):
            prompt_cls = PromptItemDispatcher.from_type(prompt_dict.get("prompt_type"))
            if prompt_cls is not None:
                return prompt_cls
            # infer from fields
            if prompt_dict.get("audio_prompt"):
                return PromptItemCont
            elif prompt_dict.get("remix_prompt"):
                return PromptItemCover
            return PromptItemVocal

        return infer_prompt_cls(prompt_dict).from_dict(prompt_dict, load_audio, audio_prompt_cache_dir)

    @staticmethod
    def from_request(
        request: dict,
        style_text_format: str,
        index: int = 0,
        audio_prompt_cache_dir: str = DEFAULT_AUDIO_PROMPT_CACHE_DIR,
    ) -> PromptItemBase:
        """
        request example:
        {
            "tags_music": {
                "genre": [...],
                "genre_extra": [...],
                "extra": [...],
                "mood": [...],
                "scene": [...],
                "speaker": [...],
                "voice": [...],
                "lang": [...],
                "sinking": [...]
                "speaker_id": ...,  # optional
            },
            "speaker_id": ...,  # optional (override tags_music.speaker_id)
            "lyrics": ...,
            "duration": ...,
            "freeform_text": ...,
            "prompt_lyrics": ...,
            "extra": {
                "audio_prompt": ...,
                "frontend_results": ...,
                "remix_mir": ...,
            }
        }
        """
        def infer_prompt_cls(request: dict):
            extra = request.get("extra", {})
            prompt_cls = PromptItemDispatcher.from_type(extra.get("prompt_type"))
            if prompt_cls is not None:
                return prompt_cls
            # infer from fields
            if extra.get("audio_prompt"):
                return PromptItemCont
            elif extra.get("remix_mir"):
                return PromptItemCover
            return PromptItemVocal

        tag_order = list(DataStyleText.TAG_INDEX[style_text_format].keys())
        tags_music = request["tags_music"]
        prompt_cls = infer_prompt_cls(request)
        item =  prompt_cls.from_dict(
            {
                "style_text": format_style_text([tags_music.get(k, []) for k in tag_order]),
                "speaker_id": request.get("speaker_id", tags_music.get("speaker_id")),  # request.speaker_id | request.tags_music.speaker_id
                "lyrics": request.get("lyrics"),
                "total_duration": request["duration"],
                "freeform_text": request.get("freeform_text"),
                "frontend_results": request.get("frontend_results"),
                "remix_mir": request.get("remix_mir"),
                **request.get("extra", {"index": f"default_{index}"}),
            },
            load_audio=True,  # prompt has been preprocessed, always load audio if there is any
            audio_prompt_cache_dir=audio_prompt_cache_dir,
        )
        return item

def deterministic_hash(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def download_wav(wav_path: str, cache_dir: Path):
    """Download wav file from web url or hdfs to local dir, unless the wav_path is already a local path"""
    def is_web_url(path: str):
        parsed_url = urlparse(path)
        return parsed_url.scheme in ('http', 'https')

    def get_local_path(path: str) -> Path:
        path_hash = deterministic_hash(path)
        filename = path_hash + ".wav"
        return cache_dir / filename

    local_path = get_local_path(wav_path)
    if is_web_url(wav_path):
        if not local_path.exists():
            with urllib.request.urlopen(wav_path) as response, open(local_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        return str(local_path)
    elif hh.ishdfs(wav_path):
        if not local_path.exists():
            hh.get(wav_path, str(local_path))
        return str(local_path)
    # it is local path already
    return wav_path


def load_and_normalize_wav_optional(wav_path: Optional[str], additional_transforms=(), audio_prompt_cache_dir=DEFAULT_AUDIO_PROMPT_CACHE_DIR):
    if wav_path is None or not wav_path.strip():
        return None
    cache_dir = Path(audio_prompt_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_wav_path = download_wav(wav_path, cache_dir)
    audio_transforms = Compose([ToTensor(), SetAudioDimensions(), *additional_transforms])
    return audio_transforms(load_wav(local_wav_path, sr=PROMPT_SAMPLE_RATE, mono=False))


def parse_freeform_text(freeform_text: Optional[str]) -> Optional[str]:
    freeform_text = freeform_text.strip() if freeform_text else None
    if not freeform_text:
        return None
    return freeform_text


def parse_style_text(
    style_text: Optional[Union[list, str]] = None,
    n_categories: Optional[Union[int, list[int]]] = None,
) -> list[list[str]]:
    if not style_text:
        return None
    if isinstance(style_text, list):
        result = style_text
    else:
        if "|" not in style_text:  # in case the input is just one genre tag
            style_text = style_text + "||"
        result = [[tag.strip() for tag in cat.split(",")] for cat in style_text.split("|")]
    if n_categories is not None:
        if not isinstance(n_categories, Iterable):
            n_categories = [n_categories]
        if not any((len(result) == nc) for nc in n_categories):
            raise ZhInferError(f"Invalid style_text: {style_text}")
    return result


def parse_style_text_or_individual_category_keys(prompt_dict: dict) -> list[list[str]]:
    """Check if style_text or text_prompt column exists and parse them into style_text. \
    If not, parse individual category keys into style_text.
    """
    n_categories = [3, 4, 5, 9, 10, 13]
    style_text = parse_style_text(prompt_dict.get("style_text", prompt_dict.get("text_prompt")), n_categories=n_categories)
    if style_text:
        return style_text
    style_string = "|".join(prompt_dict.get(tag, "") for tag in STR_TAG_CATEGORY_ORDER)
    return parse_style_text(style_string, n_categories=n_categories)


def format_style_text(style_text: Optional[list[list[str]]]) -> Optional[str]:
    if not style_text:
        return None
    return "|".join([",".join(sublist) for sublist in style_text])


def style_text_to_genre(style_text: list[list[str]]) -> str:
    # The first category's first tag
    return style_text[0][0]


def parse_total_duration(total_duration: Optional[Union[str, float]]) -> Optional[float]:
    if total_duration is None:
        return None
    if isinstance(total_duration, str):
        total_duration = total_duration.strip()
        if not total_duration:
            return None
        return float(total_duration)
    return total_duration

def parse_mir_info(instrument: str,
                   bpm: str, 
                   key: str, 
                   mode: str, 
                   mir_info: Optional[dict] = None,
                   mir_filters: Optional[list[str]] = 'root_convert_to_maj') -> MIRInfo:
    key_default, root_default, mode_default = "X", "empty ", ""
    bpm_default = -1
    inst_default = "empty_instrument"

    def parse_key(root, mode, mir_filters):
        if root == '' or root is None:
            root = root_default
        elif root not in KEY_ID_MAP:
            if len(root) == 1 or '#' in root:
                root = root.upper()
            elif len(root) == 2:
                root = AUDIO_TAGS_KEY_SPECIAL_MAP[root[0].upper()+root[1].lower()]
            else:
                raise ZhInferError(f"Invalid root: {root}")
            
        if mode == '' or mode is None:
            mode = mode_default
        elif mode not in MODE_ID_MAP:
            raise ZhInferError(f"Invalid mode: {mode}")
        
        if mir_filters is not None and 'root_convert_to_maj' in mir_filters and mode in ['Minor', 'Min']:
            ori_root_id = ROOT_ID_MAP[root]
            maj_root_id = (ori_root_id + 3) % 12
            root = ID_ROOT_MAP[maj_root_id] if maj_root_id != 0 else ID_ROOT_MAP[12]
            key = root

        key = ":".join([root, mode.replace("Major", "Maj").replace("Minor", "Min")]) if root != '' else key_default
        return key, root, mode
    
    def parse_tempo(bpm):
        def is_number(s):
            pattern = r'^[-+]?[0-9]*\.?[0-9]+$'
            return bool(re.match(pattern, s))
        
        tempo_value, tempo_text = bpm_default, ""

        if isinstance(bpm, str):
            if is_number(bpm):
                tempo_value = round(float(bpm))
                tempo_text = tempo_to_label(tempo_value)
            elif bpm in TEMPO_LABELS:
                tempo_text = bpm
            elif bpm in COARSE_TEMPO_LABELS:
                # randomly map coarse label to fine-grained label
                tempo_text = random.choice(COARSE_TEMPO_LABEL2TEMPO_LABEL[bpm])
        elif isinstance(bpm, int) or isinstance(bpm, float):
            tempo_value = round(bpm)
            tempo_text = tempo_to_label(tempo_value)
        return tempo_value, tempo_text
    
    def parse_inst(inst: str, mir_filters):
        if inst == '' or inst is None:
            return [], inst_default, [], 0
        
        raw_insts = [tagging_inst_to_38(_i.strip().replace(" ", "_")) for _i in inst.strip().split(",")]
        inst_vocab = VOCAB2ID_INST_V1.to_dict()
        insts = [inst for inst in raw_insts if inst in inst_vocab]
        insts_id = [inst_vocab[inst] for inst in insts]
        if len(insts) > 0:
            main_inst = insts[0] 
            main_inst_id = inst_vocab[main_inst]
        else:
            main_inst, main_inst_id = [], 0

        if mir_filters is not None:
            candidate_insts = copy.deepcopy(insts)
            for _inst in ["Vocal", "Drums", "Bass", "Choir_and_Voice"]:
                if f'main_inst_exclude_{_inst}' in mir_filters and main_inst == _inst:
                    candidate_insts.remove(_inst)
                    main_inst = None if len(candidate_insts) == 0 else candidate_insts[0]
                    main_inst_id = None if main_inst is None else inst_vocab[main_inst]
        
        return raw_insts, main_inst, insts_id, main_inst_id
    
    if mir_info is not None:  
        return [[''], [''], [''], ['']], mir_info["key"], mir_info["tempo"], mir_info["main_inst"], mir_info["insts_id"], mir_info["main_inst_id"]

    key, root, mode = parse_key(key, mode, mir_filters)
    tempo, tempo_text = parse_tempo(bpm)
    insts, main_inst, insts_id, main_inst_id = parse_inst(instrument, mir_filters)

    return [insts, [tempo_text], [root], [mode]], key, tempo, main_inst, insts_id, main_inst_id


def format_total_duration(total_duration: Optional[float]) -> Optional[str]:
    if total_duration is None:
        return None
    return f"total_duration: {total_duration}"


def format_bpm(bpm: Optional[int]) -> Optional[str]:
    if bpm is None or bpm < 0:
        return None
    return f"BPM:{bpm}"


def format_inst(inst: Optional[list[str]]) -> Optional[str]:
    if inst is None or (isinstance(inst, list) and len(list) <= 0):
        return None
    return ",".join(inst)


def parse_lyrics_prompt(lyrics_prompt: Optional[str]) -> Optional[str]:
    if lyrics_prompt is None:
        return None
    lyrics_prompt = lyrics_prompt.strip()
    if not lyrics_prompt:
        return None
    return lyrics_prompt


def parse_speaker_id(speaker_id: Optional[int]) -> Optional[int]:
    if speaker_id is None:
        return None
    speaker_id = int(speaker_id)
    return speaker_id


def parse_remix_mir(remix_mir: Optional[Union[str, dict]]) -> Optional[dict]:
    if not remix_mir:
        return None
    if isinstance(remix_mir, str):
        return json.loads(remix_mir)
    return remix_mir


def parse_frontend_results(
    frontend_results: Optional[Union[str, list[Union[str, dict[str, str]]]]]
) -> Optional[list[dict[str, str]]]:
    if not frontend_results:
        return None
    if isinstance(frontend_results, str):
        frontend_results = frontend_results.split("\n")
    dict_list = []
    for line in frontend_results:
        if isinstance(line, str):
            text, phonemes = line.split("#", 1)
            dict_list.append({
                "text": text,
                "phonemes": phonemes,
            })
        else:
            dict_list.append(line)
    return dict_list


def format_style_text(style_text: list[list[str]]) -> str:
    return "|".join([",".join(cat) for cat in style_text])


def zip_prompts(prompts: dict[str, Any]) -> list[dict[str, Any]]:
    n_prompts = min(len(v) for v in prompts.values())
    return [{k: prompts[k][i] for k in prompts.keys()} for i in range(n_prompts)]


def prompt_path_to_items(
    prompt_path: Union[dict, str, Path],
    load_audio: bool = True,
    audio_prompt_cache_dir: Optional[str] = None,
    max_items: Optional[int] = None,
) -> list[PromptItemBase]:
    """Load prompts from a dict or a CSV file path"""
    if isinstance(prompt_path, dict): # prompt path is already an item list
        # Format expects { 'style_audio': [], 'style_text': [], 'lyrics': [] }
        df = pd.DataFrame.from_dict(prompt_path)
    else:
        df = pd.read_csv(Path(prompt_path))
    # NOTE: Please make sure the numerical columns are always filled.
    # Remove columns that only contain na. Columns can be partially filled.
    # The caller is responsible for data validation.
    df = df.dropna(axis='columns', how='all')
    df = df.fillna("")  # Replace na with ""
    prompt_dicts = zip_prompts(df.to_dict('list'))
    if max_items is not None:
        if max_items <= 0:
            raise ZhInferError(f"Invalid max_items {max_items}")
        prompt_dicts = prompt_dicts[:max_items]
    return [PromptItemDispatcher.from_dict(
        d,
        load_audio=load_audio,
        audio_prompt_cache_dir=audio_prompt_cache_dir
    ) for d in prompt_dicts]


def translate_style_text_vocab(vocab: str) -> str:
    """Deduct style_text_format from the vocab string"""
    prefixes = ["sa_tag", "multi_tag", "9_cat", "10_cat", "13_cat"]
    for prefix in prefixes:
        if vocab.startswith(prefix):
            return prefix
    raise ValueError(f"Unsupported tranform target: {vocab}") 


def run_additional_transforms(
    lyrics: str,
    lyrics_replace: Optional[str] = None,
    remix_prompt: Optional[Any] = None,
    text_to_phoneme_transform: Optional[Callable[[str], str]] = None,
    remix_mir_transform: Optional[RemixMIRTransform] = None,
    duration_scaling_factor: float = 1.0,
    offset: float = 0.0,
    remix_mir_mode: str = "regular",
    language: Optional[list] = ['Chinese'],
) -> dict:
    """Call additional services/models to get extra information."""
    def lyrics_to_frontend_results(transform: Callable[[str], str], language: Optional[list]=['Chinese']) -> list[dict[str, str]]:
        phrases = [Phrase.parse(text=line) for line in lyrics.split("\n")]
        line = []
        if "Japanese" in language:
            lang = "ja"
        else:
            lang = "zh_en"
        for phrase in phrases:
            line.append({
                "text": phrase.format_text(),
                "phonemes": transform(phrase.text, lang=lang) if phrase.has_utterance else None,
            })
        return line

    # Additional process (service call)
    results = {}
    if text_to_phoneme_transform is not None:
        results["frontend_results"] = lyrics_to_frontend_results(text_to_phoneme_transform, language=language)
    if remix_mir_transform is not None:
        results["remix_mir"] = remix_mir_transform(
            wav=remix_prompt,
            text=lyrics,
            text_replace=lyrics_replace,
            duration_scaling_factor=duration_scaling_factor,
            offset=offset,
            mode=remix_mir_mode,
        )

    return results

def run_additional_transforms_inst(
    lyrics: str,
) -> dict:
    """Call additional services/models to get extra information."""
    def lyrics_to_frontend_results() -> list[dict[str, str]]:
        phrases = [Phrase.parse(text=line) for line in lyrics.split("\n")]
        line = []
        for phrase in phrases:
            line.append({
                "text": phrase.format_text(),
                "phonemes": None,
            })
        return line

    # Additional process (service call)
    results = {}
    results["frontend_results"] = lyrics_to_frontend_results()

    return results


def run_preprocess(
    prompt_items: list[PromptItemBase],
    prompt_sample_rate: int = PROMPT_SAMPLE_RATE,
    audio_prompt_cache_dir: str = DEFAULT_AUDIO_PROMPT_CACHE_DIR,
    random_seed: Optional[int] = None,
) -> list[dict]:
    def is_remix_mir_required(prompts: list[PromptItemBase]) -> bool:
        return any(isinstance(prompt, PromptItemCover) for prompt in prompts)

    def is_phoneme_required(prompts: list[PromptItemBase]) -> bool:
        return not all(isinstance(prompt, PromptItemCover) for prompt in prompts)

    preprocess_module = PreprocessModule()
    # Check all the items, avoid intializing unnecessary transforms
    remix_mir_transform = RemixMIRTransform(sample_rate=prompt_sample_rate) if is_remix_mir_required(prompt_items) else None
    text_to_phoneme_transform = SamiTextToPhonemeTransform.init_cached() if is_phoneme_required(prompt_items) else None

    return [
        prompt_item.load_audio(audio_prompt_cache_dir).preprocess_to_request(
            preprocess_module=preprocess_module,
            text_to_phoneme_transform=text_to_phoneme_transform,
            remix_mir_transform=remix_mir_transform,
            random_seed=random_seed,
        ) for prompt_item in prompt_items
    ]


class ZhInferTransforms(BaseTransforms):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, prompt_item: PromptItemBase) -> DataSampleZhVocal:
        return prompt_item.to_data_sample(*self.args, **self.kwargs)


class ZhInferPromptCacheManager:
    CACHE_SUFFIX = ".cache.json"

    def __init__(self,
        processed_requests: list[dict],
        style_text_format: str,
        creation_time: Optional[str] = None,
        preprocess_version: Optional[str] = None,
        random_seed: Optional[int] = None,
        original_prompt_hash: Optional[str] = None,  # hash of the original (unprocessed) prompts
    ):
        self.processed_requests = processed_requests
        self.style_text_format = style_text_format
        self.creation_time = creation_time if creation_time else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # The following variables will be used to calculate the cache hash
        self.preprocess_version = preprocess_version
        self.random_seed = random_seed
        self.original_prompt_hash = original_prompt_hash

    @property
    def hash(self) -> str:
        prompt_str = json.dumps({
            "preprocess_version": self.preprocess_version,
            "random_seed": self.random_seed,
            "original_prompt_hash": self.original_prompt_hash,
        })
        return deterministic_hash(prompt_str)

    @property
    def suggested_filename(self) -> str:
        return f"{self.creation_time}{ZhInferPromptCacheManager.CACHE_SUFFIX}"

    def to_dict(self) -> dict:
        return {
            "cache_info": {
                "creation_time": self.creation_time,
                "preprocess_version": self.preprocess_version,
                "random_seed": self.random_seed,
                "original_prompt_hash": self.original_prompt_hash,
                "style_text_format": self.style_text_format,
            },
            "prompts": self.processed_requests,
        }
    
    def write(self, cache_fp: Union[str, Path]) -> None:
        Path(cache_fp).parent.mkdir(exist_ok=True, parents=True)
        with open(cache_fp, "w", encoding="utf-8") as fout:
            json.dump(self.to_dict(), fout, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_dict(cls, d) -> "ZhInferPromptCacheManager":
        try:
            return cls(
                processed_requests=d["prompts"],
                **d["cache_info"],
            )
        except (KeyError, TypeError) as e:
            raise ZhInferError(f"Invalid cache file: {e}")

    @classmethod
    def load_from_path(cls, path: Union[str, Path]) -> "ZhInferPromptCacheManager":
        with open(path) as fin:
            d = json.load(fin)
        return cls.load_from_dict(d)
    
    @classmethod
    def load_from_dir(cls, path: Union[str, Path], target_cache_hash: str) -> Optional["ZhInferPromptCacheManager"]:
        cache_fps = Path(path).glob(f"*{ZhInferPromptCacheManager.CACHE_SUFFIX}")
        for cache_fp in cache_fps:
            if not cache_fp.exists():
                continue
            cache = ZhInferPromptCacheManager.load_from_path(cache_fp)
            if target_cache_hash == cache.hash:
                logger.info(f"Prompt cache found at path {str(cache_fp)}, load")
                return ZhInferPromptCacheManager.load_from_path(cache_fp)
        logger.info(f"Did not find prompt cache in directory {str(path)}")
        return None

    @classmethod
    def new(
        cls,
        original_prompt_items: list[PromptItemBase],
        style_text_format: str,
        processed_requests: Optional[list[dict]]=None,
        preprocess_version: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> "ZhInferPromptCacheManager":
        def get_prompt_items_hash(prompt_items: list[PromptItemBase]) -> str:
            def get_dict_hash(_dict: list[dict]) -> str:
                return deterministic_hash(json.dumps(_dict))
            return get_dict_hash(json.dumps([prompt_item.to_dict() for prompt_item in prompt_items]))

        original_prompt_hash = get_prompt_items_hash(original_prompt_items)
        return cls(
            processed_requests=processed_requests if processed_requests else [],
            style_text_format=style_text_format,
            preprocess_version=preprocess_version,
            random_seed=random_seed,
            original_prompt_hash=original_prompt_hash,
        )

    def to_prompt_items(self, audio_prompt_cache_dir: str) -> list[PromptItemBase]:
        return [PromptItemDispatcher.from_request(
            request=request,
            style_text_format=self.style_text_format,
            audio_prompt_cache_dir=audio_prompt_cache_dir,
        ) for request in self.processed_requests]


def inference_dataset_from_prompt(
    prompt_path: Union[dict, str, list[dict], Path],
    conditions: str = "style_text,lyrics_tokens",
    batch_size: int = 8,
    max_phone_len: int = 2000,
    style_text_transform_target: str = "sa_tag",
    enable_cfg: bool = True,
    cfg_targets: Optional[list[str]] = None,  # default
    lyrics_tokenizer: str = "sami_phoneme_v3",
    prompt_sample_rate: int = PROMPT_SAMPLE_RATE,
    frame_rate: int = 25,
    audio_prompt_cache_dir: str = DEFAULT_AUDIO_PROMPT_CACHE_DIR,
    max_items: Optional[int] = None,
    enable_offline_preprocess: bool = False,
    cache_preprocessed_prompt: bool = False,
    prompt_cache_dir: str = DEFAULT_PROMPT_CACHE_DIR,
    random_seed: Optional[int] = None,
) -> DataPipeline:
    """
    Args:
        prompt_path:
            - A path to a CSV prompt file.
            - A path to a JSON prompt file. Using JSON format will always bypass the internal preprocessing logic, regardless of the value of enable_offline_preprocess.
            - A dict that can be read by pandas.DataFrame.from_dict.
            - A list of dicts where each dict is a request that has been processed by a separate service in advance.
              See the docstring of `request_dict_to_item` for the dict format. enable_offline_preprocess will be ignored in this option.
        conditions: Inference conditions, separated by comma.
        batch_size: Batch size.
        max_phone_len: The maximum token length of the phoneme sequence, including section tags and singer tags. Any phoneme sequence
            that is longer than it will be cropped down to this value.
        style_text_transform_target: The target decides the target vocab and how to rewrite. The available options are the keys of STYLE_TEXT_REWRITE_FNS.
        enable_cfg: Whether to use CFG.
        cfg_targets: A list of CFG rewrite targets. All available options are in AVAILABLE_CFG_TARGETS.
        lyrics_tokenizer: The lyrics tokenizer name, which decides the vocab of the tokenizer. The value should be the same as the one used for model training.
            "sami_phoneme_legacy" or "sami_phoneme_v2".
        prompt_sample_rate: The sample rate of audio prompt. Support 44100 only.
        frame_rate: Audio token frame rate.
        audio_prompt_cache_dir: If prompt audios in the CSV file are provided as URLs, the downloaded audio will be placed under this directory.
        max_items: The maximum number of prompt items that will be used from the top. All prompt items will be used if it is None.
        enable_offline_preprocess: Whether to use offline preprocess by preprocess_py. preprocess_py must be properly setup in advance.
            If the prompt is already a JSON file, this argument will be set to False internally.
        cache_preprocessed_prompt: If enable_offline_preprocess is True, and set this argument to True, the processed prompt set will be cached under promprt_cache_dir.
            If a future prompt's content matches the cache, load the cache and bypass preprocess.
        promprt_cache_dir: The directory to save the prompt caches.
    Return: a DataPipeline object.

    See PromptItem.from_dict for the CSV/dict format.
    """
    def remap_cfg_targets(cfg_targets: Optional[list[str]]) -> Optional[list[str]]:
        def remap_section_tag(cfg_target: str) -> str:
            if cfg_target != "section_tag":
                return cfg_target
            logger.warning('CFG target "section_tag" is deprecated and replaced by "lyrics"')
            return "lyrics"
        if not cfg_targets:
            return None
        return [remap_section_tag(cfg_target) for cfg_target in cfg_targets]

    def prompt_path_to_json_hook(prompt_path: Union[str, Any]):
        """If prompt_path is a path to a JSON file, load the JSON file."""
        if not (isinstance(prompt_path, str) or isinstance(prompt_path, Path)):
            return prompt_path
        path = Path(prompt_path)
        if path.exists() and path.suffix == ".json":
            try:  # check if it is a cache file first
                prompt_manager = ZhInferPromptCacheManager.load_from_path(prompt_path)
                logger.info("prompt_path points to a cache JSON file, load")
                return prompt_manager.processed_requests
            except ZhInferError:  # if not, load as a normal JSON file
                logger.info("prompt_path points to a normal JSON file, load")
                with open(prompt_path) as fin:
                    return json.load(fin)
        return prompt_path

    def is_prompt_path_request(prompt_path: Any) -> bool:
        return isinstance(prompt_path, list) or isinstance(prompt_path, dict)
    
    def prompt_path_dict_to_list(prompt_path: Union[dict, list[dict]]) -> list[dict]:
        if isinstance(prompt_path, dict):
            logger.info("prompt_path is a dict, converted it to list")
            prompt_path = [prompt_path]
        return prompt_path

    def load_prompt_items_from_cache(original_prompt_items: list[PromptItemBase], bypass: bool) -> tuple[list[PromptItemBase], bool]:
        if bypass:
            return original_prompt_items, False
        target_hash = ZhInferPromptCacheManager.new(
            original_prompt_items=original_prompt_items,
            style_text_format=style_text_format,
            random_seed=random_seed
        ).hash
        cache = ZhInferPromptCacheManager.load_from_dir(prompt_cache_dir, target_hash)
        if cache is not None:
            return cache.to_prompt_items(audio_prompt_cache_dir), True
        return prompt_items, False

    def write_cache(original_prompt_items: list[PromptItemBase], processed_requests: list[dict], bypass: bool) -> None:
        if bypass:
            return
        cache = ZhInferPromptCacheManager.new(
            original_prompt_items=original_prompt_items,
            style_text_format=style_text_format,
            processed_requests=processed_requests,
            random_seed=random_seed,
        )
        cache_fp = Path(prompt_cache_dir) / cache.suggested_filename
        logger.info(f"Save preprocessed prompt cache to {str(cache_fp)}")
        cache.write(cache_fp)

    def load_prompt_audios(prompt_items: list[PromptItemBase]) -> list[PromptItemBase]:
        return [prompt_item.load_audio(audio_prompt_cache_dir=audio_prompt_cache_dir) for prompt_item in prompt_items]
    
    style_text_format = translate_style_text_vocab(style_text_transform_target)

    # Load prompt
    prompt_path = prompt_path_to_json_hook(prompt_path=prompt_path)
    is_prompt_request = is_prompt_path_request(prompt_path)
    if is_prompt_request:
        logger.info("prompt_path is a request")
        prompt_path = prompt_path_dict_to_list(prompt_path)
        prompt_items = [
            PromptItemDispatcher.from_request(
                request=request,
                style_text_format=style_text_format,
                index=index,
                audio_prompt_cache_dir=audio_prompt_cache_dir,
            ) for index, request in enumerate((prompt_path[:max_items] if max_items is not None else prompt_path))
        ]
        if enable_offline_preprocess:
            logger.info("Request is supposed to be preprocessed, disable offline preprocess")
            enable_offline_preprocess = False
    else:
        logger.info("prompt_path points to a CSV file")
        prompt_items = prompt_path_to_items(
            prompt_path=prompt_path,
            load_audio=False,  # Do not load audio here to save memory
            max_items=max_items,
        )

    # Cache loading / Preprocess
    if enable_offline_preprocess:
        logger.info("Enable offline preprocess")
        original_prompt_items = prompt_items
        prompt_items, cache_loaded = load_prompt_items_from_cache(original_prompt_items, bypass=not cache_preprocessed_prompt)
        if not cache_loaded:  # call external transform function in preprocess_py offline
            logger.info("Cache not found or cache disabled, run preprocess")
            requests = run_preprocess(
                prompt_items=original_prompt_items,
                prompt_sample_rate=prompt_sample_rate,
                audio_prompt_cache_dir=audio_prompt_cache_dir,
                random_seed=random_seed,
            )
            write_cache(original_prompt_items, requests, bypass=not cache_preprocessed_prompt)
            prompt_items = [PromptItemDispatcher.from_request(
                request=request,
                style_text_format=style_text_format,
                index=index
            ) for index, request in enumerate(requests)]
    elif not is_prompt_request:
        logger.info("Load audios in the CSV prompt file without preprocessing")
        prompt_items = load_prompt_audios(prompt_items)

    dataset = WebPipeline(prompt_items, pipeline=[])
    batch_fn = default_batch_fn(batch_size, collation_fn=partial(collate_fn_zh, app_type=None, conditions=conditions, enable_cfg=enable_cfg))
    tokenizer = PHONEME_TOKENIZERS[lyrics_tokenizer]()

    return transform_dataset(
        dataset,
        # Wrap the rest of operations (PromptItem -> DataSample conversion, including lyrics tokenization)
        # into a Transform that can then be used by transform_dataset.
        segment_transforms=[ZhInferTransforms(
            tokenizer=tokenizer,
            max_phone_len=max_phone_len,
            frame_rate=frame_rate,
            style_text_format=style_text_format,
            enable_cfg=enable_cfg,
            cfg_targets=remap_cfg_targets(cfg_targets),
        )],
        # Any transform after batching (mainly tensor padding) has been included in the collate_fn
        batch_transforms=[],
        batch_fn=batch_fn,
    )

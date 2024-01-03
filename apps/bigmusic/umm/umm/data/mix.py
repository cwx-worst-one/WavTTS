import logging
import math
import random
import sys
from string import punctuation

try:
    from zhon.hanzi import punctuation as punctuation_zh
except:
    print("[WARNING] Failed to import zhon.hanzi.punctuation")
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import phonemizer
import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import AutoTokenizer, BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

from task.bigmusic.umm.umm.data import INDEX

try:
    from task.bigmusic.umm.umm.data.sami_tokenizer import convert_labels_to_text_id
except:
    print("[WARNING] Failed to import convert_labels_to_text_id")
import logging

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset import ra_wds
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
    FastNormalizeAudio,
    LoudnessCheck,
)
from samantha.utils.webdataset import return_self

logger = logging.getLogger(__name__)


def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    return text.translate(str.maketrans("", "", nlp_punctuation)).strip()


def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    # text = []
    token = []
    # tag = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        # text.append(batch[idx]["text"])
        token.append(batch[idx].get("token", torch.zeros(0).long()))
        # tag.append(batch[idx]["tag"])
    return {
        "audio": torch.stack(audio, dim=0),
        # "text": text,
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
        # "tag": tag,
    }


def collate_mss_audio_and_token_fn(
    batch: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Randomly pad MSS audios (full mix, instrumental, vocal) and corresponding tokens.

    @hanoihantrakul 11/10/2023
    To ensure the 3 tracks are randomly padded in the same places, the easiest way to do this
    is to stack the audio as (3, len_in_samples), apply RandomPad() and then unstack the audio.
    """
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)

    audio, audio_vocal, audio_inst, token = (
        [],
        [],
        [],
        [],
    )  # Ignore `text` and `tag` keys
    for x in batch:
        # e.g. x["audio"].shape, x["audio_inst"].shape, x["audio_vocal"].shape are all (1, 54000)
        audio_stacked = torch.cat((x["audio"], x["audio_vocal"], x["audio_inst"]))
        assert audio_stacked.size() == (3, x["audio"].size(-1))
        audio_stacked = random_pad(
            audio_stacked
        )  # like padding a 3 channel audio i.e (3, 54000)
        audio.append(audio_stacked[0])
        audio_vocal.append(audio_stacked[1])
        audio_inst.append(audio_stacked[2])
        token.append(x.get("token", torch.zeros(0).long()))

    return {
        "audio": torch.stack(audio, dim=0),
        "audio_vocal": torch.stack(audio_vocal, dim=0),
        "audio_inst": torch.stack(audio_inst, dim=0),
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
    }


def collate_audio_text(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    text = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        text.append(batch[idx]["text"])
    return {"audio": torch.stack(audio, dim=0), "text": text}


def collate_audio(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
    return {"audio": torch.stack(audio, dim=0)}


class BaseTransforms:
    """Base class for all data transforms"""

    name = "BaseTransforms"

    def __init__(self, log_interval: int = 100):
        self.count = 0
        self.skipped = 0
        self.messages = {}
        self.log_interval = log_interval

    def _update_stats(self, skipped: bool, message: Optional[str] = None):
        self.count += 1
        if skipped:
            self.skipped += 1
        if message is not None:
            message = f"[{self.name}] {message}"
            if message not in self.messages:
                self.messages[message] = 0
            self.messages[message] += 1
        # Print
        if self.count > 0 and self.count % self.log_interval == 0:
            worker_id = torch.utils.data.get_worker_info()
            if worker_id is not None:
                worker_id = worker_id.id
            else:
                worker_id = "Undefined"
            print(
                f"[{worker_id}] "
                f"Skipped {self.skipped}/{self.count} items, "
                f"Messages: {self.messages}",
                file=sys.stderr,
                flush=True,
            )

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        raise NotImplementedError()


class MCCTransforms(BaseTransforms):
    name = "MCCTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ) -> None:
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.lyrics_confidence = lyrics_confidence
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        # Apply AED filtering if applicable
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False, "Not AED Filtered"
        # Avoid sound effect if applicable
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False, "Sound Effect"
        # Apply license-based filtering if applicable
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False, "Excluded License"
        # Apply AudioMetrics filtering if applicable
        if self.audio_metrics_filtered:
            is_good, msg = self.is_audio_metrics_good(metadata.get("audio_metrics", {}))
            if not is_good:
                return False, msg
        return True, None

    def is_audio_metrics_good(self, audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        # Clipping
        clip = audio_metrics.get("clipping", {})
        if clip.get("rate", 0) >= 5e-5:
            return False, "clipping"
        for ch in ["left", "right"]:
            if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
                return False, "clipping"
        # Loudness
        loudness = audio_metrics.get("loudness", {})
        if (
            loudness.get("integrated_loudness", -7) > -5
            or loudness.get("max_mom_loud", -7) >= 0
            or loudness.get("max_short_term_loud", -7) >= 0
        ):
            return False, "loudness"
        # RMS stats
        rms_stats = audio_metrics.get("rms_stats", {})
        if rms_stats.get("peak", 0) > 3:
            return False, "rms_stats"
        for ch in ["left", "right"]:
            if (
                rms_stats.get(f"{ch}_total", -10) > -5
                or rms_stats.get(f"{ch}_total", -10) < -40
                or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
            ):
                return False, "rms_stats"
        # Cutoff frequency
        cutoff_freq = audio_metrics.get("cutoff_frequency", {})
        for ch in ["left", "right"]:
            if (
                cutoff_freq.get(f"rel_{ch}", 48000) < 15000
                and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
                and cutoff_freq.get(f"band_std_{ch}", 10) < 5
            ):
                return False, "cutoff_frequency"
        # Phase
        phase = audio_metrics.get("phase_check", {})
        if (
            phase.get("has_phase_issue", False)
            or abs(phase.get("rms_downmix_diff", 0.1)) > 3
        ):
            return False, "phase_check"
        return True, None

    def __call__(self, x: Dict[str, Any]) -> Generator:
        raise NotImplementedError()


class MCCVocalTransforms(MCCTransforms):
    name = "MCCVocalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(item["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        result = lyrics.get("result", None)
        if result is None or len(result) != 1:
            self._update_stats(skipped=True, message="No result")
            return
        utterances = result[0].get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return

        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class MCCInstrumentalTransforms(MCCTransforms):
    name = "MCCInstrumentalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )

    def __call__(self, item: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(item["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        if audio.size(-1) < self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        if self.max_num_crops is not None:
            max_num_crops = self.max_num_crops
        else:
            max_num_crops = audio.size(-1) // (self.max_duration * self.sample_rate)

        durations = []
        starts = []
        for _ in range(max_num_crops):
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            durations.append(duration)
            starts.append(random.randint(0, audio.size(-1) - duration))

        for i in range(max_num_crops):
            clip = audio[:, starts[i] : starts[i] + durations[i]]
            if not self.is_loud(clip):
                self._update_stats(skipped=True, message="Not loud enough")
                return
            output_dict = {"audio": clip, "text": "", "tag": "instrumental"}
            if self.tokenizer is not None:
                if self.tokenizer == "tts_chinese_frontend_model":
                    token = torch.zeros(0).long()
                else:
                    encoded_text = self.tokenizer(
                        "",
                        add_special_tokens=False,
                        # padding="longest",
                        return_tensors="pt",
                    )
                    token = encoded_text["input_ids"].squeeze(dim=0)
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class MSSTransforms(BaseTransforms):
    """
    MSS data consists of the full mix, vocal mix and instrumental mix as well as
    the lyrics and utterances found in the vocal.

    In this transform, the vocal utterances start and end time have to be checked and filtered
    to ensure the model sees complete vocal utterances and no "half sentences".

    MSSTransforms() follows a overlapping logic to MCCVocalTransforms():
        - Access the lyrics and utterances
        - Filter the utterances and check start and end times
        - Extract the same patch of audio from full mix, instrumental and vocal track
        - Tokenize the full English text
        - Pack data into an output_dict

    Args:
        min_duration (float): Minumum audio clip length (after cropping according to a detected utterance)
        max_duration (float): Maximum audio clip length (after cropping according to a detected utterance)
        lyrics_confidence (float): Minimum confidence to determine if lyric should be included in training
        frame_rate (int): Frame rate of the tokenizer. Used to align start and end times with audio
        tokenizer: When available, this is normally the "wordpiece" tokenizer BertTokenizer.from_pretrained("bert-large-uncased")

    Yields:
        output_dict:
            "audio": clip of full audio corresponding to a detected utterance. Kept the `audio` name to make reusing other parts of DataModule() code easier
            "audio_inst": clip of MSS instrumental corresponding to a detected utterance
            "audio_vocal": clip of MSS vocal corresponding to a detected utterance
            "tag": mss tag for task ID
            "utterance_text": text of detected utterance
            "token": language model tokens of the detected utterance
    """

    name = "MSSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        min_duration: float = 5,
        max_duration: float = 30,
        lyrics_confidence: float = 0.8,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.lyrics_confidence = lyrics_confidence
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        # Compose default audio transforms
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)

    def filter_lyrics(self, utterance):
        """
        @hanoihantrakul 11/7/2023
        This is a method copy pasted from MCCVocalTransforms().
        Unfortunately, inheritance is tricky because the nested keys changes per dataset.
        """
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(
            utterance["additions"]["confidence"]
        )  # different from MCCVocalTransforms()
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def _get_lyrics(self, item):
        """Every HDFS dataset has a specific set of nested keys to access the root `lyrics` key."""
        return item["__index_data__"].get("lyrics", None)

    def _try_to_extract_utterances(self, lyrics):
        """Contains dataset-specific logic for accessing utterances nested under the root `lyrics` key"""
        utterances = lyrics.get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return
        return utterances

    def _get_utterance_start_end_idxs(self, selected_utterance):
        """Utterance start and end time expressed in audio samples."""
        start = int(float(selected_utterance["start_time"]) / 1000 * self.sample_rate)
        end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
        return start, end

    def _try_to_tokenize_text(self, text):
        """Tokenize the text. Skip sample if no tokens were produced."""
        assert self.tokenizer is not None
        encoded_text = self.tokenizer(
            text,
            add_special_tokens=False,
            # padding="longest",
            return_tensors="pt",
        )
        token = encoded_text["input_ids"].squeeze(dim=0)
        if token.size(-1) == 0:
            self._update_stats(skipped=True, message="Token zero length")
            return
        else:
            return token

    def _token_is_longer_than_audio_clip_bounds(self, token, clip):
        """Check if the tokens extend beyond the bounds of the audio clip."""
        return (
            token.size(-1)
            > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
        )

    def _ensure_lengths_are_consistent(self, audio_dict, tolerance_in_samples=100):
        """The instrumental and vocal MSS tracks are often ~64 samples longer than the full mix."""
        # The three tracks should be similar lengths to begin with
        assert (
            audio_dict["inst"].size(-1) - audio_dict["full"].size(-1)
            < tolerance_in_samples
        )
        assert (
            audio_dict["vocal"].size(-1) - audio_dict["full"].size(-1)
            < tolerance_in_samples
        )

        # Trim to shortest length
        shortest_len = min(v.size(-1) for v in audio_dict.values())
        audio_dict = {k: v[:, :shortest_len] for k, v in audio_dict.items()}
        assert audio_dict["full"].size() == audio_dict["inst"].size()
        assert audio_dict["full"].size() == audio_dict["vocal"].size()
        assert audio_dict["inst"].size() == audio_dict["vocal"].size()
        return audio_dict

    def __call__(self, item: Dict[str, Any]) -> Generator:
        # Find the lyrics key
        lyrics = self._get_lyrics(item)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return

        # Find utterances from the lyrics
        utterances = self._try_to_extract_utterances(lyrics)
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return
        random.shuffle(filtered_utterances)

        # Impose an upper limit of how many contiguous (i.e. highly correlated and low diversity)
        # utterances can be included in the dataset
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        # Load audio
        try:
            _audio_dict = {
                "full": self.base_transform(item["audio.npy"]),
                "inst": self.base_transform(item["acc.npy"]),
                "vocal": self.base_transform(item["vocal.npy"]),
            }
            _audio_dict = self._ensure_lengths_are_consistent(_audio_dict)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Use filtered utterances to extract audio
        for selected_utterance in filtered_utterances:
            start_idx, end_idx = self._get_utterance_start_end_idxs(selected_utterance)
            if end_idx > _audio_dict["full"].size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return

            # Use utterance start and end idx to extract corresponding patch of audio
            output_dict = {
                "audio": _audio_dict["full"][:, start_idx:end_idx],
                "audio_inst": _audio_dict["inst"][:, start_idx:end_idx],
                "audio_vocal": _audio_dict["vocal"][:, start_idx:end_idx],
                "tag": "mss",
            }
            # Tokenize the text associated with this patch of audio
            utterance_text = normalize_text(str(selected_utterance["text"]))
            output_dict.update(utterance_text=utterance_text)
            if self.tokenizer is not None:
                token = self._try_to_tokenize_text(utterance_text)
                clip = output_dict["audio_vocal"]
                if self._token_is_longer_than_audio_clip_bounds(token, clip):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                else:
                    output_dict.update(token=token)

            self._update_stats(skipped=False)
            yield output_dict


class MSSTransformsV2(MSSTransforms, MCCTransforms):
    """
    This Transform is very similar to MSSTransforms() with the addition of
    metadata and audio_metrics filtering logic from MCCTransforms().

    @hanoihantrakul 11/10/2023
    Rather than combine this all into one MSSTransforms, it was much easier to
    create and test a new class MSSTransformsV2.
    """

    name = "MSSTransformsV2"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        min_duration: float = 5,
        max_duration: float = 30,
        lyrics_confidence: float = 0.8,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        MSSTransforms.__init__(
            self,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            lyrics_confidence=lyrics_confidence,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            **kwargs,
        )
        # Configure MCCTransforms() with settings default for MSS dataset.
        # These are near identical to settings for MCCVocalTransforms(MCCTransforms)
        MCCTransforms.__init__(
            self,
            sample_rate=sample_rate,
            audio_key="audio.npy",
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=0.05,
            loudness_ratio_threshold=0.2,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=False,
            aed_filtered=False,
            audio_metrics_filtered=True,
            avoid_sound_effect=True,  # although MSS dataset shouldn't contain this, it is a safety check
            exclude_licenses=[],  # MSS data doesn't contain this key
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )

    def __call__(self, item: Dict[str, Any]) -> Generator:
        # Check metadata and audio metrics. New logic compared to MSSTransforms()
        is_good, message = self.is_metadata_good(item["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

        # TODO: @hanoihantrakul, the code below is copy/pasted from MSSTransforms().__call__(). Refactor to reuse.
        # Find the lyrics key
        lyrics = self._get_lyrics(item)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return

        # Find utterances from the lyrics
        utterances = self._try_to_extract_utterances(lyrics)
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return
        random.shuffle(filtered_utterances)

        # Impose an upper limit of how many contiguous (i.e. highly correlated and low diversity)
        # utterances can be included in the dataset
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        # Load audio
        try:
            _audio_dict = {
                "full": self.base_transform(item["audio.npy"]),
                "inst": self.base_transform(item["acc.npy"]),
                "vocal": self.base_transform(item["vocal.npy"]),
            }
            _audio_dict = self._ensure_lengths_are_consistent(_audio_dict)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Use filtered utterances to extract audio
        for selected_utterance in filtered_utterances:
            start_idx, end_idx = self._get_utterance_start_end_idxs(selected_utterance)
            if end_idx > _audio_dict["full"].size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return

            # Use utterance start and end idx to extract corresponding patch of audio
            output_dict = {
                "audio": _audio_dict["full"][:, start_idx:end_idx],
                "audio_inst": _audio_dict["inst"][:, start_idx:end_idx],
                "audio_vocal": _audio_dict["vocal"][:, start_idx:end_idx],
                "tag": "mss",
            }
            # Tokenize the text associated with this patch of audio
            utterance_text = normalize_text(str(selected_utterance["text"]))
            output_dict.update(utterance_text=utterance_text)
            if self.tokenizer is not None:
                token = self._try_to_tokenize_text(utterance_text)
                clip = output_dict["audio_vocal"]
                if self._token_is_longer_than_audio_clip_bounds(token, clip):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                else:
                    output_dict.update(token=token)

            self._update_stats(skipped=False)
            yield output_dict


class KaraokeTransforms(BaseTransforms):
    name = "KaraokeTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        utterances = lyrics.get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None and callable(self.tokenizer):
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class LibriLightASRTransforms(BaseTransforms):
    name = "LibriLightASRTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"])
        end_time = float(utterance["end_time"])
        delta = end_time - start_time
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        utterances = item.get("__index_data__", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(float(selected_utterance["start_time"]) * self.sample_rate)
            end = int(float(selected_utterance["end_time"]) * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class LibriTTSTransforms(BaseTransforms):
    name = "LibriTTSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        text = normalize_text(item["normalized_text.txt"])
        output_dict = {"audio": audio, "text": text, "tag": "vocal"}
        if self.tokenizer is not None and callable(self.tokenizer):
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=False,
                # padding="longest",
                return_tensors="pt",
            )
            token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class SpeechZhTransforms(BaseTransforms):
    name = "SpeechZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        text = item["__index_data__"].get("text", None)
        if text is None or len(text) == 0:
            self._update_stats(skipped=True, message="No text")
            return
        output_dict = {"audio": audio, "text": text, "tag": "speech"}
        if self.tokenizer is not None:
            if self.tokenizer == "tts_chinese_frontend_model":
                labels = list(
                    filter(
                        lambda x: x != "", item["__index_data__"]["labels"].split("\n")
                    )
                )
                labels, _, _ = convert_labels_to_text_id(labels)
                token = torch.from_numpy(labels[0]).long()
            else:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class DouyinMusicTransforms(BaseTransforms):
    name = "DouyinMusicTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        audio_npy = item[self.audio_key]
        if audio_npy.shape[-1] < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio_npy.shape[-1] > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        try:
            audio = self.base_transform(audio_npy)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        text = item["__index_data__"].get("text", None)
        if text is None or len(text) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        output_dict = {"audio": audio, "text": text, "tag": "vocal"}
        if self.tokenizer is not None:
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=False,
                # padding="longest",
                return_tensors="pt",
            )
            token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class VocalZhTransforms(BaseTransforms):
    name = "VocalZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.lyrics_confidence = lyrics_confidence
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        result = lyrics.get("result", None)
        if result is None or len(result) != 1:
            self._update_stats(skipped=True, message="No result")
            return
        utterances = result[0].get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return

        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                if self.tokenizer == "tts_chinese_frontend_model":
                    labels = list(
                        filter(
                            lambda x: x != "", selected_utterance["phoneme"].split("\n")
                        )
                    )
                    labels, _, _ = convert_labels_to_text_id(labels)
                    token = torch.from_numpy(labels[0]).long()
                else:
                    encoded_text = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        # padding="longest",
                        return_tensors="pt",
                    )
                    token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms: BaseTransforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)


class LibriLightDataset(WebPipeline):
    name = "LibriLight"
    data_sample_rate = 16000

    def __init__(
        self,
        urls,
        sample_rate=24000,
        min_duration: int = 5,
        max_duration: int = 30,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"map": [self.transform]}]
        super().__init__(dataset, pipeline)

    def transform(self, item) -> Dict[str, Any]:
        audio = self.base_transform(item["audio.npy"])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            return
        return {"audio": audio}


class LibriLightASRDataset(WebPipeline):
    name = "LibriLightASR"

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/speech/librilight_asr_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriLightASRTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class BigTTSTransforms(BaseTransforms):
    name = "BigTTSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "wav",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        self.resampler = {}
        self.fast_normalizer = FastNormalizeAudio()
        self.normalize_audio = normalize_audio
        self.base_transform = Compose(base_transforms)

    def do_resample(self, src_sample_rate, x):
        if src_sample_rate == self.sample_rate:
            return x

        if src_sample_rate not in self.resampler:
            self.resampler[src_sample_rate] = Resample(
                src_sample_rate, self.sample_rate
            )
        return self.resampler[src_sample_rate](x)

    def remove_punc(self, text):
        punctuation_all = list(punctuation_zh + punctuation)
        for punc in punctuation_all:
            text = text.replace(punc, "")
        return text

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            output_dict = {}
            text = self.remove_punc(normalize_text(item["text"]))
            if self.tokenizer is not None and callable(self.tokenizer):
                encoded_text = self.tokenizer(
                    text, add_special_tokens=False, return_tensors="pt"
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                output_dict.update(token=token)
            audio = self.base_transform(item[self.audio_key])
            audio = self.do_resample(item["src_sample_rate"], audio)
            if self.normalize_audio:
                audio = self.fast_normalizer(audio)
            if audio.size(-1) < self.min_duration * self.sample_rate:
                self._update_stats(skipped=True, message="Audio too short")
                return
            if audio.size(-1) > self.max_duration * self.sample_rate:
                self._update_stats(skipped=True, message="Audio too long")
                return
            if self.tokenizer is not None and callable(self.tokenizer):
                if (
                    token.size(-1)
                    > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
            output_dict.update({"audio": audio, "text": text, "tag": "vocal"})
            yield output_dict
            self._update_stats(skipped=False)
        except Exception as exn:
            item_info = (
                f"{item.keys()=} "
                f"{item.get('__dataset_name__', 'none')=} "
                f"{item.get('__index_url__', 'none')=} "
                f"{item.get('__data_url__', 'none')=} "
                f"{item.get('uttid', 'none')=} "
                f"{item.get('src_sample_rate', 'none')=}"
            )
            logger.warning(item_info, exc_info=exn)


class BigTTSDataset(WebPipeline):
    name = "BigTTS"

    def __init__(
        self,
        data_id: int = 193,  # 181: en_4.2wh, 182: en_4.2wh_cn_3.2wh
        # url_pattern: str = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/tts_Len_S11labs-rp2900_P1/package/wav_1.0_web_dataset_2/data/part=00003/shard-00010.tar', # for debug, datasets 181/182 are too large.
        url_pattern: str = None,
        sample_rate: int = 24000,
        audio_key: str = "wav",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(
            data_id=data_id, data_urls=url_pattern, sample_limit_per_file=0.5, **kwargs
        )
        transforms = BigTTSTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class LibriTTSDataset(WebPipeline):
    name = "LibriTTS"
    data_sample_rate = 24000

    def __init__(
        self,
        urls: str = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriTTSTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MCCBaseDataset(WebPipeline):
    name = "MCCBaseDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        if self.name.startswith("MCCVocal"):
            transforms = MCCVocalTransforms(
                sample_rate=sample_rate,
                audio_key=audio_key,
                min_duration=min_duration,
                max_duration=max_duration,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                lyrics_confidence=lyrics_confidence,
                normalize_audio=normalize_audio,
                aed_filtered=aed_filtered,
                audio_metrics_filtered=audio_metrics_filtered,
                avoid_sound_effect=avoid_sound_effect,
                exclude_licenses=exclude_licenses,
                max_num_crops=max_num_crops,
                tokenizer=tokenizer,
                frame_rate=frame_rate,
            )
        elif self.name.startswith("MCCInstrumental"):
            transforms = MCCInstrumentalTransforms(
                sample_rate=sample_rate,
                audio_key=audio_key,
                min_duration=min_duration,
                max_duration=max_duration,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                lyrics_confidence=lyrics_confidence,
                normalize_audio=normalize_audio,
                aed_filtered=aed_filtered,
                audio_metrics_filtered=audio_metrics_filtered,
                avoid_sound_effect=avoid_sound_effect,
                exclude_licenses=exclude_licenses,
                max_num_crops=max_num_crops,
                tokenizer=tokenizer,
                frame_rate=frame_rate,
            )
        else:
            raise KeyError("Invalid MCC dataset name")

        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MCCVocalDataset(MCCBaseDataset):
    name = "MCCVocal"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str],
        weights: List[int] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            **kwargs,
        )


class MCCVocalDatasetGroupAGenreBalanced(MCCBaseDataset):
    name = "MCCVocalGroupAGenreBalanced"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = [
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-blues.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-childhood.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-classical.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-country.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-devotional.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-easy-listening.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-electronic.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-folk.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-hip-hop-rap.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-jazz.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-metal.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-pop.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-r-b-soul.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-reggae.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-rock.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-soundtrack.txt",
        ],
        weights: List[int] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            **kwargs,
        )


class MCCInstrumentalDataset(MCCBaseDataset):
    name = "MCCInstrumental"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str],
        weights: List[int] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            **kwargs,
        )


class KaraokeDataset(WebPipeline):
    name = "Karaoke"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/karaoke_for_singsong_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = KaraokeTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MSSDataset(WebPipeline):
    """Dataset object wrapping the MSSTransforms."""

    name = "MSSDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = [
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-blues_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-childhood_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-classical_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-country_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-devotional_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-easy-listening_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-electronic_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-folk_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-hip-hop-rap_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-jazz_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-metal_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-pop_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-pop_url2index_l1out.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-r-b-soul_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-reggae_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-rock_url2index.txt",
            "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/mcc_vocal_A_1m_mss/url2index/vocal-A-soundtrack_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-blues_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-childhood_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-classical_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-country_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-devotional_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-easy-listening_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-electronic_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-folk_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-hip-hop-rap_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-jazz_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-metal_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-pop_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-pop_url2index_l1out.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-r-b-soul_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-reggae_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-rock_url2index.txt",
            # "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc_vocal_A_1m_mss/url2idx_v1/url2index/vocal-A-soundtrack_url2index.txt",
        ],  # Prepared by Weituo Hao using MSS system developed by Wei-Tsung Lu
        sample_rate: int = 24000,
        min_duration: float = 5,
        max_duration: float = 30,
        lyrics_confidence: float = 0.8,
        max_num_crops: int = None,
        frame_rate: int = 25,
        tokenizer=None,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MSSTransformsV2(
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            lyrics_confidence=lyrics_confidence,
            max_num_crops=max_num_crops,
            frame_rate=frame_rate,
            tokenizer=tokenizer,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        # Need to parse multiple url2index files using MultiIterableDataset()
        dataset = MultiIterableDataset(
            datasets=[IndexedWebDataset(url2index=url, **kwargs) for url in url2index]
        )
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DouyinMusicDataset(WebPipeline):
    name = "DouyinMusic"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):

        print(f"[{self.name}] initializing...")
        transforms = DouyinMusicTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalZhDataset(WebPipeline):
    name = "VocalZh"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = VocalZhTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        if isinstance(url2index, list):
            print(f"[{self.name}] MultiIterableDataset constructing...")
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
            print(f"[{self.name}] MultiIterableDataset constructed.")
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SpeechZhDataset(WebPipeline):
    name = "SpeechZhDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 2,
        max_duration: float = 30,
        normalize_audio: bool = True,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = SpeechZhTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer is not None:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            vocal_zh = VocalZhDataset(
                url2index=[
                    INDEX[region]["MCCVocal-Zh-A"],
                    INDEX[region]["MCCVocal-Zh-B"],
                    INDEX[region]["MCCVocal-Zh-C"],
                    INDEX[region]["HotGalaxy"],
                    INDEX[region]["Soda"],
                ],
                weights=[1, 4, 80, 50, 50],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            vocal_en = MCCVocalDataset(
                url2index=INDEX[region]["MCCVocal"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(
                    MultiIterableDataset(datasets=[vocal_zh, vocal_en], weights=[1, 1]),
                    wds.shuffle(shuffle_buffer_size),
                )
            )
        if weights[1] > 0:
            speech_zh = SpeechZhDataset(
                url2index=[
                    INDEX[region]["FanqieShort"],
                    INDEX[region]["FanqieLong"],
                    INDEX[region]["XimalayaShort"],
                    INDEX[region]["XimalayaLong"],
                    INDEX[region]["XiaoyuzhouShort"],
                    INDEX[region]["XiaoyuzhouLong"],
                ],
                weights=[20, 20, 6, 3, 6, 3],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            speech_en = LibriLightASRDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(
                    MultiIterableDataset(
                        datasets=[speech_zh, speech_en], weights=[1, 1]
                    ),
                    wds.shuffle(shuffle_buffer_size),
                )
            )
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
            )
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        fast_dev: bool = False,
        small: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "wordpiece_multilingual":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-uncased"
            )
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if fast_dev:
            self.train_dataset = DataPipeline(
                LibriTTSDataset(
                    urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    handler=wds.warn_and_continue,
                ),
                self.bucketize,
            )
        else:
            if weights[0] > 0:
                if small:
                    mcc_vocal = MCCVocalDatasetGroupAGenreBalanced(
                        sample_rate=sample_rate,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_num_crops=max_num_crops,
                        normalize_audio=normalize_audio,
                        tokenizer=self.tokenizer,
                        frame_rate=self.frame_rate,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    )
                else:
                    mcc_vocal = MCCVocalDataset(
                        url2index=INDEX[region]["MCCVocal"],
                        sample_rate=sample_rate,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_num_crops=max_num_crops,
                        normalize_audio=normalize_audio,
                        tokenizer=self.tokenizer,
                        frame_rate=self.frame_rate,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    )
                datasets.append(
                    DataPipeline(mcc_vocal, wds.shuffle(shuffle_buffer_size))
                )
            if weights[1] > 0:
                librilight = LibriLightASRDataset(
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_num_crops=max_num_crops,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                )
                datasets.append(
                    DataPipeline(librilight, wds.shuffle(shuffle_buffer_size))
                )
            if weights[2] > 0:
                mcc_instrumental = MCCInstrumentalDataset(
                    url2index=INDEX[region]["MCCInstrumental"],
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_num_crops=max_num_crops,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                )
                datasets.append(
                    DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
                )
            weights = [i for i in weights if i != 0]
            self.train_dataset = DataPipeline(
                MultiIterableDataset(
                    datasets=datasets, weights=[i for i in weights if i != 0]
                ),
                self.bucketize,
            )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_mss_audio_and_token_fn,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        if tokenizer is not None:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None
            collate_fn = collate_audio
        self.frame_rate = frame_rate
        self.batcher = self._prepare_bucket_batcher(
            sample_rate, batch_size, min_duration, max_duration
        )

        # Define the Train Dataset
        mss_dataset = MSSDataset(
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            max_num_crops=max_num_crops,
            frame_rate=frame_rate,
            tokenizer=self.tokenizer,
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
            handler=wds.warn_and_continue,
        )
        self.train_dataset = DataPipeline(mss_dataset, self.bucketize)

    def _prepare_bucket_batcher(
        self, sample_rate, batch_size, min_duration, max_duration
    ):
        """
        The BucketBatcher() object tries to make sure all samples in a batch are of
        roughly equal length to minimize the need to zeropad shorter sequences and waste
        compute resources.

        This logic was copied from MixWebDataModule().
        """
        # `batch_size` here is not well named. It should really be called `batch_max_length`
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        bucket_batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        return bucket_batcher

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class ParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_id: int = None,
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        fast_dev: bool = False,
        small: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
        bsz_evaluator: Optional[str] = None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "wordpiece_multilingual":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-uncased"
            )
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "seed":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "apps/bigmusic/umm/umm/data/tokenizer_bbpe64k-0303"
            )
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        if bsz_evaluator:
            bsz_evaluator = eval(bsz_evaluator)
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
            bsz_evaluator=bsz_evaluator,
        )
        datasets = []
        bigtts = BigTTSDataset(
            data_id=data_id,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            max_num_crops=max_num_crops,
            normalize_audio=normalize_audio,
            tokenizer=self.tokenizer,
            frame_rate=self.frame_rate,
            resampled=True,
            shardshuffle=True,
            # use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )
        datasets.append(DataPipeline(bigtts, wds.shuffle(shuffle_buffer_size)))
        self.train_dataset = DataPipeline(
            MultiIterableDataset(datasets=datasets, weights=[1]), self.bucketize
        )

        self.validation_dataset = []

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=16,
            pin_memory=True,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class DouyinMusicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        languages: List[str] = ["en"],
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 50,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_audio_text,
        weights: List[int] = None,
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )

        datasets = [
            DataPipeline(
                DouyinMusicDataset(
                    url2index=INDEX[region]["DouyinMusic"][language],
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                ),
                wds.shuffle(shuffle_buffer_size),
            )
            for language in languages
        ]
        if weights is None:
            weights = [1 for _ in range(len(datasets))]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(datasets=datasets, weights=weights), self.bucketize
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixZhWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            vocal = VocalZhDataset(
                url2index=[
                    INDEX[region]["MCCVocal-Zh-A"],
                    INDEX[region]["MCCVocal-Zh-B"],
                    INDEX[region]["MCCVocal-Zh-C"],
                    INDEX[region]["HotGalaxy"],
                    INDEX[region]["Soda"],
                ],
                weights=[1, 4, 80, 50, 50],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(vocal, wds.shuffle(shuffle_buffer_size)))
        if weights[1] > 0:
            speech = SpeechZhDataset(
                url2index=[
                    INDEX[region]["FanqieShort"],
                    INDEX[region]["FanqieLong"],
                    INDEX[region]["XimalayaShort"],
                    INDEX[region]["XimalayaLong"],
                    INDEX[region]["XiaoyuzhouShort"],
                    INDEX[region]["XiaoyuzhouLong"],
                ],
                weights=[20, 20, 6, 3, 6, 3],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(speech, wds.shuffle(shuffle_buffer_size)))
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
            )
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class SodaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            vocal = VocalZhDataset(
                url2index=INDEX[region]["Soda"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(vocal, wds.shuffle(shuffle_buffer_size)))
        if weights[1] > 0:
            speech = SpeechZhDataset(
                url2index=[
                    INDEX[region]["FanqieShort"],
                    INDEX[region]["FanqieLong"],
                    INDEX[region]["XimalayaShort"],
                    INDEX[region]["XimalayaLong"],
                    INDEX[region]["XiaoyuzhouShort"],
                    INDEX[region]["XiaoyuzhouLong"],
                ],
                weights=[20, 20, 6, 3, 6, 3],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(speech, wds.shuffle(shuffle_buffer_size)))
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
            )
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixLangDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            vocal = VocalZhDataset(
                url2index=[INDEX[region]["Soda"], INDEX[region]["MCCVocal-En-500k"]],
                weights=[1, 1],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(vocal, wds.shuffle(shuffle_buffer_size)))
        if weights[1] > 0:
            speech_zh = SpeechZhDataset(
                url2index=[
                    INDEX[region]["FanqieShort"],
                    INDEX[region]["FanqieLong"],
                    INDEX[region]["XimalayaShort"],
                    INDEX[region]["XimalayaLong"],
                    INDEX[region]["XiaoyuzhouShort"],
                    INDEX[region]["XiaoyuzhouLong"],
                ],
                weights=[20, 20, 6, 3, 6, 3],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            speech_en = LibriLightASRDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(
                    MultiIterableDataset(
                        datasets=[speech_zh, speech_en], weights=[1, 1]
                    ),
                    wds.shuffle(shuffle_buffer_size),
                )
            )
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
            )
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MusicCollectorDataset(WebPipeline):
    name = "MusicCollector"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = VocalZhTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MusicCollectorWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1],
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []

        url2index = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/youtube_sft/playlist/1CIaSnLlv5FgE0V1ZFkWAU/url2index.txt"
        sample_rate = 24000
        mc_vocal = MusicCollectorWebDataModule(
            url2index=url2index,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            max_num_crops=max_num_crops,
            normalize_audio=normalize_audio,
            tokenizer=self.tokenizer,
            frame_rate=self.frame_rate,
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )

        datasets.append(DataPipeline(mc_vocal, wds.shuffle(shuffle_buffer_size)))
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        valid_url2index = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/youtube_sft/playlist/1CIaSnLlv5FgE0V1ZFkWAU/url2index.txt"
        valid_pipeline = WebPipeline(
            MusicCollectorWebDataModule(
                url2index=valid_url2index,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [valid_pipeline]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

import torch
import random
import webdataset as wds
from functools import partial

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Generator
from torchaudio_augmentations import Compose
from recipes.sacodec.datasets.datamodule import DataModule, SemanticTokenLengthTransform, default_bucket_batcher_fn
from recipes.musiclm.datamodules.webdataset import wds_to_dict
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.utils.webdataset import return_self
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.dataset import MultiIterableDataset
import librosa

import io
import random
import json
from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    Pad,
    LoudnessCheck,
    ReadMP3,
)
from recipes.musiclm.transforms.base import TransformBase
import torchaudio


def semantic_collate_fn(batch, mixed_ratio=0.0):
    batch["target_audio"] = batch["audio"]
    del batch["audio"]
    batch["style_audio"] = batch["target_audio"]
    batch["style_text"] = batch["text"]
    del batch["text"]
    if random.random() < mixed_ratio:
        batch["conditions"] = "style_audio,duration"
    else:
        batch["conditions"] = "style_text,duration"
    return batch

def collate_fn(batch, mixed_ratio=0.0):
    if random.random() < mixed_ratio:
        batch["conditions"] = "style_audio,duration"
    else:
        batch["conditions"] = "style_text,duration"
    return batch


class MCC40MDataset(WebPipeline):

    def __init__(
        self,
        url2index: Union[str, int],
        sample_rate: int,
        duration: Union[float, List[float]],
        audio_key: str = "mp3",
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        avoid_vocal: bool = False,
        max_vocal_threshold: float = 0.5,
        text_type: Optional[str] = None,
        crop_type: str = 'section',
        max_num_crops: Optional[Union[int, List[int]]] = None,
        additional_transforms: Optional[List] = None,
        handler: Callable = wds.warn_and_continue,
        mono: bool = None,
        data_sample_rate: int = None,
        **kwargs,
    ):
        is_parquet = isinstance(url2index, int)
        if is_parquet:
            dataset = ParquetDataset(data_id=url2index, **kwargs)
        else:
            dataset = IndexedWebDataset(url2index=url2index, handler=handler, **kwargs)

        audio_transforms = MCCTransforms(
            duration=duration,
            sample_rate=sample_rate,
            audio_key=audio_key,
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            text_type=text_type,
            crop_type=crop_type,
            max_num_crops=max_num_crops,
            mono=mono,
            data_sample_rate=data_sample_rate
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        additional_transforms = [wds.map(t) for t in additional_transforms] if additional_transforms else []
        pipeline = []
        if not is_parquet:
            pipeline.append("decode")
        pipeline.append(
            {"compose": [preprocessor.train_buffer_preprocessor, *additional_transforms]}
        )
        super().__init__(dataset, pipeline)


class WrappedMCC40MDataset(MultiIterableDataset):

    def __init__(
        self,
        url2index_list: list,
        num_samples: int = -1,
        seed: int = 2023,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        datasets = [
            MCC40MDataset(
                url2index=url2index,
                **kwargs,
            ) for url2index in url2index_list
        ]

        if weights is None:
            weights=[1.0 for _ in range(len(datasets))]
        assert len(weights) == len(datasets)
        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=weights,
            seed=seed,
        )


class InstrumentalWebDataModule(DataModule):
    def __init__(
        self,
        dataset_name: str = "SSTK_US_1M",
        val_split: str = "SSTK_EVAL_US",
        sample_rate: int = 24000,
        duration: Union[float, List[float]] = 30.0,
        batch_size: int = 16,
        val_batch_size: int = 16,
        shuffle_buffer_size: int = 0,
        num_workers: int = 6,
        pin_memory: bool = True,
        collate_fn: Callable = collate_fn,
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        avoid_vocal: bool = False,
        max_vocal_threshold: float = 0.5,
        text_type: Optional[str] = None,
        crop_type: str = 'section',
        validation_crop_type: str = 'single_crop',
        max_num_crops: Optional[Union[int, List[int]]] = 3,
        additional_transforms: Optional[List] = None,
        keys=["audio", "text", "structure", "intensity", "start_time", "song_duration", "duration", "sections", "meta_song_id"],
        mixed_ratio: float = 0.0,
        use_pipe: bool = False,
        seed: int = 555,
        mono: bool = None, # None - inferred from sample rate
        data_sample_rate: int = None,
    ):
        if dataset_name == "SSTK_US":
            train_urls_and_weights = [(106, 1.0)]
        elif dataset_name == "SSTK_US_GENRE_BALANCED":
            train_urls_and_weights = [(146, 1.0)]
        elif dataset_name == "SSTK_US_SFT":
            train_urls_and_weights = [(373, 1.0)]
        elif dataset_name == "SSTK_US_SFT_44k":
            train_urls_and_weights = [(443, 1.0)]
        elif dataset_name == "SSTK_US_1M":
            train_urls_and_weights = [(341, 1.0)]
        elif dataset_name == "UMM_US":
            train_urls_and_weights = [(551, 1.0)]
        elif dataset_name == "SSTK_US_1M_44k":
            train_urls_and_weights = [(442, 1.0)]
        elif dataset_name == "SSTK_US_44k":
            train_urls_and_weights = [(710, 1.0)]
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_name}")
        if not isinstance(duration, (list, tuple)):
            duration = [duration]
        train_dataset = WrappedMCC40MDataset(
            url2index_list=[x[0] for x in train_urls_and_weights],
            weights=[x[1] for x in train_urls_and_weights],
            sample_rate=sample_rate,
            duration=duration,
            audio_key="audio.npy",
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            text_type=text_type,
            crop_type=crop_type,
            max_num_crops=max_num_crops,
            additional_transforms=additional_transforms,
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            seed=seed,
            mono=mono,
            data_sample_rate=data_sample_rate
        )
        train_dataset = WebPipeline(
            train_dataset,
            pipeline=[{"compose": [
                wds_to_dict(*keys),
                wds.map(SemanticTokenLengthTransform(sample_rate=sample_rate, audio_key="audio")),
                wds.shuffle(shuffle_buffer_size),
                default_bucket_batcher_fn(
                    sample_rate, duration, batch_size, lyrics_frame_rate=0, max_duration=max(duration)
                ),
            ]}],
        )

        if val_split == "SSTK_US":
            val_urls_and_weights = [(346, 1.0)]
        elif val_split == "SSTK_US_44k":
            val_urls_and_weights = [(444, 1.0)]
        else:
            raise NotImplementedError(f"Unknown val split: {val_split}")

        validation_dataset = WrappedMCC40MDataset(
            url2index_list=[x[0] for x in val_urls_and_weights],
            weights=[x[1] for x in val_urls_and_weights],
            sample_rate=sample_rate,
            duration=duration,
            audio_key="audio.npy",
            normalize_audio=normalize_audio,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            avoid_vocal=avoid_vocal,
            max_vocal_threshold=max_vocal_threshold,
            text_type=text_type,
            crop_type=validation_crop_type,
            max_num_crops=max_num_crops,
            additional_transforms=additional_transforms,
            resampled=False,
            shardshuffle=False,
            use_pipe=use_pipe,
            seed=seed,
            nodesplitter=return_self,
            mono=mono
        )
        validation_dataset = WebPipeline(
            validation_dataset,
            pipeline=[{"compose": [
                wds_to_dict(*keys),
                wds.map(SemanticTokenLengthTransform(sample_rate=sample_rate, audio_key="audio")),
                default_bucket_batcher_fn(
                    sample_rate, duration, val_batch_size, lyrics_frame_rate=0, max_duration=None
                ),
            ]}],
        )

        collate_fn = partial(collate_fn, mixed_ratio=mixed_ratio)

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,
            collate_fn=collate_fn,
            do_shuffle=False,
        )
        

class MCCTransforms(TransformBase):
    def __init__(
        self,
        duration: Union[int, List[int]],
        sample_rate: int,
        audio_key: str = "mp3",
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        avoid_vocal: bool = False,
        max_vocal_threshold: float = 0.5,
        text_type: Optional[str] = None,
        crop_type: str = 'section',
        max_num_crops: Optional[Union[int, List[int]]] = None,    # if None, auto set based on audio length
        mono: bool = None,
        data_sample_rate: int = None,
    ) -> None:
        super().__init__()

        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.avoid_vocal = avoid_vocal
        self.max_vocal_threshold = max_vocal_threshold
        self.text_type = text_type
        self.crop_type = crop_type
        self.n_samples = [int(d * sample_rate) for d in duration]
        self.crop_step_size = [(10 * sample_rate) for _ in range(len(self.n_samples))] # TODO: remove this

        if max_num_crops is None:
            max_num_crops = [None] * len(self.n_samples)
        elif not isinstance(max_num_crops, (list, tuple)):
            max_num_crops = [max_num_crops] * len(self.n_samples)
        assert len(max_num_crops) == len(self.n_samples)
        self.max_num_crops = max_num_crops

        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold,
        )
        self.read_mp3 = ReadMP3(self.sample_rate)

        if mono is None:
            mono = (self.sample_rate == 24000)

        base_transforms = []
        if audio_key == "mp3":
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms.append(ToTensor(mono=mono))
        if data_sample_rate is not None and data_sample_rate != sample_rate:
            base_transforms.append(torchaudio.transforms.Resample(data_sample_rate, sample_rate))

        base_transforms += [
            SetAudioDimensions(),
            NormalizeAudioToFloat32(),
        ]
        if normalize_audio:
            base_transforms.append(NormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        # Apply SFT filtering if the data is there
        if "filter_label" in metadata:
            if metadata["filter_label"]["high_quality"] != "yes":
                return False, "Not High Quality (SFT)"
        return True, None

    def get_vocal_data(self, metadata: Dict[str, Any]):
        vad = metadata.get("vad", {})
        voice_proportion = vad.get("voice_proportion", 0)
        text = ""
        for key in ["description", "title", "keywords", "genres", "instruments"]:
            if key in metadata: text += str(metadata[key])
        has_vocal_metadata = 'vocal' in text.lower()
        return voice_proportion, has_vocal_metadata

    def get_text(self, metadata, text_type):
        if "human_label" in metadata:
            hvals = [l for l in metadata["human_label"].values() if isinstance(l, str) and len(l)]
            human_labels = ", ".join(hvals)
            return human_labels
        if text_type == "sstk_dropout":
            return format_text_sstk_dropout(metadata)
        else:
            raise ValueError(f"Unknown text type: {text_type}")

    def maybe_convert_parquet(self, x: Dict[str, Any]):
        if "__index_data__" in x:   # WebDataset, nothing to do
            return x
        x[self.audio_key] = x["wav"]
        del x["wav"]
        x["__index_data__"] = json.loads(x["meta"])
        del x["meta"]
        x["__url__"] = x["__data_url__"]
        del x["__data_url__"]
        return x
    
    # def get_duration(self, metadata, audio):
    #     if 'duration' in metadata:
    #         return float(metadata['duration'])
    #     elif 'raw' in metadata and 'duration' in metadata['raw']:
    #         return float(metadata['raw']['duration'])
    #     else:
    #         return audio.shape[-1] / self.sample_rate

    def yield_single_cropped(self, audio, metadata, text):
        duration = audio.shape[-1] // self.sample_rate

        n_samples = self.n_samples[-1]
        if audio.size(1) < n_samples: 
            self._update_stats(skipped=True, message="No Valid Crop")
            return

        st_sample = 0
        en_sample = st_sample + n_samples
        cropped_audio = audio[:, st_sample : en_sample]
        if not self.is_loud(cropped_audio):
            return
        output = {
            "audio": cropped_audio,
            "metadata": metadata,
            "start_time": st_sample / self.sample_rate,
            "sample_rate": self.sample_rate,
            "song_duration": duration,
            "duration": n_samples // self.sample_rate,
            "text": text,
            "meta_song_id": metadata.get("asset_id", 0),
            "sections": ['none', st_sample / self.sample_rate]
        }
        yield output
        self._update_stats(skipped=False)


    def yield_random_cropped(self, audio, metadata, text):
        def get_window_ids(audio, n_samples, crop_step_size):
            num_windows = 1 + (audio.size(1) - n_samples) // crop_step_size
            window_ids = list(range(num_windows))
            return audio, window_ids, num_windows
        duration = audio.shape[-1] // self.sample_rate

        # Choose n_samples at random
        candidates = [
            (n, m, c) for n, m, c in zip(
                self.n_samples,
                self.max_num_crops,
                self.crop_step_size,
            ) if audio.size(1) >= n
        ]
        if len(candidates) == 0:
            self._update_stats(skipped=True, message="Audio Length Not Suitable")
            return
        random.shuffle(candidates)
        n_samples, max_num_crops, crop_step_size = candidates[0]
        crop_wing_span = n_samples // crop_step_size
        audio = Pad(n_samples=n_samples)(audio)
        
        # Determine possible crop starting points
        audio, window_ids, num_windows = get_window_ids(
            audio, n_samples=n_samples, crop_step_size=crop_step_size
        )
        random.shuffle(window_ids)
        # Return up to max_num_crops
        if max_num_crops is None:
            max_num_crops = max(1, audio.size(1) // n_samples)
        num_crops = 0
        taboo = set()

        for wid in window_ids:
            if wid in taboo:
                continue
            st_sample = wid * crop_step_size
            en_sample = st_sample + n_samples
            cropped_audio = audio[:, st_sample : en_sample]
            if not self.is_loud(cropped_audio):
                continue
            output = {
                "audio": cropped_audio,
                "metadata": metadata,
                "start_time": st_sample / self.sample_rate,
                "sample_rate": self.sample_rate,
                "song_duration": duration,
                "duration": n_samples // self.sample_rate,
                "text": text,
                "meta_song_id": metadata.get("asset_id", 0),
                "sections": ['none', st_sample / self.sample_rate]
            }
            yield output
            num_crops += 1
            if num_crops >= max_num_crops:
                break
            # Avoid overlapping segments
            for overlapped_wid in range(
                max(0, wid - crop_wing_span + 1),
                min(num_windows, wid + crop_wing_span),
            ):
                taboo.add(overlapped_wid)
        if num_crops == 0:
            self._update_stats(skipped=True, message="No Valid Crop")
        else:
            self._update_stats(skipped=False)

    def yield_section_cropped(self, audio, metadata, text):
        def merge_raw_segments(raw_segments, start_thresh=0.1):
            for idx in range(len(raw_segments)-1):
                seg_cur = raw_segments[idx]
                seg_next = raw_segments[idx+1]
                if seg_cur['interval'][-1] > seg_next['interval'][0]:
                    # print('Fixing raw segments:', seg_cur['interval'], raw_segments)
                    seg_cur['interval'][-1] = seg_next['interval'][0]

            valid_sections = [section for idx, section in enumerate(raw_segments) if idx == 0 or section["start_prob"] > start_thresh]
            return [(s['label'], s['interval'][0], s['interval'][1]) for s in valid_sections]
        
        segments = merge_raw_segments(metadata['deepchorus']['raw_segments']) # better transition probability logic
        # segments = [(s['label'], s['interval'][0], s['interval'][1]) for s in metadata['deepchorus']['segments']] # (label, start, end)

        song_duration = audio.shape[-1] / self.sample_rate
        valid_durations = [d for d in self.duration if song_duration + 10 > d ] # 10s leeway
        if song_duration > 30: # remove short audio
            valid_durations = [d for d in valid_durations if d > 30]
        if not valid_durations:
            self._update_stats(skipped=True, message="Audio Length Not Suitable")
            return
        target_duration = random.choice(valid_durations)
        
        def overlap(tuples, search):
            for t in tuples:
                if(t[1]>search[0] and t[0]<search[1]):
                    return True
            return False

        def is_valid_segment(s):
            label, start, end = s
            return start + target_duration <= song_duration + 10

        def format_segment(start_time, segments, target_duration):
            segment_list = []
            segment_start = None
            for s in segments:
                label, start, end = s
                if start >= start_time and start < start_time + target_duration:
                    if segment_start is None:
                        segment_start = start
                    segment_list.extend([label, start - segment_start]) # normalize by start time
            return segment_list
    
        
        valid_segments = [s for s in segments if is_valid_segment(s)]
        random.shuffle(valid_segments)

        ranges = []
        filtered_segments = []
        for s in valid_segments:
            label, start, end = s
            clip_range = (start, start+target_duration)
            if overlap(ranges, clip_range):
                continue
            else:
                filtered_segments.append(s)
                ranges.append(clip_range)

        if not filtered_segments:
            return self.yield_random_cropped(audio, metadata, text)

        max_num_crops = max(self.max_num_crops)
        num_crops = 0
        for (label, start, end) in filtered_segments:
            st_sample = round(start * self.sample_rate)
            en_sample = round((start + target_duration) * self.sample_rate)
            cropped_audio = audio[:, st_sample:en_sample] # no need to pad audio, collate_fn + SemanticTokenLengthTransform handles padding
            if not self.is_loud(cropped_audio):
                continue
            output = {
                "audio": cropped_audio,
                "metadata": metadata,
                "start_time": st_sample / self.sample_rate,
                "sample_rate": self.sample_rate,
                "song_duration": song_duration,
                "duration": target_duration,
                "text": text,
                "sections": format_segment(start, segments, target_duration)
            }
            yield output
            num_crops += 1
            if num_crops >= max_num_crops:
                break
        if num_crops == 0:
            self._update_stats(skipped=True, message="No Valid Crop")
        else:
            self._update_stats(skipped=False)

    def __call__(self, x: Dict[str, Any]) -> Generator:
        x = self.maybe_convert_parquet(x)
        is_good, message = self.is_metadata_good(x["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return
        vocal_ratio, has_vocal_metadata = self.get_vocal_data(x["__index_data__"])
        if self.avoid_vocal and (vocal_ratio > self.max_vocal_threshold or has_vocal_metadata):
            self._update_stats(skipped=True, message="Has Vocal")
            return

        try:
            audio = self.base_transform(x[self.audio_key])
        except Exception as e:
            print(f"[MP3 decoding error] {e}")
            self._update_stats(skipped=True, message="MP3 Decoding Error")
            return
        
        metadata = x["__index_data__"]
        text = self.get_text(metadata, self.text_type)

        def trim_silence(audio, metadata, min_sec=0.5, top_db=70):
            sample_rate = self.sample_rate
            window_length = int(sample_rate * min_sec)
            hop_length = int(window_length / 4)
            new_wav, index = librosa.effects.trim(audio, top_db=top_db, frame_length=window_length, hop_length=hop_length)
            offset = index[0] / sample_rate
            if 'deepchorus' in metadata:
                raw_segments = metadata['deepchorus']['raw_segments']
                for segment in raw_segments:
                    segment['interval'] = [max(x-offset, 0) for x in segment['interval']]
            return new_wav, metadata
        
        audio, metadata = trim_silence(audio, metadata)

        if 'deepchorus' in metadata and self.crop_type == 'section':
            yield from self.yield_section_cropped(audio, metadata, text)
        elif self.crop_type == "single_crop":
            yield from self.yield_single_cropped(audio, metadata, text)
        else:
            yield from self.yield_random_cropped(audio, metadata, text)

def sample_pct(arr, dropout=0.5, min_examples=1):
    random.shuffle(arr)
    if len(arr) * (1 - dropout) <= min_examples:
        return arr
    return [a for idx, a in enumerate(arr) if random.random() >= dropout]


def format_text_sstk_dropout(metadata):
    text_fields = {}
    for key in ["description", "title", "keywords", "genres", "instruments"]:
        v = metadata.get(key)
        if v is None or v == "\\N" or len(v.strip()) == 0:
            continue
        text_fields[key] = v.strip()

    if "bpm" in metadata:
        try:
            bpm = int(metadata['bpm'])
            text_fields["bpm"] = f'{bpm} BPM'
        except Exception as e: pass

    if len(text_fields) == 0:
        return ""
    
    
    if random.random() <= 0.3:
        if random.random() < 0.8 and "description" in text_fields:
            return text_fields["description"]
        elif "title" in text_fields:
            return text_fields["title"]
        
    keywords = []
    if "keywords" in text_fields:
        kw = [t.strip() for t in text_fields["keywords"].split(",")]
        kw = sample_pct(kw, 0.75, min_examples=6)
        keywords.extend(kw)
    if "genres" in text_fields:
        g = [t.strip() for t in text_fields["genres"].split(",")]
        g = sample_pct(g, 0.3, min_examples=1)
        keywords.extend(g)
    if "instruments" in text_fields:
        i = [t.strip() for t in text_fields["instruments"].split(",")]
        i = sample_pct(i, 0.3, min_examples=1)
        keywords.extend(i)
    if "bpm" in text_fields and random.random() < 0.1:
        keywords.append(text_fields["bpm"])

    keywords = list(set(keywords))
    random.shuffle(keywords)
    if random.random() < 0.5:
        keywords = [k.lower() for k in keywords]
    else:
        keywords = [k.capitalize() for k in keywords]
    if random.random() < 0.5:
        return ", ".join(keywords)
    else:
        return " ".join(keywords)

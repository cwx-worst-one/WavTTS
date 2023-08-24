import io
import random
from string import punctuation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

import samantha.utils.hdfs_helper as hh
from recipes.musiclm.transforms.audio import (
    FastNormalizeAudio,
    LoudnessCheck,
    NormalizeAudio,
    ReadMP3,
)
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self


def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    return text.translate(str.maketrans("", "", nlp_punctuation))


def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    text = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        text.append(batch[idx].get("text", None))
    return {"audio": torch.stack(audio, dim=0), "text": text}


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


class LibriTTSDataset(WebPipeline):
    name = "LibriTTS"
    data_sample_rate = 24000

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
        text = normalize_text(item["normalized_text.txt"])
        # TODO: assumed 25Hz
        if len(text) <= 0 or len(text) > audio.size(-1) / self.sample_rate * 25:
            return
        yield {"audio": audio, "text": text}


class MCCInstrumentalDataset(WebPipeline):
    name = "MCCInstrumental"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores/npy_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        normalize_audio: bool = True,
        # filtering
        aed_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
            # base_transforms.append(NormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"map": [self.transform]}]
        super().__init__(dataset, pipeline)

    def is_audio_metrics_good(self, audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        # Clipping
        clip = audio_metrics.get("clipping", {})
        if clip.get("rate", 0) >= 5e-5:
            return False
        for ch in ["left", "right"]:
            if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
                return False
        # Loudness
        loudness = audio_metrics.get("loudness", {})
        if (
            loudness.get("integrated_loudness", -7) > -5
            or loudness.get("max_mom_loud", -7) >= 0
            or loudness.get("max_short_term_loud", -7) >= 0
        ):
            return False
        # RMS stats
        rms_stats = audio_metrics.get("rms_stats", {})
        if rms_stats.get("peak", 0) > 3:
            return False
        for ch in ["left", "right"]:
            if (
                rms_stats.get(f"{ch}_total", -10) > -5
                or rms_stats.get(f"{ch}_total", -10) < -40
                or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
            ):
                return False
        # Cutoff frequency
        cutoff_freq = audio_metrics.get("cutoff_frequency", {})
        for ch in ["left", "right"]:
            if (
                cutoff_freq.get(f"rel_{ch}", 48000) < 15000
                and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
                and cutoff_freq.get(f"band_std_{ch}", 10) < 5
            ):
                return False
        # Phase
        phase = audio_metrics.get("phase_check", {})
        if (
            phase.get("has_phase_issue", False)
            or abs(phase.get("rms_downmix_diff", 0.1)) > 3
        ):
            return False
        return True

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False
        audio_metrics_is_good = self.is_audio_metrics_good(
            metadata.get("audio_metrics", {})
        )
        if self.audio_metrics_filtered and not audio_metrics_is_good:
            return False
        return True

    def transform(self, item) -> Dict[str, Any]:
        if not self.is_metadata_good(item["__index_data__"]):
            return
        audio = self.base_transform(item[self.audio_key])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            return
        duration = (
            random.randint(self.min_duration, self.max_duration) * self.sample_rate
        )
        start = random.randint(0, audio.size(-1) - duration)
        audio = audio[:, start : start + duration]
        if not self.is_loud(audio):
            return
        return {"audio": audio}


class MCCVocalDataset(MCCInstrumentalDataset):
    name = "MCCVocal"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt",
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
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            **kwargs,
        )
        self.lyrics_confidence = lyrics_confidence

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["additions"]["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        return True

    def transform(self, item) -> Dict[str, Any]:
        if not self.is_metadata_good(item["__index_data__"]):
            return
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            return
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            print(f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.sample_rate:
            print(f"Too short: {audio.size(-1)}")
            return
        utterances = item["__index_data__"]["lyrics"].get("utterances", None)
        if utterances is None:
            return
        random.shuffle(utterances)
        filtered_utterances = filter(self.filter_lyrics, utterances)
        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            clip = audio[:, start:end]
            if clip.size(-1) < self.sample_rate:
                print(f"Too short: {clip.size(-1)}")
                return
            if not self.is_loud(clip):
                return
            text = normalize_text(selected_utterance["text"])
            # TODO: assumed 25Hz
            if len(text) <= 0 or len(text) > clip.size(-1) / self.sample_rate * 25:
                return
            yield {"audio": clip, "text": text}


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
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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


class MixWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
    ):
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=False,
            batch_size=batch_size,
            length_fn=lambda x: x["audio"].shape[-1],
        )

        librilight = LibriLightDataset(
            urls="pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/librilight/large/{00000..01650}.tar",
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
        )
        mcc_instrumental = MCCInstrumentalDataset(
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=True,
            handler=wds.warn_and_continue,
        )
        mcc_vocal = MCCVocalDataset(
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=True,
            handler=wds.warn_and_continue,
        )

        train_dataset = WebPipeline(
            MultiIterableDataset(
                datasets=[mcc_vocal, mcc_instrumental, librilight], weights=[2, 1, 2]
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = WebPipeline(
            LibriLightDataset(
                urls="pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/speech/librilight/large/01651.tar",
                sample_rate=sample_rate,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                nodesplitter=return_self,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class VocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: Tuple[int, int] = (1, 1),
        region: str = "US",
        use_pipe: bool = True,
        val_resampled: bool = False,
    ):
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=False,
            batch_size=batch_size,
            length_fn=lambda x: x["audio"].shape[-1],
        )
        if region == "US":
            mcc_vocal_index = (
                "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt"
            )
            mcc_vocal_val_index = (
                "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx_val.txt"
            )
        elif region == "CN":
            mcc_vocal_index = "recipes/datasets/mcc/mcc60m_index_train.txt"
            mcc_vocal_val_index = "recipes/datasets/mcc/mcc60m_index_val.txt"
            # hh.get(
            #     "hdfs://haruna/home/byte_speech_sv/zongyu.yin/assets/mcc60m_index_train.txt",
            #     mcc_vocal_index,
            # )
            # hh.get(
            #     "hdfs://haruna/home/byte_speech_sv/zongyu.yin/assets/mcc60m_index_val.txt",
            #     mcc_vocal_val_index,
            # )
        else:
            raise KeyError(f"Wrong region: {region}")
        mcc_vocal = MCCVocalDataset(
            url2index=mcc_vocal_index,
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )
        libritts = LibriTTSDataset(
            urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            handler=wds.warn_and_continue,
        )

        train_dataset = WebPipeline(
            MultiIterableDataset(datasets=[mcc_vocal, libritts], weights=weights),
            pipeline=[{"compose": [self.bucketize]}],
        )
        if val_resampled:
            validation_dataset = [
                WebPipeline(
                    MCCVocalDataset(
                        url2index=mcc_vocal_val_index,
                        sample_rate=sample_rate,
                        resampled=True,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
                WebPipeline(
                    LibriTTSDataset(
                        urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
                        sample_rate=sample_rate,
                        resampled=True,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
            ]
        else:
            validation_dataset = [
                WebPipeline(
                    MCCVocalDataset(
                        url2index=mcc_vocal_val_index,
                        sample_rate=sample_rate,
                        nodesplitter=return_self,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
                WebPipeline(
                    LibriTTSDataset(
                        urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
                        sample_rate=sample_rate,
                        nodesplitter=return_self,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
            ]

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            for i in item:
                batch = self.batcher.collate_batch(i)
                if batch is not None:
                    yield batch

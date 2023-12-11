import os
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import webdataset as wds
from torchaudio_augmentations import Compose
from recipes.datasets.base import (
    BaseAudioTransform,
    BatchedStr,
    DataResult,
    WebDataModuleBase,
)
from samantha.dataio.data_bucket import data_bucket
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.transforms.audio import ToTensor, RandomResizedCrop
from samantha.utils.logger import RankedLogger
from samantha.utils.webdataset import return_self
from recipes.datasets.filters import Filters

from samantha.transforms.tokenizers.phoneme import LyricPhonemeTokenizer

logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class MelDataResult(DataResult):
    audio: Optional[torch.Tensor] = None
    mel: Optional[torch.Tensor] = None
    audio_input_length: Optional[int] = None
    input_length: Optional[int] = None

@dataclass
class M1DataResult(DataResult):
    lyrics: BatchedStr
    lyrics_normalized_text: BatchedStr
    lyrics_tokens: torch.Tensor
    style_text: BatchedStr
    conditions: BatchedStr
    audio: Optional[torch.Tensor] = None
    mel: Optional[torch.Tensor] = None
    audio_input_length: Optional[int] = None
    input_length: Optional[int] = None

class MelDataModule(WebDataModuleBase):

    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        lyrics_tokenizer: LyricPhonemeTokenizer,
        hop_size: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        min_seconds: int,
        max_seconds: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        nodesplitter=wds.shardlists.single_node_only,
        filters: Filters = Filters([]),
    ):
        self._sample_rate = sample_rate
        self._lyrics_tokenizer = lyrics_tokenizer
        self._hop_size = hop_size
        self._min_seconds = min_seconds
        self._frame_rate = self._sample_rate // hop_size

        self._min_audio_frames = int(self._frame_rate * min_seconds)
        self._max_audio_frames = int(self._frame_rate * max_seconds)
        self._max_bucket_size = int(batch_size * self._max_audio_frames)

        self._buckets_frames = list(map(lambda i: i * self._frame_rate, list(range(min_seconds, max_seconds + 1))))

        self._filters = filters

        batcher = BucketBatcher(
            buckets=self._buckets_frames,
            batch_size=batch_size,
            dynamic_batch=False,
            # maximum_bucket_size=self._max_bucket_size,
            length_fn=lambda x: x.mel.shape[0], # [frame, mels]
        )

        self.to_tensor = Compose(
            [
                ToTensor(),
            ]
        )
        self.audio_transform = BaseAudioTransform()

        dataset = (
            IndexedWebDataset(
                url2index=url2index,
                resampled=resampled,
                shardshuffle=shardshuffle,
                use_pipe=False,
                nodesplitter=nodesplitter,
            )
            .decode()
            .compose(self.transform)
        )

        super().__init__(
            train_dataset=dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batcher=batcher,
        )

    @property
    def min_duration(self) -> float:
        return self._min_audio_frames / self._frame_rate

    @property
    def max_duration(self) -> float:
        return self._max_audio_frames / self._frame_rate

    def transform(self, items: Iterable[Dict[str, Any]]) -> Iterable[MelDataResult]:
        for item in items:
            index = item["__index_data__"]

            mel = item["mel.npy"]
            mel = self.to_tensor(mel)

            audio = item.get("audio.npy")
            if audio is not None:
                audio = self.audio_transform(audio)

            if mel.shape[0] < self._min_audio_frames:
                # logger.warning(mel.shape[0], "Audio too short")
                continue

            if mel.shape[0] > self._max_audio_frames:
                # logger.warning(mel.shape[0], "Audio too long")
                continue

            lyrics_tokens = item["lyrics_tokens.npy"]
            lyrics_tokens = self.to_tensor(lyrics_tokens)
            
            input_length = index.get("input_length")

            shard = os.path.basename(item["__url__"])
            key = item["__key__"]

            if self._filters(index):
                continue

            # pad lyrics_tokens, as in the original training:
            n_pad_tokens = self._lyrics_tokenizer.max_phone_len - lyrics_tokens.shape[0]
            lyrics_tokens = torch.nn.functional.pad(lyrics_tokens, (0, n_pad_tokens), value=self._lyrics_tokenizer.pad_token_id)
            yield M1DataResult(
                mel=mel,
                audio=audio,
                audio_input_length=input_length,
                shard=shard,
                key=key,
                lyrics=index["lyrics"],
                lyrics_normalized_text=index["lyrics_normalized_text"],
                style_text=index["style_text"],
                conditions=index["conditions"],
                lyrics_tokens=lyrics_tokens,
            )


class CombinedMelDataModule(WebDataModuleBase):
    _mel_pad_silence_value = -100.0

    def __init__(
        self,
        sample_rate: int,
        train_url2index: str,
        validation_url2index: str,
        lyrics_tokenizer: LyricPhonemeTokenizer,
        hop_size: int,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        min_seconds: int,
        max_seconds: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        filters: Filters = Filters([]),
    ):
        self._sample_rate = sample_rate

        train_url2index = data_bucket(train_url2index)
        validation_url2index = data_bucket(validation_url2index)

        mel_datamodule = MelDataModule(
            url2index=train_url2index,
            sample_rate=sample_rate,
            lyrics_tokenizer=lyrics_tokenizer,
            hop_size=hop_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            resampled=resampled,
            shardshuffle=shardshuffle,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            num_workers=num_workers,
            pin_memory=pin_memory,
            filters=filters,
        )

        self._max_audio_frames = mel_datamodule._max_audio_frames

        train_dataset = mel_datamodule.train_dataset
        
        mel_datamodule = MelDataModule(
            url2index=validation_url2index,
            sample_rate=sample_rate,
            lyrics_tokenizer=lyrics_tokenizer,
            hop_size=hop_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            resampled=False,
            shardshuffle=False,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            num_workers=num_workers,
            pin_memory=pin_memory,
            nodesplitter=return_self,
            filters=filters,
        )
        validation_dataset = mel_datamodule.train_dataset
        
        test_dataset = deepcopy(validation_dataset)
        predict_dataset = deepcopy(validation_dataset)

        batcher = mel_datamodule.batcher
        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batcher=batcher,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:

                ## mel:
                mel_max_length = max(
                    [item["mel"].shape[0] for item in batch]
                )

                ## lyrics tokens
                ## this is already done in the transform, as per the original training recipe
                # lyrics_tokens_max_length = max(
                #     [item["lyrics_tokens"].shape[0] for item in batch]
                # )

                ## audio (optional)
                # audio_max_length = max(
                #     [item["audio"].shape[1] for item in batch]
                # )
                for idx in range(len(batch)):

                    ## mel:
                    mel = batch[idx].get("mel")
                    pad_frames = mel_max_length - mel.shape[0]
                    padded_mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_frames), value=self._mel_pad_silence_value)
                    batch[idx]["mel"] = padded_mel

                    ## lyrics tokens
                    # lyrics_tokens = batch[idx].get("lyrics_tokens")
                    # batch[idx]["lyrics_tokens"] = torch.nn.functional.pad(lyrics_tokens, (0, lyrics_tokens_max_length - lyrics_tokens.shape[0]))

                    ## audio (optional)
                    # audio = batch[idx].get("audio")
                    # batch[idx]["audio"] = torch.nn.functional.pad(audio, (0, audio_max_length - audio.shape[1]))

                yield batch

from typing import List
from recipes.datasets.base import WebDataModuleBase, audio_batcher
from recipes.mir.datasets.mix import DialectDataset, DialectDataResult, ChineseGenreDataset, ChineseGenreDataResult
from webdataset.pipeline import DataPipeline
from transformers import BertTokenizer
from samantha.utils.webdataset import return_self



class DialectDataModule(WebDataModuleBase):
    _sample_rate: int = 24000
    _buckets_sec: List[int] = [10, 20, 30]
    _pin_memory: bool = True
    _epoch_size: int = 1746

    def __init__(self, batch_size: int, shuffle_buffer_size: int, num_workers: int = 4):
        batcher = audio_batcher(self._sample_rate, batch_size, self._buckets_sec, length_fn=lambda x: x.input_audio.shape[-1])

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        train_dataset = DataPipeline(DialectDataset(
            split="train",
            tokenizer=tokenizer,
            sample_rate=self._sample_rate,
            min_duration=10,
            max_duration=30,
            normalize_audio=False,
            resampled=True,
            shardshuffle=True,
        ))

        test_dataset = DataPipeline(DialectDataset(
            split="validation",
            tokenizer=tokenizer,
            sample_rate=self._sample_rate,
            min_duration=10,
            max_duration=30,
            normalize_audio=False,
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self

        ))

        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            predict_dataset=test_dataset,
            batcher=batcher,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
        )


class ChineseGenreDataModule(WebDataModuleBase):
    _sample_rate: int = 24000
    _buckets_sec: List[int] = [10, 20, 30]
    _pin_memory: bool = True
    _epoch_size: int = 6536

    def __init__(self, batch_size: int, shuffle_buffer_size: int, num_workers: int = 4):
        batcher = audio_batcher(self._sample_rate, batch_size, self._buckets_sec, length_fn=lambda x: x.input_audio.shape[-1])

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        train_dataset = DataPipeline(ChineseGenreDataset(
            split="train",
            tokenizer=tokenizer,
            sample_rate=self._sample_rate,
            min_duration=10,
            max_duration=30,
            resampled=True,
            shardshuffle=True,
        ))

        test_dataset = DataPipeline(ChineseGenreDataset(
            split="validation",
            tokenizer=tokenizer,
            sample_rate=self._sample_rate,
            min_duration=10,
            max_duration=30,
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self

        ))

        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            predict_dataset=test_dataset,
            batcher=batcher,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            epoch_size=self._epoch_size
        )

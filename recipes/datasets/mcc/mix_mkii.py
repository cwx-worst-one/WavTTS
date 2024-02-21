import json
import logging
import math
import random
import re
import sys
from string import punctuation

try:
    from zhon.hanzi import punctuation as punctuation_zh
except:
    print("[WARNING] Failed to import zhon.hanzi.punctuation")
import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer
from webdataset.pipeline import DataPipeline

from recipes.musiclm.transforms.audio import FastNormalizeAudio
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
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
    token = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        token.append(batch[idx].get("token", torch.zeros(0).long()))
    return {
        "audio": torch.stack(audio, dim=0),
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
    }


def collate_fn_mss(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    audio = []
    audio_vocal = []
    audio_vocal_perturb = []
    audio_inst = []
    token = []
    for idx in range(len(batch)):
        audio.append(batch[idx]["audio"][0])
        audio_vocal.append(batch[idx]["audio_vocal"][0])
        if "audio_vocal_perturb" in batch[0]:
            audio_vocal_perturb.append(batch[idx]["audio_vocal_perturb"][0])
        audio_inst.append(batch[idx]["audio_inst"][0])
        token.append(batch[idx].get("token", torch.zeros(0).long()))
    ret = {
        "audio": torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)[:, None],
        "audio_vocal": torch.nn.utils.rnn.pad_sequence(
            audio_vocal, batch_first=True, padding_value=0)[:, None],
        "audio_inst": torch.nn.utils.rnn.pad_sequence(
            audio_inst, batch_first=True, padding_value=0)[:, None],
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
    }
    if len(audio_vocal_perturb) > 0:
        ret.update({
            "audio_vocal_perturb": torch.nn.utils.rnn.pad_sequence(
                audio_vocal_perturb, batch_first=True, padding_value=0)[:, None],
        })
    return ret



def collate_audio(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
    return {"audio": torch.stack(audio, dim=0)}


def collate_all(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    token = []
    text = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        token.append(batch[idx].get("token", torch.zeros(0).long()))
        text.append(batch[idx]["text"])
    return {
        "audio": torch.stack(audio, dim=0),
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
        "text": text,
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


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms: BaseTransforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)


class MixTransforms(BaseTransforms):
    name = "MixTransforms"
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

    def resample(self, src_sample_rate, x):
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

    def get_audio(self, item):
        name = item["__dataset_name__"]
        if re.match("music_.*", name):
            src_sample_rate = self.data_sample_rate
        else:
            src_sample_rate = item["src_sample_rate"]
        audio = self.base_transform(item[self.audio_key])
        audio = self.resample(src_sample_rate, audio)
        if self.normalize_audio:
            audio = self.fast_normalizer(audio)
        return audio

    def get_text_token(self, text):
        text = self.remove_punc(normalize_text(str(text)))
        encoded_text = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )
        token = encoded_text["input_ids"].squeeze(dim=0)
        return token, text

    def __call__(self, item: Dict[str, Any]) -> Generator:
        text = item["text"]
        try:
            audio = self.get_audio(item)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        output_dict = {"audio": audio}
        if self.tokenizer is not None and callable(self.tokenizer):
            token, normalized_text = self.get_text_token(text)
            if (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
            output_dict.update(text=normalized_text)
        yield output_dict
        self._update_stats(skipped=False)


class MixDataset(WebPipeline):
    name = "Mix"

    def __init__(
        self,
        data_id: int = None,
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
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        transforms = MixTransforms(
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
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_ids,
        data_weights,
        val_data_id: int = 793,
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
        tokenizer: str = None,
        frame_rate: int = 25,
        bsz_evaluator: Optional[str] = None,
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

        def get_dataset(id):
            return MixDataset(
                data_id=id,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )

        datasets = [get_dataset(id) for id in data_ids]
        if len(datasets) > 1:
            self.train_dataset = DataPipeline(
                MultiIterableDataset(datasets=datasets, weights=data_weights),
                wds.shuffle(shuffle_buffer_size),
                self.bucketize,
            )
        else:
            self.train_dataset = DataPipeline(
                datasets[0], wds.shuffle(shuffle_buffer_size), self.bucketize
            )
        self.validation_dataset = DataPipeline(
            MixDataset(
                data_id=val_data_id,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            self.bucketize,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=16,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixTransformsMSS(MixTransforms):
    def __init__(self, *args, **kwargs):
        audio_perturb = kwargs.pop('audio_perturb', False)
        super().__init__(*args, **kwargs)
        self.audio_perturb = audio_perturb
        if audio_perturb:
            from recipes.umm.utils.nansy_utils import AudioPerturb
            self.ap = AudioPerturb()

    def get_audio(self, item, key=None):
        name = item["__dataset_name__"]
        if re.match("music_.*", name):
            src_sample_rate = self.data_sample_rate
        else:
            src_sample_rate = item["src_sample_rate"]
        if key is None:
            key = self.audio_key
        audio = self.base_transform(item[key])
        audio = self.resample(src_sample_rate, audio)
        if self.normalize_audio:
            audio = self.fast_normalizer(audio)
        return audio

    def __call__(self, item: Dict[str, Any]) -> Generator:
        text = item["text"]
        try:
            audio = self.get_audio(item)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        output_dict = {"audio": audio}
        if self.tokenizer is not None and callable(self.tokenizer):
            token, normalized_text = self.get_text_token(text)
            if (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
            output_dict.update(text=normalized_text)
        try:
            output_dict.update(audio_vocal=self.get_audio(item, 'vocal'))
            output_dict.update(audio_inst=self.get_audio(item, 'acc'))
            if self.audio_perturb:
                if random.random() > 0.5:
                    output_dict['audio_vocal_perturb'] = self.ap(output_dict['audio_vocal'][0])[None]
                else:
                    output_dict['audio_vocal_perturb'] = output_dict['audio_vocal']
        except Exception as e:
            self._update_stats(skipped=True, message=f"No acc or vocal in audio: {e}")
            return

        yield output_dict
        self._update_stats(skipped=False)



class MixMSSDataset(WebPipeline):
    name = "MixMSSDataset"

    def __init__(
        self,
        data_id: int = None,
        url_pattern: str = None,
        sample_rate: int = 24000,
        audio_key: str = "wav",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        audio_perturb=False,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        transforms = MixTransformsMSS(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            audio_perturb=audio_perturb
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MixMSSDataModule(MixDataModule):
    def __init__(
        self,
        data_ids,
        data_weights,
        val_data_id: int = 793,
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn_mss,
        tokenizer: str = None,
        frame_rate: int = 25,
        bsz_evaluator: Optional[str] = None,
        audio_perturb=False
    ):
        pl.LightningDataModule.__init__(self)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        if tokenizer is not None:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
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

        def get_dataset(id):
            return MixMSSDataset(
                data_id=id,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
                extra_fields_in_data=['vocal', 'acc'],
                audio_perturb=audio_perturb
            )

        datasets = [get_dataset(id) for id in data_ids]
        if len(datasets) > 1:
            self.train_dataset = DataPipeline(
                MultiIterableDataset(datasets=datasets, weights=data_weights),
                wds.shuffle(shuffle_buffer_size),
                self.bucketize,
            )
        else:
            self.train_dataset = DataPipeline(
                datasets[0], wds.shuffle(shuffle_buffer_size), self.bucketize
            )
        self.validation_dataset = DataPipeline(
            MixMSSDataset(
                data_id=val_data_id,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                extra_fields_in_data=['vocal', 'acc']
            ),
            self.bucketize,
        )

import logging
import math
import random
import sys
from string import punctuation
from zhon.hanzi import punctuation as punctuation_zh
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
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer, AutoTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

from recipes.datasets.mcc import INDEX
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id
from recipes.musiclm.transforms.audio import FastNormalizeAudio, LoudnessCheck
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
import logging
import numpy as np
import pyworld as pw

from recipes.datasets.mcc.mix import (BaseTransforms, Compose, Resample, normalize_text, 
        WebPipeline, ParquetDataset, WebDatasetBufferPreprocessor)
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)


logger = logging.getLogger(__name__)

def get_f0_wds(wav, sr, hop_size):
    assert wav.shape[0] == 1
    wav = wav.squeeze(0)
    _f0, t = pw.dio(wav.numpy().astype(np.double), sr, frame_period=hop_size / sr * 1000)
    f0 = pw.stonemask(wav.numpy().astype(np.double), _f0, t, sr)
    f0 = torch.from_numpy(f0).float().unsqueeze(0)
    return f0

def f0_normalize(f0):
    _f0 = f0.clone()
    f0 = torch.log1p(f0)
    f0_mean = torch.mean(f0[_f0!=0])
    f0_std = torch.std(f0[_f0!=0])
    f0[_f0!=0] = (f0[_f0!=0] - f0_mean) / f0_std
    return f0


def get_vuv(f0):
    vuv = f0.clone()
    vuv[vuv!=0] = 1
    return vuv


def get_f0_vuv(wav, sr, hop_size):
    f0 = get_f0_wds(wav, sr=sr, hop_size=hop_size)

    if f0 is None:
        return None, None

    vuv = get_vuv(f0)

    f0 = f0_normalize(f0)
    
    if torch.isnan(f0).any() or torch.isinf(f0).any():
        return None, None
    
    return f0, vuv


def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    # text = []
    token = []
    # tag = []
    f0 = []
    vuv = []

    for idx in range(len(batch)):
        audio_x = random_pad(batch[idx]["audio"])
        f0_x, vuv_x = get_f0_vuv(audio_x, batch[idx]['sample_rate'], batch[idx]["hop_length"])
        if f0_x is None:
            print('F0 is None, audio shape: ', audio_x.shape)
            continue
        audio.append(audio_x)
        f0.append(f0_x)
        vuv.append(vuv_x)
        # text.append(batch[idx]["text"])
        token.append(batch[idx].get("token", torch.zeros(0).long()))
        # tag.append(batch[idx]["tag"])
    return {
        "audio": torch.stack(audio, dim=0),
        # "text": text,
        "f0": torch.nn.utils.rnn.pad_sequence(
            f0, batch_first=True, padding_value=0
        ),
        "vuv": torch.nn.utils.rnn.pad_sequence(
            vuv, batch_first=True, padding_value=0
        ),
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
        # "tag": tag,
    }


class BigTTSTransforms(BaseTransforms):
    name = "BigTTSTransforms"
    data_sample_rate = 24000

    def __init__(
            self,
            sample_rate: int = 24000,
            hop_length: int = 300,
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
        self.hop_length = hop_length
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
            self.resampler[src_sample_rate] = Resample(src_sample_rate, self.sample_rate)
        return self.resampler[src_sample_rate](x)

    def remove_punc(self, text):
        punctuation_all = list(punctuation_zh + punctuation)
        for punc in punctuation_all:
            text = text.replace(punc, '')
        return text

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            output_dict = {}
            text = self.remove_punc(normalize_text(item["text"]))
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
                '''
                elif (
                        token.size(-1)
                        > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                '''
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
            
            output_dict.update({"audio": audio, "text": text, "tag": "vocal",
                "sample_rate": self.sample_rate, "hop_length": self.hop_length})
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
            data_id: int = 193, # 181: en_4.2wh, 182: en_4.2wh_cn_3.2wh
            # url_pattern: str = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/tts_Len_S11labs-rp2900_P1/package/wav_1.0_web_dataset_2/data/part=00003/shard-00010.tar', # for debug, datasets 181/182 are too large.
            url_pattern: str = None,
            sample_rate: int = 24000,
            hop_length: int = 300,
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
        # if url_pattern is None:
        #     urls = parse_data_urls(data_id=data_id)
        # else:
        #     urls = parse_data_urls(data_urls=url_pattern)
        # dataset = ra_wds.WebDataset(urls=urls, **kwargs)

        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)

        transforms = BigTTSTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            hop_length=hop_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")



class ParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_id: int = None,
        sample_rate: int = 24000,
        hop_length: int = 300,
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
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == 'seed':
            self.tokenizer = AutoTokenizer.from_pretrained("recipes/umm/tokenizer_bbpe64k-0303")
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
        bigtts = BigTTSDataset(
            data_id=data_id,
            sample_rate=sample_rate,
            hop_length=hop_length,
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
        datasets.append(
            DataPipeline(bigtts, wds.shuffle(shuffle_buffer_size))
        )
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[1]
            ),
            self.bucketize,
        )

        self.validation_dataset = []

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
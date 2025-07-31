from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
import pytorch_lightning as pl
import torch
import webdataset as wds
from functools import partial
from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline

from samantha.dataio.batching import BucketBatcher
from torch.utils.data._utils.collate import collate, collate_tensor_fn, default_collate_fn_map
import random
import torch.distributed as dist
from recipes.bigmusic.datasets.utils.ddp_utils import distributed_subset

########################## Data Modules ##########################

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_buffer_size: int = 0,
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
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        if isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=2,
                    collate_fn=self.collate_fn,
                    pin_memory=self.pin_memory
                )
                for val in self.validation_dataset
            ]
        else:
            return DataLoader(
                self.validation_dataset,
                batch_size=None,
                num_workers=2,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory
            )

    def predict_dataloader(self):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dataset = distributed_subset(self.predict_dataset)
        else:
            dataset = self.predict_dataset
        return [DataLoader(dataset, batch_size=None, num_workers=2, collate_fn=self.collate_fn, pin_memory=self.pin_memory)]


# Segment Transforms
class SemanticTokenLengthTransform():
    def __init__(self, sample_rate=24000, semantic_frame_rate=25, audio_key="target_audio"):
        self.sample_rate = sample_rate
        self.semantic_frame_rate = semantic_frame_rate
        self.audio_key = audio_key

    def __call__(self, item):
        target_audio = item[self.audio_key]
        audio_length = target_audio.shape[-1]
        seq_length = audio_length * self.semantic_frame_rate // self.sample_rate
        return { **item, 'target_tokens_length': seq_length }

def crop_pad_to_seq_length(seq: torch.Tensor, target_seq_len, dtype=None, padding_value=0, start=0):
    *dims, input_seq_len = seq.shape
    dtype = seq.dtype if dtype is None else dtype
    pad_x = torch.full((*dims, target_seq_len), fill_value=padding_value, dtype=dtype, device=seq.device)
    seq_slice = seq[..., start:start+target_seq_len]
    pad_x[..., :seq_slice.shape[-1]] = seq_slice
    return pad_x

def random_crop_pad_to_seq_length(seq: torch.Tensor, target_seq_len, dtype=None, padding_value=0):
    *dims, input_seq_len = seq.shape
    start_idx = random.randint(0, max(0, input_seq_len - target_seq_len))
    return crop_pad_to_seq_length(seq, target_seq_len=target_seq_len, dtype=dtype, padding_value=padding_value, start=start_idx)


def pad_collate_tensor_fn(batch, *, collate_fn_map):
     max_length = max([x.shape[-1] for x in batch])
     batch = [crop_pad_to_seq_length(x, max_length, x.dtype, padding_value=0) for x in batch]
     return collate_tensor_fn(batch, collate_fn_map=collate_fn_map)

def simple_collate_fn(batch, *, collate_fn_map):
    return batch

def dictionary_collate(batch, remove_invalid=True):
    """Fixes pytorch's default collate which cannot handle dictionaries or null fields."""
    lyrics_collate_fn_map = {
        **default_collate_fn_map,
        torch.Tensor: pad_collate_tensor_fn,
        list: simple_collate_fn,
        type(None): simple_collate_fn,
    }
    if remove_invalid:
        def remove_invalid_fields(item):
            def invalid_field(field): return field is None or isinstance(field, dict)
            return { k:v for k,v in item.items() if not invalid_field(v)}
        batch = [remove_invalid_fields(item) for item in batch]
    return collate(batch, collate_fn_map=lyrics_collate_fn_map)

class MusicBucketBatcher(BucketBatcher):
    def __init__(
        self,
        *args,
        collate_fn = dictionary_collate,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn

    def __call__(self, it):
        for item in it:
            batch = self.collate_batch(item)
            if batch is not None and len(batch):
                yield self.collate_fn(batch)

def default_bucket_batcher_length_fn(item, sample_rate=24000, semantic_frame_rate=25):
    if 'target_tokens_length' in item:
        target_seq_length = item['target_tokens_length'] # value returned from SemanticTokenLengthTransform
    elif 'target_audio' in item:
        target_seq_length = int(item['target_audio'].shape[-1] / sample_rate * semantic_frame_rate) # fallback case
    elif 'audio' in item:
        target_seq_length = int(item['audio'].shape[-1] / sample_rate * semantic_frame_rate) # fallback case
    else:
        raise ValueError("Batched item must target audio to determine length function")

    if 'lyrics_tokens_length' in item:
        input_seq_length = item['lyrics_tokens_length']
    elif 'lyrics_tokens' in item:
        input_seq_length = len(item['lyrics_tokens']) # fallback case
    else:
        input_seq_length = 0 # instrumental case
    return target_seq_length + input_seq_length

def default_bucket_batcher_fn(
    sample_rate,
    sample_duration,
    batch_size,
    semantic_frame_rate=25,
    lyrics_frame_rate=14,
    max_duration=None,
):
    token_frame_rate = semantic_frame_rate + lyrics_frame_rate
    sample_duration = sample_duration if isinstance(sample_duration, (list, tuple)) else [sample_duration]
    buckets_samples = [d * token_frame_rate for d in sorted(set(sample_duration))]
    length_fn = partial(default_bucket_batcher_length_fn, sample_rate=sample_rate, semantic_frame_rate=semantic_frame_rate)
    if max_duration is None:
        maximum_bucket_size = max(buckets_samples) * batch_size
    else:
        maximum_bucket_size = max_duration * token_frame_rate * batch_size
    return MusicBucketBatcher(
        buckets=buckets_samples,
        maximum_bucket_size=maximum_bucket_size,
        dynamic_batch=True,
        length_fn=length_fn,
    )
    
def default_batch_fn(batch_size, collation_fn=dictionary_collate):
    return wds.batched(batch_size, collation_fn=collation_fn)

def transform_dataset(dataset, segment_transforms=(), batch_transforms=(), batch_fn=None, shuffle_buffer_size=None):
    if isinstance(dataset, DataPipeline):
        data_pipeline = dataset
    else:
        data_pipeline = DataPipeline(dataset)
    if shuffle_buffer_size is not None:
        data_pipeline.append(wds.shuffle(shuffle_buffer_size))
    for segment_transform in segment_transforms:
        data_pipeline.append(wds.map(segment_transform))
    if batch_fn is not None:
        data_pipeline.append(batch_fn)
        for batch_transform in batch_transforms:
            data_pipeline.append(wds.map(batch_transform))
    return data_pipeline

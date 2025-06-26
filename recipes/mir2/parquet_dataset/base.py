import logging
import json
import hashlib
import torch
import torch.nn.functional as F
from abc import ABC
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from random import randrange

import webdataset as wds
from webdataset import filters, shardlists,  warn_and_continue
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

from samantha.dataio.parquet.extension import setup_sampler
from samantha.dataio.parquet.shardlists import ResampledShards, SimpleShardList
from samantha.dataio.utils import resolve_data_urls
from samantha.dataio.preprocess import AudioLengthModifier


logger = logging.getLogger(__name__)

class YieldState:
    def __init__(self):
        pass

    def __call__(self, item):
        # now yield None to adapt cruise lite dataloader
        if item is not None:
            return item, (None, None)

class ParquetDataset(DataPipeline, FluidInterface):
    def __init__(
        self,
        data_id: list = None,
        data_urls: List[Dict[str, str]] = None,
        handler: Callable[[Exception], bool] = warn_and_continue,
        resampled: bool = False,
        shardshuffle: Optional[Any] = None,
        detshuffle: bool = False,
        nodesplitter=shardlists.single_node_only,
        sample_limit_per_file: int = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_id = data_id
        self.data_urls = data_urls

        if type(data_id) is list:
            self.urls = []
            for did in data_id:
                self.urls += resolve_data_urls(data_id=did, data_urls=None)
        else:
            self.data_urls = data_urls
            self.urls = resolve_data_urls(data_id=self.data_id, data_urls=self.data_urls)

        if resampled:
            self.append(
                ResampledShards(self.urls, replacement=kwargs.get("replacement", False))
            )
            self.append(shardlists.split_by_node)
            self.append(shardlists.split_by_worker)
        else:
            self.append(SimpleShardList(self.urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle:
                shardshuffle = 100
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))

        self.append(
            setup_sampler(
                handler, sample_limit_per_file, extra_fields_in_data, sample_config
            )
        )

        sample_shuffle_buffer_size = kwargs.get("sample_shuffle_buffer_size", 0)
        if sample_shuffle_buffer_size > 1:
            self.append(filters.shuffle(sample_shuffle_buffer_size))

        item_transform = kwargs.get("item_transform")
        if item_transform is not None:
            self.map(item_transform)

        batch_size = kwargs.get("batch_size_in_worker")
        if batch_size is not None:
            self.append(filters.batched(batchsize=batch_size, collation_fn=None))

        yield_state = kwargs.get("yield_state", False)
        if yield_state:
            yield_state_func = YieldState()
            self.map(yield_state_func)

    def load_state_dict(self, dataset_state):
        pass


class CustomLengthModifier(AudioLengthModifier):
    """
    Handle the edge case that source_frames equals to target_frames,
    which will cause exceptions in AudioLengthModifier
    """
    def __init__(
            self,
            sampling_rate: int,
            target_duration_sec: float,
            is_random_crop: bool = True,
            ):
        super().__init__(sampling_rate, target_duration_sec, is_random_crop)

        if target_duration_sec < 0:
            self._raise_config_error("target_duration_sec must be positive.")

    def __call__(self, audio: torch.Tensor):
        self._check_audio(audio)
        num_channels, source_frames = audio.shape
        target_frames = int(self.target_duration_sec * self.sampling_rate)

        start_idx = 0
        if source_frames < target_frames:
            output_audio = F.pad(
                    input=audio,
                    pad=(0, target_frames - source_frames),
                    mode="constant",
                    value=0,
                    )
            padding_mask = F.pad(
                    input=torch.zeros_like(audio),
                    pad=(0, target_frames - source_frames),
                    mode="constant",
                    value=1,
                    )
        else:
            if self.is_random_crop and (source_frames > target_frames):
                start_idx = randrange(source_frames - target_frames)
            output_audio = audio.clone()[:, start_idx : start_idx + target_frames]
            padding_mask = torch.zeros((num_channels, target_frames))
        return output_audio, padding_mask, start_idx

class TaggingDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, infer_batch_size=None, *args, **kwargs):
        # infer_batch_size is only used in test_preprocess
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._audio_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._val_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=False
        )
        self.sampling_rate = sampling_rate
        self.target_duration_sec = target_duration_sec
        self.chunk_len = int(self.sampling_rate * self.target_duration_sec)
        self.infer_batch_size = infer_batch_size

    def train_preprocess(self, x):
        out = x.copy()
        np_audio = self._sample_train_audio(x)
        out["audio.npy"] = torch.tensor(np_audio, dtype=torch.float32)
        return out

    def val_preprocess(self, x):
        out = x.copy()
        np_audio = self._sample_val_audio(x)
        out["audio.npy"] = torch.tensor(np_audio, dtype=torch.float32)
        return out

    def test_preprocess(self, x):
        # NOTE (vibertthio): This assume the input is mono and the first dimension is the channel dimension
        out = x.copy()
        np_audio = torch.from_numpy(x['audio.npy'].squeeze(0))

        # NOTE (vibertthio): Add padding to make sure all audio is used, including the last chunk
        np_audio = torch.nn.functional.pad( np_audio, (0, self.chunk_len//2) )
        np_audio = np_audio.unfold(0, self.chunk_len, self.chunk_len//2)
        if np_audio.shape[0] > self.infer_batch_size:
            np_audio = np_audio[:self.infer_batch_size, :]

        out["audio.npy"] = np_audio.type(torch.float32)
        return out

    def _sample_train_audio(self, x):
        np_audio = x["audio.npy"]
        audio_duration_in_s = (
            np_audio.shape[0] / self._audio_length_modifier.sampling_rate
        )

        # sample from audio
        audio, _, start_idx = self._audio_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1)),
        )

        sampled_np_audio = np.squeeze(audio.numpy())

        return sampled_np_audio

    def _sample_val_audio(self, x):
        np_audio = x["audio.npy"]
        audio_duration_in_s = (
            np_audio.shape[0] / self._val_length_modifier.sampling_rate
        )

        # sample from audio
        audio, _, start_idx = self._val_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1)),
        )

        sampled_np_audio = np.squeeze(audio.numpy())

        return sampled_np_audio


    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Factory command to create the MSS model.
        This method gets the appropriate MSS model class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the MSS model to create.
        Returns:
            An instance of the MSS model that is created.
        """

        if name not in cls.registry:
            logging.warning("Executor %s does not exist in the registry", name)
            return None

        exec_class = cls.registry[name]
        executor = exec_class(*args, **kwargs)
        return executor


class TaggingPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = TaggingDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec
        self._hop_factor = 1  # for beat

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def test_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)

class BaseParquetDataset(ParquetDataset):
    def __init__(
        self,
        data_id: int = 3309,
        batch_size: int = 12,
        infer_batch_size: int = 32,
        resampled: bool =True,
        shuffle_buffer: int = 96,
        sample_rate: int = 24000,
        sample_len: float = 29.1,
        nodesplitter = shardlists.split_by_node,
        validation: float = False,
        test: bool = False,
        debug: bool = False,
        replacement: bool = False,
        num_iter: int = None,
        dataset_preprocessors = None,  # add a new parameter for flexible preprocessors
        to_tuple = None,
    ):
        super().__init__(
            data_id=data_id, 
            resampled=resampled,
            nodesplitter=nodesplitter,
            replacement=replacement,
            #sample_config={"name": "ParquetSampleV2"},
        )
        self.validation = validation
        self.test = test
        self.sample_rate = sample_rate

        if dataset_preprocessors is not None:
            self.dataset_preprocessors = dataset_preprocessors
        else:
            self.dataset_preprocessors = TaggingDatasetMixin(
                sample_len, sample_rate, infer_batch_size=infer_batch_size,
            )

        if debug:
            self.append(wds.map(self.preproc_each_sample,)) 
        else:
            self.append(wds.map(self.preproc_each_sample, handler=warn_and_continue,))

        if to_tuple is not None:
            self.append(wds.to_tuple(*to_tuple))
        else:
            self.append(wds.to_tuple('audio.npy', 'tags', 'uuid_int', 'uuid', 'index_url'))
        if not (validation or test):
            self.append(wds.shuffle(shuffle_buffer))
        
        # self.append(filters.batched(batchsize=batch_size, collation_fn=None))
        

        if test:
            # Collate for test is different than training, because each batch in test uses one audio
            # On the other hand, training and validation both use multiple audios.
            # Therefore, collate function is different for each
            self.append(wds.batched(batch_size, collation_fn=self.collate_test))
        else:
            self.append(wds.batched(batch_size, collation_fn=self.collate))

        # Do not artificially set the sample number of epoch to 
        # avoid breaking the internal logic of the dataloader.
        if validation and num_iter is not None:
            self.with_epoch(num_iter)


    def export_json(self, sample, meta):
        import soundfile as sf
        import os
        
        uttid = sample['uttid']
        # with open(f"tmp/uttid_{self.data_id}.lst", "a") as file:
        #     file.write(f"{uttid}\n")
        
        # TODO: make this configurable, so it could be synced with the pl_module and the callbacks
        outdir = f"tmp/{self.data_id}.audios"
        os.makedirs(outdir, exist_ok=True)

        out_path_wav = os.path.join(outdir, f"{self.data_id}.{uttid}.wav")
        if os.path.exists(out_path_wav) == False:
            with open(out_path_wav, 'wb') as f:
                f.write(sample["wav"])

        out_path = os.path.join(outdir, f"{self.data_id}.{uttid}.json")
        if os.path.exists(out_path) == False:
            json.dump(meta, open(out_path, 'w'))
        
        

    def process_label_and_audio(self):
        raise NotImplementedError()


    def preproc_each_sample(self, sample):
        # print(f' ******** sample keys: {sample.keys()}')
        
        # TODO: make this configurable or move it into a callback
        # self.export_json(sample, meta=json.loads(sample['meta']))
    
        item = self.process_label_and_audio(sample)

        # 如果处理结果为 None，直接跳过该样本
        if item is None:
            # 使用 webdataset 的 filter 机制跳过该样本
            raise wds.filters.FilterException("Invalid sample")

        if self.validation:
            item = self.dataset_preprocessors.val_preprocess(item)
        elif self.test:
            item = self.dataset_preprocessors.test_preprocess(item)
        else:
            item = self.dataset_preprocessors.train_preprocess(item)
        return item
     

    def get_id(self, x):
        # Python hash function is not guaranteed to be consistent across runs, so we use hashlib
        _hash = int(hashlib.sha256(str(x).encode("utf-8")).hexdigest(), 16) % 10**8
        return _hash

    def collate_test(self, batch):
        audios, tags, uuids_int, uuid, index_url = batch[0] # only take the first in batch, so batch_size should be 1
        num_chunks = audios.shape[0]
        for tag_type in tags:
            tags[tag_type] = torch.tensor(tags[tag_type]).repeat(num_chunks, 1)
        
        uuids_int = torch.tensor(uuids_int).repeat(num_chunks, 1)
        uuids = [uuid] * num_chunks
        index_urls = [index_url] * num_chunks

        #print(audios.shape, [v.shape for _, v in tags.items()], uuids.shape)
        return [audios, tags, uuids_int, uuids, index_urls]

    def collate(self, batch):
        tag_type_batch = {}
        audios = []
        uuids_int = []
        uuids = []
        index_urls = []
        for audio, tags, uuid_int, uuid, index_url in batch: 
            for tag_type in tags:
                if tag_type not in tag_type_batch:
                    tag_type_batch[tag_type] = []
                tag_type_batch[tag_type].append(tags[tag_type])
            audios.append(audio)
            uuids_int.append(uuid_int)
            uuids.append(uuid)
            index_urls.append(index_url)
        
        new_batch = {}
        for tag_type in tag_type_batch:
            new_batch[tag_type] = np.stack(tag_type_batch[tag_type])
        audios = torch.stack(audios)
        uuids_int = torch.tensor(uuids_int)
        return [audios, new_batch, uuids_int, uuids, index_urls]

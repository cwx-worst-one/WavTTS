import copy
import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Union

import pytorch_lightning as pl
import torch
from cruise import CruiseConfig, CruiseDataModule, last_cli
from cruise.data_module.lite.batcher import DefaultBatcher
from cruise.data_module.lite.dataloader import LiteCruiseDataLoader, LiteCruiseDataset
from cruise.data_module.lite.multi_iterable_dataset import MultiIterableDataset
from mariana.utils.audio.audio_logger import AudioLogger

import samantha  # noqa: F401, import for solve mariana import path.
from samantha.dataio.batching import SimpleBatcher, setup_batcher_fn

# from mariana.data.audio.compose import build_draw_batch_fn, build_item_augmentation
from samantha.dataio.bigmusic.bigmusic_compose import (
    build_draw_batch_fn,
    build_item_augmentation,
)
from samantha.dataio.utils import resolve_data_urls

# Not ready
# from mariana.data.utils.path_utils import (
#     file_pattern,
#     format_file_list,
#     get_file_key,
#     resolve_data_urls,
# )


logger = AudioLogger()

_default_dataloader_config = {
    "num_workers": 1,
    "drop_last": False,
    "pin_memory": True,
    "torch_num_threads": 2,
    "persistent_workers": True,
    "prefetch_factor": 32,
    "batch_prefetch_factor": 8,
    "prefetch_retry": 5,
    "cuda_cache_size": 2,
    "gpu_prefetch": True,  # debug without gpu
    "save_ckpt_interval": int(1e9 + 7),
    "bitwise_resume": False,
    "auto_source_len": True,
    "force_enable_gc": True,
    "terminate_timeout": 30,
    "shuffle_buffer_size": 1,
}
_default_val_dataloader_config = _default_dataloader_config
_default_predict_dataloader_config = _default_dataloader_config
_default_train_dataloader_config = {
    **_default_dataloader_config,
    "num_workers": 4,
    "bitwise_resume": True,
}


_default_parquet_dataset_config = {
    "read_batch_size": 40,
    "index_file_list": None,
    # index file通过这个参数传递
    # 如果有extra feature也放在这里传进去
    "sampler": "ParquetSampler",
    "columns": None,
    "use_threads": True,
    "native_parallel_file_num": 1,
    # "split_path_by_rank": True,  # useless
    "filter_by_index": True,
    "split_path_list_by_rank": True,  # control by `split_path_list`
    "split_path_list_by_process": True,  # control by `split_path_list`
}


class _LiteMultiTransformDataset(LiteCruiseDataset):
    def __init__(self, data_ids, **_kwargs):
        super().__init__(**_kwargs)
        assert data_ids is None or len(data_ids) == len(self.data_sources)
        self.data_ids = data_ids

    def create_dataset(self):
        """Override this method to support different item transform for different datasets."""
        dataset = super().create_dataset()
        if isinstance(dataset, MultiIterableDataset):
            dataset_list = dataset._datasets
        else:
            dataset_list = [dataset]
        assert len(dataset_list) == len(self.data_ids)
        if isinstance(self.item_transform, Dict):
            for ds, data_id in zip(dataset_list, self.data_ids):
                delattr(ds, "item_transform")
                setattr(ds, "item_transform", self.item_transform[data_id])
        return dataset


_LITE_USE_PL_MODULE = int(os.getenv("LITE_USE_PL_MODULE", "0")) != 0

if _LITE_USE_PL_MODULE:
    BaseDataModule = pl.LightningDataModule
else:
    BaseDataModule = CruiseDataModule


class MusicLiteDataModule(BaseDataModule):
    def __init__(
        self,
        train_dataset_ids: Union[int, List[int], str, List[str]] = None,
        train_data_urls: List[Dict[str, str]] = None,
        train_dataset_weights: List[float] = None,
        train_dataset_config: Dict[str, Any] = _default_parquet_dataset_config,
        train_dataloader_config: Dict[str, Any] = _default_train_dataloader_config,
        train_item_transform: Any = None,
        train_batch_transform: Any = None,
        train_device_transform: Any = None,
        train_batcher: Any = None,
        val_dataset_ids: Union[int, List[int], str, List[str]] = None,
        val_data_urls: List[Dict[str, str]] = None,
        val_dataset_weights: List[float] = None,
        val_dataset_config: Dict[str, Any] = _default_parquet_dataset_config,
        val_dataloader_config: Dict[str, Any] = _default_val_dataloader_config,
        val_item_transform: Any = None,
        val_batch_transform: Any = None,
        val_device_transform: Any = None,
        val_batcher: Any = None,
        predict_dataset_ids: Union[int, List[int], str, List[str]] = None,
        predict_data_urls: List[Dict[str, str]] = None,
        predict_dataset_weights: List[float] = None,
        predict_dataset_config: Dict[str, Any] = _default_parquet_dataset_config,
        predict_dataloader_config: Dict[str, Any] = _default_predict_dataloader_config,
        predict_item_transform: Any = None,
        predict_batch_transform: Any = None,
        predict_device_transform: Any = None,
        predict_batcher: Any = None,
        seed: int = 12236,
        shuffle: bool = True,  # only control in train dataset
        save_ckpt_interval: int = None,
        bitwise_resume: bool = True,
        data_key_in_source_urls: str = "data",
        split_path_list: bool = True,  # only control in train dataset
    ):
        super().__init__()
        if _LITE_USE_PL_MODULE:
            assert isinstance(self, pl.LightningDataModule)
            self.save_hyperparameters()
        else:
            assert isinstance(self, CruiseDataModule)
            self.save_hparams()

        self.train_data_sources_types = self._make_data_sources_and_types(
            train_dataset_ids,
            train_data_urls,
            train_dataset_weights,
            self.hparams.data_key_in_source_urls,
        )
        self.val_data_sources_types = self._make_data_sources_and_types(
            val_dataset_ids, val_data_urls, val_dataset_weights, "data"
        )
        self.predict_data_sources_types = self._make_data_sources_and_types(
            predict_dataset_ids, predict_data_urls, predict_dataset_weights, "data"
        )

    def _make_data_sources_and_types(
        self, dataset_ids, data_urls, dataset_weights=None, data_key="data"
    ):
        if dataset_ids is not None:
            assert data_urls is None, "dataset_ids and data_urls cannot be both set"
            dataset_ids = [dataset_ids] if isinstance(dataset_ids, int) else dataset_ids
            data_urls_lists = [
                resolve_data_urls(data_id=dataset_id) for dataset_id in dataset_ids
            ]
        elif data_urls is not None:
            data_urls_lists = [resolve_data_urls(data_urls=data_urls)]
        else:
            data_urls_lists = None

        data_sources = defaultdict(list)
        source_types = []
        key_data = data_key
        key_index = "index"
        if data_urls_lists:
            for dataset_urls in data_urls_lists:
                # convert list of dict to dict of list
                dataset_urls_dict = {}
                for url in dataset_urls:
                    for k, v in url.items():
                        if k not in dataset_urls_dict:
                            dataset_urls_dict[k] = []
                        dataset_urls_dict[k].append(v)
                if key_data not in dataset_urls_dict:
                    raise ValueError(
                        f"key `{key_data}` not found in data_urls but set in data_key_in_source_urls"
                    )

                for k, v in dataset_urls_dict.items():
                    if k in (key_index, key_data):
                        data_sources[k].append(v)
                        if k == key_data:
                            # only support parquet data for now
                            source_types.append("parquet")
                    # TODO: 支持其他的离线特征
        assert dataset_weights is None or len(dataset_weights) == len(source_types), (
            "dataset_weights must be set with the same length as data_sources,"
            " is there a lengths mismatch or dataset have mixed source types?"
        )
        return data_sources, source_types, dataset_weights

    def _dataloader_factory(self, stage: Literal["train", "val", "predict"]):
        data_source_types = getattr(self, f"{stage}_data_sources_types")
        if data_source_types is None:
            return None
        data_sources, source_types, dataset_weights = data_source_types
        if data_sources is None or len(data_sources) == 0:
            return None
        dataset_config = getattr(self.hparams, f"{stage}_dataset_config")
        if dataset_config is not None:
            dataset_config = {**_default_parquet_dataset_config, **dataset_config}
        else:
            dataset_config = _default_parquet_dataset_config
        if stage in ["val", "predict"]:
            # 文件数量比较小的时候，需要关闭按文件分割，不然会自动重复数据
            dataset_config["split_path_list_by_rank"] = False
            dataset_config["split_path_list_by_process"] = False
        else:
            # 训练的场景下，如果训练数据小，也要关闭按文件分割，不然会自动重复数据
            dataset_config["split_path_list_by_rank"] = self.hparams.split_path_list
            dataset_config["split_path_list_by_process"] = self.hparams.split_path_list

        data_ids = getattr(self.hparams, f"{stage}_dataset_ids")

        # setup batcher
        batcher_cfg = copy.deepcopy(getattr(self.hparams, f"{stage}_batcher"))
        logger.info(f"setup batcher for {stage} stage with {batcher_cfg}")
        try:
            batcher = setup_batcher_fn(batcher_cfg)
        except Exception:
            logger.error(f"setup batcher for {stage} stage failed, use default batcher")
            batcher = SimpleBatcher()  # default

        batch_transform = getattr(self.hparams, f"{stage}_batch_transform")
        device_transform = getattr(self.hparams, f"{stage}_device_transform")
        dataloader_config = getattr(self.hparams, f"{stage}_dataloader_config")
        default_dataloader_config = globals().get(
            f"_default_{stage}_dataloader_config", None
        )
        assert (
            default_dataloader_config is not None
        ), f"default dataloader config for {stage} stage not found"
        dataloader_config = {**default_dataloader_config, **dataloader_config}

        if stage == "train":
            dataloader_config["bitwise_resume"] = self.hparams.bitwise_resume
            save_ckpt_interval = self.hparams.save_ckpt_interval
            if save_ckpt_interval is None:
                if hasattr(self, "trainer") and self.trainer is not None:
                    save_ckpt_interval = self.trainer.val_check_interval
                else:
                    save_ckpt_interval = int(1e9)
            dataloader_config["save_ckpt_interval"] = save_ckpt_interval
        else:
            dataloader_config["bitwise_resume"] = False
            dataloader_config["save_ckpt_interval"] = int(1e9)

        key_index = "index"
        if stage == "train":
            data_urls = data_sources[self.hparams.data_key_in_source_urls]
        else:
            data_urls = data_sources["data"]
        index_urls = data_sources.get(key_index, None)

        transform_config = getattr(self.hparams, f"{stage}_item_transform")
        if not callable(transform_config) and transform_config:
            item_transform = build_item_augmentation(transform_config)
        else:
            item_transform = transform_config
        batch_transform_config = getattr(self.hparams, f"{stage}_batch_transform")
        if not callable(batch_transform_config) and batch_transform_config:
            batch_transform = build_draw_batch_fn(batch_transform_config)
        else:
            batch_transform = batch_transform_config

        dataset_configs = [
            {**dataset_config, "index_file_list": index_urls[i] if index_urls else None}
            for i in range(len(data_urls))
        ]
        if _LITE_USE_PL_MODULE:
            world_size = self.trainer.world_size if self.trainer else 1
            rank = self.trainer.global_rank if self.trainer else 0
        else:
            # cruise will help us get this
            world_size = None
            rank = None
        lite_dataset = _LiteMultiTransformDataset(
            data_ids=data_ids,
            data_sources=data_urls,
            source_types=source_types,
            multiplex_weights=dataset_weights,
            dataset_cfg=dataset_configs,
            item_transform=item_transform,
            shuffle=self.hparams.shuffle if stage == "train" else False,
            seed=self.hparams.seed,
            world_size=world_size,
            rank=rank,
        )
        dataloader = LiteCruiseDataLoader(
            dataset=lite_dataset,
            batcher=batcher,
            batch_transform=batch_transform,
            device_transform=device_transform,
            **dataloader_config,
        )
        return dataloader

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        if not hasattr(self, "_train_dataloader"):
            self.train_dataloader()
        if state_dict and self._train_dataloader is not None:
            self._train_dataloader.__setstate__(state_dict)

    def state_dict(self):
        if self._train_dataloader is not None:
            return self._train_dataloader.__getstate__()
        return {}

    def train_dataloader(self):
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = self._dataloader_factory(
                "train"
            )  # keep it for save state dict
        return self._train_dataloader

    def val_dataloader(self):
        # TODO: sometimes we need multi-dataloaders for multi-datasets
        # so we need to return a list of dataloaders instead of a single dataloader with mixed datasets
        return self._dataloader_factory("val")

    def predict_dataloader(self):
        # TODO: predict dataset can not be constructed by dataset id and may not use parquet file.
        # redefine it here
        if not hasattr(self, "_predict_dataloader"):
            self._predict_dataloader = self._dataloader_factory("predict")
        return self._predict_dataloader

    def teardown(self, stage=None):
        if hasattr(self, "_train_dataloader"):
            logger.info(f"{__class__.__name__} teardown _train_dataloader")
            self._train_dataloader.terminate()
            self._train_dataloader = None
        if hasattr(self, "_val_dataloader"):
            logger.info(f"{__class__.__name__} teardown _val_dataloader")
            self._val_dataloader.terminate()
            self._val_dataloader = None
        if hasattr(self, "_predict_dataloader"):
            logger.info(f"{__class__.__name__} teardown _predict_dataloader")
            self._predict_dataloader.terminate()
            self._predict_dataloader = None


class LiteTransformCompose:
    def __init__(
        self, transforms: List[Callable], max_skip_rate: float = 1.0, log_interval=1000
    ) -> None:
        """A compose class for transforms."""
        self.transforms = transforms
        self.max_skip_rate = max_skip_rate
        self.log_interval = log_interval

        self.transform_total = 0
        self.transform_skip = 0
        self.profile = int(os.getenv("LITE_TRANSFORM_PROFILE", 0))
        self.profile_counter = defaultdict(float)

    def _check_skip_rate(self):
        skip_rate = self.transform_skip / self.transform_total
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            worker_num = worker_info.num_workers
        else:
            worker_id = 0
            worker_num = 1
        if skip_rate > self.max_skip_rate:
            logger.warning(
                f"Worker-{worker_id}/{worker_num}: Skip rate {skip_rate} > {self.max_skip_rate}, "
                f"skip {self.transform_skip} / {self.transform_total} samples"
            )
            if int(os.getenv("LITE_TRANSFORM_RAISE_SKIP_RATE", 0)) != 0:
                raise RuntimeError("Skip rate too high")

    def __call__(self, item: Dict[str, Any], **kwargs) -> Any:
        """note: kwargs will pass some info from dataloader"""
        self.transform_total += 1
        for t in self.transforms:
            try:
                _tik = time.perf_counter() if self.profile else 0
                item = t(item)
                _tok = time.perf_counter() if self.profile else 0
                transfrom_name = t.__class__.__name__
                if self.profile:
                    self.profile_counter[transfrom_name] += _tok - _tik

            except Exception as e:
                logger.debug(f"Exception in transform {t.__class__.__name__}: {e}")
                if int(os.getenv("LITE_TRANSFORM_DEBUG", 0)) != 0:
                    import traceback

                    traceback.print_exc()
                self.transform_skip += 1
                item = None
                break
        if self.transform_total % self.log_interval == 0:
            self._check_skip_rate()
            if self.profile:
                logger.info(f"Transform time cost: {self.profile_counter}")
        return item

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string


class LiteTransformComposeSpecificKey(LiteTransformCompose):
    def __init__(self, transforms=[], keys: str = "audio", **kwargs) -> None:
        if not isinstance(transforms, list):
            transforms = [transforms]
        super().__init__(transforms, **kwargs)
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys

    def __call__(self, item: Dict[str, Any], **kwargs) -> Any:
        for key in self.keys:
            if key in item:
                item[key] = super().__call__(item[key], **kwargs)
            else:
                logger.warning(f"key {key} not found in item")
        return item

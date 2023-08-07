from typing import Any, Dict, List

from cruise.data_module import DistributedCruiseDataLoader
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.utils import parse_data_urls, sort_data_sources


class ProcessorBase:
    r"""ProcessorBase class for :class:`UniDataModule`. The datamodule each step will
    call :func:`process` to transform a batch of training samples.

    BucketBatcher is embedded in the Processor, so all the batches are drawn from
    the batcher.

    Args:
        batcher_config (Dict): bucket batcher config.
    """

    def __init__(self, batcher_config: Dict = None):
        self.batcher = BucketBatcher(**batcher_config)

    def process_one(self, sample: Any):
        r"""Processing one training sample.

        Args:
            sample (Any): one training sample

        Returns:
            Any: processed sample
        """
        raise NotImplementedError

    def process(self, samples, collate_last=False):
        r"""Processing a list of processed samples by `self.transform`.

        Args:
            samples (List[Any]): a batch of processed samples
            collate_last (bool): whether collate remain buffer samples

        Returns:
            List[Any]: processed batch, could a list of tensors.

        """

        if collate_last:
            for batch in self.batcher.collect_last_batch():
                if batch:
                    yield self.post_processing(batch)

        for sample in samples:
            sample = self.process_one(sample)
            batch = self.batcher.collate_batch(sample)
            if batch is None:
                continue
            yield self.post_processing(batch)

    def post_processing(self, processed_batch):
        r"""In this function, user can do things like padding, clipping etc. to
        the batch. The default will be identity transform.

        Args:
            processed_batch (List[Any]): a list of output of `process_one`

        Returns:
            Any: post processed batch
        """
        return processed_batch

    batch_transform = process
    transform = None


class UniDataModule(LightningDataModule):
    r"""Unified datamodule for mix loading parquet/webdataset/kv. It will internally
    apply the proper loader based on the dataset format.

    User can apply this data module via yaml file

    .. code-block::yaml

        pl_datamodule:!new:samantha.dataio.datamodule.UniDataModule
            processor: !new:<a-subclass-of-processor-base>
                batcher_config: xxx
            train_data_path:
                - hdfs:://dataset/train/sub1/*.tar
                - hdfs:://dataset/train/sub2/*.tar
            valid_data_path:
                - hdfs:://dataset/valid/sub1/*.tar
                - hdfs:://dataset/valid/sub2/*.tar

    Args:
        processor (ProcessorBase): an object of subclass of :class:`ProcessorBase`.
        train_num_workers (int): number of training workers.
        valid_num_workers (int): number of valid workers.
        train_data_id (int): training dataset id registered in platform. the datamodule
            will retrieve the hdfs path using the id via sdk (`bytedance.easycycle`).
        valid_data_id (int): valid dataset id.
        train_data_path (List[str]): train dataset path, it can be a regex pattern,
            eg: hdfs:://dataset/*.tar. This must be mutually exclusive with
            `train_data_id`.
        valid_data_path (List[str]): valid dataset path, it can be a regex pattern.
            This must be mutually exclusive with `valid_data_id`.
        drop_last (bool): set to ``True`` to drop the last incomplete batch, if the
            dataset size is not divisible by the batch size.
        repeat (bool): repeat sampling the dataset or not.
        **kwargs: other dataloader related parameters. see
            :class:`DistributedCruiseDataLoader`.
    """

    def __init__(
        self,
        processor: ProcessorBase = None,
        train_num_workers: int = 4,
        valid_num_workers: int = 1,
        train_data_id: int = None,
        valid_data_id: int = None,
        train_data_path: List[str] = None,
        valid_data_path: List[str] = None,
        drop_last: bool = False,
        repeat: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.processor = processor
        self._parse_data_paths()
        self.kwargs = kwargs

    def _parse_data_paths(self):
        if self.hparams.train_data_id and self.hparams.train_data_path:
            raise MisconfigurationException(
                f"Combination of parameters train_data_id={self.hparams.train_data_id} "
                f"and train_data_path={self.hparams.train_dat_path} should be mutually "
                f"exclusive."
            )
        if self.hparams.valid_data_id and self.hparams.valid_data_path:
            raise MisconfigurationException(
                f"Combination of parameters valid_data_id={self.hparams.valid_data_id} "
                f"and valid_data_path={self.hparams.valid_dat_path} should be mutually "
                f"exclusive."
            )

        self.hparams.train_data_path = parse_data_urls(
            data_id=self.hparams.train_data_id, data_urls=self.hparams.train_data_path
        )

        if self.hparams.valid_data_path or self.hparams.valid_data_id:
            self.hparams.valid_data_path = parse_data_urls(
                data_id=self.hparams.valid_data_id,
                data_urls=self.hparams.valid_data_path,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.hparams.train_data_path is None:
            return None
        data_sources, data_types = sort_data_sources(self.hparams.train_data_path)
        loader = DistributedCruiseDataLoader(
            data_sources=data_sources,
            source_types=data_types,
            processor=self.processor,
            batch_sizes=[1] * len(data_sources),
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.train_num_workers,
            num_readers=[32] * len(data_sources),
            keys_or_columns=None,
            decode_fn_list=None,
            repeat=self.hparams.repeat,
            predefined_steps=-1,
            **self.kwargs,
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.hparams.valid_data_path is None:
            return None
        data_sources, data_types = sort_data_sources(self.hparams.valid_data_path)
        loader = DistributedCruiseDataLoader(
            data_sources=data_sources,
            source_types=data_types,
            processor=self.processor,
            batch_sizes=[1] * len(data_sources),
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.valid_num_workers,
            num_readers=[32] * len(data_sources),
            keys_or_columns=None,
            decode_fn_list=None,
            repeat=False,
            predefined_steps=-1,
            **self.kwargs,
        )
        return loader

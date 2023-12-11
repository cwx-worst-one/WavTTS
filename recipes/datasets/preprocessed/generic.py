from copy import deepcopy
from typing import Any, Dict, Iterable, Optional
import webdataset as wds
from recipes.datasets.base import (
    DataResult,
    WebDataModuleBase,
)
from samantha.dataio.data_bucket import data_bucket
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.transforms.audio import ToTensor
from samantha.utils.logger import RankedLogger
from samantha.utils.webdataset import return_self

from recipes.mi1.models.tagging import MI1_MusicTaggingInput

logger = RankedLogger(__name__, rank_zero_only=True)


class GenericPreprocessedDataModule(WebDataModuleBase):

    def __init__(
        self,
        sample_rate: int,
        train_url2index: str,
        validation_url2index: str,
        test_url2index: str,
        batch_size: int,
        shuffle_buffer_size: int,
        resampled: bool,
        shardshuffle: bool,
        num_workers: int = 8,
        pin_memory: bool = True,
        batch_size_valid: Optional[int] = None,
        batch_size_test: Optional[int] = None,
    ):
        self._sample_rate = sample_rate

        train_url2index = data_bucket(train_url2index)
        validation_url2index = data_bucket(validation_url2index)
        test_url2index = data_bucket(test_url2index)

        self.to_tensor = ToTensor()


        train_dataset = (
            IndexedWebDataset(
                url2index=train_url2index,
                resampled=resampled,
                shardshuffle=shardshuffle,
                use_pipe=False,
                nodesplitter=wds.shardlists.single_node_only if resampled else return_self,
            )
            .decode()
            .compose(self.transform)
        )

        validation_dataset = (
            IndexedWebDataset(
                url2index=validation_url2index,
                resampled=False,
                shardshuffle=False,
                use_pipe=False,
                nodesplitter=return_self,
            )
            .decode()
            .compose(self.transform)
        )

        test_dataset = (
            IndexedWebDataset(
                url2index=test_url2index,
                resampled=False,
                shardshuffle=False,
                use_pipe=False,
                nodesplitter=return_self,
            )
            .decode()
            .compose(self.transform)
        )
        
        predict_dataset = deepcopy(test_dataset)

        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size_valid=batch_size_valid,
            batch_size_test=batch_size_test,
        )

    def transform(self, items: Iterable[Dict[str, Any]]) -> Iterable[MI1_MusicTaggingInput]:
        for item in items:
            yield MI1_MusicTaggingInput(
                hidden_states=self.to_tensor(item["hidden_states.npy"]),
                tag=self.to_tensor(item["tag.npy"])
            )
from typing import Dict, List, Union

import webdataset as wds
from torch.utils.data import IterableDataset


class WebPipeline(IterableDataset):
    r"""WebPipeline is a tool to apply a sequence of webdataset transformations
    to a webdataset or any iterable dataset.

    WebPipeline supports defining pipeline stages using function name strings,
    which can simplify the pipeline declaration in yaml config files.

    .. code-block:: yaml

        pipeline: !new:samantha.dataio.webdataset.WebPipeline
            dataset: !ref <dataset>
            pipeline:
                - decode
                - shuffle: 100
                - to_tuple: "audio.npy label.txt"]
                - compose: !name:<custom_transform>
                - map: !name:<custom_fn>
                - batched: 32

    Supported stage specs of pipeline in :func:`__init__` are

    - str: wds_method
    - dict of element: {wds_method: args(optional), kwargs(optional)}

    .. code-block:: yaml

        - batched
        - batched: 32
        - batched: [32]
        - batched: dict(batchsize=32, collation_fn=<custom_fn>)

    Args:
        dataset (Union[WebDataset, Iterable]): target dataset for transformations
        pipeline: (List[Union[str, Dict]]): list of transformations to apply to
            the dataset
    """

    def __init__(
        self,
        dataset: Union[wds.WebDataset, IterableDataset],
        pipeline: List[Union[str, Dict]],
    ):
        super().__init__()
        if isinstance(dataset, wds.WebDataset):
            dataset = dataset
        else:
            dataset = wds.FluidWrapper(dataset)
        self.data_pipeline = self._build_pipeline(dataset, pipeline)

    def _apply_stage(self, dataset, func_name, args, kwargs):
        if not hasattr(dataset, func_name):
            # Check if func_name is a valid webdataset function
            raise ValueError(
                f"Invalid WebPipeline stage: {func_name}, `{func_name}` is not a "
                "valid webdataset function"
            )
        # Apply the transformation
        return getattr(dataset, func_name)(*args, **kwargs)

    def _build_pipeline(self, dataset, pipeline):
        for f in pipeline:
            func_name = ""
            args = []
            kwargs = {}
            if isinstance(f, str):
                # If f is a string, it is the name of a webdataset function
                func_name = f
            elif isinstance(f, dict):
                # If f is a dict, it is a mapping of function name to args
                func_name = list(f.keys())[0]
                vals = f[func_name]
                if isinstance(vals, list):
                    args = vals
                elif isinstance(vals, dict):
                    kwargs = vals
                else:
                    args = [vals]
            else:
                raise ValueError(
                    f"Invalid WebPipeline stage: {f}, must be a string or dict"
                )
            dataset = self._apply_stage(dataset, func_name, args, kwargs)
        return dataset

    def __iter__(self):
        return iter(self.data_pipeline)

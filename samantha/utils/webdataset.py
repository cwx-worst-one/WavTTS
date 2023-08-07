import functools
from typing import Any, List

from pytorch_lightning.utilities import rank_zero_deprecation
from webdataset import WebDataset


def return_self(x: Any) -> Any:
    return x


def apply_webdataset_pipeline(
    wds_dataset: WebDataset, pipeline: List[functools.partial]
):
    r"""Applies a sequence of webdataset transformations to a dataset.

    Args:
        wds_dataset (WebDataset): target dataset for transformations
        pipeline: (List[FluidInterface]): List of transformations. Each transformation
            in the list should be a ``Callable`` object composed of a `FluidInterface`
            class method bound to its args. For example,
            `functools.partial(FluidInterface.decode, 'rgb')`
            or, when defined in HyperPyYaml,
            `!name:webdataset.WebDataset.decode ["rgb"]`.
    """

    rank_zero_deprecation(
        "`apply_webdataset_pipeline` is deprecated from v0.3,"
        "Use `samantha.dataio.webdataset.WebPipeline` instead."
    )

    for f in pipeline:
        if isinstance(f, functools.partial):
            wds_dataset = f.func(wds_dataset, *f.args)
        else:
            wds_dataset = f(wds_dataset)

    return wds_dataset

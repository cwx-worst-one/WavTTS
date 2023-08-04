import io
import logging
import os
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union

import fsspec
import torch
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_warn

import samantha.utils.hdfs_helper as hh
from samantha.utils.distributed import rank_zero_first

logger = logging.getLogger(__name__)


def _check_ckpt_path(path):
    path = str(path)
    exists = True
    if hh.ishdfs(path):
        exists = hh.exists(path)
    else:
        fs = get_filesystem(path, skip_instance_cache=False)
        exists = fs.exists(path)
    if not exists:
        raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")


def _load(
    path_or_url: Union[IO, _PATH], map_location: _MAP_LOCATION_TYPE = None
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict
            specifying how to remap storage locations.
    """
    if not isinstance(path_or_url, (str, Path)):  # pragma: no cover
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)

    path_or_url = str(path_or_url)

    if path_or_url.startswith("http"):  # pragma: no cover
        return torch.hub.load_state_dict_from_url(
            str(path_or_url), map_location=map_location
        )

    if hh.ishdfs(path_or_url):
        local_path = os.path.basename(path_or_url)
        with rank_zero_first(is_global=False):
            if not os.path.exists(local_path):
                logger.info(f"Downloading checkpoint from {path_or_url}")
                if not hh.get(path_or_url, local_path):
                    raise ConnectionError(
                        f"Failed to download checkpoint from {path_or_url} to"
                        f" {local_path}"
                    )
        path_or_url = local_path

    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb", skip_instance_cache=True) as f:
        return torch.load(f, map_location=map_location)


def _atomic_save(checkpoint: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method,
            but can deal with anything which ``torch.save`` accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
    """
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer, pickle_protocol=4)
    with fsspec.open(filepath, "wb") as f:
        value = bytesbuffer.getvalue()
        # slice the buffer to 512M
        chunk_size = 1 << 29
        for i in range(0, len(value), chunk_size):
            f.write(value[i : i + chunk_size])


class LargeTorchCheckpointIO(CheckpointIO):
    r"""TorchCheckpointIO optimized for large model ckpt.

    The ``TorchCheckpointIO`` from pytorch_lightning fails to save
    large models(e.g. >1G)
    to HDFS and very large models(e.g.>4G) to local.

    .. note::

        The main issues are

        - ``torch.save`` default pickle protocol can't save a file larger than 4G
        - ``f.write(bytesbuffer.getvalue())`` exceed the max array size of
          HDFS java client

        In this implementation, we

        - explicitly use pickle protocol 4 to allow large file saving
        - slice ``bytebuffer`` into chunks before using ``f.write()``

    To use this customized checkpoint io, simply add it to your trainer.

    .. code-block:: yaml

        plugins:
            - !new:samantha.plugins.torch_io.LargeTorchCheckpointIO

        trainer: !new:pytorch_lightning.Trainer
            plugins: !ref <plugins>

    """

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` "
                "with `storage_options` arg is not supported for "
                f"`{self.__class__.__name__}`. Please implement your "
                "custom `CheckpointIO` to define how you'd like"
                "to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            _atomic_save(checkpoint, path)
        except AttributeError as err:  # pragma: no cover
            key = "hyper_parameters"
            checkpoint.pop(key, None)
            rank_zero_warn(
                f"Warning, `{key}` dropped from checkpoint. "
                f"An attribute is not picklable: {err}"
            )
            _atomic_save(checkpoint, path)

    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        _check_ckpt_path(path)
        return _load(path, map_location=map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            logger.debug(f"Removed checkpoint: {path}")

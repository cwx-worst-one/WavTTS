"""Core data pipeline module for torchspeech."""
import logging
from torch.utils.data import Dataset
from .datablock import DataBlock

LOG = logging.getLogger(__name__)

class DataPipe(Dataset):
    r"""TorchSpeech data pipeline.

    This implements PyTorch Dataset, but is intended for infinite length data
    sets. It does not support random indexing and by extension shuffling.
    To use with multiple workers, see DataPipeLoader.

    :type \*blocks: DataBlock
    :param \blocks: Sequence of DataBlocks that make up the pipeline. First \
            block item can be any iterable object.

    .. note:: The first data block in the pipeline is expected to generate
        records independently.  If it is subclassed from DataBlock it should
        not consume any items from its input iterator.  Otherwise it may be
        any generic iterable object.

    :Example:

    >>> import torchspeech.utils.data as td
    >>>
    >>> def multiply_by_ten(x):
    >>>    return 10*x
    >>>
    >>> pipe = DataPipe(
    >>>     range(5),
    >>>     td.Repeater(3),
    >>>     td.Transform(multiply_by_ten))
    >>>
    >>> for item in pipe:
    >>>    print(item)
    >>>
    0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40
    """

    def __init__(self, *blocks):

        self._blocks = list(blocks)
        self._last_block = None
        self._output_iterator = None

        # Check argument integrity
        for index, block in enumerate(self._blocks):
            if not isinstance(block, DataBlock) and index > 0:
                err = (
                    "DataPipe item #%d is not derived from DataBlock. Only "
                    "first item may be a generic iterator." % (index + 1)
                )

                raise ValueError(err)

            elif not hasattr(block, "__iter__"):
                err = "DataPipe item #%d is not iterable." % (index + 1)
                raise ValueError(err)

        # Connect blocks together
        self._last_block = []
        for block in self._blocks:
            if isinstance(block, DataBlock):
                block.set_input(self._last_block)

            self._last_block = block

    def blocks(self):
        """
        A generator for returning the blocks in the pipeline.
        """
        for block in self._blocks:
            yield block

    def __iter__(self):
        """
        :returns: an iterator for the data pipeline.  This will cause blocks
        in the pipeline to initialize.
        """
        self._output_iterator = iter(self._last_block)
        return self

    def __next__(self):
        """
        :returns: next item from the pipeline.
        """
        return next(self._output_iterator)

    def __len__(self):
        """
        Implementation for PyTorch dataset compatibility.  Returns infinite
        length place-holder.  The data pipe class has no fixed length.

        :returns: fixed value: 1e18
        """
        return 1000000000000000000

    def __getitem__(self, index):
        """
        Implementation for PyTorch dataset compatibility.  Return next item
        in the data pipe.  Random access on the pipe is not permitted and the
        index parameter is ignored.

        :returns: next item from the pipeline (ignores index).
        """
        if self._output_iterator is None:
            self._output_iterator = iter(self._last_block)

        return next(self)

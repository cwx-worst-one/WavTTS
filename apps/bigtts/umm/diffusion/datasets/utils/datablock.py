"""DataBlock base class modules."""

import logging

LOG = logging.getLogger(__name__)


class DataBlock():
    """
    Base class for TorchSpeech data processing pipeline blocks.
    
    Inherited classes should override the __next__ and reset methods to
    give specialized functionality.
    """

    def __init__(self):
        self._input_iterable = []
        self._input_iterator = iter([])

    def set_input(self, input_iterable):
        """
        Set the input iterable of this block.

        :type input_iterable: iterable
        :param input_iterable: A python iterable that this datablock will draw \
        its input samples from.
        :returns: None.
        """
        self._input_iterable = input_iterable

    def next_input(self):
        """
        Return next item from the input iterator.  If the input object was
        never set or it has run out of items, then StopIteration exeption
        will be thrown.

        :returns: Next item pulled from the upstream input block.
        """
        return next(self._input_iterator)

    def __iter__(self):
        """
        Return an iterator on this DataBlock. This resets the data block state
        and will invalidate any previous iterators returned by this object.
        Do not override this method.

        :returns: The initialized data block.
        """
        
        self.reset_input_iterator()
        LOG.debug('Create new iterator and reset state for block: %s', 
            self.__class__.__name__) 
        self.reset()
        return self

    def __next__(self):
        """
        Override in derived classes to reset internal state.  Default is 1-to-1
        passthrough of input iterator.

        :returns: Next item produced by this data block.
        """
        return self.next_input()

    def reset(self):
        """
        Optional override in derived classes to reset internal state. Lazy
        initialization can be put here.
        Default is no-op.
        """

    def reset_input_iterator(self):
        """
        Reset the input iterator.

        :returns: None
        """
        self._input_iterator = iter(self._input_iterable)


class DataBlockSharded(DataBlock):
    """
    Base class for data blocks that wish to have access to the worker rank and
    size. Rank and worker number is with respect to a single training process.
    """

    def __init__(self):
        super().__init__()
        self._worker_rank = 0
        self._worker_size = 1

    def set_shard_info(self, worker_rank, worker_size):
        """Set the worker information.

        :type worker_rank: int
        :param worker_rank: The workers rank.
        :type worker_size: int
        :param worker_size: Number of workers active for the training process. 
        :returns: None.
        """
        self._worker_rank = worker_rank
        self._worker_size = worker_size

    def worker_rank(self):
        """
        :returns: The worker rank.
        """
        return self._worker_rank

    def worker_size(self):
        """
        :returns: The number of active workers for the training process.
        """
        return self._worker_size

"""Package for common utility data blocks"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .datablock import DataBlock


class Filter(DataBlock):
    """
    A data block for filtering records via user supplied predicate. For
    complicated filtering scenarios, consider creating a custom data block
    class.

    *Input record format:* Any object.

    *Output record format:* Same as input format.

    :type predicate_fn: function
    :param predicate_fn: A function that returns a Boolean indicating whether \
    a record should be filtered. The function should be compatible with the \
    input records' format.
    
    :type max_gap: int
    :param max_gap: Maximum number of filterings to tolerate in a row. This \
    is used to prevent badly formatted predicates causing infinite looping.
    
    :Example:

    >>> def only_odd(x):
    >>>    return x % 2 == 1
    >>> 
    >>> pipe = DataPipe(
    >>>     range(100),
    >>>     Filter(only_odd))
    >>> 
    >>> print(list(pipe))
    [1,3,5,7...]
    """

    def __init__(self, predicate_fn, max_gap=10000):
        super().__init__()
        self._predicate_fn = predicate_fn
        self._max_gap = max_gap

    def __next__(self):
        """
        :returns: Next item that meets filter criteria.
        """

        if self._max_gap is None:
            while True:
                candidate = self.next_input()
                if self._predicate_fn(candidate):
                    return candidate

        for _ in range(self._max_gap):
            candidate = self.next_input()
            if self._predicate_fn(candidate):
                return candidate

        raise ValueError(
            'User filter has been applied %d times, but no items returned.' %
            self._max_gap)


class Transform(DataBlock):
    """
    A datablock for doing simple transformation via user supplied function.
    For more complicated transforms, consider creating a new data block class.

    *Input record format:* Any object.

    *Output record format:* Any object (determined by transform).

    :type predicate_fn: function
    :param predicate_fn: A function that returns a Boolean indicating whether \
    a record should be filtered. The function should be compatible with the \
    input records' format.
    :type keep nones: boolean
    :param keep_nones: Set to False to remove any 'None' item returned from \
    the transform.

    :Example:

    >>> def times_10(x):
    >>>     return x * 10
    >>> 
    >>> pipe = DataPipe(
    >>>     range(5),
    >>>     Transform(times10))
    >>> 
    >>> print(list(pipe))
    [0,10,20,30,40]
    """

    def __init__(self, transform_fn, keep_nones=True):
        super().__init__()
        self._transform_fn = transform_fn
        self._keep_nones = keep_nones

    def __next__(self):
        """
        :returns: Next transformed input item.
        """

        while True:
            candidate = self._transform_fn(self.next_input())
            if self._keep_nones or candidate is not None:
                return candidate


class CullFields(DataBlock):
    """
    A field removal data block. This can help python free up additional memory
    and / or reduce batching overhead by sending less data to be presented to
    the main training process.

    *Input record format:* String keyed dictionary.

    *Output record format:* String keyed dictionary.

    :type \*fields: Sequence of strings
    :param \*fields: Field names which should be removed.

    :Example:
     
    >>> pipe = DataPipe(
    >>>     ... elided ...
    >>>     FilterByUserDevice(column='UserDevice', ...),
    >>>     ExtractFeatures(audio='BinaryContent', ...),
    >>>     # Remove unecessary data before we pool up records
    >>>     CullFields('BinaryContent','UserDevice')
    >>>     RandomizeBuffer(10000))
    """

    def __init__(self, *fields):
        super().__init__()
        self._fields = set(fields)

    def __next__(self):
        """
        :returns: Copy of the next input record with specified fields removed.
        """
        item = self.next_input().copy()
        for field in self._fields:
            if field in item:
                del item[field]

        return item


class Repeater(DataBlock):
    """
    Data block for repeating finite iterators. When the input iterator runs
    out of items, it will be reset and re-iterated over. If any upstream block
    is an infinite iterator, this block will have no effect.

    *Input record format:* Any object.

    *Output record format:* Same as input format.

    :type count: int
    :param count: Number of times to repeat content from upstream data blocks. \
    Setting to zero will cause infinite repetitions.
    
    :Example:
    
    >>> pipe = DataPipe(
    >>>     range(5),
    >>>     Repeater(3)
    >>> 
    >>> print(list(pipe))
    [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
    """

    def __init__(self, count=0):
        super().__init__()
        self._max_count = count
        self._epoch = 0

    def __next__(self):
        """:returns: next item."""

        while True:
            try:
                return self.next_input()

            except StopIteration:
                self._epoch += 1
                if self._max_count > 0 and self._epoch >= self._max_count:
                    raise

            self.reset_input_iterator()

    def reset(self):
        """Resets epoch counter.
        
        :returns: None
        """
        self._epoch = 0


class RandomizeBuffer(DataBlock):
    """
    A data block for drawing samples randomly from a buffer. A large pool of
    items is read from the input. When an item is requested it is randomly
    sampled from this pool.

    *Input record format:* Any object.

    *Output record format:* Same as input format.

    :type buffer_len: int
    :param buffer_len: Size of the randomization pool. Larger pools better \
    approximate global randomization but at the cost of increased memory.

    """

    def __init__(self, buffer_len):
        super().__init__()
        self._buffer_len = buffer_len
        self._buffer = []

    def __next__(self):
        """
        :returns: next item sampled from the buffer. This will attempt to 
        refill the buffer with a new item from the input. 
        """

        try:
            while len(self._buffer) < self._buffer_len:
                self._buffer.append(self.next_input())

        except StopIteration:
            if not self._buffer:
                raise

        next_index = np.random.randint(len(self._buffer))
        next_item = self._buffer[next_index]
        self._buffer[next_index] = self._buffer[-1]
        del self._buffer[-1]

        return next_item


class PrefetchBuffer(DataBlock):
    """
    Data block for prefetching. Prefetch works by processing items in advance
    on a separate thread.  It can be used to hide latency or computation
    that comes in spikes.

    *Input record format:* Any object.

    *Output record format:* Same as input format.

    :type buffer_size: int
    :param buffer_size: Number of items to prefetch at a time.
    :type unroll: boolean
    :param unroll: When batch_size > 1 will individual items one at a time \
    rather than as a bulk batch.
    :type post_fetch_op: function
    :param post_fetch_op: function to apply to returned pre-fetch. This \
    operation will be run on background thread and should be thread-safe. \
    The function should take a pre-fetch as input and return a desired output \
    item. No-op if None.

    :Example:
    
    >>> # Prefetch network IO on background thread so we don't starve the pipe.
    >>> pipe = DataPipe(... elided ...
    >>>                 ReadNetworkFileIntoMemory(...),
    >>>                 PrefetchBuffer())
    """

    def __init__(self, buffer_size=1, unroll=False, post_fetch_op=None):
        super().__init__()
        self._buffer_size = max(1, buffer_size)
        self._prefetch = None
        self._executor = None
        self._post_fetch_op = post_fetch_op
        self._unroll = unroll
        self._unroll_iter = iter([])

        if buffer_size <= 1:
            self._unroll = False

    def __next__(self):
        """
        :returns: next item.
        """

        if self._prefetch is None:
            self._queue_next_prefetch()

        if self._unroll:
            while True:
                try:
                    return next(self._unroll_iter)
                except StopIteration:
                    pass

                self._unroll_iter = iter(self._queue_next_prefetch())
        else:

            return self._queue_next_prefetch()

    def _fetch_op(self):
        if self._buffer_size <= 1:
            result = self.next_input()
        else:
            result = [None] * self._buffer_size
            for i in range(self._buffer_size):
                result[i] = self.next_input()

        if self._post_fetch_op is not None:
            return self._post_fetch_op(result)

        return result

    def _queue_next_prefetch(self):
        """
        Enqueues a new prefetch. If there was an existing prefetch thread
        return its result.
        """
        if self._prefetch is None:
            prefetch_result = None
        else:
            prefetch_result = self._prefetch.result()

        self._prefetch = self._executor.submit(self._fetch_op)
        return prefetch_result

    def reset(self):
        """
        Reset thread pool executor. Any previous pre-fetch is lost.
        
        :returns: None.
        """
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch = None

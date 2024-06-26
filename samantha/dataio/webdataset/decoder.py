import re
from io import BytesIO

from webdataset.autodecode import Continue, decoders


def gzfilter(key, data, sample_key, sample):
    """Decode .gz files.

    This decodes compressed files and the continues decoding.

    :param key: file name extension
    :param data: binary data
    """
    import gzip

    if not key.endswith(".gz"):
        return None
    decompressed = gzip.open(BytesIO(data)).read()
    return Continue(key[:-3], decompressed)


def basichandlers(key, data, sample_key, sample):
    """Handle basic file decoding.

    This function is usually part of the post= decoders.
    This handles the following forms of decoding:

    - txt -> unicode string
    - cls cls2 class count index inx id -> int
    - json jsn -> JSON decoding
    - pyd pickle -> pickle decoding
    - pth -> torch.loads
    - ten tenbin -> fast tensor loading
    - mp messagepack msg -> messagepack decoding
    - npy -> Python NPY decoding

    :param key: file name extension
    :param data: binary data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)

    if extension in decoders:
        return decoders[extension](data)

    return None


class DecoderWithSampleKey:
    """Decode samples using a list of handlers.
    NOTE: this adds the `__key__` variable for each decoding function,
    which can be useful if it's required for decoding.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, handlers, pre=None, post=None, only=None, partial=False):
        """Create a Decoder.

        :param handlers: main list of handlers
        :param pre: handlers called before the main list (.gz handler by default)
        :param post: handlers called after the main list (default handlers by default)
        :param only: a list of extensions; when give, only ignores files with those extensions
        :param partial: allow partial decoding (i.e., don't decode fields that aren't of type bytes)
        """
        if isinstance(only, str):
            only = only.split()
        self.only = only if only is None else set(only)
        if pre is None:
            pre = [gzfilter]
        if post is None:
            post = [basichandlers]
        assert all(callable(h) for h in handlers), f"one of {handlers} not callable"
        assert all(callable(h) for h in pre), f"one of {pre} not callable"
        assert all(callable(h) for h in post), f"one of {post} not callable"
        self.handlers = pre + handlers + post
        self.partial = partial

    def decode1(self, key, data, sample_key, sample):
        """Decode a single field of a sample.

        :param key: file name extension
        :param data: binary data
        :param sample_key: key of the data sample (unique in tar)
        """
        key = "." + key
        for f in self.handlers:
            result = f(key, data, sample_key, sample)
            if isinstance(result, Continue):
                key, data = result.key, result.data
                continue
            if result is not None:
                return result
        return data

    def decode(self, sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        sample_key = sample["__key__"]  # retrieve the sample key first
        for k, v in list(sample.items()):
            if k[0:2] == "__":
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8")
                    except Exception as e:
                        print(f"Can't decode v of k = {k} as utf-8: v = {v} ({e})")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v, sample_key, sample)
                else:
                    result[k] = v
            else:
                # assert isinstance(v, bytes), f"k,v = {k}, {v}" # TODO: commented for Parquet compat...
                result[k] = self.decode1(k, v, sample_key, sample)
        return result

    def __call__(self, sample):
        """Decode an entire sample.

        :param sample: the sample
        """
        assert isinstance(sample, dict), (len(sample), sample)
        return self.decode(sample)

'''
binary data's parser.
supported: pickle, protobuf.
'''

import pickle
import numpy as np
from dataloader import ParseLoads
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class PickleParser:
    """parse serialized bytes to python object."""

    def __call__(self, binary_data, **_kwargs):
        '''unserialize binary data by pickle.'''
        if binary_data is None:
            return None

        try:
            # pickle will try to import 3rd party library,
            # in order to construct python object.
            # But it may fail, when some libraries can't be founded.
            item = pickle.loads(binary_data)
            item['file_type'] = 'tensorbundle'
            return item
        except Exception:
            return binary_data


@PREPROCESS.register_module()
class NumpyParser:
    """parse serialized bytes to python object."""

    def __init__(self, dtype='np.int16', out_key='wav'):
        '''init.'''
        self.dtype = eval(dtype)
        self.out_key = out_key

    def __call__(self, binary_data, **kwargs):
        '''unserialize binary data by pickle.'''
        if binary_data is None:
            return None
        try:
            item_out = dict()
            item = np.frombuffer(bytes(binary_data), dtype=self.dtype)
            item_out[self.out_key] = item
            item_out['file_type'] = 'tensorbundle'
            return item_out
        except Exception:
            return binary_data


@PREPROCESS.register_module()
class ProtoParser:
    """parse serialized bytes to python object."""

    def __call__(self, binary_data, **kwargs):
        '''unserialize binary data by Protobuf.'''
        if binary_data is None:
            return None
        try:
            parse_data = ParseLoads(binary_data)
            item = dict()
            for data in parse_data:
                name = bytes.decode(data[0])
                feature = data[1]
                item[name] = feature
            item['file_type'] = 'tfrecord'
            return item
        except Exception:
            return binary_data


@PREPROCESS.register_module()
class DecordRaw:
    """
        decode raw data
        Supported data types use @PREPROCESS.register_module()
    class properties.
        Use set to store, because the search efficiency is high
    """

    SUPPORTED_DTYPE = set(['int16', 'float32', 'bytes'])

    def __init__(self, key2type):
        '''
        key2type is a dict of key to type ,such as 'frames' to 'int16'
        used to decode raw data of key to wanted type
        key must in the raw data
        now it only support int16, float32, bytes
        '''
        self.key2type = key2type
        for d_type in key2type.values():
            assert d_type in DecordRaw.SUPPORTED_DTYPE

    def __call__(self, item, **_kwargs):
        '''decode data in item'''

        if item.get('file_type', 'tensorbundle') == 'tensorbundle':
            return item

        if self.key2type is None or item is None:
            return item

        for name, d_type in self.key2type.items():
            if name not in item:
                continue
            feature = item.pop(name)
            if d_type == 'int16':
                item[name] = np.fromstring(feature, dtype=np.int16)
            elif d_type == 'float32':
                item[name] = np.fromstring(feature, dtype=np.float32)
            elif d_type == 'bytes':
                item[name] = bytes.decode(feature)
        return item

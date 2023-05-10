'''
decompress data.

HDFS IO is heavy, we'll use more compresss tool to reduce HDFS file size.
'''
try:
    import lilcom
except ImportError:
    lilcom = None
from .preprocess import PREPROCESS


@PREPROCESS.register_module()
class LilcomDecompress:
    '''
    lilcom is used by kaldi io to compress fbank.
    contributed by liuyi.ai.
    '''

    def __init__(self, in_key='src', out_key='src'):
        '''init.'''
        if lilcom is None:
            raise ImportError('Dolphin preprocess LilcomDecompress: cannot import lilcom')
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **_kwargs):
        '''
        do call.
        lilcom.decompress take a byte_string as input,
        and output a float numpy array if success, otherwise raise Exception.
        '''
        if item is None or self.in_key not in item:
            return None
        # The item should contain a key claiming the compress method.
        if 'compress' not in item or item['compress'] != 'lilcom':
            return item
        src = item.pop(self.in_key, None)
        src = lilcom.decompress(src)
        item[self.out_key] = src
        return item

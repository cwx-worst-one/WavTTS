'''
unit test of get_meta
'''
from core.dataset import get_meta


def _test(path=None):
    '''
    test get meta
    '''
    if path is None:
        path = (
            'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/'
            'datasets/dolphin/librispeech_wav/meta'
        )
    metas = get_meta(path)
    assert metas is not None


if __name__ == "__main__":
    _test()

'''test cmvn by python and by C++'''
import time
import numpy as np
from core.dataset.preprocess import fbank


def _get_data():
    '''get data'''
    data = {'src': np.random.rand(400, 80).astype(np.float32)}
    return data


def test_cmvn():
    '''test cmvn'''
    data = _get_data()
    start = time.time()
    fn = fbank.DynamicCmvn(
        key='src',
        cmvn_type='sliding_dynamic_C',
        norm_var=False,
        center=False,
        min_cmn_window=100,
        cmn_window=300,
    )
    for _ in range(10):
        out1 = fn(data)
    time1 = time.time() - start
    start = time.time()
    fn1 = fbank.DynamicCmvn(
        key='src',
        cmvn_type='sliding_dynamic',
        norm_var=False,
        center=False,
        min_cmn_window=100,
        cmn_window=300,
    )
    for _ in range(10):
        out2 = fn1(data)
    time2 = time.time() - start
    assert out1['src'].shape == out2['src'].shape and out1['src'].dtype == out2['src'].dtype
    assert np.allclose(out1['src'], out2['src'], rtol=1e-05, atol=1e-08, equal_nan=False)
    print("time1: {} \t time2: {}\n".format(time1, time2))

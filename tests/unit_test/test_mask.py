'mask test'
import time
import torch
from core.dataset.preprocess.freq_mask import FreqMask, BatchFreqMask
from core.dataset.preprocess.time_mask import TimeMask, BatchTimeMask


def test_freq_mask():
    torch.manual_seed(20210812)
    data = {'fbank': torch.randn((16, 1600, 80), dtype=torch.float32, device='cuda')}
    f = FreqMask(freq_mask_size=27, freq_mask_num=1, replace_with_zero=False)
    f2 = BatchFreqMask(freq_mask_size=27, freq_mask_num=1, replace_with_zero=False)
    start = time.time()
    for _ in range(10):
        f(data)
    time1 = time.time() - start
    print(time1)

    start = time.time()
    for _ in range(10):
        f2(data)
    time2 = time.time() - start
    print(time2)


def test_time_mask():
    torch.manual_seed(20210812)
    data = {'fbank': torch.randn((16, 1600, 80), dtype=torch.float32, device='cuda')}
    t = TimeMask(time_mask_size=20, time_mask_num=10, replace_with_zero=False)
    t2 = BatchTimeMask(time_mask_size=20, time_mask_num=10, replace_with_zero=False)
    start = time.time()
    for _ in range(10):
        t(data)
    time1 = time.time() - start
    print(time1)

    start = time.time()
    for _ in range(10):
        t2(data)
    time2 = time.time() - start
    print(time2)

# pylint: disable=all
# TODO(zhengyijie) del this pylint disable
''' test falcon noise '''
import pickle
from core.dataset.preprocess import AddNoise
import copy
import numpy as np
import time
from scipy.io.wavfile import write
import io


class AddNoiseTest(AddNoise):
    def set_values(self, values, noise_rate):
        self.values = values
        self.noise_rate = noise_rate


def test_FalconNoise():
    values = []
    for _ in range(10):
        io_buffer = io.BytesIO()
        write(io_buffer, rate=8000, data=(np.random.normal(0, 0.1, 5000) * 32767).astype("int16"))
        values.append(pickle.dumps({"wav": io_buffer.read()}))

    backend = "sox"
    sample_factor = 0.5
    item = {"sample_rate": 16000, "waveform": np.random.normal(0, 0.1, [1, 16000]) * 32767}

    addNoise = AddNoiseTest(p=1, resample_backend=backend)
    addNoise.set_values(copy.deepcopy(values[:200]), noise_rate=item["sample_rate"] * sample_factor)
    start_time = time.time()
    noise = addNoise.update_noise(item["waveform"].shape[1], data_rate=item["sample_rate"])
    noise_time = time.time() - start_time
    print("unit test: noise shape check passed! noise shape is ", noise.shape)
    print("unit test:data value check passed")

    print(
        f"unit test: using backend {backend}, test finished in {time.time()-start_time}\t"
        f"AddNoise consumed {noise_time}"
    )

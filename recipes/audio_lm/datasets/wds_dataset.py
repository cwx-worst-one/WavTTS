import numpy as np
import webdataset as wds
from pydub import AudioSegment
from torch.utils.data import IterableDataset


class AudioLMWdsDataset(IterableDataset):
    def __init__(
        self,
        hdfs_pattern,
        number_shards,
        sample_rate=24000,
        sample_duration=10,
        batch_size=1,
        mono=True,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.segment_size = self.durations * self.sample_rate
        self.mono = mono
        urls = [f"pipe: hdfs dfs -cat {hdfs_pattern % i}" for i in range(number_shards)]
        self.dataset = (
            wds.WebDataset(
                urls=urls, shardshuffle=True, detshuffle=True, resampled=True
            )
            .decode()
            .map(self._filter_by_length)
            .map(self._resample_and_downmix)
            .map(self._random_slice)
            .to_tuple()
            .batched(batch_size)
        )

    def __iter__(self):
        return iter(self.dataset)

    def _filter_by_length(self, item):
        audio = item["npy"]
        audio_len = audio.shape[1]
        if (audio_len >= self.segment_size - 0.05 * self.sample_rate) and (
            audio_len <= self.sample_rate * 360
        ):
            return item

    def _resample_and_downmix(self, item):

        src_sr = item["json"]["sr"]
        src_nc = item["json"]["nc"]
        tgt_nc = 1 if self.mono else src_nc
        if src_sr == self.sample_rate and src_nc == tgt_nc:
            return item["npy"]

        if src_sr != self.sample_rate:
            audio = AudioSegment(
                data=item["npy"], sample_width=2, frame_rate=src_sr, channels=src_nc
            )
            audio = audio.set_channels(1 if self.mono else src_nc).set_frame_rate(
                self.sample_rate
            )
            audio = np.asarray(audio.get_array_of_samples())
            return audio

    def _random_slice(self, audio):
        audio_len = audio.shape[1]
        if audio_len < self.segment_size - 0.05 * self.sample_rate:
            return
        if audio_len < self.segment_size:
            audio = np.pad(audio, ((0, 0), (0, self.segment_size - audio_len)))
        else:
            beg = np.random.randint(low=0, high=audio_len - self.segment_size + 1)
            audio = audio[:, beg : beg + self.segment_size]

        audio = audio.astype(np.float32) / 32768.0
        if np.sqrt(np.mean(audio**2)) <= 1e-4:
            return
        return audio.reshape((-1)) if self.mono else audio


if __name__ == "__main__":

    import copy

    import psutil
    import torch.utils.data
    from tqdm import tqdm

    def profile_memory(iterator):
        count = 0
        initial = psutil.virtual_memory().used
        print(f"Begin: initial memory {initial >> 20} MiB")
        for _ in iterator:
            count += 1
            if count % 1024 == 0:
                print(
                    f"iterate {count} samples, cost memory "
                    f"{(psutil.virtual_memory().used - initial) >> 20} MiB"
                )
        print(
            f"End: iterate {count} samples, cost memory "
            f"{(psutil.virtual_memory().used - initial) >> 20} MiB"
        )

    hdfs_pattern = "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/wds_pool/15m_filtered/part_0/%05d.tar"  # noqa
    dataset = AudioLMWdsDataset(
        hdfs_pattern=hdfs_pattern, number_shards=100, batch_size=1
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=copy.deepcopy(dataset), batch_size=None
    )

    # profile_memory(iter(dataset))
    # profile_memory(dataloader)
    for idx, _ in enumerate(tqdm(dataloader)):
        if idx > 10240:
            break

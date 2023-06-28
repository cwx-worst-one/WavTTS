import webdataset as wds
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(
        self,
        hdfs_urls,
        sample_rate=24000,
        sample_duration=10,
        batch_size=8,
        mono=True,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.segment_size = self.durations * self.sample_rate
        self.mono = mono
        urls = f"pipe: hdfs dfs -cat {hdfs_urls}"
        self.dataset = (
            wds.WebDataset(
                urls=urls, shardshuffle=True, detshuffle=True, resampled=True
            )
            .decode() # 根据已知的后缀名或类型进行解码
            .map(self._resample_and_downmix) # 应用自定义处理函数
            .map(self._random_slice) # 应用自定义处理函数
            .batched(batch_size) # 组batch
        )

    def __iter__(self):
        return iter(self.dataset)

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
    pattern = "hdfs://output/%05d.tar"
    dataset = Dataset(hdfs_pattern=pattern, number_shards=100)
    for batch in dataset:
        pass
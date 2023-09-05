import webdataset as wds
from torch.utils.data import IterableDataset

import recipes.t5_mulan.dataset.utils as utils


MCC_N2M_URLS = [
    f"pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/jingsong.gao/mulan_precompute/t5/n2m/mcc_n2m_{idx:04d}.tar"
    for idx in range(6379)
]


class MCCN2MDatasetApril(IterableDataset):
    def __init__(self, seq_len=250, **kwargs):
        self.dataset = (
            wds.WebDataset(MCC_N2M_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(utils.pad_seq_embeds(seq_len=seq_len))
            .map(self._process)
        )

    def _process(self, data):
        data["music_id"] = data["meta.json"]["music_id"]
        del data["meta.json"]
        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = MCCN2MDatasetApril()
    for sample in dataset:
        print(sample.keys())
        for k in ['audio', 'text_embeds', 'text_embeds_mask']:
            print(k, sample[k].shape)
        print("music_id", type(sample["music_id"]), sample["music_id"])
        break

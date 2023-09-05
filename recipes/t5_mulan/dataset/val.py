import webdataset as wds
from torch.utils.data import IterableDataset

import recipes.t5_mulan.dataset.utils as utils


VAL_PATH = "pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/jingsong.gao/mulan_precompute/t5/val/kaggle_val.tar"

class KaggleValDataset(IterableDataset):
    def __init__(self, seq_len=250, **kwargs):
        self.dataset = (
            wds.WebDataset(VAL_PATH, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(utils.pad_seq_embeds(
                text_field="text_embeds.npy",
                text_output_field="text_embeds",
                seq_len=seq_len,
            ))
            .map(utils.pad_seq_embeds(
                text_field="aspect_list_embeds.npy",
                text_output_field="aspect_list_embeds",
                seq_len=seq_len,
            ))
            .map(self._process)
        )
    
    def _process(self, data):
        data["music_id"] = utils.fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = KaggleValDataset()
    for sample in dataset:
        print(sample.keys())
        for k in ['audio', 'text_embeds', 'text_embeds_mask', 'aspect_list_embeds', 'aspect_list_embeds_mask']:
            print(k, sample[k].shape)
        print("music_id", type(sample["music_id"]), sample["music_id"])
        break

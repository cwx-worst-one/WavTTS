import os
import sys
import glob
import tqdm
import random
import librosa
import warnings
import numpy as np
from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder

warnings.filterwarnings("ignore")


class FMADataset(AbstractWebDatasetGeneratorBasedBuilder):
    """
    FMA Dataset
    """

    SAMPLING_RATE = 24000
    SAMPLE_LENGTH = 24000 * 30
    MAX_LENGTH = 24000 * 300
    AUDIO_NUMPY_DTYPE = np.float32

    def __init__(self, current_shard, shard_size=10000, start_point=0) -> None:
        super().__init__()
        self.current_shard = current_shard
        self.shard_size = shard_size
        self.start_point = start_point
        self.dataset_name = "fma_24kHz_%d_%d" % (current_shard, start_point)
        
    def _split_data(self, data_dir):
        # random shuffle and get keys
        random.seed(42)
        # fl = glob.glob("/mnt/bn/transcription/fma/fma_full/*/*.mp3")
        fl = glob.glob("/mnt/bn/transcription-49c6ce75/fma/fma_full/*/*.mp3")
        random.shuffle(fl)
        sub_fl = fl[self.current_shard * self.shard_size + self.start_point: (self.current_shard + 1) * self.shard_size]
        keys = [os.path.basename(fn)[:-4] for fn in sub_fl]
        return {"train": {"keys": [keys, sub_fl]}}

    @property
    def DATASET_NAME(self):
        return self.dataset_name

    def _load_example(self, key, path):
        return_dict = []
        np_audio, _ = librosa.load(path, sr=FMADataset.SAMPLING_RATE)
        np_audio = np_audio[:FMADataset.MAX_LENGTH]
        if len(np_audio) < FMADataset.SAMPLE_LENGTH: # pad audio
            sliced_audio = np.pad(np_audio, [0, (FMADataset.SAMPLE_LENGTH - len(np_audio))])
            sliced_dict = {
                "audio.npy": sliced_audio,
                "__key__": key + "_0",
                "dataset.txt": "FMA_24kHz"
            }
            return_dict.append(sliced_dict)
        else:
            num_chunk, remainder = divmod(len(np_audio), FMADataset.SAMPLE_LENGTH)
            for i in range(num_chunk):
                sliced_audio = np_audio[i*FMADataset.SAMPLE_LENGTH:(i+1)*FMADataset.SAMPLE_LENGTH]
                sliced_dict = {
                    "audio.npy": sliced_audio,
                    "__key__": key + "_%d" % i,
                    "dataset.txt": "FMA_24kHz"
                }
                return_dict.append(sliced_dict)
            if remainder > 0:
                sliced_audio = np_audio[-FMADataset.SAMPLE_LENGTH:]
                sliced_dict = {
                    "audio.npy": sliced_audio,
                    "__key__": key + "_%d" % num_chunk,
                    "dataset.txt": "FMA_24kHz"
                }
                return_dict.append(sliced_dict)
        return return_dict



    def _generate_examples(self, keys):
        keys, paths = keys
        for key, path in tqdm.tqdm(zip(keys, paths)):
            sliced_dicts = self._load_example(key, path)
            for single_dict in sliced_dicts:
                yield single_dict


if __name__ == "__main__":
    current_shard = int(sys.argv[1])
    shard_size = int(sys.argv[2])
    start_point = int(sys.argv[3])
    ds = FMADataset(current_shard, shard_size, start_point)
    ds.create_dataset()

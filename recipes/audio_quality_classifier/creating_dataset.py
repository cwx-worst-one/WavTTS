import webdataset as wds
import soundfile as sf
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader
from recipes.musiclm.datasets.mcc import WrappedMCC40MDataset

dataset = WrappedMCC40MDataset(
    url2index_list=[
       '/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy/npy_url2idx.txt',
    ],
    weights=[
        1.0,
    ],
    sample_rate=24000,
    duration=2,
    audio_key="audio.npy",
    min_volume_threshold=0.05,
    loudness_ratio_threshold=0.2,
    ar_filtering=None,
    max_num_crops=3,
    avoid_vocal=True,
    max_vocal_threshold=0.5,
    audio_metrics_filtered=False,
    resampled=True,
    shardshuffle=True,
    seed=1
)

dataset_batched = DataPipeline(
            dataset,
            wds.shuffle(256), #256,
            wds.to_tuple("audio"),
            wds.batched(64),
)
dataloder = DataLoader(dataset_batched, batch_size=None, num_workers=8)

cnt = 0

for batch in dataloder:
    for wav in batch[0]:
        sf.write(f"mcc40m_segments/{cnt}.wav", wav.numpy().T, 24000)
        cnt += 1
    if cnt > 1000000:
        break
    



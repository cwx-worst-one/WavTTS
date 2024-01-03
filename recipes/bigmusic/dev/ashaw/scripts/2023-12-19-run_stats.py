import os

import pytorch_lightning as pl

import torch

os.chdir('/mnt/bn/ashaw-us/repos/samantha')

from recipes.bigmusic.datasets.lyrics import LyricsDataset, LyricsDataModule, DefaultDatasets, transform_dataset

from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt

from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform
from samantha.dataio.webdataset.extension import IndexedWebDataset
from recipes.bigmusic.datasets.transforms.lyrics_segment import is_valid_lyrics, extract_metadata_and_utterances, lyrics_to_segments
from recipes.bigmusic.datasets.lyrics import LyricsDataset

from recipes.bigmusic.datasets.index_lists import INDEX
from IPython.display import Audio
import webdataset as wds
from recipes.bigmusic.datasets.transforms.lyrics_segment import _words_to_segment, lyrics_to_segments

from recipes.bigmusic.datasets.lyrics import LyricsDataset, LyricsDataModule
ldm = LyricsDataModule.from_dataset_type(
    dataset_types="mcc60m_vocalB_style_mixed_tag_audio_album_date",
    sample_duration=[20,30], batch_size=4, 
    num_workers=8, 
)

tdl = ldm.train_dataloader()

it = iter(tdl)

from tqdm import tqdm

for item in tqdm(it):
    pass
    

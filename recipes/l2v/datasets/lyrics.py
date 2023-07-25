from typing import Callable, List, Optional
import torch
import webdataset as wds
import pytorch_lightning as pl
# from recipes.musiclm.transforms.musiclm import MCCTransforms
from recipes.l2v.datasets.transforms.lyrics import LyricsTransforms, LyricsTokenTransform, LyricsChromaTransform
from recipes.musiclm.preprocess import WebDatasetBufferPreprocessor
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader
from samantha.dataio.dataset import MultiIterableDataset
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import phoneme_padding_value
from recipes.musiclm.datamodules.webdataset import return_self
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer

default_lyrics_max_seq_len = 150
default_audio_max_seq_len = 24000*10

def pad_batch(batch, padding_value=0):
    return torch.nn.utils.rnn.pad_sequence([torch.as_tensor(t) for t in batch], batch_first=True, padding_value=padding_value)

def pad_to_seq_length(batch, seq_len, dtype, padding_value=0):
    batch_pad = torch.full((len(batch), seq_len), fill_value=padding_value, dtype=dtype)
    for idx, item in enumerate(batch):
        batch_pad[idx, :len(item)] = torch.as_tensor(item[:seq_len])
    return batch_pad

class LyricsCollator(object):
    def __init__(self, lyrics_padding_value=phoneme_padding_value, lyrics_max_seq_len=default_lyrics_max_seq_len, audio_max_seq_len=default_audio_max_seq_len):
        self.lyrics_padding_value = lyrics_padding_value
        self.lyrics_max_seq_len = lyrics_max_seq_len
        self.audio_max_seq_len = audio_max_seq_len

    def __call__(self, batches):
        return lyrics_collate(batches, self.lyrics_padding_value, 
                              lyrics_max_seq_len=self.lyrics_max_seq_len, 
                              audio_max_seq_len=self.audio_max_seq_len)

def lyrics_collate(batches, lyrics_padding_value, lyrics_max_seq_len=None, audio_max_seq_len=None):
    item_pairs = [
        tuple([item.get(item_key) for item_key in ['mulan_audio', 'target_audio', 'vocal_audio', 'lyrics', 'lyrics_tokens', 'mulan_text', 'vocal_chroma']])
        for item in batches]
    mulan_audio, target_audio, vocal_audio, lyrics, tokens, mulan_text, vocal_chroma = zip(*item_pairs)

    def _pad_audio(audio):
        if audio[0] is not None:
            if audio_max_seq_len is not None:
                return pad_to_seq_length(audio, audio_max_seq_len, dtype=torch.float, padding_value=0)
            else:
                return pad_batch(audio, padding_value=0)
        else:
            return None

    mulan_audio = _pad_audio(mulan_audio)
    target_audio = _pad_audio(target_audio)
    vocal_audio = _pad_audio(vocal_audio)

    if vocal_chroma[0] is not None:
        vocal_chroma = torch.stack([torch.as_tensor(vc) for vc in vocal_chroma])
    else:
        vocal_chroma = None

    if tokens[0] is not None:
        if lyrics_max_seq_len is not None:
            tokens = pad_to_seq_length(tokens, lyrics_max_seq_len, dtype=torch.long, padding_value=lyrics_padding_value)
        else: 
            tokens = pad_batch(tokens, padding_value=lyrics_padding_value)
    else:
        tokens = None
    return {
        'mulan_audio': mulan_audio,
        'target_audio': target_audio,
        'vocal_audio': vocal_audio,
        'vocal_chroma': vocal_chroma,
        'lyrics_tokens': tokens,
        'lyrics': lyrics,
        'mulan_text': mulan_text,
    }

class WavCollator(object):
    "Converts l2v dataloader to work with musiclm models"
    def __init__(self, audio_max_seq_len=None):
        self.audio_max_seq_len = audio_max_seq_len

    def __call__(self, batches):
        wavs = [item['target_audio'] for item in batches]
        return { 'audio': wav_collate(wavs, audio_max_seq_len=self.audio_max_seq_len) }

class InstrumentalCollator(WavCollator):
    "Converts musiclm dataloader to work with l2v models"
    def __call__(self, batches):
        wavs = [item['audio'] for item in batches]
        wavs = wav_collate(wavs, audio_max_seq_len=self.audio_max_seq_len)
        return {
            'mulan_audio': wavs,
            'target_audio': wavs,
        }

def wav_collate(wavs, audio_max_seq_len=None):
    wavs = [wav if len(wav.shape) == 1 else wav[0] for wav in wavs]
    if audio_max_seq_len is not None:
        wavs = pad_to_seq_length(wavs, audio_max_seq_len, dtype=torch.float, padding_value=0)
    else:
        wavs = pad_batch(wavs, padding_value=0)
    return wavs

class LyricsDataset(WebPipeline):
    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        sample_duration: float,
        segment_transforms,
        audio_keys: dict,
        audio_format: str = "mp3",
        max_num_segments: int = None,
        handler: Callable = wds.warn_and_continue,
        **kwargs,
    ):
        dataset = IndexedWebDataset(
            url2index=url2index,
            handler=handler,
            **kwargs,
        )
        audio_transforms = LyricsTransforms(
            sample_rate=sample_rate,
            sample_duration=sample_duration,
            audio_keys=audio_keys,
            audio_format=audio_format,
            segment_transforms=segment_transforms,
            max_num_segments=max_num_segments,
            url2index=url2index,
        )
        preprocessor = WebDatasetBufferPreprocessor(
            sample_rate=sample_rate,
            transforms=audio_transforms,
        )
        pipeline=[
            "decode",
            {"compose": [preprocessor.train_buffer_preprocessor]},
        ]
        super().__init__(dataset, pipeline)

class WrappedLyricsDataset(MultiIterableDataset):
    def __init__(
        self,
        dataset_list: list,
        sample_rate: int,
        sample_duration: float,
        segment_transforms: list,
        handler: Callable = wds.warn_and_continue,
        num_samples: int = 10_000_000,
        weights: List[int] = None,
        seed: int = 2023,
        **kwargs,
    ):
        datasets = []
        for item in dataset_list:
            dataset = LyricsDataset(
                url2index=item['url2index'],
                sample_rate=sample_rate,
                sample_duration=sample_duration,
                segment_transforms=segment_transforms,
                audio_keys=item["audio_keys"],
                audio_format=item["audio_format"],
                handler=handler,
                **kwargs,
            )
            datasets.append(dataset)
        if weights is None:
            weights = [1.0 for _ in range(len(datasets))]
        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=weights,
            seed=seed,
        )

class LyricsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @classmethod
    def from_dataset_type(
        cls, 
        dataset_type: str,
        batch_size: int,
        sample_rate=24000,
        sample_duration=10,
        shuffle_buffer_size: int = 100,
        lyrics_max_seq_len: int = 150,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        lyrics_tokenizer = PhonemeTokenizer(allow_unknown=True)
        segment_transforms = [LyricsTokenTransform(lyrics_tokenizer, lyrics_max_seq_len)]
        lyrics_collator = LyricsCollator(lyrics_tokenizer.pad_id, lyrics_max_seq_len=lyrics_max_seq_len, audio_max_seq_len=sample_rate*sample_duration)
        if dataset_type == 'mixture_only':
            train_dataset = create_mixture_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
        elif dataset_type == 'conditional_vocals':
            segment_transforms = [LyricsTokenTransform(lyrics_tokenizer, lyrics_max_seq_len), LyricsChromaTransform()]
            train_dataset = create_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
        elif dataset_type == 'multitask':
            train_dataset = create_multitask_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
        validation_dataset = create_validation_dataset(sample_rate, sample_duration, batch_size, segment_transforms, lyrics_collator)
        return LyricsDataModule(train_dataset=train_dataset, validation_dataset=validation_dataset, num_workers=num_workers, pin_memory=pin_memory)

    @classmethod
    def wav_dataset(
        cls, 
        batch_size: int,
        sample_rate=24000,
        sample_duration=10,
        shuffle_buffer_size: int = 100,
        num_workers: int = 8,
    ):
        wav_collator = WavCollator(audio_max_seq_len=sample_rate*sample_duration)
        segment_transforms = None # no need to tokenize lyrics
        train_dataset = create_mixture_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, wav_collator)
        validation_dataset = create_validation_dataset(sample_rate, sample_duration, batch_size, segment_transforms, wav_collator)
        return LyricsDataModule(train_dataset=train_dataset, validation_dataset=validation_dataset, num_workers=num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=None, num_workers=self.num_workers, pin_memory=self.pin_memory)


## Multitask 

def to_batched_dataset(dataset, batch_size, collation_fn, shuffle_buffer_size=None):
    if shuffle_buffer_size is None:
        return DataPipeline(
            dataset,
            wds.batched(batch_size, collation_fn=collation_fn),
        )
    return DataPipeline(
        dataset,
        wds.shuffle(shuffle_buffer_size),
        wds.batched(batch_size, collation_fn=collation_fn),
    )

def create_mixture_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, collator):
    lyrics_dataset = WrappedLyricsDataset(
        [
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_train.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'full.mp3', 'target_audio': 'full.mp3'},
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso.tar_to_index.tsv", 
                "audio_format": "m4a", # resso is m4a format for some reason
                "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc9m.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            },
            # {
            #     "url2index": "/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_8m_url2index.tsv", 
            #     "audio_format": "mp3",
            #     "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            # },  # This is old partial 60m - currently used to run baseines
        ],
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        segment_transforms=segment_transforms,
        resampled=True,
        shardshuffle=True,
        use_pipe=True,
        # weights=[0.05, 0.15, 0.2, 0.6] # 50k, 250k, 1m, 6m
        weights=[0.05, 0.25, 1, 20] # 50k, 250k, 1m, 35m
    )
    return to_batched_dataset(lyrics_dataset, batch_size, collator, shuffle_buffer_size)

def create_mixture_mcc_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, collator):
    lyrics_dataset = WrappedLyricsDataset(
        [
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc9m.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'mp3', 'target_audio': 'mp3'},
            },
        ],
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        segment_transforms=segment_transforms,
        resampled=True,
        shardshuffle=True,
        use_pipe=True,
        weights=[0.3, 0.7]
    )
    return to_batched_dataset(lyrics_dataset, batch_size, collator, shuffle_buffer_size)

def create_mixture_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, collator):
    lyrics_dataset = WrappedLyricsDataset(
        [
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset_acc/karaoke_train.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'mulan_audio': 'mss_acc', 'vocal_audio': 'mss_vocal'},
            },
        ],
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        segment_transforms=segment_transforms,
        resampled=True,
        shardshuffle=True,
        weights=[0.2, 0.8]
    )
    return to_batched_dataset(lyrics_dataset, batch_size, collator, shuffle_buffer_size)

def create_vocal_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, collator):
    lyrics_dataset = WrappedLyricsDataset(
        [
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/webdataset_acc/karaoke_train.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'target_audio': 'vocal.mp3', 'vocal_audio': 'vocal.mp3' },
            },
            {
                "url2index": "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv", 
                "audio_format": "mp3",
                "audio_keys": { 'target_audio': 'mss_vocal', 'vocal_audio': 'mss_vocal'},
            },
        ],
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        segment_transforms=segment_transforms,
        resampled=True,
        shardshuffle=True,
        weights=[0.1, 0.9]
    )
    return to_batched_dataset(lyrics_dataset, batch_size, collator, shuffle_buffer_size)

def create_instumental_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size):
    from recipes.musiclm.datasets.mcc import WrappedMCC40MDataset
    mcc40m_ds = WrappedMCC40MDataset(
        [
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/a.tsv",
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/b.tsv",
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/c.tsv",
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/d.tsv",
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/e.tsv",
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/f.tsv",
        ],
        sample_rate=sample_rate,
        duration=sample_duration,
        audio_key='mp3',
        exclude_licenses = [],
        loudness_ratio_threshold=0.2,
        max_num_crops=3,
        resampled=True,
        shardshuffle=True,
        use_pipe=False
    )

    instrumental_collator = InstrumentalCollator()

    return to_batched_dataset(mcc40m_ds, batch_size, instrumental_collator, shuffle_buffer_size)

def create_validation_dataset(sample_rate, sample_duration, batch_size, segment_transforms, collator):
    lyrics_dataset = LyricsDataset(
        "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_valid.tar_to_index.tsv",
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        segment_transforms=segment_transforms,
        audio_keys={ "target_audio": "full.mp3", "mulan_audio": "full.mp3" },
        audio_format="mp3",
        resampled=False,
        shardshuffle=False,
        nodesplitter=return_self
    )
    return to_batched_dataset(lyrics_dataset, batch_size, collator)

def create_multitask_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator):
    mixture_batch_ds = create_mixture_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
    vocal_batch_ds = create_vocal_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
    instrumental_batch_ds = create_instumental_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size)
    combined_ds = MultiIterableDataset(
        [mixture_batch_ds, vocal_batch_ds, instrumental_batch_ds], 
        num_samples=10_000_000, seed=2023,
        weights=[0.75, 0.05, 0.2]
    )
    return combined_ds

def create_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator):
    mixture_batch_ds = create_mixture_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
    vocal_batch_ds = create_vocal_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
    mixture_mcc9m_batch_ds = create_mixture_mcc_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, segment_transforms, lyrics_collator)
    instrumental_batch_ds = create_instumental_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size)
    combined_ds = MultiIterableDataset(
        [mixture_batch_ds, vocal_batch_ds, mixture_mcc9m_batch_ds, instrumental_batch_ds], 
        num_samples=10_000_000, seed=2023,
        weights=[0.15, 0.15, 0.55, 0.15]
    )
    return combined_ds

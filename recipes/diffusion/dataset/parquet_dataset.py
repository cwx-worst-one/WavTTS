from collections import defaultdict
from typing import Iterator, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
import webdataset as wds
import pickle
import pyloudnorm as pyln
import librosa
import io

from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline
from webdataset import shardlists
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline

from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from recipes.bigmusic.datasets.symbolic_music.fixed_length_trans5stem_and_lyric2audio_codec import FixedLengthTrans5StemsAndLyric2AudioCodec

class ParquetDatasetWrapper(WebPipeline):
    def __init__(
        self,
        data_id=408,
        resampled: bool = True,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        nodesplitter=wds.split_by_node,
        n_channels:int = 1,
        dataset_samplerate: int= 44100,
    ):
        dataset = ParquetDataset(
            data_id=data_id, resampled=resampled, nodesplitter=nodesplitter
        )
        self.audio_duration = audio_duration
        self.audio_random_crop = audio_random_crop
        self.meter = pyln.Meter(dataset_samplerate)
        self.dataset_samplerate = dataset_samplerate
        self.n_channels = n_channels
        pipeline = [{"compose": self.decode}]
        super().__init__(dataset, pipeline)

    def __iter__(self):
        return iter(self.data_pipeline)

    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            # from byte to npy
            wav_npy, sr = librosa.load(io.BytesIO(item["wav"]), sr=None, mono=(self.n_channels==1))
            wav_tensor = torch.as_tensor(wav_npy, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor[None, :]
            audio_duration_samples = int(self.audio_duration * self.dataset_samplerate)
            # pad audio to the length of duration
            if wav_tensor.shape[-1] < audio_duration_samples:
                pad_len = audio_duration_samples - wav_tensor.shape[-1]
                wav_tensor = torch.nn.functional.pad(
                    wav_tensor, (0, pad_len), "constant", 0
                )

            if self.audio_random_crop:
                # random crop
                start = torch.randint(
                    0, wav_tensor.shape[-1] - audio_duration_samples + 1, (1,)
                ).item()
                wav_tensor = wav_tensor[:, start : start + audio_duration_samples]
            else:
                # use first [audio_duration second]
                wav_tensor = wav_tensor[:, :audio_duration_samples]

            # loudness detection
            db = self.meter.integrated_loudness(wav_tensor.detach().cpu().numpy().T)
            if db < -50 or np.isneginf(db):
                continue

            yield {"audio": wav_tensor, "meta_song_id": item["uttid"]}


class ParquetDataModule(pl.LightningDataModule):
    data_sample_rate = None

    def __init__(
        self,
        train_data_id: int,
        valid_data_id: int,
        train_batch_size: int,
        valid_batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        n_channels: int = 1,
        dataset_samplerate: int = 44100,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        prefetch_factor=2,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = ParquetDatasetWrapper(
            data_id=train_data_id,
            resampled=resampled,
            audio_duration=audio_duration,
            audio_random_crop=audio_random_crop,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
        )
        self.validation_dataset = ParquetDatasetWrapper(
            data_id=valid_data_id,
            resampled=True,
            audio_duration=audio_duration,
            audio_random_crop=False,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
        )
        self.predict_dataset = self.train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def collate_fn(self, batch):
        keys = batch[0].keys()
        batch_dict = defaultdict(list)
        for b in batch:
            for k in keys:
                batch_dict[k].append(b[k])
        batch_dict["audio"] = torch.stack(batch_dict["audio"])
        return dict(batch_dict)

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(self.train_batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.valid_batch_size, collation_fn=self.collate_fn),
        )

        return DataLoader(
            validation_dataset_batched,
            batch_size=None,
            num_workers=1,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            predict_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


class OfflineFeatureParquetDatasetWrapper(ParquetDatasetWrapper):
    KEYS_REQUIRED=["vocoder_emb","umm_token","uttid","loudness"]
    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            assert all([k in item for k in self.KEYS_REQUIRED])

            meta_song_id = item["uttid"]
            vocoder_embs = item["vocoder_emb"]
            if vocoder_embs is None:
                print(
                    f"[WARNING] get 'vocoder_emb=None' from dataset (uttid={meta_song_id})"
                )
                continue
            else:
                vocoder_embs = torch.from_numpy(pickle.loads(vocoder_embs))
                if vocoder_embs.ndim == 2:
                    vocoder_embs = vocoder_embs[None, :]

            umm_tokens = item["umm_token"]
            if umm_tokens is None:
                print(
                    f"[WARNING] get 'umm_token=None' from dataset (uttid={meta_song_id})"
                )
                continue
            else:
                umm_tokens = torch.from_numpy(pickle.loads(umm_tokens))
                if umm_tokens.ndim == 1:
                    umm_tokens = umm_tokens[None, :]
            
            loudness = pickle.loads(item["loudness"])
            if len(loudness)!=2:
                print(f"[WARINING] invalid 'loudness' with length {len(loudness)}(2) ")
                continue
            loudness, dur = loudness
            if not (isinstance(loudness,list) and isinstance(dur,float)):
                print(f"[WARINING] invalid 'loudness' with {type(loudness)}(list) and {type(dur)}(float)")
                continue

            try:
                assert len(umm_tokens) == len(vocoder_embs) == len(loudness)
                min_bs = len(umm_tokens)
            except:
                min_bs = min(len(umm_tokens), len(vocoder_embs), len(loudness))

            yield {
                "meta_song_id": meta_song_id,
                "condition_tokens": umm_tokens[:min_bs],
                "vocoder_embs": vocoder_embs[:min_bs],
                "loudness": loudness[:min_bs],
                "duration": dur,
            }


class OfflineLeadsheetParquetDatasetWrapper(ParquetDatasetWrapper):
    KEYS_REQUIRED=["uttid", "vocoder_emb", "meta"]
    def __init__(self, data_id=408, resampled: bool = True, audio_duration: int = 30, audio_random_crop: bool = True, nodesplitter=wds.split_by_node, n_channels: int = 1, dataset_samplerate: int = 44100):
        super().__init__(data_id, resampled, audio_duration, audio_random_crop, nodesplitter, n_channels, dataset_samplerate)
        self.config = FixedLengthTrans5StemsAndLyric2AudioCodec.Config(
        # TODO:hzy:move to config yaml
        lyrics_seq_len=0,
        leadsheet_seq_len=7000,
        semantic_frame_rate=25,
        audio_max_duration=audio_duration,
        sample_rate=dataset_samplerate,
        audio_key='wav',
        conditions="remi_leadsheet_tokens",
        include_utterance_phoneme_tokens=False
        )
        self.codec = FixedLengthTrans5StemsAndLyric2AudioCodec(self.config)

    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            assert all([k in item for k in self.KEYS_REQUIRED])

            # TODO:hzy:extract vocoder_embs in 1838/1839 datasets
            # Extract vocoder_embs
            meta_song_id = item["uttid"]
            vocoder_embs = item["vocoder_emb"]
            if vocoder_embs is None:
                print(
                    f"[WARNING] get 'vocoder_emb=None' from dataset (uttid={meta_song_id})"
                )
                continue
            else:
                vocoder_embs = torch.from_numpy(pickle.loads(vocoder_embs))
                if vocoder_embs.ndim == 2:
                    vocoder_embs = vocoder_embs[None, :]
            
            # Extract leadsheet tokens
            dfs_dict = BMDfsDictBuilder(item)\
                .pre_load_meta()\
                .add_df_note(subsets=["vocal", "piano", "guitar", "bass", "drums"])\
                .add_df_lyrics()\
                .add_df_beat()\
                .add_df_section()\
                .add_df_chord()\
                .quantize_chord_to_beat()\
                .quantize_section_to_downbeat()\
                .add_audio(audio_key=self.config.audio_key)\
                .create_output()
            tokens = self.codec.encode_leadsheet(dfs_dict)

            ## Convert leadsheet tokens to required model input format
            leadsheet_tokens = self.codec.chop_or_pad(arr=tokens[:-1],
                target_len=self.config.leadsheet_seq_len - 1,
            )
            leadsheet_tokens = torch.LongTensor(np.append(leadsheet_tokens, self.codec.indexer["eos"]))

            # Extract audio array
            audio, sample_rate = dfs_dict["audio"]
            audio = audio.squeeze(0)
            dur =  len(audio)/sample_rate
            wav_tensor = torch.as_tensor(audio, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor[None, :]
            audio_duration_samples = int(self.audio_duration * self.dataset_samplerate)

            # pad audio to the length of duration
            if wav_tensor.shape[-1] < audio_duration_samples:
                pad_len = audio_duration_samples - wav_tensor.shape[-1]
                wav_tensor = torch.nn.functional.pad(
                    wav_tensor, (0, pad_len), "constant", 0
                )

            if self.audio_random_crop:
                # random crop
                start = torch.randint(
                    0, wav_tensor.shape[-1] - audio_duration_samples + 1, (1,)
                ).item()
                wav_tensor = wav_tensor[:, start : start + audio_duration_samples]
            else:
                # use first [audio_duration second]
                wav_tensor = wav_tensor[:, :audio_duration_samples]

            # loudness detection
            loudness = self.meter.integrated_loudness(wav_tensor.detach().cpu().numpy().T)
            if loudness < -50 or np.isneginf(loudness):
                continue

            if not (isinstance(loudness,list) and isinstance(dur,float)):
                print(f"[WARINING] invalid 'loudness' with {type(loudness)}(list) and {type(dur)}(float)")
                continue

            try:
                assert len(leadsheet_tokens) == len(vocoder_embs) == len(loudness)
                min_bs = len(leadsheet_tokens)
            except:
                min_bs = min(len(leadsheet_tokens), len(vocoder_embs), len(loudness))

            yield {
                "meta_song_id": meta_song_id,
                "condition_tokens": leadsheet_tokens[:min_bs],
                "vocoder_embs": vocoder_embs[:min_bs],
                "loudness": loudness[:min_bs],
                "duration": dur,
            }


class OfflineFeatureParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_id: int,
        valid_data_id: int,
        train_batch_size: int,
        valid_batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        dataset_samplerate: int = 44100,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        prefetch_factor=2,
        datasetcalss = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = datasetcalss(
            data_id=train_data_id,
            resampled=resampled,
            audio_duration=audio_duration,
            audio_random_crop=audio_random_crop,
            dataset_samplerate=dataset_samplerate,
        )
        self.validation_dataset = datasetcalss(
            data_id=valid_data_id,
            resampled=True,
            audio_duration=audio_duration,
            audio_random_crop=False,
            dataset_samplerate=dataset_samplerate,
        )
        self.predict_dataset = self.train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def collate_fn(self, batch):
        keys = batch[0].keys()
        batch_dict = defaultdict(list)
        for b in batch:
            for k in keys:
                batch_dict[k].append(b[k])
        batch_dict["condition_tokens"] = torch.stack(
            batch_dict["condition_tokens"], dim=0
        )  # [b,n,t]
        batch_dict["vocoder_embs"] = torch.stack(
            batch_dict["vocoder_embs"], dim=0
        )  # [b,n,dim,t]
        batch_dict["loudness"] = np.stack(batch_dict["loudness"],axis=0)
        return dict(batch_dict)

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(self.train_batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.valid_batch_size, collation_fn=self.collate_fn),
        )

        return DataLoader(
            validation_dataset_batched,
            batch_size=None,
            num_workers=1,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            predict_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    # datamodule = ParquetDataModule(794, 793, 4, 4, 2, dataset_samplerate=24000)
    
    # train_loader = datamodule.val_dataloader()

    # for d in train_loader:
    #     # print(d.keys())
    #     print(d["audio"].shape)
    #     # assert 1==2

    # dataset=OfflineFeatureParquetDatasetWrapper(865)
    # # dataset=OfflineFeatureParquetDatasetWrapper(856)
    # # dataset=OfflineFeatureParquetDatasetWrapper(839)
    # # dataset=OfflineFeatureParquetDatasetWrapper(808)
    # for data in dataset:
    #     # print(data.keys())
    #     for k,v in data.items():
    #         try:
    #             print(k, v.shape)
    #         except:
    #             print(k, v)
    # print("OfflineFeatureParquetDatasetWrapper")
    # # OfflineFeatureParquetDataModule(856,856,16,16)



    dataset=OfflineLeadsheetParquetDatasetWrapper(1838, dataset_samplerate=24000)
    for data in dataset:
        print(data.keys())
        for k,v in data.items():
            try:
                print(k, v.shape)
            except:
                print(k, v)
    print("OfflineLeadsheetParquetDatasetWrapper")

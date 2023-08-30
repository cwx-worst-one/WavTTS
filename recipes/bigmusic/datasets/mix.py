import io
import random
from string import punctuation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import webdataset as wds
import random
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline
import logging, phonemizer
from recipes.bigmusic.datasets.lyrics import LyricsDataset
from recipes.datasets.mcc.mix import (
    LibriTTSDataset,
    MCCInstrumentalDataset,
    MCCVocalDataset,
    DataModule
)

from recipes.musiclm.utils.dist import local_zero_first
from recipes.musiclm.transforms.audio import (
    FastNormalizeAudio,
    LoudnessCheck,
    NormalizeAudio,
    ReadMP3,
)
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self
from transformers import Wav2Vec2PhonemeCTCTokenizer


def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")    
    return text.translate(str.maketrans("", "", nlp_punctuation))

def pad_crop(sequence, seq_len, dtype, padding_value=0):
    # in item_pad_idx, 0 indicates the values are padded.
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    item_pad_idx = torch.full((seq_len,), fill_value=0, dtype=int)
    item_pad_idx[:len(sequence)] = torch.ones_like(torch.as_tensor(sequence[:seq_len]), dtype=int)
    return item_pad, item_pad_idx

def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    # TODO: (QQ) make these constants configurable.
    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    MAX_PHONE_LEN = 200
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    default_lyrics_token = torch.full((MAX_PHONE_LEN,), PHONE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    style_text = []
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     
    for idx in range(len(batch)):        
        audio.append(random_pad(batch[idx]["audio"]))
        if isinstance(batch[idx]["style_text"], Tuple):
            mulan_label = batch[idx]["style_text"][0]
        else:
            mulan_label = batch[idx]["style_text"]
        style_text.append(mulan_label)        
        normalized_text.append(batch[idx]["normalized_text"])
        phoneme_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("lyrics_tokens", default_lyrics_token)), 
            MAX_PHONE_LEN, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        speaker_id.append(batch[idx].get("speaker_id", default_speaker_id))        
    return {
        "target_audio": torch.stack(audio, dim=0), 
        "style_text": style_text,
        "normalized_text": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        }

class LibrilightDataset(WebPipeline):
    name = "LibrilightASR"
    data_sample_rate = 24000
    

class MCCInstrumentalDataset(MCCInstrumentalDataset):
    name = "DecoderMCCInstrumental"
    def transform(self, item) -> Dict[str, Any]:
        mood = item["__index_data__"].get("final_mood")            
        genre = item["__index_data__"].get("final_genre")
        output = super().transform(item)
        if output is None:
            return
        output["style_text"] = normalize_text(" ".join([x for x in [mood, genre] if (x is not None) and (x !='nan') ])),
        output["normalized_text"] = ""
        yield output     


class MCCVocalDataset(MCCInstrumentalDataset):
    name = "DecoderMCCVocal"    
    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        **kwargs,
    ):
        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
        phonemizer.logger.get_logger().setLevel(logging.ERROR)

        super().__init__(
            url2index=url2index,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            **kwargs,
        )
        self.lyrics_confidence = lyrics_confidence

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["additions"]["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        return True

    def group_utterances(self, utterances, min_duration, max_duration):
        utterances = [(u['start_time']/1000, u['end_time']/1000, u['text']) for u in utterances]        
        for i in range(len(utterances)-1):
            if utterances[i][1] > utterances[i+1][0]:
                logging.warning("utterances need to be non-overlapping")
                return []
        utterances = [u for u in utterances if u[1]-u[0] > 0]
        segs = []
        i = 0
        s, cur_seg = i, []        
        while i < len(utterances):
            if utterances[i][1] - utterances[s][0] < min_duration:
                cur_seg.append(utterances[i])                
                i += 1
            else:
                j = i
                while j < len(utterances) and utterances[j][1] - utterances[s][0] <= max_duration:
                    j += 1
                if j == i:
                    i = s + 1
                else:
                    k = random.randint(i+1, j)
                    cur_seg.extend([utterances[k] for k in range(i, k)])
                    segs.append([cur_seg[0][0], cur_seg[-1][1], 
                                ' '.join([u[2] for u in cur_seg])])
                    i = k
                # Move to the next vocal starting point
                while i < len(utterances):
                    if len(utterances[i][2]) >= 2:
                        break
                    i += 1
                if i < len(utterances):
                    s, cur_seg = i, []        
        return segs
        
    def transform(self, item) -> Dict[str, Any]:
        if not self.is_metadata_good(item["__index_data__"]):
            return
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            return
        mood = item["__index_data__"].get("final_mood")
        genre = item["__index_data__"].get("final_genre")
        audio = self.base_transform(item[self.audio_key])
        utterances = item["__index_data__"]["lyrics"].get("utterances", None)
        if utterances is None:
            return       
        segments = self.group_utterances(utterances, self.min_duration, self.max_duration)
        for segment in segments:                
            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]
            if not self.is_loud(clip):
                continue
            normalized_text = normalize_text(segment[2])
            phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]                                
            yield {
                "audio": clip, 
                "style_text": normalize_text(" ".join(['vocal'] + [x for x in [mood, genre] if (x is not None) and (x !='nan') ])),                    
                "normalized_text": normalized_text,
                "lyrics_tokens": phoneme_tokens,
            }                

        # random.shuffle(utterances)
        # filtered_utterances = filter(self.filter_lyrics, utterances)
        # for selected_utterance in filtered_utterances:
        #     start = int(
        #         float(selected_utterance["start_time"]) / 1000 * self.sample_rate
        #     )
        #     end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
        #     clip = audio[:, start:end]
        #     if not self.is_loud(clip):
        #         return
        #     normalized_text = normalize_text(selected_utterance["text"])
        #     # TODO: assumed 25Hz
        #     if len(normalized_text) <= 0 or len(normalized_text) > clip.size(-1) / self.sample_rate * 25:
        #         return
        #     phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]
        #     yield {
        #         "audio": clip, 
        #         "style_text": normalize_text(" ".join([x for x in [mood, genre] if (x is not None) and (x !='nan') ])),
        #         "normalized_text": normalized_text,
        #         "lyrics_token": phoneme_tokens,
        #     }


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_buffer_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )
                for val in self.validation_dataset
            ]
        else:
            return DataLoader(
                self.validation_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class MixWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [
            5,
            6,
            7,
            8,
            9,
            10,  
            11,
            12,
            13,
            14,
            15,          
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
    ):
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=False,
            batch_size=batch_size,
            length_fn=lambda x: x["audio"].shape[-1],
        )

        mcc_vocal = MCCVocalDataset(
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=True,
            handler=wds.reraise_exception,
        )
        mcc_instrumental = MCCInstrumentalDataset(
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=True,
            handler=wds.reraise_exception,
        )

        train_dataset = WebPipeline(            
            mcc_vocal,
            # MultiIterableDataset(
            #     datasets=[mcc_vocal, mcc_instrumental], weights=[2, 1]                
            # ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = WebPipeline(
            MCCVocalDataset(
                url2index="/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx_val.txt",
                sample_rate=sample_rate,
                nodesplitter=return_self,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                use_pipe=True,
                handler=wds.reraise_exception,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            for i in item:
                batch = self.batcher.collate_batch(i)
                if batch is not None:
                    yield batch
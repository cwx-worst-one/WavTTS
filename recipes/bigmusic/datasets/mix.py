import json
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
import pytorch_lightning as pl
import torch
import webdataset as wds
from functools import partial
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset import shardlists
from webdataset.pipeline import DataPipeline
import logging, phonemizer
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN
from recipes.bigmusic.datasets.utils.zh_datasets import ZhMetaTransform
from recipes.bigmusic.datasets.utils.zh_meta import SongSlice, ZhMetaParseError, ZhMetaTransformError
from recipes.bigmusic.utils.format_utils import normalize_text

from recipes.datasets.mcc.mix import (
    INDEX,
    WebDatasetBufferPreprocessor,
    BaseTransforms,
    DataModule
)
from recipes.datasets.mcc.sami_tokenizer import SamiOfflineTokenizer, SamiTokenizerError

from recipes.musiclm.utils.dist import local_zero_first
from recipes.musiclm.transforms.audio import FastNormalizeAudio
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    Pad,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self

MAX_STYLE_LEN = 16

# Yilin: No need to use line break for Chinese. A full stop `。` will be
# added 
# LINE_BREAK_PHONE_TOKEN = get_line_break_id()


def pad_crop(sequence, seq_len, dtype, padding_value=0):
    # in item_pad_idx, 0 indicates the values are padded.
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    item_pad_idx = torch.full((seq_len,), fill_value=0, dtype=int)
    item_pad_idx[:len(sequence)] = torch.ones_like(torch.as_tensor(sequence[:seq_len]), dtype=int)
    return item_pad, item_pad_idx

def collate_fn(batch: List[torch.Tensor], conditions="style_text,lyrics_tokens") -> Dict[str, torch.Tensor]:
    # collate_fn batches the examples based on the target_audio length.
    PHONE_PAD_ID = 0
    max_phone_len = int(batch[0].get("max_phone_len", MAX_PHONE_LEN))
    max_length = max([x["target_audio"].shape[-1] for x in batch])
    if 'acc' in batch[0]:
        max_length = max([x["acc"].shape[-1] for x in batch] + [max_length])
    if 'vocal' in batch[0]:
        max_length = max([x["vocal"].shape[-1] for x in batch] + [max_length])
    random_pad = Pad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    target_audio = []
    acc_audio = []
    vocal_audio = []
    style_text = []
    normalized_text = []
    lyrics_tokens = []
    target_tokens_length = []
    speaker_id = []
    dataset_name = []

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    for idx in range(len(batch)):
        audio = batch[idx]["target_audio"]
        if audio.ndim == 1:
            audio = audio[None, :]
                  
        target_audio.append(random_pad(audio))

        style_label = batch[idx].get("style_text")
        if isinstance(style_label, Tuple):
            style_label = style_label[0]
        style_text.append(style_label)

        speaker_id.append(batch[idx]["artist_id"])
        
        construct = lambda key, default: \
            batch[idx][key].clone().detach() if key in batch[idx] and batch[idx][key] is not None \
                else default.clone().detach()
        
        text = batch[idx].get("normalized_text")
        if not text:
            text = batch[idx].get("lyrics")
        normalized_text.append(text)
        
        target_tokens_length.append(batch[idx]["target_tokens_length"])
        phoneme_tokens, _ = pad_crop(
            construct("lyrics_tokens", default_lyrics_token),
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)

        style_metadata.append(batch[idx].get("style_metadata"))
        song_id.append(batch[idx].get("song_id"))
        shard.append(batch[idx].get("shard"))
        worker_id.append(batch[idx].get("worker_id"))
        dataset_name.append(batch[idx].get("dataset_name"))
        if 'acc' in batch[idx]:
            acc_audio.append(random_pad(batch[idx]['acc']))
        if 'vocal' in batch[idx]:
            vocal_audio.append(random_pad(batch[idx]['vocal']))

    stacked_audio = torch.stack(target_audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)    
    batch = {
        "target_audio": torch.stack(target_audio, dim=0),        
        "style_text": style_text,
        "normalized_text": normalized_text,
        "lyrics": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "target_tokens_length": torch.as_tensor(target_tokens_length),
        "speaker_id": torch.as_tensor(speaker_id).unsqueeze(1),
        "conditions": conditions,
        "dataset_name": dataset_name,

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }
    if "style_tag" in conditions or "style_audio" in conditions:
        batch["style_audio"] = stacked_audio
    if len(acc_audio) > 0:
        batch['audio_acc'] = torch.stack(acc_audio, dim=0)
        batch['audio_vocal'] = torch.stack(vocal_audio, dim=0)
    return batch



########################## Chinese Datasets ######################


class VocalTransforms(BaseTransforms):
    name = "VocalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: Optional[int] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        index_key: str = "__index_data__",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,  # TODO: Remove it, not used
        loudness_ratio_threshold: float = 0.2,  # TODO: Remove it, not used
        lyrics_field: str = "lyrics",  # TODO: Remove it, not used
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer=None,   # TODO: Remove it, not used
        frame_rate: int = 25,
        read_structure_tags: bool = False,  # TODO: Remove it, not used
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,   
        infer_structure_tags: bool = False,  # TODO: Remove it, not used
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,  # TODO: Remove it, not used
        tag_taxonomy_lang: str = "SA",  # TODO: Remove it, not used.
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        extra_audio_keys=[]
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        # self.min_volume_threshold = min_volume_threshold
        # self.loudness_ratio_threshold = loudness_ratio_threshold
        self.lyrics_confidence = lyrics_confidence
        self.audio_key = audio_key
        self.index_key = index_key
        # self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        # self.infer_structure_tags = infer_structure_tags
        # self.read_structure_tags = read_structure_tags
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        # There are two fields for lyrics: "lyrics" (ASR) and "lyrics_gt" (Original).
        # self.yrics_field is the top priority, fall back to the other field if the field is empty.
        # self.lyrics_field = lyrics_field
        self.sinking_threshold = sinking_threshold
        # self.quality_filter = quality_filter
        # self.tag_taxonomy_lang = tag_taxonomy_lang
        self.extra_audio_keys = extra_audio_keys

        self.meta_transform = ZhMetaTransform.from_data_id(
            data_id,
            sinking_threshold=sinking_threshold,
            lyrics_confidence=self.lyrics_confidence,
            segment_method=self.segment_method,
            max_seg_per_track=self.max_seg_per_track,
            duration_range=(self.min_duration, self.max_duration),
        )
        self.phrase_dropout = partial(
            SongSlice.dropout,
            line_break_dropout_rate=line_break_dropout_rate,
            section_tag_dropout_rate=section_tag_dropout_rate,
        )

        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        meta = item[self.index_key]
        if isinstance(meta, str):
            meta = json.loads(meta)

        # Get track level audio
        try:
            audio = self.base_transform(item[self.audio_key])
            extra_audio = [self.base_transform(item[k]) for k in self.extra_audio_keys]
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Parse and transform meta
        try:
            _, song_slices, style_text, artist_id = self.meta_transform(meta)
        except ZhMetaParseError as pe:
            self._update_stats(skipped=True, message=f"ParseError: {pe}")
            return
        except ZhMetaTransformError as te:
            self._update_stats(skipped=True, message=f"TransformError: {te}")
            return

        # Yield one example per segment
        self._update_stats(skipped=False)
        for song_slice in song_slices:
            clip, extra_clip = song_slice.slice_audio(audio, self.sample_rate, extra_audio)
            reformatted_phrases = self.phrase_dropout(phrases=song_slice.phrases)

            # NOTE: `normalize_text` removes `:` for the singer tag.
            normalized_text = normalize_text(
                "\n".join([s.format_text() for s in reformatted_phrases]), enable_punctuation=True
            )

            tokenizer = SamiOfflineTokenizer()
            # If a phrase fails, skip the entire slice. Otherwise it could worsen the phoneme missing issue
            # in the training data.
            try:
                text_tokens = torch.from_numpy(tokenizer.tokenize_phrases(reformatted_phrases)).long()
            except SamiTokenizerError as e:
                self._update_stats(skipped=True, message=f"Error tokenizing phrases: {e}")
                continue

            yield {
                "target_audio": clip,
                "target_tokens_length": int(clip.shape[-1] / self.sample_rate * self.frame_rate),
                "style_text": style_text,
                "artist_id": artist_id,
                "normalized_text": normalized_text,
                "lyrics_tokens": text_tokens,
                "max_phone_len": self.segment_max_phone_len,
            } | {k: v for k, v in zip(self.extra_audio_keys, extra_clip)}


class VocalDataset(WebPipeline):
    name = "VocalDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        region: str = "CN",
        dataset_names: List[str] = ["Soda"],
        dataset_weights: List[float] = [1],        
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer: Any = "tts_chinese_frontend_model",
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,
        tag_taxonomy_lang: str = "SA",
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        extra_audio_keys=[],
        **kwargs,
    ):
        assert region in INDEX
        print(f"[{self.name}] initializing...")        

        transforms = VocalTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_field=lyrics_field,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,            
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            sinking_threshold=sinking_threshold,
            quality_filter=quality_filter,
            tag_taxonomy_lang=tag_taxonomy_lang,
            line_break_dropout_rate=line_break_dropout_rate,
            section_tag_dropout_rate=section_tag_dropout_rate,
            extra_audio_keys=extra_audio_keys
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")

        if isinstance(dataset_names, list):            
            url2index = [INDEX[region][dataset_name] for dataset_name in dataset_names]
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=dataset_weights,
            )
        else:
            if dataset_names == "SodaTest":
                url2index = "hdfs://haruna/home/byte_speech_sv/data/soda_valid/val_url2index.txt"
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
            else:
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_names], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalParquetDataset(WebPipeline):
    name = "VocalParqueDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 365,  # QQ music: 365 WYY_music: TODO
        url_pattern: str = None,
        sample_rate: int = 24000,
        audio_key: str = "wav",
        index_key: str = "meta",
        min_duration: int = 10,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        lyrics_field: str = "lyrics",
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer: Any = "tts_chinese_frontend_model",
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        infer_structure_tags: bool = False,
        read_structure_tags: bool = False,
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,
        tag_taxonomy_lang: str = "SA",
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        extra_audio_keys=[],
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern,
                                 extra_fields_in_data=extra_audio_keys, **kwargs)
        
        transforms = VocalTransforms(
            data_id=data_id,
            sample_rate=sample_rate,
            audio_key=audio_key,
            index_key=index_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_field=lyrics_field,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,            
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            infer_structure_tags=infer_structure_tags,
            read_structure_tags=read_structure_tags,
            sinking_threshold=sinking_threshold,
            quality_filter=quality_filter,
            tag_taxonomy_lang=tag_taxonomy_lang,
            line_break_dropout_rate=line_break_dropout_rate,
            section_tag_dropout_rate=section_tag_dropout_rate,
            extra_audio_keys=extra_audio_keys
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


########################## Data Modules ##########################


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
        do_shuffle: bool = True,    # set to False if shuffling is already done at dataset level
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.do_shuffle = do_shuffle

    def train_dataloader(self):
        if self.do_shuffle:
            train_dataset = DataPipeline(
                self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
            )
        else:
            train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
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


class MixVocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,             
        use_pipe: bool = True,
        tokenizer: str = "tts_chinese_frontend_model",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region: str = "CN",
        conditions: str = "style_text,lyrics_tokens",
        wds_dataset_names: List[str] = [],
        wds_dataset_weights: List[int] = [],
        parquet_dataset_ids: List[int] = [],
        parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.8,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        infer_structure_tags: bool = False,
        read_structure_tags: bool = False,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,
        tag_taxonomy_lang: str = "SA",
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        extra_audio_keys=[]
    ):
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = partial(collate_fn, conditions=conditions)

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        self.wds_vocal_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_names:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_vocal_datasets = [VocalDataset(
                region=region,
                dataset_names=wds_dataset_names,
                dataset_weights=wds_dataset_weights,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                use_soda_gt_lyrics=True,
                lyrics_field=lyrics_field,
                lyrics_confidence=lyrics_confidence,
                tokenizer=self.tokenizer,                
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,            
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                quality_filter=quality_filter,
                frame_rate=frame_rate,
                tag_taxonomy_lang=tag_taxonomy_lang,
                line_break_dropout_rate=line_break_dropout_rate,
                section_tag_dropout_rate=section_tag_dropout_rate,
                )]
        self.parquet_vocal_datasets = []
        if parquet_dataset_ids:
            for parquet_id in parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        lyrics_field=lyrics_field,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,
                        infer_structure_tags=infer_structure_tags,
                        read_structure_tags=read_structure_tags,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        quality_filter=quality_filter,
                        frame_rate=frame_rate,
                        tag_taxonomy_lang=tag_taxonomy_lang,
                        line_break_dropout_rate=line_break_dropout_rate,
                        section_tag_dropout_rate=section_tag_dropout_rate,
                        extra_audio_keys=extra_audio_keys
                    ))

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.wds_vocal_datasets + self.parquet_vocal_datasets, 
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalParquetDataset(
                data_id=1528,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                lyrics_field=lyrics_field,
                lyrics_confidence=lyrics_confidence,
                tokenizer=self.tokenizer,
                infer_structure_tags=infer_structure_tags,
                read_structure_tags=read_structure_tags,
                resampled=False,
                shardshuffle=True,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                quality_filter=quality_filter,
                frame_rate=frame_rate,
                tag_taxonomy_lang=tag_taxonomy_lang,
                line_break_dropout_rate=line_break_dropout_rate,
                section_tag_dropout_rate=section_tag_dropout_rate,
                extra_audio_keys=[]
        ),
            pipeline=[{"compose": [self.bucketize]}],
        )] 

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixLangVocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        use_pipe: bool = True,
        tokenizer: str = "tts_chinese_frontend_model",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        zh_parquet_dataset_ids: List[int] = [],
        zh_parquet_dataset_weights: List[int] = [],
        en_parquet_dataset_ids: List[int] = [],        
        en_parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.8,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,
        tag_taxonomy_lang: str = "SA",
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
    ):        
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        
        self.parquet_vocal_datasets = []
        if zh_parquet_dataset_ids:
            for parquet_id in zh_parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        lyrics_field=lyrics_field,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,                
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        quality_filter=quality_filter,
                        frame_rate=frame_rate,
                        tag_taxonomy_lang=tag_taxonomy_lang,
                        line_break_dropout_rate=line_break_dropout_rate,
                        section_tag_dropout_rate=section_tag_dropout_rate,
                        ))

        if en_parquet_dataset_ids:
            for parquet_id in en_parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        lyrics_field=lyrics_field,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,                
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        quality_filter=quality_filter,
                        frame_rate=frame_rate,
                        tag_taxonomy_lang=tag_taxonomy_lang,
                        line_break_dropout_rate=line_break_dropout_rate,
                        section_tag_dropout_rate=section_tag_dropout_rate,
                        ))

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.parquet_vocal_datasets, 
                                 weights=zh_parquet_dataset_weights+en_parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalDataset(
                region='CN',
                dataset_names="SodaTest",
                dataset_weights=1,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                lyrics_field=lyrics_field,
                lyrics_confidence=lyrics_confidence,
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                tokenizer=self.tokenizer,                
                use_pipe=use_pipe,
                resampled=False,
                nodesplitter=return_self,                
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                quality_filter=quality_filter,
                frame_rate=frame_rate,
                tag_taxonomy_lang=tag_taxonomy_lang,
                line_break_dropout_rate=line_break_dropout_rate,
                section_tag_dropout_rate=section_tag_dropout_rate,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )] 

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
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class SftWebDataModule(DataModule):
    # This DataModule supports both Artist SFT and Lyrics SFT.
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        conditions: str = "style_text,speaker_id,lyrics_tokens",
        use_pipe: bool = True,
        tokenizer: str = "phoneme",
        frame_rate: int = 25,
        normalize_audio: bool = False,        
        parquet_datasets: dict = {},
        use_dynamic_batch: str = False,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.8,
        read_structure_tags: bool = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sample_limit_per_file: int = 1000,
        sinking_threshold: float = 0.51,
        quality_filter: bool = False,
        tag_taxonomy_lang: str = "SA",
        line_break_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
    ):        
        print(conditions)
        print(parquet_datasets)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = partial(collate_fn, conditions=conditions)
        assert parquet_datasets

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        self.parquet_vocal_datasets = {}
        for split, pd_id_weights in parquet_datasets.items():
            pds, pdws = [], []
            is_train = True if split == 'train' else False
            nodesplitter = shardlists.single_node_only if split == 'train' else return_self
            for pd_id, pd_weight in pd_id_weights:
                pdws.append(pd_weight)
                pds.append(
                    VocalParquetDataset(
                        data_id=pd_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        lyrics_field=lyrics_field,
                        lyrics_confidence=lyrics_confidence,
                        read_structure_tags=read_structure_tags,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        tokenizer=self.tokenizer,                
                        resampled=is_train,
                        shardshuffle=is_train,
                        nodesplitter=nodesplitter,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sample_limit_per_file=sample_limit_per_file,
                        sinking_threshold=sinking_threshold,
                        quality_filter=quality_filter,
                        frame_rate=frame_rate,
                        tag_taxonomy_lang=tag_taxonomy_lang,
                        line_break_dropout_rate=line_break_dropout_rate,
                        section_tag_dropout_rate=section_tag_dropout_rate,
                        ))
            self.parquet_vocal_datasets[split] = WebPipeline(            
                MultiIterableDataset(datasets=pds, weights=pdws),
                pipeline=[{"compose": [self.bucketize]}],
            )
        
        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=self.parquet_vocal_datasets["train"],
            validation_dataset=self.parquet_vocal_datasets["test"],
            predict_dataset=self.parquet_vocal_datasets["train"],  # TODO
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class LyricsSftWebDataModule(SftWebDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            parquet_datasets={"train":[(1026, 1)], "test":[(1012, 1)]},
            **kwargs,
        )


class ArtistSftWebDataModule(SftWebDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            parquet_datasets={"train":[(1473, 1)], "test":[(1012, 1)]},
            **kwargs,
        )

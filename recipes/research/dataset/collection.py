from typing import Any, Dict, List, Optional

import torch

from recipes.research.dataset.parquet_dataset import (
    AudioParquetDataset,
    FeatureParquetDataset,
    IndexParquetDataset,
)
from recipes.research.diff.text_processor import (
    BPMTextProcessor,
    MulanPretrainMetadataTextProcessor,
    WordProcessor,
)
from samantha.data.audio.dataset import AudioFolderDataset
from samantha.data.audio.webdataset import AudioWebDataset
from samantha.transforms.audio import Loudness


class Billboardv2WebDataset(AudioWebDataset):


    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            url2index="/mnt/bd/billboard-44k/data/*/url2index.txt",
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            shuffle_buffer_size=0,
            crop_from_start=crop_from_start,
            nitems=500000,
        )


class MusDBTrainAudioFolderDataset(AudioFolderDataset):

    def __init__(self, sample_rate: int, channels: int, segment_duration: float):
        super().__init__(
            root="/mnt/bd/janne-download/data/music/musdb18hq/train",
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle=True,
            num_workers=4,
        )


class MusDBTestAudioFolderDataset(AudioFolderDataset):

    def __init__(self, sample_rate: int, channels: int, segment_duration: float):
        super().__init__(
            root="/mnt/bd/janne-download/data/music/musdb18hq/test",
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle=False,
            num_workers=4,
        )


class ShutterStockv1ParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=206,
            data_type="music_instrumental",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
        )

class EveryNoisev1ParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=234,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
        )


class ShutterStockParquetDataset(AudioParquetDataset):


    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        self.bpm_processor = BPMTextProcessor(min_bpm=40, max_bpm=200, bpm_step=10)

        self.description_processor = WordProcessor(min_word_len=0)
        self.keyword_processor = WordProcessor(min_word_len=3)
        self.genre_processor = WordProcessor(min_word_len=0)
        self.instrument_processor = WordProcessor(min_word_len=2)
        self.text_processor = MulanPretrainMetadataTextProcessor()

        super().__init__(
            data_id=241,
            data_type="music_instrumental",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=1515728,
        )

    def process_text(self, index: Dict[str, Any]) -> Optional[str]:

        descriptions = index["description"]
        keywords = index["keywords"]
        genres = index["genres"]
        instruments = index["instruments"]
        bpms = index["bpm"]
        quality_label = index["quality_label"]

        if instruments == "\\N":
            instruments = ""

        bpms = self.bpm_processor.process(bpms)
        descriptions = self.description_processor.process(descriptions, shuffle=False)
        keywords = self.keyword_processor.process(keywords, shuffle=False)  # TODO

        genres = self.genre_processor.process(
            genres, shuffle=False
        )  # in case of primary/secondary genres, don't shuffle
        instruments = self.instrument_processor.process(
            instruments, shuffle=False
        )  # TODO

        processed_text = self.text_processor.process(
            descriptions, keywords, genres, instruments
        )

        processed_text = f"{quality_label} {processed_text}"
        return processed_text


class ShutterStockIndexParquetDataset(IndexParquetDataset):

    def __init__(self, resampled: bool, shardshuffle: bool):
        super().__init__(
            data_id=241,
            shuffle_buffer_size=0,
            resampled=resampled,
            shardshuffle=shardshuffle,
            nitems=1515728,
        )


class ShutterStockFeature7c355eaParquetDataset(FeatureParquetDataset):

    def __init__(
        self,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        self.bpm_processor = BPMTextProcessor(min_bpm=40, max_bpm=200, bpm_step=10)

        self.description_processor = WordProcessor(min_word_len=0)
        self.keyword_processor = WordProcessor(min_word_len=3)
        self.genre_processor = WordProcessor(min_word_len=0)
        self.instrument_processor = WordProcessor(min_word_len=2)
        self.text_processor = MulanPretrainMetadataTextProcessor()

        super().__init__(
            data_id=262,
            sample_rate=44100,  # TODO: this MUST be included into the tensor metadata somehow..! very fault prone
            channels=2,
            frame_rate=21.533203125,  # TODO: this MUST be included into the tensor metadata somehow..! very fault prone
            data_type="music_instrumental",
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=1515728,
        )

    def process_text(self, index: Dict[str, Any]) -> Optional[str]:

        descriptions = index["description"]
        keywords = index["keywords"]
        genres = index["genres"]
        instruments = index["instruments"]
        bpms = index["bpm"]
        quality_label = index.get("quality_label", None)

        if instruments == "\\N":
            instruments = ""

        bpms = self.bpm_processor.process(bpms)
        descriptions = self.description_processor.process(descriptions, shuffle=False)
        keywords = self.keyword_processor.process(keywords, shuffle=False)  # TODO

        genres = self.genre_processor.process(
            genres, shuffle=False
        )  # in case of primary/secondary genres, don't shuffle
        instruments = self.instrument_processor.process(
            instruments, shuffle=False
        )  # TODO

        processed_text = self.text_processor.process(
            descriptions, keywords, genres, instruments
        )

        if quality_label is not None:
            processed_text = f"{quality_label} {processed_text}"
        return processed_text

    def filter(self, feature: torch.Tensor) -> bool:
        if feature.std() > 2.0:
            return True
        return False


class EveryNoiseFeature7c355eaParquetDataset(FeatureParquetDataset):

    def __init__(
        self,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        self.bpm_processor = BPMTextProcessor(min_bpm=40, max_bpm=200, bpm_step=10)

        self.description_processor = WordProcessor(min_word_len=0)
        self.keyword_processor = WordProcessor(min_word_len=3)
        self.genre_processor = WordProcessor(min_word_len=0)
        self.instrument_processor = WordProcessor(min_word_len=2)
        self.text_processor = MulanPretrainMetadataTextProcessor()

        super().__init__(
            data_id=264,
            sample_rate=44100,  # TODO: this MUST be included into the tensor metadata somehow..! very fault prone
            channels=2,
            frame_rate=21.533203125,  # TODO: this MUST be included into the tensor metadata somehow..! very fault prone
            data_type="music_vocal",
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=845442,
        )

    def process_text(self, index: Dict[str, Any]) -> Optional[str]:
        genre = index["everynoise_genre_merged"]

        text = f"high quality {genre}"
        return text

    def filter(self, feature: torch.Tensor) -> bool:
        if feature.std() > 2.0:
            return True
        return False


class EveryNoiseParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=234,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            shuffle_buffer_size=0,
            segment_duration=segment_duration,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=931405,
        )


class EveryNoiseGenreParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=240,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            shuffle_buffer_size=0,
            segment_duration=segment_duration,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=845442,
        )

    def process_text(self, index: Dict[str, Any]) -> Optional[str]:
        genre = index["everynoise_genre_merged"]
        return genre


class PlaylistV5ParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=250,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=2160769,
        )


class PlaylistV5_24khz_ParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
    ):
        super().__init__(
            data_id=259,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=n_segments_per_read,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=2160769,
        )


class HQAudioParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
    ):
        super().__init__(
            data_id=271,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=n_segments_per_read,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=4607902,
        )



class MCCVocalAB(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        resampled: bool,
        shardshuffle: bool,
        segment_duration: float,
        min_audio_duration: float,
        max_audio_duration: float,
    ):
        super().__init__(
            data_id=254,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            shuffle_buffer_size=0,
            resampled=resampled,
            shardshuffle=shardshuffle,
            segment_duration=segment_duration,
            min_audio_duration=min_audio_duration,
            max_audio_duration=max_audio_duration,
            use_lyrics=True,
        )

from samantha.transforms.audio import Filters, MinMaxFilter

class MultiLingualBigASR48LangParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        resampled: bool,
        shardshuffle: bool,
        buckets_sec: List[float],
        batch_size: int,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
    ):
        self.filters = Filters([
            MinMaxFilter(Loudness(), min=-50, max=-1.0),
        ])
        super().__init__(
            data_id=212,
            data_type="speech",
            sample_rate=sample_rate,
            channels=channels,
            shuffle_buffer_size=0,
            resampled=resampled,
            shardshuffle=shardshuffle,
            buckets_sec=buckets_sec,
            min_audio_duration=min_audio_duration,
            max_audio_duration=max_audio_duration,
            batch_size=batch_size,
            audio_filters=self.filters,
        )


class MultiLingualBigTTSParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        resampled: bool,
        shardshuffle: bool,
        buckets_sec: List[float],
        batch_size: int,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
    ):
        self.filters = Filters([
            MinMaxFilter(Loudness(), min=-50, max=-1.0),
        ])
        super().__init__(
            data_id=273,
            data_type="speech",
            sample_rate=sample_rate,
            channels=channels,
            shuffle_buffer_size=0,
            resampled=resampled,
            shardshuffle=shardshuffle,
            buckets_sec=buckets_sec,
            min_audio_duration=min_audio_duration,
            max_audio_duration=max_audio_duration,
            batch_size=batch_size,
            audio_filters=self.filters,
        )


class FrankSinatraParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        use_lyrics: bool,
        crop_from_start: bool = False,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
    ):
        super().__init__(
            data_id=266,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            use_lyrics=use_lyrics,
            min_audio_duration=min_audio_duration,
            max_audio_duration=max_audio_duration,
            nitems=1299,
        )

    def process_text(self, index: Dict[str, Any]) -> Optional[str]:
        text = "high quality Frank Sinatra"
        return text

class RadioheadParquetDataset(AudioParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        segment_duration: float,
        resampled: bool,
        shardshuffle: bool,
        crop_from_start: bool = False,
    ):
        super().__init__(
            data_id=269,
            data_type="music_vocal",
            sample_rate=sample_rate,
            channels=channels,
            pad=True,
            segment_duration=segment_duration,
            shuffle_buffer_size=0,
            n_segments_per_read=1,
            resampled=resampled,
            shardshuffle=shardshuffle,
            crop_from_start=crop_from_start,
            nitems=184,
        )

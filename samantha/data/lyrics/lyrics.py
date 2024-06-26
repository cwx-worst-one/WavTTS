import random
from collections import Counter
from functools import cached_property
from itertools import chain
from typing import Dict, List, Tuple

from typing_extensions import Self


class LyricWord:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        text: str,
        language: str,
        confidence: float,
    ):
        # NOTE: if there was no end time detected, replace it with the last known end time (the last start time)
        if end_time is None:
            end_time = start_time

        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.language = language
        self.confidence = confidence

    def __repr__(self):
        return str(self.__dict__)


class LyricChunk:
    def __init__(
        self,
        start_time: float,
        end_time: float,
        text: str,
        language: str,
        confidence: float,
    ):
        # NOTE: if there was no end time detected, replace it with the last known end time (the last start time)
        if end_time is None:
            end_time = start_time

        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.language = language
        self.confidence = confidence
        self._words = []

    def __repr__(self):
        return str(self.__dict__)

    @property
    def words(self) -> List[LyricWord]:
        return self._words

    def add_word(self, w: LyricWord):
        self._words.append(w)

    def remove_word(self, idx: int) -> LyricWord:
        return self._words.pop(idx)


class Lyrics:
    def __init__(self):
        self._chunks = []
        self._raw_text = None

    @classmethod
    def load_from_dict(cls, lyrics_dict):
        cls = cls()
        cls._raw_text = lyrics_dict["text"]
        for c in lyrics_dict["chunks"]:
            chunk = LyricChunk(
                c["timestamp"][0],
                c["timestamp"][1],
                c["text"],
                c["language"],
                c["confidence"],
            )
            for word in c["words"]:
                word = LyricWord(
                    word["timestamp"][0],
                    word["timestamp"][1],
                    word["text"],
                    word["language"],
                    word["confidence"],
                )
                chunk.add_word(word)
            cls.add_chunk(chunk)

        # NOTE: filter out last timestamp if it has no start and end time:
        last_timestamp = cls.timestamps[-1]
        if last_timestamp[0] is None and last_timestamp[1] is None:
            cls.remove_chunk(-1)
        return cls

    @property
    def chunks(self) -> List[LyricChunk]:
        return self._chunks

    @property
    def words(self) -> List[LyricWord]:
        words = list(chain(*[c.words for c in self.chunks]))
        w_lyrics = Lyrics()
        w_lyrics._chunks = words
        return w_lyrics

    @property
    def timestamps(self) -> List[Tuple[float, float]]:
        return [(c.start_time, c.end_time) for c in self.chunks]

    @property
    def durations(self) -> List[float]:
        return [c.end_time - c.start_time for c in self.chunks]

    @property
    def raw_text(self) -> str:
        return self._raw_text

    @property
    def text(self) -> List[str]:
        return [c.text for c in self.chunks]

    @property
    def start_time(self) -> float:
        return self.chunks[0].start_time

    @property
    def end_time(self) -> float:
        return self.chunks[-1].end_time

    @property
    def language(self):
        languages = Counter([c.language for c in self.chunks])
        return languages.most_common(1)[0][0]

    def add_chunk(self, c: LyricChunk):
        self._chunks.append(c)

    def remove_chunk(self, idx: int) -> LyricChunk:
        return self._chunks.pop(idx)

    @staticmethod
    def find_timestamps_within_duration_range(
        timestamps: List[Tuple[float, float]], min_duration: float, max_duration
    ):
        # Flatten the list of timestamps to a single list of points, ensuring they are unique and sorted
        points = sorted(set([point for segment in timestamps for point in segment]))

        # List to store potential start and end points that fit within the min and max duration
        suitable_segments = []

        # Identify all segments that fit within the min and max duration
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                start_point = points[i]
                end_point = points[j]

                # If the segment duration fits within the specified range, add it to suitable segments
                if min_duration <= end_point - start_point <= max_duration:
                    suitable_segments.append((start_point, end_point))

        return suitable_segments

    def get_segments(self, min_duration: float, max_duration: float) -> Self:
        # Extract all timestamps from the lyrics data for selection
        all_timestamps = self.timestamps
        suitable_segments = self.find_timestamps_within_duration_range(
            all_timestamps, min_duration, max_duration
        )

        chunks = []
        for segment in suitable_segments:
            start_point, end_point = segment

            lyrics = Lyrics()

            for c in self.chunks:
                if c.start_time >= start_point and c.end_time <= end_point:
                    lyrics.add_chunk(c)
                elif c.end_time > end_point:
                    break

            if len(lyrics.chunks):
                chunks.append(lyrics)
        return chunks

import re
from random import Random
from typing import List, Optional, Union

import numpy as np


class BPMTextProcessor:

    def __init__(self, min_bpm: int, max_bpm: int, bpm_step: int):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.bpm_step = bpm_step
        self.bpm_bins = np.arange(min_bpm, max_bpm + 1, bpm_step)

    def process(self, bpm: str) -> Optional[str]:
        try:
            bpm = int(bpm)
            if self.min_bpm < bpm < self.max_bpm:
                bpm = self.bpm_bins[(np.abs(self.bpm_bins - bpm)).argmin()]
                bpm = str(bpm)
            else:
                bpm = None
        except Exception as e:
            bpm = None
        return bpm

    def __call__(self, batch_bpm: List[str]) -> List[Optional[str]]:
        return [self.process(bpm) for bpm in batch_bpm]


class WordProcessor:

    def __init__(self, min_word_len: int, seed: Optional[int] = None):
        self.min_word_len = min_word_len
        self.gen = Random(seed)

    def process_word(self, word: str):
        word = word.lower()
        word = word.strip()
        word = re.sub(r"[^A-Za-z0-9 /]+", "", word)
        if len(word) < self.min_word_len:
            return None
        return word

    def process(
        self, words: Optional[str], shuffle: bool, seed: Optional[int] = None
    ) -> str:
        if words is None:
            return None

        words = words.replace(",", ", ")
        words = [self.process_word(w) for w in words.split(" ")]
        words = [w for w in words if w is not None]
        if not len(words):
            return None

        if seed is not None:
            self.gen.seed(seed)

        if shuffle:
            self.gen.shuffle(words)
        return words

    def __call__(
        self, batch_words: List[str], shuffle: bool, seed: Optional[int] = None
    ) -> List[str]:
        return [self.process(words, shuffle, seed) for words in batch_words]


class SimpleTextProcessor:

    def __init__(
        self,
        sample_n_keywords: int,
        sample_n_genres: int,
        sample_n_instruments: int,
        seed: Optional[int] = None,
    ):
        self.sample_n_keywords = sample_n_keywords
        self.sample_n_genres = sample_n_genres
        self.sample_n_instruments = sample_n_instruments
        self.gen = Random(seed)

    def sample_if_exists(
        self, choices: Optional[List[str]], n: int
    ) -> Optional[List[str]]:
        if choices is None:
            return None

        n = min(len(choices), n)
        return self.gen.sample(choices, k=n)

    def flatten_if_exists(
        self, words: Optional[List[str]], delimiter: str
    ) -> Optional[str]:
        if words is None:
            return None
        return delimiter.join(words)

    def process(
        self,
        keywords: List[str],
        genres: List[str],
        insts: List[str],
        bpm: str,
        seed: Optional[int] = None,
    ) -> str:

        if seed is not None:
            self.gen.seed(seed)

        keywords = self.sample_if_exists(keywords, self.sample_n_keywords)
        genres = self.sample_if_exists(genres, self.sample_n_genres)
        insts = self.sample_if_exists(insts, self.sample_n_instruments)

        # flatten:
        keywords = self.flatten_if_exists(keywords, delimiter=", ")
        genres = self.flatten_if_exists(genres, delimiter=", ")
        insts = self.flatten_if_exists(insts, delimiter=", ")

        if bpm is not None:
            bpm = f"{bpm} bpm"
        delimiter = ", "

        # post-process:
        # 1. re-order blocks:
        block = [keywords, genres, insts, bpm]
        self.gen.shuffle(block)

        # 2 filter out None
        block = [b for b in block if b is not None]

        # 3. create final string
        final_str = f"{delimiter}".join(block)
        return final_str

    def __call__(
        self,
        batch_keywords: List[str],
        batch_genres: List[str],
        batch_instruments: List[str],
        batch_bpms: List[str],
        seed: int = 42,
    ) -> List[str]:
        return [
            self.process(keywords, genres, insts, bpm, seed + idx)
            for idx, (keywords, genres, insts, bpm) in enumerate(
                zip(batch_keywords, batch_genres, batch_instruments, batch_bpms)
            )
        ]


class MetadataTextProcessor:
    """
    This class creates text prompts from the metadata by concatenating a random subset of the metadata as a string.
    This allows for specific properties to be specified during inference, while not requiring these properties to be
    present at all times. For half of the samples, we include the metadata-type (e.g., Instruments or Moods) and join
    them with a delimiting character (e.g., Instruments: Guitar, Drums, Bass Guitar|Moods: Uplifting, Energetic).
    For the other half, we do not include the metadata-type and join the properties with a comma
    (e.g., Guitar, Drums, Bass Guitar, Uplifting, Energetic). For metadata-types with a list of values, we shuffle the
    list. Hence, we perform a variety of random transformations of the resulting string, including two variants of
    delimiting character (“,” and “|”), shuffling orders and transforming between upper and lower case
    """

    def __init__(
        self,
        sample_n_keywords: int,
        sample_n_genres: int,
        sample_n_instruments: int,
        seed: Optional[int] = None,
    ):
        self.sample_n_keywords = sample_n_keywords
        self.sample_n_genres = sample_n_genres
        self.sample_n_instruments = sample_n_instruments
        self.gen = Random(seed)

    def sample_if_exists(
        self, choices: Optional[List[str]], n: int
    ) -> Optional[List[str]]:
        if choices is None:
            return None

        n = min(len(choices), n)
        return self.gen.sample(choices, k=n)

    def add_type_if_exists(self, metadata_type: str, words: str) -> Optional[str]:
        if words is None:
            return None

        words = f"{metadata_type}: {words}"
        return words

    def flatten_if_exists(
        self, words: Optional[List[str]], delimiter: str
    ) -> Optional[str]:
        if words is None:
            return None
        return delimiter.join(words)

    def random_capitalize(self, words: str) -> str:
        if self.gen.random() > 0.5:
            return words.title()
        return words

    def process(
        self,
        keywords: List[str],
        genres: List[str],
        insts: List[str],
        bpm: str,
        seed: Optional[int] = None,
    ) -> str:

        if seed is not None:
            self.gen.seed(seed)

        keywords = self.sample_if_exists(keywords, self.sample_n_keywords)
        genres = self.sample_if_exists(genres, self.sample_n_genres)
        insts = self.sample_if_exists(insts, self.sample_n_instruments)

        # flatten:
        keywords = self.flatten_if_exists(keywords, delimiter=", ")
        genres = self.flatten_if_exists(genres, delimiter=", ")
        insts = self.flatten_if_exists(insts, delimiter=", ")

        if self.gen.random() > 0.5:
            keywords = self.add_type_if_exists("moods", keywords)
            genres = self.add_type_if_exists("genres", genres)
            insts = self.add_type_if_exists("instruments", insts)
            bpm = self.add_type_if_exists("bpm", bpm)
            delimiter = "|"
        else:
            if bpm is not None:
                bpm = f"{bpm} bpm"
            delimiter = ", "

        # post-process:
        # 1. re-order blocks:
        block = [keywords, genres, insts, bpm]
        self.gen.shuffle(block)

        # 2 filter out None
        block = [b for b in block if b is not None]

        # 3. create final string
        final_str = f"{delimiter}".join(block)

        # 4. capitalize
        final_str = self.random_capitalize(final_str)
        return final_str

    def __call__(
        self,
        batch_keywords: List[str],
        batch_genres: List[str],
        batch_instruments: List[str],
        batch_bpms: List[str],
        seed: int = 42,
    ) -> List[str]:
        return [
            self.process(keywords, genres, insts, bpm, seed + idx)
            for idx, (keywords, genres, insts, bpm) in enumerate(
                zip(batch_keywords, batch_genres, batch_instruments, batch_bpms)
            )
        ]


class MulanPretrainMetadataTextProcessor:

    def __init__(self, seed: Optional[int] = None):
        self.gen = Random(seed)

    def flatten_if_exists(
        self, words: Optional[List[str]], delimiter: str
    ) -> Optional[str]:
        if words is None:
            return None
        return delimiter.join(words)

    def process(
        self,
        descriptions: List[str],
        keywords: List[str],
        genres: List[str],
        insts: List[str],
        seed: Optional[int] = None,
    ) -> str:

        if seed is not None:
            self.gen.seed(seed)

        text_fields = [descriptions, keywords, genres, insts]
        text_fields = [t for t in text_fields if t is not None and t != ""]

        selected_fields = self.gen.sample(text_fields, k=self.gen.randint(1, 2))

        selected_fields = [" ".join(s) for s in selected_fields]
        selected_text = " ".join(selected_fields)

        ## post-process:
        ## replace multiple spaces:
        selected_text = re.sub(" +", " ", selected_text)

        return selected_text

    def __call__(
        self,
        batch_descriptions: List[str],
        batch_keywords: List[str],
        batch_genres: List[str],
        batch_instruments: List[str],
        seed: Optional[int] = None,
    ) -> List[str]:
        return [
            self.process(descriptions, keywords, genres, insts, seed + idx)
            for idx, (descriptions, keywords, genres, insts) in enumerate(
                zip(batch_descriptions, batch_keywords, batch_genres, batch_instruments)
            )
        ]

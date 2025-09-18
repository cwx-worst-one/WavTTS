import random

import zhconv

from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *


class TmpLyricsCapitalize(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = ["meta", "lyrics"],
        out_key: str = "lyrics",
        aug_rate: float = 0.5,
        line_aug_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.aug_rate = aug_rate
        self.line_aug_rate = line_aug_rate

    def call(self, item, **kwargs):
        meta, lyrics = item
        language = get_nested_value(
            meta, "standard_music_meta.sa_tagging_model.language"
        )
        if (
            not language
            or language[0] not in ["Cantonese", "English"]
            or random.random() > self.aug_rate
        ):
            return lyrics

        language = language[0]
        if language == "Cantonese":
            new_lyrics = zhconv.convert(lyrics, locale="zh-tw")
        elif language == "English":
            new_lyrics = []
            for line in lyrics.split("\n"):
                if random.random() > self.aug_rate:
                    new_lyrics.append(line)
                else:
                    if line[0] == "[":
                        new_line = "[" + line[1:].capitalize()
                    else:
                        new_line = line.capitalize()
                    new_lyrics.append(new_line)
            new_lyrics = "\n".join(new_lyrics)
        else:
            new_lyrics = lyrics

        return new_lyrics


class TmpAddNonVocal(MusicMetaRWTransform):
    def __init__(
        self, in_key: str = ["meta", "keywords"], out_key: str = "keywords", **kwargs
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)

    def call(self, item, **kwargs):
        meta, keywords = item
        language = get_nested_value(
            meta, "standard_music_meta.sa_tagging_model.language"
        )

        if not language or language[0] != "Non-vocal":
            return keywords

        keywords.append("Non-vocal")

        return keywords

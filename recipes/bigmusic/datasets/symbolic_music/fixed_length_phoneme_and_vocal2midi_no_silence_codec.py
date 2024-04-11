from __future__ import annotations
import os
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl

from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.fixed_length_phoneme_and_vocal2midi_codec import (
    FixedLengthPhonemeAndVocal2MidiCodec,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder
)


class FixedLengthPhonemeAndVocal2MidiNoSilenceCodec(FixedLengthPhonemeAndVocal2MidiCodec):
    def encode(self, dfs_dict: Dict[str, Any]) -> np.ndarray:
        try:
            leadsheet_tokens = self.encode_leadsheet(dfs_dict)
            ## We decode the leadsheet tokens and only keep bars with lyrics or notes
            bar_tokens = []
            keep = False
            result_tokens = []
            for t in leadsheet_tokens:
                if self.indexer.inverse[t] in ["bar", "eos"]:
                    if keep:
                        result_tokens.extend(bar_tokens)
                    keep = False
                    bar_tokens = []
                if self.indexer.inverse[t].startswith("stem"):
                    keep = True
                bar_tokens.append(t)
            leadsheet_tokens = self.chop_or_pad(result_tokens, self.config.leadsheet_seq_len - 1)
            leadsheet_tokens = np.append(leadsheet_tokens, self.indexer["eos"])
            if self.config.include_utterance_phoneme_tokens:
                lyrics_tokens = self.chop_or_pad(
                    self.encode_lyrics(dfs_dict)[:-1],
                    self.config.lyrics_seq_len - 1,
                )
                lyrics_tokens = np.append(lyrics_tokens, self.indexer["eol"])
                return np.concatenate([lyrics_tokens, leadsheet_tokens])
            return leadsheet_tokens
        except Exception as e:
            print(repr(e))
            from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

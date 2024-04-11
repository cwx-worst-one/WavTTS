from recipes.bigmusic.datasets.symbolic_music.fl_p2t5_codec import (
    FLP2T5Codec
)


class FLT5Codec(FLP2T5Codec):
    class Config(FLP2T5Codec.Config):
        lyrics_seq_len: int = 0
        include_utterance_phoneme_tokens: bool = False
        include_phoneme: bool = False
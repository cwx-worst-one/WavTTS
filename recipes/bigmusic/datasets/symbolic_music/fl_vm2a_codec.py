from recipes.bigmusic.datasets.symbolic_music.base import (
    IndexerConfig,
    SymbolicMusicCodecBase,
)


class FLVM2ACodec(SymbolicMusicCodecBase):
    class Config(IndexerConfig):
        pass

        
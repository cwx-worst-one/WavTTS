from functools import partial

from samantha.models.zoo.musicfm import MusicFM
from samantha.models.zoo.umm import UMM

MODEL_ZOO = {
    "musicfm": MusicFM,
    "umm-en-stage1-layer12": partial(UMM, model_version="en-stage1", layer_idx=12),
    "umm-en-stage2-layer12": partial(UMM, model_version="en-stage2", layer_idx=12),
    "umm-en-stage3-layer12": partial(UMM, model_version="en-stage3", layer_idx=12),
    "umm-zh-stage3-layer12": partial(UMM, model_version="zh-stage3", layer_idx=12),
    "umm-en-zh-stage3-layer12": partial(
        UMM, model_version="en-zh-stage3", layer_idx=12
    ),
}

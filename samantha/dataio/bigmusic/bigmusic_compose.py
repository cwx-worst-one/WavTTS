import copy
import importlib
import inspect

from mariana.data.audio.compose import BatchCompose, Compose

import samantha

# Use a shortcut to import all transforms, making it easier to test
from samantha.dataio.bigmusic.music_batch_transform import *

# NOTE: mariana's WavResample casts the type to int before casting it back,
# avoid using it if wav has been cast to float.
from samantha.dataio.bigmusic.music_meta_transform import (
    MusicWavResample as WavResample,
)
from samantha.dataio.bigmusic.music_meta_transform import *

# from samantha.dataio.bigmusic.music_meta_transform import (
#     LyricsTransform,
#     DummyItemTransform,
#     LyricsTransform,
#     FilterByLyricsConfidence,
#     ConvertMetaToDict,
#     StructureParser,
#     StyleTagParser,
#     FreeformTextParser,
#     SliceSplit,
#     SliceReformat,
# )


class ComposeWithSkipNum(Compose):
    def __call__(self, item, *args, **kwargs):
        return super().__call__(item, *args, **kwargs, skip_num=self.skip_num)


def get_aug_fn(aug_cfg, meta_data=None):
    """get augmentation object from config.
    Args:
        aug_cfg(dict): augmentation config, must have `type`.

    Returns:
        callable object: augmenation object.
    """
    aug_cfg_bak = copy.deepcopy(aug_cfg)
    cls_name = aug_cfg_bak.pop("type")
    segs = cls_name.rsplit(".", 1)
    if len(segs) == 1:
        cls = globals()[cls_name]
    else:
        module = importlib.import_module(segs[0])
        cls = getattr(module, segs[1])

    args = inspect.getfullargspec(cls.__init__).args
    if meta_data:
        if "bpe_code" in args:
            meta_data["bpe_code"] = meta_data.get("total.code")
        for key in args:
            if key in meta_data.keys() and key not in aug_cfg_bak.keys():
                aug_cfg_bak[key] = meta_data[key]
    obj = cls(**aug_cfg_bak)
    return obj


def build_item_augmentation(aug_cfg_list, meta_data=None):
    """get augmentation object from config.
    Args:
        aug_cfg_list(list): augmentation config list.

    Returns:
        Composed callable object: augmenation objects.
    """
    aug_list = []

    for aug_cfg in aug_cfg_list:
        aug_fn = get_aug_fn(aug_cfg, meta_data=meta_data)
        aug_list.append(aug_fn)

    return ComposeWithSkipNum(aug_list)


def build_draw_batch_fn(cfgs, meta_data=None):
    """
    build draw batch function.
    Args:
        cfgs(list): collate function list

    Return:
        callable object: draw batch fn.
    """
    collate_list = []

    for cfg in cfgs:
        collate_fn = get_aug_fn(cfg, meta_data=meta_data)
        collate_list.append(collate_fn)

    if len(collate_list) <= 0:
        return None

    return BatchCompose(collate_list)

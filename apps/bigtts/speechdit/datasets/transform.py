import io
import logging
import librosa
import math
import pickle
from typing import Optional

import numpy as np
import torch
from torchaudio.transforms import Resample

from samantha.dataio.lite.transform import ItemTransformBase
from samantha.dataio.lite.utils.masking import WavMasking
from samantha.dataio.lite.utils.parquet import get_meta_obj
from samantha.dataio.lite.utils.phone_to_id import PhoneToId
from samantha.dataio.lite.utils.wav_util import get_bn, get_mel, get_text_info, get_wav

logger = logging.getLogger(__name__)


class Transform1(ItemTransformBase):
    def __init__(
        self,
        max_length=30,
        use_text=True,
        cfg_drop_rate=0.1,
        text_drop_id=1,
        bn_config=None,
        use_phone_lang=False,
        p_drop_bn_ctx=0,
        masking=None,
    ):
        self.max_wav_len = int(max_length * 24000)
        self.use_text = use_text
        self.cfg_drop_rate = cfg_drop_rate
        self.text_drop_id = text_drop_id
        self.bn_config = bn_config
        self.use_phone_lang = use_phone_lang
        self.p_drop_bn_ctx = p_drop_bn_ctx
        self.masking = masking

        self.phone2id = PhoneToId()
        self.bn_hz = bn_config["sampling_rate"] // bn_config["hop_size"]

    def __call__(self, item):
        # sourcery skip: dict-assign-update-to-union, dict-literal, merge-dict-assign
        # load wav
        meta_obj = get_meta_obj(item)
        data_dict = dict()
        bn = get_bn(item)

        #wav, sr = librosa.load(io.BytesIO(item["wav"]), sr=None, mono=False)
        #print(bn.shape, wav.shape, sr)
        #exit()

        if bn.shape[1] != self.bn_config["bn_dim"] or bn.shape[0] < self.bn_hz * 0.3:
            return None

        data_dict["bn"] = bn
        data_dict["bn"] = (
            data_dict["bn"] - self.bn_config["bn_norm_mean"]
        ) / self.bn_config["bn_norm_std"]

        if self.cfg_drop_rate > 0:
            flag_drop = np.random.rand() <= self.cfg_drop_rate
        else:
            flag_drop = False

        data_dict["flag_drop"] = flag_drop

        text_info = get_text_info(
            item,
            meta_obj,
            self.use_text,
            flag_drop,
            self.text_drop_id,
            self.use_phone_lang,
        )
        if text_info is None:
            return None
        else:
            data_dict.update(text_info)


        # mask bn.
        data_dict = self.masking.masking(data_dict, "bn", flag_drop)  # update ctx & ctx_mask
        data_dict["bn"] = data_dict["bn"].transpose(0, 1)
        data_dict["bn_ctx"] = data_dict["bn_ctx"].transpose(0, 1)
        data_dict["bn_ctx_mask"] = data_dict["ctx_mask"]

        if flag_drop:
            data_dict["prompt_bn"] = data_dict["bn"]
        else:
            unmask_idx = torch.where(data_dict["bn_ctx_mask"]!=1)[0]
            data_dict["prompt_bn"] = data_dict["bn_ctx"][:, unmask_idx]

        if self.p_drop_bn_ctx > 0:
            if np.random.rand() <= self.p_drop_bn_ctx:
                data_dict["bn_ctx"][:] = self.bn_config["bn_padding"]
        return data_dict

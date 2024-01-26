import logging
import math
import pickle

import numpy as np
import torch

from samantha.dataio.lite.transform import ItemTransformBase
from samantha.dataio.lite.utils.masking import WavMasking
from samantha.dataio.lite.utils.parquet import get_meta_obj
from samantha.dataio.lite.utils.phone_to_id import PhoneToId
from samantha.dataio.lite.utils.wav_util import (
    get_bn,
    get_mel,
    get_text_info,
    get_wav
)

logger = logging.getLogger(__name__)


class VoiceBoxTransform(ItemTransformBase):
    def __init__(self,
        max_length=30,
        mel_config=None,
        mel_norm_mean=-5.8843,
        mel_norm_std=2.2615,
        mel_padding=-2,
        umm_hop_size=600,
        mask_p_drop_x=0.3,
        p_drop_audio_frames=(0.7, 1.0),
        mask_use_alignment=True,
        wav_divide=2400,
        use_text=False,
        text_drop_rate=0,
        wav_amp_aug=None,
        token_spec_aug=None,
        use_bn=False,
        bn_config=None,
        mel_vocoder_type="hifigan",
        use_phone_lang=False,
        token_type="umm",
    ):

        self.max_wav_len = int(max_length * 24000)
        self.phone2id = PhoneToId()
        self.wav_amp_aug = wav_amp_aug
        self.token_spec_aug = token_spec_aug
        self.mel_config = mel_config
        self.mel_norm_mean = mel_norm_mean
        self.mel_norm_std = mel_norm_std
        self.umm_hop_size = umm_hop_size
        self.umm_hz = 24000 // umm_hop_size
        self.use_text = use_text
        self.text_drop_rate = text_drop_rate
        self.wav_divide = wav_divide
        self.use_bn = use_bn
        self.bn_config = bn_config
        self.mel_vocoder_type = mel_vocoder_type
        self.token_type = token_type
        assert token_type in ["umm", "zvq"]
        self.use_phone_lang = use_phone_lang
        self.shot = 0
        self.total = 0
        self.audio_resampler = {}

        self.hop_ms = mel_config['hop_size'] / mel_config['sampling_rate']
        self.masking = WavMasking(
            p_drop_x=mask_p_drop_x,
            p_drop_audio_frames=p_drop_audio_frames,
            padding_value=bn_config["bn_padding"] if self.use_bn else mel_padding,
        )
        self.mask_use_alignment = mask_use_alignment

        if self.use_bn:
            self.bn_hz = 24000 // bn_config["hop_size"]
            lcm_umm_bn = math.lcm(self.umm_hz, self.bn_hz)
            self.lcm_umm_factor = lcm_umm_bn // self.umm_hz
            self.lcm_bn_factor = lcm_umm_bn // self.bn_hz


    def __call__(self, item): 
        # sourcery skip: dict-assign-update-to-union, dict-literal, merge-dict-assign
        # load wav
        meta_obj = get_meta_obj(item)
        data_dict = dict()
        data_dict["token"] = torch.as_tensor(
            pickle.loads(item[f"{self.token_type}_token"]), dtype=torch.long
        )
        if self.use_bn:
            bn = get_bn(item)
            assert bn.shape[1] == 64
            if self.bn_hz == self.umm_hz:
                max_bn_len = min(bn.shape[0], data_dict["token"].shape[0])
                max_umm_len = max_bn_len
            else:
                # align BN with UMM
                if (bn.shape[0] / self.bn_hz) < (data_dict["token"].shape[0] / self.umm_hz):
                    max_lcm_len = int(bn.shape[0] * self.lcm_bn_factor)
                else:
                    max_lcm_len = int(data_dict["token"].shape[0] * self.lcm_umm_factor)
                max_lcm_len -= max_lcm_len % (self.lcm_umm_factor*self.lcm_bn_factor)

                max_bn_len = max_lcm_len // self.lcm_bn_factor
                max_umm_len = max_lcm_len // self.lcm_umm_factor
            acoustic_len = max_bn_len
            data_dict["bn"] = bn[:max_bn_len, :]
            data_dict["bn"] = (
                (data_dict["bn"] - self.bn_config['bn_norm_mean']) /
                self.bn_config['bn_norm_std']
            )
        else:
            wav = get_wav(
                item, self.max_wav_len, self.audio_resampler,
                self.wav_amp_aug, self.mel_config
            )
            if wav is None:
                return None
            mel, max_mel_len, max_umm_len = get_mel(
                wav, self.wav_divide, self.mel_config,
                self.umm_hop_size, self.mel_norm_mean,
                self.mel_norm_std, self.mel_vocoder_type
            )
            data_dict['mel'] = mel.transpose(1, 0)
            acoustic_len = mel.shape[1]
        # data_dict['acoustic_len'] = acoustic_len

        text_info = get_text_info(
            item, meta_obj, acoustic_len, self.hop_ms, self.mask_use_alignment,
            self.use_text, self.text_drop_rate, self.use_phone_lang)
        if text_info is None:
            return None
        else:
            data_dict.update(text_info)

        # align token len with mel/bn length; spec_aug token;
        if "token" in data_dict:
            data_dict["token"] = data_dict["token"][:max_umm_len]

        if self.use_bn:
            # mask bn.
            data_dict = self.masking.masking(data_dict, "bn") # update ctx & ctx_mask
            data_dict['bn'] = data_dict["bn"].transpose(0, 1)
            data_dict['bn_ctx'] = data_dict["bn_ctx"].transpose(0, 1)
            data_dict['bn_ctx_mask'] = data_dict["ctx_mask"]
        else:
            data_dict = self.masking.masking(data_dict, "mel") # update ctx & ctx_mask
            data_dict['mel_ctx'] = data_dict['mel_ctx'].transpose(0, 1)[:, :max_mel_len]
            data_dict['mel'] = data_dict['mel'].transpose(0, 1)[:, :max_mel_len]
            data_dict['mel_ctx_mask'] = data_dict['ctx_mask'][:max_mel_len]

        return data_dict

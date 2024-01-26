import logging
import random

import numpy as np
import torch

from samantha.dataio.lite.transform import CollatorBase
from samantha.dataio.lite.utils.pad import collate_1d, collate_2d
from samantha.dataio.lite.utils.wav_util import sequence_mask

logger = logging.getLogger(__name__)


class VoiceBoxCollator(CollatorBase):
    def __init__(self,
            tokenizer_pad,
            mel_config=None,
            mel_padding=-2,
            max_crop_second=30.0,
            bn_config=None,
            bn_padding=-5,
            use_bn=True
    ):
        self.tokenizer_pad = tokenizer_pad

        self.use_bn = use_bn
        if use_bn:
            self.feat_hz = mel_config["sampling_rate"] // bn_config["hop_size"]
            self.feat_padding = bn_padding
        else:
            self.feat_hz = mel_config["sampling_rate"] // mel_config["hop_size"]
            self.feat_padding = mel_padding

        self.max_crop_len = int(max_crop_second * self.feat_hz)

    def __call__(self, batches):
        results = [item for item in batches if item is not None]
        if not results:
            return None
        token_lens = [b['token'].shape[0] for b in batches]
        max_token_len = max(token_lens)
        token = collate_1d(
            [b["token"] for b in batches],
            pad_idx=self.tokenizer_pad,
            max_len=max_token_len
        )
        ret_dict = {'token': token}
        ret_dict["token_mask"] = sequence_mask(torch.from_numpy(np.array(token_lens)),
                  max_len=ret_dict["token"].shape[1])

        # pad bn/mel
        feat_name = "bn" if self.use_bn else "mel"
        feat_lens = [b[feat_name].shape[1] for b in batches]
        max_feat_len = max(feat_lens)
        min_feat_len = min(feat_lens)

        feats = collate_2d(
            [b[feat_name] for b in batches],
            pad_idx=self.feat_padding,
            max_len=max_feat_len)

        feat_ctx = collate_2d(
            [b[f'{feat_name}_ctx'] for b in batches],
            pad_idx=self.feat_padding,
            max_len=max_feat_len)

        feat_ctx_mask = collate_1d(
            [b[f'{feat_name}_ctx_mask'] for b in batches],
            pad_idx=0.0,
            max_len=max_feat_len)

        ret_dict[feat_name] = feats.transpose(1, 2)  # [B, T, C]
        ret_dict[f"{feat_name}_ctx"] = feat_ctx.transpose(1, 2)
        ret_dict[f"{feat_name}_lens"] = torch.from_numpy(np.array(feat_lens))
        ret_dict[f"{feat_name}_mask"] = sequence_mask(
            seq_lens=ret_dict[f"{feat_name}_lens"],
            max_len=max_feat_len)
        ret_dict[f"{feat_name}_ctx_mask"] = feat_ctx_mask

        utt_ids = [b['utt_id'] for b in batches]

        if "phone" in batches[0]:
            text_lens = [b['phone'].shape[0] for b in batches]
            max_text_len = max(text_lens)
            phones = collate_1d(
                [torch.LongTensor(b['phone']) for b in batches],
                pad_idx=0, max_len=max_text_len
            )
            tones = collate_1d(
                [torch.LongTensor(b['tone']) for b in batches],
                pad_idx=0, max_len=max_text_len
            )
            word_segs = collate_1d(
                [torch.LongTensor(b['word_seg']) for b in batches],
                pad_idx=0, max_len=max_text_len
            )
            ret_dict["phone"] = phones
            ret_dict["tone"] = tones
            ret_dict['word_seg'] = word_segs
            ret_dict['text_lens'] = torch.from_numpy(np.array(text_lens))
            ret_dict["text_mask"] = sequence_mask(ret_dict["text_lens"],
                                    max_len=max_text_len)
            ret_dict["text_mel_mask"] = sequence_mask(
                ret_dict[f"{feat_name}_lens"] + ret_dict['text_lens'],
                max_len=torch.max(
                    ret_dict[f"{feat_name}_lens"] + ret_dict['text_lens'])
            )
            if "lang" in batches[0]:
                ret_dict["lang"] = collate_1d(
                    [torch.LongTensor(b['lang']) for b in batches],
                    pad_idx=0, max_len=max_text_len
                )

        crop_len = max(self.feat_hz, np.random.rand() * min_feat_len)
        crop_len = int(min(self.max_crop_len , crop_len))
        start_point = random.randint(0,  max(min_feat_len - crop_len - 1, 0))
        ret_dict[f"prompt_{feat_name}"] = (
            ret_dict[feat_name][:, start_point:start_point+crop_len, :].transpose(1, 2)
        )

        return ret_dict

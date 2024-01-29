from collections import defaultdict

import torch
import numpy as np

from samantha.dataio.lite.transform import CollatorBase


class ContinuousCollator(CollatorBase):
    def __init__(
        self,
        tokenizer_pad,
        block_sparse=False,
        use_bpe=False,
        use_extra_tag=False,
        min_crop_ratio=0.15,
        max_crop_ratio=0.6,
    ):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse
        self.use_bpe = use_bpe
        self.use_extra_tag = use_extra_tag
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio

    def __call__(self, batches):
        results = []
        for item in batches:
            if item is not None:
                results.append(item)

        if len(results) == 0:
            return None
        ret_dict = defaultdict(list)

        text_lens = []
        bn_lens = []
        byt5_lens = []
        wav_lens = []
        spk_embd_masks = []
        # utt_ids = []
        # tag_ids = []
        for x in results:
            text_lens.append(x["phone"].shape[0])
            if self.use_bpe:
                byt5_lens.append(x["byt5"].shape[0])
            else:
                byt5_lens = None
            bn_lens.append(x["bn"].shape[0])
            wav_lens.append(x["wav"].shape[0])
            spk_embd_masks.append(x["spk_embd_mask"])
            # bpe_lens.append(x["bpe_seq"].shape[0])
            # utt_ids.append(x["utt_id"])
            # if x["tag_id"] is not None:
            #     tag_ids.append(x["tag_id"])

        max_text_len = max(text_lens)
        max_bn_len = max(bn_lens)
        max_wav_len = max(wav_lens)
        if self.use_bpe:
            max_byt5_len = max(byt5_lens)

        text_lens = torch.from_numpy(np.asarray(text_lens))
        bn_lens = torch.from_numpy(np.asarray(bn_lens))
        wav_lens = torch.from_numpy(np.asarray(wav_lens))
        if self.use_bpe:
            byt5_lens = torch.from_numpy(np.asarray(byt5_lens))

        max_seq_len = max(text_lens + bn_lens)

        ret_dict["text_lens"] = text_lens
        ret_dict["bn_lens"] = bn_lens
        ret_dict["byt5_lens"] = byt5_lens
        ret_dict["wav_lens"] = wav_lens / max_wav_len
        ret_dict["spk_embd_masks"] = spk_embd_masks
        # ret_dict["utt_id"] = utt_ids
        # if len(tag_ids) > 0:
        #     ret_dict["tag_id"] = np.asarray(tag_ids)

        min_crop_len = int(self.min_crop_ratio * min(bn_lens))
        max_crop_len = int(self.max_crop_ratio * min(bn_lens))
        crop_len = np.random.randint(min_crop_len, max_crop_len)
        crop_bns = []

        # length padding
        for x in results:
            for k, v in x.items():
                if v is None:
                    ret_dict[k] = None
                    continue
                if k == "__key__":
                    continue
                if k == "bn":
                    crop_begin = np.random.randint(0, v.shape[0] - crop_len)
                    crop_bns.append(v[crop_begin : crop_begin + crop_len])
                    v = np.pad(
                        v,
                        ((0, max_bn_len - v.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "stop_token":
                    v = np.pad(
                        v,
                        (0, max_seq_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "lang_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "spk_seq":
                    v = np.pad(
                        v,
                        (0, max_bn_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k == "byt5":
                    if self.use_bpe:
                        v = np.pad(
                            v,
                            ((0, max_byt5_len - v.shape[0]), (0, 0)),
                            mode="constant",
                            constant_values=self.pad,
                        )
                elif k == "wav":
                    v = np.pad(
                        v,
                        (0, max_wav_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                elif k not in ["utt_id", "tag_id", "spk_embd_mask"]:
                    v = np.pad(
                        v,
                        (0, max_text_len - v.shape[0]),
                        mode="constant",
                        constant_values=self.pad,
                    )
                ret_dict[k].append(v)

        for k in ret_dict.keys():
            if k == "bn":
                ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k == "wav":
                ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k == "byt5":
                if self.use_bpe:
                    ret_dict[k] = np.stack(ret_dict[k], axis=0)
            elif k != "utt_id":
                if ret_dict[k] is not None:
                    ret_dict[k] = np.asarray(ret_dict[k])
            else:
                continue
            if ret_dict[k] is not None:
                try:
                    ret_dict[k] = torch.from_numpy(ret_dict[k])
                except Exception:
                    print("ret_dict[k]: ", ret_dict[k])

        ret_dict["crop_bn"] = torch.from_numpy(np.stack(crop_bns, axis=0))

        to_long_list = [
            "phone",
            "tone",
            "stop_token",
            "lang_seq",
            "spk_seq",
            "tag_id",
            "phonetone",
        ]
        for x in to_long_list:
            if x in ret_dict:
                if ret_dict[x] is not None:
                    ret_dict[x] = ret_dict[x].long()
        return ret_dict

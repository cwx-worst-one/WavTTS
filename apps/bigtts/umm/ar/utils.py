import logging
from itertools import zip_longest
from typing import List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def process_eos_indexes(semantic_samples, semantic_module, sample_rate=24000):
    semantic_frame_rate = semantic_module.extra_params.semantic_frame_rate
    eos_id = semantic_module.target_embedder.eos_id
    eos_index_list = []
    if eos_id is not None:
        eos_padding_id = 0
        eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
        semantic_samples[eos_mask] = eos_padding_id
        print(f"semantic_samples: {semantic_samples}")
        token2wav_rate = int(sample_rate / semantic_frame_rate)
        eos_index_list = (
            (semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0
        ).bool().sum(axis=1) * token2wav_rate
    return semantic_samples, eos_index_list


def truncate_wav_to_eos(wavs, eos_index_list):
    truncated_wavs = []
    for i, (eos, wav) in enumerate(zip_longest(eos_index_list, wavs)):
        if eos is not None:
            wav = wav[:eos]
        truncated_wavs.append(wav)
    return truncated_wavs


def get_split_emb(
    _emb: torch.Tensor,
    prompt_lens: torch.Tensor,
    crop_len: int,
    minlen: int,
    least_one=False,
):
    device = _emb.device
    B, T, C = _emb.shape

    count = prompt_lens // crop_len
    remain = prompt_lens % crop_len
    slice_count = (
        (count + (remain > minlen)) if minlen > 0 else count
    )  # only get greater than minlen

    if not slice_count.any():
        if least_one:
            # select at least one
            i = torch.nonzero(remain > 0)[0][0]
            slice_count[i] = remain[i]
        else:
            return None, None

    T_new = (T + crop_len - 1) // crop_len * crop_len  # 确保是crop_len的整数倍，方便后面切分
    # 直接pad原始tensor为crop len的整数倍，方便后面处理
    # new_emb = torch.zeros(B, T_new, C, device=_emb.device, dtype=_emb.dtype)
    new_emb = F.pad(_emb, (0, 0, 0, T_new - T), "constant")

    B_index = torch.arange(B, device=device)
    # T_index = torch.arange(T, device=device)
    T_new_index = torch.arange(T_new, device=device)
    prompt_lens_trunc = count * crop_len

    assert T_new % crop_len == 0, "is padding wrong?"
    select_index = T_new_index[None, :] < (slice_count * crop_len)[:, None]

    # 处理padding, 长度足够延伸长度
    extend_cond = (prompt_lens > crop_len) & (remain > 0)
    if extend_cond.any():
        extend_left_mask = (
            extend_cond[:, None]
            & (T_new_index[None, :] >= (prompt_lens_trunc)[:, None])
            & (T_new_index[None, :] < (prompt_lens_trunc + crop_len)[:, None])
        )
        extend_right_mask = (
            extend_cond[:, None]
            & (T_new_index[None, :] >= (prompt_lens - crop_len)[:, None])
            & (T_new_index[None, :] < (prompt_lens)[:, None])
        )
        new_emb[extend_left_mask] = new_emb[extend_right_mask]
    # 处理padding，长度不够延伸，那需要重复自身
    # 先选出需要处理的
    repeat_cond = (prompt_lens < crop_len) & (remain > 0)  # & (prompt_lens > minlen)
    if repeat_cond.any():
        repeat_mask = (
            repeat_cond[:, None]
            & (T_new_index[None, :] >= prompt_lens_trunc[:, None])
            & (T_new_index[None, :] < (prompt_lens_trunc + crop_len)[:, None])
        )
        to_repeat = new_emb[repeat_mask].view(-1, crop_len, C)
        to_repeat_len = remain[repeat_cond]
        # repeat 赋值操作
        for ii in range(to_repeat.size(0)):
            copy_time = crop_len // to_repeat_len[ii] + 1
            to_repeat[ii] = to_repeat[ii][: to_repeat_len[ii], :].repeat(copy_time, 1)[
                -crop_len:, :
            ]
        # 填回
        new_emb[repeat_mask] = to_repeat.view(-1, C)

    M = slice_count.sum()
    select_emb = new_emb[select_index].view(M, crop_len, -1)
    scatter_index = B_index[:, None].repeat(1, T_new)[select_index].view(M, crop_len)
    return select_emb, scatter_index


def reorder_attr(obj, attr_order_list):
    """hepler function to reset attr"""
    names = []
    values = []
    for attr_name in attr_order_list:
        if hasattr(obj, attr_name):
            names.append(attr_name)
            values.append(getattr(obj, attr_name))
            delattr(obj, attr_name)
    for name, value in zip(names, values):
        setattr(obj, name, value)


def sequence_mask(seq_lens, max_len=None, device="cpu"):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    mask = mask.int()
    return mask


class TokenBuffer:
    def __init__(
        self, max_length, init_temperature=0.9, max_temperature=10000, max_retry_times=1
    ):
        self.max_length = max_length
        self.init_temperature = init_temperature
        self.max_temperature = max_temperature
        self.max_retry_times = max_retry_times
        self.buffer: List[int] = []

    def put(self, token: int):
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
        self.buffer.append(token)

    def is_duplicate(self):
        if len(self.buffer) < self.max_length:
            return False
        first = self.buffer[0]
        return first in self.buffer[1:]

    def process_duplicate(self, step, sampler, **sampler_kwargs):
        if not self.is_duplicate():
            return None
        predict_token = None
        high_temperature = self.init_temperature
        while self.is_duplicate():
            count = 0
            while count < self.max_retry_times and self.is_duplicate():
                logger.warning(
                    f"TimeStep={step}|Fall into silence loop. {self.buffer=}, temperature={high_temperature}"
                )
                predict_token = sampler(temp=high_temperature, **sampler_kwargs)
                predict_token = predict_token[:, None]
                self.put(predict_token.item())
                count += 1
            high_temperature = min(high_temperature + 0.1, self.max_temperature)
        return predict_token


def _emb_select(x: torch.Tensor, indices: torch.Tensor, crop_len: int, h_len: int):
    """split and pad x through indices"""
    return x.view(-1, h_len)[indices].view(-1, crop_len, h_len)


def _custom_cat(
        lyrics_tokens:torch.Tensor,
        sos_ids:torch.Tensor,
        target_ids:torch.Tensor,
        input_lens:torch.Tensor,
        target_lens:torch.Tensor,
        bsz:int,
        t: int,
):
    """
    Fuse the lyrics_tokens, sos_ids, target_ids into one tensor.

    The orginal code is:
    ```python
        h = torch.zeros([bsz, t], device=device).long()
        for i in range(bsz):
            h[i, : input_lens[i] + 1 + 1 + target_lens[i]+1] = torch.cat(
                (
                    torch.zeros([1]).to(sos_ids.device), # placeholder
                    lyrics_tokens[i, :input_lens[i]],
                    # torch.zeros([prompt_lens[i]]).to(sos_ids.device), # query placeholder
                    sos_ids[i, :],
                    target_ids[i, :target_lens[i]+1]))
    ```
    """
    device = target_ids.device
    h = torch.zeros([bsz, t], device=device, dtype=torch.long)

    tN = torch.arange(t, device=device)
    tN_ = tN[None, :]
    input_lens_ = input_lens[:, None]
    target_lens_ = target_lens[:, None]
    # 0, 1, input_lens+1, input_lens+2, input_lens+2 + (target_lens + 1)
    # h[0] is zeros, ignore
    h_lyrics_mask = (tN_ >= torch.ones((bsz, 1), device=device, dtype=torch.long)) & (tN_ < (input_lens_ + 1))
    lyrics_mask = torch.arange(lyrics_tokens.size(1), device=device)[None, :] < input_lens_
    h[h_lyrics_mask] = lyrics_tokens[lyrics_mask]
    h_sos_ids_mask = torch.arange(bsz, device=device)*t + input_lens + 1
    h.flatten()[h_sos_ids_mask] = sos_ids[:, 0]   # sos_ids: [bsz, 1]
    h_target_mask = (tN_ >= (input_lens_ + 2)) & (tN_ < (input_lens_+target_lens_+3))
    target_mask = torch.arange(target_ids.size(1), device=device)[None, :] < (target_lens_ + 1)

    h[h_target_mask] = target_ids[target_mask]

    return h

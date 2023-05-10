'''
draw batch functions.
'''
# pylint:disable=too-many-lines

import io
import random
import numpy as np
import torch
import subword_nmt.apply_bpe
from core.utils import flatten_list
from core.utils.math import ceil
from core.utils.wav_util import get_frames_len, get_wav_len
from .preprocess import PREPROCESS
from .label import split_labels, read_words


FRAME_CHUNK_SIZE = 32
FBANK_DIM = 80
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001


@PREPROCESS.register_module()
class ListCollate:
    '''collate uttid.'''

    def __init__(self, key='uttid', out_key=None):
        '''init list collate function, setup list key.'''
        self.key = key
        if out_key is None:
            self.out_key = self.key
        else:
            self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.key in item for item in bucket_list) != bsz:
            return
        batch_out[self.out_key] = [item[self.key] for item in bucket_list]


@PREPROCESS.register_module()
class RefLabelCollate:
    '''reference label collate.'''

    def __init__(self, key='label', out_key=None):
        '''init.'''
        self.in_key = key
        if out_key is None:
            self.out_key = key.replace('label', 'ref')
        else:
            self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''
        do reference label collation.
        This function is for inference.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        ref_list = []
        for item in bucket_list:
            ref = item[self.in_key]
            if isinstance(ref, (list, tuple)):
                ref = ' '.join(ref)
            ref_list.append(ref)
        batch_out[self.out_key] = ref_list


@PREPROCESS.register_module()
class TimestampCollate:
    '''collate timestamp.'''

    def __init__(self, key='timestamp', sub_key=None):
        '''init list collate function, setup list key.'''
        self.key = key
        self.sub_key = sub_key

    def __call__(self, bucket_list, batch_out):
        '''
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.key in item for item in bucket_list) != bsz:
            return
        if self.sub_key is not None:
            batch_out[self.key] = [
                list(map(lambda x: x[self.sub_key], item[self.key])) for item in bucket_list
            ]
        else:
            batch_out[self.key] = [item[self.key] for item in bucket_list]


@PREPROCESS.register_module()
class DomainCollate:
    '''collate domain.'''

    def __init__(self, key='domain'):
        '''init list collate function, setup list key.'''
        self.key = key

    def __call__(self, bucket_list, batch_out):
        '''
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.key in item for item in bucket_list) != bsz:
            return

        tensor_domain = torch.zeros(bsz, dtype=torch.int64)
        for bid, item in enumerate(bucket_list):
            tensor_domain[bid] = item[self.key]
        batch_out[self.key] = tensor_domain


@PREPROCESS.register_module()
class LangCollate:
    '''collate lang.'''

    def __init__(self, key='lang'):
        '''init list collate function, setup list key.'''
        self.key = key

    def __call__(self, bucket_list, batch_out):
        '''
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.key in item for item in bucket_list) != bsz:
            return

        tensor_lang = torch.zeros(bsz, dtype=torch.int64)
        for bid, item in enumerate(bucket_list):
            tensor_lang[bid] = item[self.key]
        batch_out[self.key] = tensor_lang


@PREPROCESS.register_module()
class RareWordsCollate:
    '''rare words collate'''

    def __init__(
        self,
        key='rare_words',
        out_key='bias_words',
        tgt_dict=None,
        reorder_tgt_dict=None,
        bpe_code=None,
        distractors_path=None,
        distractors_min_count=0,
        distractors_max_count=0,
    ):
        '''init'''
        self.key = key
        self.out_key = out_key
        if reorder_tgt_dict is None:
            self.tgt_dict = tgt_dict
        else:
            self.tgt_dict = reorder_tgt_dict
        assert self.tgt_dict is not None
        assert bpe_code is not None
        self.bpe_fn = subword_nmt.apply_bpe.BPE(io.StringIO(bpe_code))
        assert distractors_max_count >= distractors_min_count
        self.distractors_min_count = distractors_min_count
        self.distractors_max_count = distractors_max_count
        self.distractors_words = []
        if distractors_path is not None:
            self.distractors_words = read_words(distractors_path, is_distractors=True)
        if distractors_min_count > 0:
            assert distractors_max_count <= len(self.distractors_words)

    def add_distractors(self, rare_words, max_len):
        '''add distractors'''
        distractors_count = random.randint(
            max(0, self.distractors_min_count - len(rare_words)),
            max(0, self.distractors_max_count - len(rare_words)),
        )
        if distractors_count == 0:
            if len(rare_words) > self.distractors_max_count:
                rare_words = random.sample(rare_words, self.distractors_max_count)
            return rare_words, max_len
        distractors_words = random.sample(self.distractors_words, distractors_count)
        distractors_bpes = self.bpe(distractors_words)
        for word in distractors_bpes:
            if word not in rare_words:
                rare_words.append(word)
                max_len = max(len(word), max_len)
        return rare_words, max_len

    def bpe(self, rare_words):
        '''bpe for rare words'''
        bpe_list = []
        for word in rare_words:
            word = " ".join(split_labels(word))
            word_bpe = self.bpe_fn.process_line(word)
            bpe_ids = self.tgt_dict.encode_line(word_bpe, append_eos=False, add_if_not_exist=False)
            bpe_list.append(bpe_ids.tolist())
        return bpe_list

    def __call__(self, bucket_list, batch_out):
        '''
        do rare words collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        max_len = 1
        rare_words = []
        for item in bucket_list:
            for word in self.bpe(item[self.key]):
                if word not in rare_words:
                    rare_words.append(word)
                    max_len = max(len(word), max_len)
        if self.distractors_min_count > 0:
            rare_words, max_len = self.add_distractors(rare_words, max_len)
        rare_token = torch.zeros(len(rare_words) + 1, max_len, dtype=torch.int64)
        for i, word in enumerate(rare_words):
            cur_len = len(word)
            rare_token[i + 1, -1 * cur_len :] = torch.LongTensor(word)
        batch_out[self.out_key] = rare_token


@PREPROCESS.register_module()
class InferenceRareWordsCollate(RareWordsCollate):
    '''rare words collate'''

    def __init__(
        self,
        in_key='rare_words',
        out_key='bias_words',
        distractors_path=None,
        tgt_dict=None,
        reorder_tgt_dict=None,
        bpe_code=None,
    ):
        super().__init__(in_key, out_key, tgt_dict, reorder_tgt_dict, bpe_code)
        self.rare_words = read_words(distractors_path, is_distractors=True)
        self.bpe_list = self.bpe(self.rare_words)
        self.rare_token = self.build_rare_token()

    def build_rare_token(self):
        '''build_rare_token'''
        max_len = 1
        rare_words = []
        for word in self.bpe_list:
            if word in rare_words:
                continue
            rare_words.append(word)
            max_len = max(len(word), max_len)
        rare_token = torch.zeros(len(rare_words) + 1, max_len, dtype=torch.int64)
        for i, word in enumerate(rare_words):
            cur_len = len(word)
            rare_token[i + 1, -1 * cur_len :] = torch.LongTensor(word)
        return rare_token

    def __call__(self, bucket_list, batch_out):
        batch_out[self.out_key] = self.rare_token


@PREPROCESS.register_module()
class FbankCollate:
    '''fbank collate.'''

    def __init__(
        self,
        frame_chunk_size=FRAME_CHUNK_SIZE,
        fbank_dim=FBANK_DIM,
        key='fbank',
        out_key=None,
        reserve_frames=0,
        mask_key='src_mask',
        strict_mode=True,
    ):
        '''
        do init.
        Args:
            reserve_frames: reserve some frames for unfold.
                            Make sure frames aligned when `src` reach LSTM module.
        '''
        self.chunk_size = frame_chunk_size
        self.fbank_dim = fbank_dim
        self.in_key = key
        if out_key is None:
            self.out_key = 'src'
        else:
            self.out_key = out_key
        self.reserve_frames = reserve_frames
        self.mask_key = mask_key
        self.strict_mode = strict_mode

    def __call__(self, bucket_list, batch_out):
        '''
        do fbank collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if self.strict_mode and sum(self.in_key in item for item in bucket_list) != bsz:
            return

        frame_length = []
        for item in bucket_list:
            if self.in_key in item:
                frame_length.append(item[self.in_key].shape[0])
            else:
                # fake feature
                frame_length.append(self.chunk_size)
        max_frame_length = max(frame_length)
        max_frame_length += self.reserve_frames
        max_frame_length = ceil(max_frame_length, self.chunk_size)
        max_frame_length -= self.reserve_frames

        tensor_fbank = torch.zeros(bsz, max_frame_length, self.fbank_dim, dtype=torch.float32)
        if self.mask_key not in batch_out:
            tensor_src_mask = torch.zeros(bsz, max_frame_length, dtype=torch.float32)

        for bid, item in enumerate(bucket_list):
            if self.in_key not in item:
                # set feat to all-zero for fake feature
                continue
            fbank = item[self.in_key]
            frame_length = fbank.shape[0]
            tensor_fbank[bid, 0:frame_length] = torch.from_numpy(fbank)
            if self.mask_key not in batch_out:
                tensor_src_mask[bid, 0:frame_length] = 1
        if self.fbank_dim == 1:
            # for TOG feature, the value is token index, and will be feed
            # into an embedding module
            batch_out[self.out_key] = tensor_fbank.squeeze(-1)
        else:
            batch_out[self.out_key] = tensor_fbank
        if self.mask_key not in batch_out:
            batch_out[self.mask_key] = tensor_src_mask


@PREPROCESS.register_module()
class CharCollate:
    '''char collation.'''

    def __init__(
        self,
        tgt_dict=None,
        key='char',
        reverse=False,
        bos_id=None,
        eos_penalty=False,
        pad_eos=False,
        las_format=False,
        set_tgt_length=True,
        **_kwargs,
    ):
        '''
        init.
        '''
        self.tgt_dict = tgt_dict
        self.key = key
        self.mask_key = key + '_mask'
        self.reverse = reverse
        self.reverse_key = key + '_rev'
        self.bos_id = bos_id
        self.eos_penalty = eos_penalty
        self.pad_eos = pad_eos
        self.las_format = las_format
        self.set_tgt_length = set_tgt_length

    def __call__(self, bucket_list, batch_out):
        '''
        do char collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        max_char_length = max(item[self.key].shape[0] for item in bucket_list)
        if self.las_format:
            max_char_length += 1

        tensor_char = torch.zeros(bsz, max_char_length, dtype=torch.int64)
        if self.pad_eos:
            tensor_char.fill_(self.tgt_dict.eos())

        if self.reverse:
            tensor_char_rev = torch.zeros(bsz, max_char_length, dtype=torch.int64)
        tensor_char_mask = torch.zeros(bsz, max_char_length, dtype=torch.float32)
        target_lengths = np.zeros((bsz,), dtype=np.int64)
        if self.eos_penalty:
            eos_alignment = torch.zeros(bsz, dtype=torch.int64)

        for bid, item in enumerate(bucket_list):
            char = item[self.key]  # NOTE: char.dtype is int32
            t_char = torch.from_numpy(char).long()
            char_length = char.shape[0]
            tensor_char[bid, 0:char_length] = t_char
            if self.reverse:
                tensor_char_rev[bid, 0 : char_length - 1] = t_char[:-1].flip(dims=[0])
                if self.bos_id is not None:
                    tensor_char_rev[bid, char_length - 1] = self.bos_id
                else:
                    tensor_char_rev[bid, char_length - 1] = self.tgt_dict.bos()
            tensor_char_mask[bid, 0:char_length] = 1
            target_lengths[bid] = char_length
            # prepare for eos penalty
            if self.eos_penalty and 'eos' in item:
                eos_alignment[bid] = item['eos']

        batch_out[self.key] = tensor_char
        batch_out[self.mask_key] = tensor_char_mask
        if self.set_tgt_length:
            batch_out['target_lengths'] = target_lengths
        if self.reverse:
            batch_out[self.reverse_key] = tensor_char_rev
        if self.eos_penalty:
            batch_out['eos'] = eos_alignment


@PREPROCESS.register_module()
class DocSentCollate:
    '''Doc collation.'''

    def __init__(
        self,
        tgt_dict=None,
        key='context_text',
        bos_id=None,
        pad_eos=False,
        **_kwargs,
    ):
        '''
        init.
        '''
        self.tgt_dict = tgt_dict
        self.key = key
        self.mask_key = key + '_mask'
        self.bos_id = bos_id
        self.pad_eos = pad_eos

    def __call__(self, bucket_list, batch_out):
        '''
        do char collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        max_sent_length, max_char_length = 0, 0
        for item in bucket_list:
            max_sent_length = max(len(item[self.key]), max_sent_length)
            for sent in item[self.key]:
                max_char_length = max(len(sent), max_char_length)
        tensor_char = torch.zeros(bsz, max_sent_length, max_char_length, dtype=torch.int64)
        tensor_char_mask = torch.zeros(bsz, max_sent_length, max_char_length, dtype=torch.float32)
        target_lengths = np.zeros((bsz, max_sent_length), dtype=np.int64)
        for bid, item in enumerate(bucket_list):
            sent_index = 0
            sents = item[self.key]
            for sent in sents:
                t_char = torch.Tensor(sent).long()
                char_length = t_char.shape[0]
                tensor_char[bid, sent_index, 0:char_length] = t_char
                tensor_char_mask[bid, sent_index, 0:char_length] = 1
                target_lengths[bid, sent_index] = char_length
                sent_index += 1

        batch_out[self.key] = tensor_char
        batch_out[self.mask_key] = tensor_char_mask
        batch_out['target_lengths'] = target_lengths


@PREPROCESS.register_module()
class PreCharCollate:
    '''previous char collate.'''

    def __init__(
        self,
        tgt_dict,
        key='char',
        rnnt_format=True,
        las_format=False,
        bos_id=None,
        reverse=False,
        eos_id=None,
        args=None,
        adaptive_tgt_key='adaptive_tgt_indices',
    ):
        '''init.
        Args:
            tgt_dict(ScpDictionary): tgt dict.
            rnnt_format(bool): control prev_char preparation for rnnt (default) or las runner.
        '''
        self.tgt_dict = tgt_dict
        self.in_key = key
        self.out_key = 'prev_' + key
        self.rnnt_format = rnnt_format
        self.las_format = las_format
        self.bos_id = bos_id
        self.reverse = reverse
        self.reverse_key = self.out_key + '_rev'
        self.eos_id = eos_id

        if rnnt_format:  # prepare for RNNT adaptive softmax
            self.split_num = args.get('jointer_split_num', 1)
            self.concat_u = 1 if args.get('concate_U', False) else 0
            self.head_size = args.get('adaptive_head_size', 0)
            self.tail_size = args.get('adaptive_tail_size', 0)
            self.tail_groups = args.get('adaptive_tail_groups', args.get('cutoff_groups', 0))
            self.adaptive_tgt_key = adaptive_tgt_key

    def __call__(self, bucket_list, batch_out):
        '''
        do char collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        # pylint:disable=too-many-branches
        bsz = len(bucket_list)
        if self.rnnt_format or self.las_format:
            max_char_length = max(item[self.in_key].shape[0] + 1 for item in bucket_list)
        else:
            max_char_length = max(item[self.in_key].shape[0] for item in bucket_list)

        tensor_pre_char = torch.zeros(bsz, max_char_length, dtype=torch.int64)
        if self.reverse:
            tensor_pre_char_rev = torch.zeros(bsz, max_char_length, dtype=torch.int64)

        for bid, item in enumerate(bucket_list):
            char = item[self.in_key]
            char_length = char.shape[0]
            if self.bos_id is not None:
                tensor_pre_char[bid, 0] = self.bos_id
            else:
                tensor_pre_char[bid, 0] = self.tgt_dict.bos()
            if self.reverse:
                if self.eos_id is not None:
                    tensor_pre_char_rev[bid, 0] = self.eos_id
                else:
                    tensor_pre_char_rev[bid, 0] = self.tgt_dict.eos()
            if self.rnnt_format:
                tensor_pre_char[bid, 1 : char_length + 1] = torch.from_numpy(char)
            else:
                tensor_pre_char[bid, 1:char_length] = torch.from_numpy(char[: char_length - 1])
                if self.reverse:
                    tensor_pre_char_rev[bid, 1:char_length] = torch.from_numpy(
                        char[: char_length - 1]
                    ).flip(dims=[0])

        batch_out[self.out_key] = tensor_pre_char
        if self.reverse:
            batch_out[self.reverse_key] = tensor_pre_char_rev

        # prepare for RNNT adaptive softmax
        if self.rnnt_format and self.head_size > 0:
            self.get_adaptive_tgt_index(bucket_list, batch_out)

    def get_adaptive_tgt_index(self, bucket_list, batch_out):
        '''
        caculate target index for adaptive softmax.
        see RNNTAdaptiveSoftmax for more info.

        ::

            adaptive_tgt_indices: list of Tensor
                0: int64 Tensor, [B, U+1, 1+concat_u]
                    for final lprob gather from head_lprob and tail_lprobs.
                    head_lprob: [sum(U+1), T, head_dim] or [B, T, U+1, head_dim]
                    tail_lprobs: list of Tensor [K, T, tail_size]
                        K target in this tail group.
                    final_lprob: [B, T, U+1, 2].
                    some memory may be wasted, but too little to worry.

                    [bid, uid, 0] is kid, index of K in this tail group input.
                    [bid, uid, 1] is buid, index of sum(U+1).
                1~tail_groups+1: int64 Tensor, [K, 2+concat_u]
                    for tail input gather from input_data.
                    input_data: [sum(U+1), T, H] or [B, T, U+1, H].
                    each tail input: [K, T, H].

                    [kid, 0] is bid, index of B.
                    [kid, 1] is uid, index of U+1.
                    [kid, 2] is buid, index of sum(U+1).
        '''
        bsz = len(bucket_list)
        split_num = min(self.split_num, bsz)
        step = (bsz + split_num - 1) // split_num
        batch_out[self.adaptive_tgt_key] = []
        for st in range(0, bsz, step):
            cur_bsz = min(step, bsz - st)
            max_char_length = max(
                item[self.in_key].shape[0] + 1 for item in bucket_list[st : st + cur_bsz]
            )
            final_gather_indices = torch.zeros(
                [cur_bsz, max_char_length, 1 + self.concat_u], dtype=torch.int64
            )  # [B, U+1, 1+concat_u]
            tail_gather_indices = [[] for _ in range(self.tail_groups)]
            buid = 0
            for bid in range(st, st + cur_bsz):
                chars = bucket_list[bid][self.in_key]
                char_len = chars.shape[0]
                for uid in range(char_len):
                    char = chars[uid]
                    if self.concat_u:
                        final_gather_indices[bid - st, uid, 1] = buid
                    if char >= self.head_size:
                        tail_group_idx = (char - self.head_size) // self.tail_size
                        final_gather_indices[bid - st, uid, 0] = len(
                            tail_gather_indices[tail_group_idx]
                        )
                        tail_gather_idx = (
                            [bid - st, uid, buid] if self.concat_u else [bid - st, uid]
                        )
                        tail_gather_indices[tail_group_idx].append(tail_gather_idx)
                    buid += 1
                if self.concat_u:
                    final_gather_indices[bid - st, char_len, 1] = buid
                buid += 1  # for rnnt format, 1 more output because `bos` in pre_char
            # each idx for one tail group, shape [this_tail_tgt_num, index_nums(2 or 3)]
            adaptive_tgt_indices = [final_gather_indices] + [
                torch.tensor(idx).long() if idx else None for idx in tail_gather_indices
            ]
            if self.split_num > 1:
                batch_out[self.adaptive_tgt_key].append(adaptive_tgt_indices)
            else:
                batch_out[self.adaptive_tgt_key] = adaptive_tgt_indices


@PREPROCESS.register_module()
class SidCollate:
    '''
    sid collate for speaker verification.
    Randomly sample length-l segment from each utterance.
    Used in SID training.
    '''

    def __init__(
        self,
        min_len=None,
        max_len=None,
        random_clip=True,
        src_key='src',
        clss_key='spk',
        input_type='fbank',
        sample_rate=16000,
        vad_key='vad',
    ):
        '''SID collate
                Args:
                    min_len: The minimum length of the sampled segments
                    max_len: The maximum length of the sampled segments
                    random_clip: randomly clip the utterance into segments
                    src_key: the feature's key
                    clss_key: the @PREPROCESS.register_module()
        class key
        '''
        self.min_len = min_len
        self.max_len = max_len
        self.src_key = src_key
        self.clss_key = clss_key
        self.random_clip = random_clip
        self.input_type = input_type
        self.sample_rate = sample_rate
        self.vad_key = vad_key

    def get_max_len(self, src):
        '''get max lens.'''
        if isinstance(src, np.ndarray):
            if self.input_type == 'wav':
                return get_frames_len(src.shape[1], self.sample_rate)
            return src.shape[0]
        if isinstance(src, dict):
            return self.get_max_len(src[self.src_key])
        if isinstance(src, (list, tuple)):
            return max(self.get_max_len(s) for s in src)
        return 0

    def get_max_vad_len(self, inputs):
        """
        get max valid frames num in inputs
        """
        if isinstance(inputs, dict):
            return sum(inputs[self.vad_key])

        if isinstance(inputs, (list, tuple)):
            return max(self.get_max_vad_len(item) for item in inputs)
        return 0

    def __call__(self, input_list, batch_out):
        '''
        called.
        input_list:
            [dict('src': numpy array, 'spk':int), dict('src': numpy array, 'spk':int),]
        The return batch contains:
            features with shape [batch_class_num * batch_size_per_class, length, dim]
            labels with shape [batch_class_num * batch_size_per_class]
            lengths (the actual utterance lengths) with shape
            [batch_class_num * batch_size_per_class]
        '''
        input_list = flatten_list(input_list)
        if self.input_type == 'wav':
            assert self.vad_key in input_list[0], "if wav train, must have vad list"
            # sampled_len: valid frames num
            if self.random_clip:
                sampled_len = random.randint(self.min_len, self.max_len)
            else:
                sampled_len = int(self.get_max_vad_len(input_list))
            # aligned_len: aligned frames num
            aligned_len = self.get_max_len(input_list)
            out_batch = self.align_all_wav(input_list, aligned_len, sampled_len)
        else:
            if self.random_clip:
                aligned_len = random.randint(self.min_len, self.max_len)
            else:
                aligned_len = self.get_max_len(input_list)
            out_batch = self.align_all_fbank(input_list, aligned_len)
        batch_out.update(out_batch)

    def align_all_fbank(self, input_list, aligned_len):
        '''return aligned batch out'''
        batch_out = dict()
        batch_arr, clss_arr, mask_arr = [], [], []
        for item in input_list:
            srcs, masks = self.align_fbank(item[self.src_key], aligned_len)
            clss = [item[self.clss_key]] * len(srcs)
            batch_arr.extend(srcs)
            clss_arr.extend(clss)
            mask_arr.extend(masks)
        batch_out['feature'] = torch.from_numpy(np.stack(batch_arr))

        # pylint: disable=not-callable
        # shape is [num of src, ]
        batch_out['label'] = torch.tensor(clss_arr, dtype=torch.int64)

        # output mask rather than length for convenience in forward.
        # shape is [num of src, aligned_len]

        batch_out['mask'] = torch.tensor(np.stack(mask_arr), dtype=torch.float32)
        # batch_out['lengths'] = torch.tensor(lens_arr, dtype=torch.int64)
        return batch_out

    def align_all_wav(self, input_list, aligned_len, sampled_len):
        """get alined batch of wav and vad"""
        batch_out = dict()
        batch_out['valid_len'] = int(sampled_len)
        batch_arr, clss_arr, mask_arr, vad_arr, frame_arr = [], [], [], [], []
        for item in input_list:
            clss = item[self.clss_key]
            src, mask, vad, frame_len = self.align_wav(
                item[self.src_key], item[self.vad_key], aligned_len, sampled_len
            )
            batch_arr.append(src)
            clss_arr.append(clss)
            mask_arr.append(mask)
            vad_arr.append(vad)
            frame_arr.append(frame_len)
        batch_out['feature'] = torch.from_numpy(np.stack(batch_arr))

        # pylint: disable=not-callable
        # shape is [num of src, ]
        batch_out['label'] = torch.tensor(clss_arr, dtype=torch.int64)

        # output mask rather than length for convenience in forward.
        # shape is [num of src, aligned_len]

        batch_out['mask'] = torch.tensor(np.stack(mask_arr), dtype=torch.float32)
        # batch_out['lengths'] = torch.tensor(lens_arr, dtype=torch.int64)
        batch_out['vad'] = torch.tensor(np.stack(vad_arr), dtype=torch.int32)
        batch_out['frames_len'] = torch.tensor(frame_arr, dtype=torch.int32)
        return batch_out

    def align_wav(self, src, vad, aligned_len, sampled_len):
        """align src and vad"""
        aligned_wav_lens = get_wav_len(aligned_len, self.sample_rate)
        frame_lens = get_frames_len(src.shape[1], self.sample_rate)
        if aligned_wav_lens < src.shape[1]:
            src = src[:, :aligned_wav_lens]
        mask_len = frame_lens
        vad_len = int(np.sum(vad))
        assert vad_len <= frame_lens, "valid frames length is bigger than all frames num"
        assert vad.shape[0] == frame_lens, "valid frames must equal to all frames length"
        mask = np.zeros((aligned_len))
        pad_shape = max(0, aligned_wav_lens - src.shape[1])
        pad_vad = aligned_len - vad.shape[0]
        src = np.concatenate(
            [src, np.zeros((src.shape[0], pad_shape), dtype=np.float32)],
            axis=1,
        )
        vad = np.concatenate(
            [vad, np.zeros(pad_vad, dtype=np.int32)],
            axis=0,
        )
        mask[:frame_lens] = 1

        if vad_len > sampled_len:
            start = random.randint(0, vad_len - sampled_len) if self.random_clip else 0
            end = start + sampled_len - 1
            cnt = 0
            for idx, icon in enumerate(vad):
                cnt += icon
                if cnt - 1 < start or cnt - 1 > end:
                    vad[idx] = 0
        return src, mask, vad, mask_len

    def align_fbank(self, src, aligned_len):
        '''align all src shape.'''
        if isinstance(src, list):
            srcs, masks = [], []
            for item in src:
                src, mask = self.align_fbank(item, aligned_len)
                srcs.extend(src)
                masks.extend(mask)
            return srcs, masks

        lens = src.shape[0]
        mask = np.zeros((aligned_len))
        if lens <= aligned_len:
            # Feature like Fbank need 2-dim mask but Wav need 1-dim mask
            pad_shape = tuple([aligned_len - lens] + list(src.shape[1:]))
            src = np.concatenate([src, np.zeros(pad_shape, dtype=np.float32)])
            mask[:lens] = 1
        elif lens > aligned_len:
            start = random.randint(0, lens - aligned_len) if self.random_clip else 0
            # Feature like Fbank is 2-dim while Wav is 1-dim, so, coding as the belows.
            src = src[start : start + aligned_len, :]
            mask[:] = 1
        return [src], [mask]


@PREPROCESS.register_module()
class BinaryTargetCollect:
    '''a signle @PREPROCESS.register_module()
    class collect'''

    def __init__(self, key="overlap", out_key="target"):
        '''init'''
        self.key = key
        self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''call'''
        tmp_list = [item[self.key] for item in bucket_list]
        target = torch.tensor(tmp_list, dtype=torch.int64)
        batch_out[self.out_key] = target


@PREPROCESS.register_module()
class LMCharCollate:
    '''char collation for lm training.'''

    def __init__(self, tgt_dict, use_eos=True):
        '''
        init.

        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        self.tgt_dict = tgt_dict
        self.use_eos = use_eos

    def __call__(self, bucket_list, batch_out):
        '''
        do char collation.

        Args:
            tgt_dict(ScpDictionary): tgt dict.
        '''
        bsz = len(bucket_list)

        max_char_length = max(item['char'].shape[0] + 1 for item in bucket_list)

        tensor_char = torch.zeros(bsz, max_char_length, dtype=torch.int64)
        tensor_char_mask = torch.zeros(bsz, max_char_length, dtype=torch.int64)

        for bid, item in enumerate(bucket_list):
            char = item['char']
            t_char = torch.from_numpy(char)
            char_length = char.shape[0]
            tensor_char[bid, 0:char_length] = t_char
            if self.use_eos:
                tensor_char[bid, char_length] = self.tgt_dict.eos()
                tensor_char_mask[bid, 0 : char_length + 1] = 1
            else:
                tensor_char_mask[bid, 0:char_length] = 1

        batch_out['char'] = tensor_char
        batch_out['char_mask'] = tensor_char_mask


@PREPROCESS.register_module()
class LMPreCharCollate:
    '''previous char collate for lm training.'''

    def __init__(self, tgt_dict=None):
        '''init.

        Args:
            tgt_dict(ScpDictionary): tgt dict.
        '''
        self.tgt_dict = tgt_dict

    def __call__(self, bucket_list, batch_out):
        '''
        do char collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        max_char_length = max(item['char'].shape[0] + 1 for item in bucket_list)
        tensor_pre_char = torch.zeros(bsz, max_char_length, dtype=torch.int64)
        tensor_char_mask = torch.zeros(bsz, max_char_length, dtype=torch.int64)

        for bid, item in enumerate(bucket_list):
            char = item['char']
            char_length = char.shape[0]
            tensor_pre_char[bid, 0] = self.tgt_dict.bos()
            tensor_pre_char[bid, 1 : char_length + 1] = torch.from_numpy(char)
            tensor_char_mask[bid, 0 : char_length + 1] = 1

        batch_out['src'] = tensor_pre_char
        batch_out['src_mask'] = tensor_char_mask


@PREPROCESS.register_module()
class CeLabelCollate:
    '''ce label collation.'''

    def __init__(self, frame_chunk_size=FRAME_CHUNK_SIZE, key='ce_label'):
        '''init.'''
        self.frame_chunk_size = frame_chunk_size
        self.key = key

    def __call__(self, bucket_list, batch_out):
        '''
        do ce label collation.

        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)

        max_ce_length = max(len(item[self.key]) for item in bucket_list)
        max_ce_length = ceil(max_ce_length, self.frame_chunk_size)

        tensor_ce = torch.zeros(bsz, max_ce_length, dtype=torch.int64)
        tensor_ce_mask = torch.zeros(bsz, max_ce_length, dtype=torch.int64)

        for bid, item in enumerate(bucket_list):
            ce_label = np.array(item[self.key])
            t_ce = torch.from_numpy(ce_label)
            # pylint: disable=unsubscriptable-object
            ce_label_length = ce_label.shape[0]
            tensor_ce[bid, 0:ce_label_length] = t_ce
            tensor_ce_mask[bid, 0:ce_label_length] = 1

        batch_out['ce_label'] = tensor_ce
        batch_out['ce_label_mask'] = tensor_ce_mask


@PREPROCESS.register_module()
class EerCollate:
    '''
    eer collate for speaker verification.
    Used in SID training.
    '''

    def __init__(
        self,
        src_key='src',
        clss_key='utt',
        input_type='fbank',
        vad_key='vad',
        **_kwargs,
    ):
        '''EER collate'''
        self.key = src_key
        self.utt_key = clss_key
        self.input_type = input_type
        self.vad_key = vad_key

    def __call__(self, input_list, batch_out):
        '''
        called.
        input_list:
            [[dict('src':numpy array, 'spk':int),], ]
            [dict('src':[numpy array], 'spk':int)]
        The return batch contains:
            features with shape [batch_class_num * batch_size_per_class, length, dim]
            labels with shape [batch_class_num * batch_size_per_class]
            lengths (the actual utterance lengths) with shape
            [batch_class_num * batch_size_per_class]
        '''
        input_list = flatten_list(input_list)
        item = input_list[0]  # eer batch only has one item

        if self.input_type == 'wav':
            frames_len = get_frames_len(item[self.key].shape[1])
            if self.vad_key not in item:
                # Generate fake vad
                vad_list = np.ones(frames_len)
                batch_out['valid_len'] = frames_len
            else:
                vad_list = item[self.vad_key]
                batch_out['valid_len'] = int(np.sum(vad_list))
            batch_out['vad'] = torch.tensor(np.stack([vad_list]), dtype=torch.int32)
            batch_out['frames_len'] = torch.tensor([frames_len], dtype=torch.int32)
        else:
            frames_len = item[self.key].shape[0]

        batch_out['utt'] = [item[self.utt_key]]
        batch_out['feature'] = torch.from_numpy(np.stack([item[self.key]]))
        batch_out['mask'] = torch.ones((1, frames_len))


@PREPROCESS.register_module()
class WaveformCollate:
    '''waveform collation.'''

    def __init__(
        self,
        wav_key='waveform',
        sample_rate_key='sample_rate',
        channel_key='channel_num',
        pad_zero_front=False,
        frame_chunk_size=FRAME_CHUNK_SIZE,
        frame_length=25,
        frame_shift=10,
        channel_num=1,
    ):
        '''init.'''
        self.wav_key = wav_key
        self.sample_rate_key = sample_rate_key
        self.channel_key = channel_key
        self.pad_zero_front = pad_zero_front
        self.frame_chunk_size = frame_chunk_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.channel_num = channel_num

    def __call__(self, bucket_list, batch_out):
        '''do wavform collate.'''

        bsz = len(bucket_list)
        if sum(self.wav_key in item for item in bucket_list) != bsz:
            return

        # all sample_rate is same
        assert len(set(item[self.sample_rate_key] for item in bucket_list)) == 1

        # complete channel key in every item
        def complete_channel_key(x):
            '''complete channel key'''
            x[self.channel_key] = 1 if self.channel_key not in x else x[self.channel_key]
            return x

        bucket_list = list(map(complete_channel_key, bucket_list))
        # all channel_num is same and equals self.channel_num
        assert len(set(item[self.channel_key] for item in bucket_list)) == 1
        assert bucket_list[0][self.channel_key] == self.channel_num

        sample_rate = bucket_list[0][self.sample_rate_key]
        window_shift = int(sample_rate * self.frame_shift * MILLISECONDS_TO_SECONDS)
        window_size = int(sample_rate * self.frame_length * MILLISECONDS_TO_SECONDS)

        # wavform shape (1, -1)
        max_samples = max(item[self.wav_key].shape[1] // self.channel_num for item in bucket_list)
        max_frames = (max_samples - window_size) // window_shift + 1
        # pad to self.frame_chunk_size
        max_frames = ceil(max_frames, self.frame_chunk_size)
        max_samples = max(max_samples, (max_frames - 1) * window_shift + window_size)

        frame_mask = torch.zeros([bsz, max_frames], dtype=torch.float32)
        if self.channel_num > 1:
            wav_mask = torch.zeros([bsz, max_samples, self.channel_num], dtype=torch.float32)
            wav_batch = torch.zeros([bsz, max_samples, self.channel_num], dtype=torch.float32)
        else:
            wav_mask = torch.zeros([bsz, max_samples], dtype=torch.float32)
            wav_batch = torch.zeros([bsz, max_samples], dtype=torch.float32)
        pad_frames, pad_samples = [], []
        for i, item in enumerate(bucket_list):
            if self.channel_num > 1:
                wav = item[self.wav_key].reshape((-1, self.channel_num))
            else:
                wav = item[self.wav_key].reshape(-1)
            samples = wav.shape[0]
            frames = (samples - window_size) // window_shift + 1
            if self.pad_zero_front:
                pad_frame = random.randint(0, max_samples - samples) // window_shift
            else:
                pad_frame = 0
            pad_sample = pad_frame * window_shift
            pad_samples.append(pad_sample)
            pad_frames.append(pad_frame)

            wav_batch[i, pad_sample : pad_sample + samples] = torch.from_numpy(wav)
            frame_mask[i, pad_frame : pad_frame + frames] = 1
            wav_mask[i, pad_sample : pad_sample + samples] = 1

        batch_out['waveform'] = wav_batch
        batch_out['wav_mask'] = wav_mask
        batch_out['src_mask'] = frame_mask
        batch_out['sample_rate'] = sample_rate
        batch_out['pad_frames'] = pad_frames
        batch_out['pad_samples'] = pad_samples
        batch_out['channel_num'] = self.channel_num


@PREPROCESS.register_module()
class TimeRange2FrameTarget:
    '''TimeRange2FrameTarget'''

    def __init__(
        self,
        downsample=320,
        in_key='time_range',
        out_key='frame_target',
        sample_rate=8000,
        wav_key='waveform',
    ):
        '''init
        downsample: number of samples of a frame after front end, for 8kHz 320 means 40ms
        in_key: the key contains time range of each side
        '''
        self.downsample = downsample
        self.in_key = in_key
        self.out_key = out_key
        self.sample_rate = sample_rate
        self.wav_key = wav_key

    def __call__(self, bucket_list, batch_out):
        '''call'''
        max_samples = max(item[self.wav_key].shape[1] for item in bucket_list)
        max_frames = ceil(max_samples, self.downsample)
        bsz = len(bucket_list)
        time_ranges1 = [item[self.in_key + '1'] for item in bucket_list]
        time_ranges2 = [item[self.in_key + '2'] for item in bucket_list]
        frame_mask1 = self.generate_frame_target(bsz, max_frames, time_ranges1)
        frame_mask2 = self.generate_frame_target(bsz, max_frames, time_ranges2)
        frame_target = frame_mask1 * frame_mask2
        batch_out[self.out_key] = frame_target

    def generate_frame_target(self, bsz, max_frames, time_ranges):
        '''generate frame-level target'''
        frame_mask = torch.zeros([bsz, max_frames], dtype=torch.int64)
        for i, tr in enumerate(time_ranges):
            for st, et in tr:
                sidx = int(st * self.sample_rate / self.downsample)
                eidx = int(et * self.sample_rate / self.downsample)
                frame_mask[i, sidx:eidx] = 1
        return frame_mask


STRESSFEAT_DIM = 262  # (257+2+1+2)


@PREPROCESS.register_module()
class StressFeatCollate:
    '''Stress feature collate.'''

    def __init__(self, frame_chunk_size=FRAME_CHUNK_SIZE, feat_dim=STRESSFEAT_DIM, key='src'):
        '''do init.'''
        self.chunk_size = frame_chunk_size
        self.feat_dim = feat_dim
        self.in_key = key
        self.out_key = 'src'

    def __call__(self, bucket_list, batch_out):
        '''
        do fbank collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.in_key in item for item in bucket_list) != bsz:
            return

        max_frame_length = max(item[self.in_key].shape[0] for item in bucket_list)
        max_frame_length = ceil(max_frame_length, self.chunk_size)

        tensor_feat = torch.zeros(bsz, max_frame_length, self.feat_dim, dtype=torch.float32)
        if 'src_mask' not in batch_out:
            tensor_src_mask = torch.zeros(bsz, max_frame_length, dtype=torch.float32)

        for bid, item in enumerate(bucket_list):
            feat = item[self.in_key]
            frame_length = feat.shape[0]
            tensor_feat[bid, 0:frame_length] = torch.from_numpy(feat)
            if 'src_mask' not in batch_out:
                tensor_src_mask[bid, 0:frame_length] = 1

        batch_out[self.out_key] = tensor_feat
        if 'src_mask' not in batch_out:
            batch_out['src_mask'] = tensor_src_mask


@PREPROCESS.register_module()
class EmotionFeatCollate:
    '''Emotion feature collate.'''

    def __init__(self, key='waveform', pad=0.0, out_key=None, mask_key=None):
        '''do init.'''
        self.in_key = key
        self.out_key = key if not out_key else out_key
        self.pad = pad
        self.mask = bool(mask_key)
        self.mask_key = mask_key

    def __call__(self, bucket_list, batch_out):
        '''
        do feature collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        for item in bucket_list:
            item[self.in_key] = item[self.in_key].reshape(1, -1)
        if sum(self.in_key in item for item in bucket_list) != bsz:
            return

        max_length = max(item[self.in_key].shape[1] for item in bucket_list)

        d_type = torch.long if isinstance(self.pad, int) else torch.float32
        tensor_feat = torch.full((bsz, max_length), self.pad, dtype=d_type)

        if self.mask:
            tensor_mask = torch.zeros(bsz, max_length, dtype=torch.float32)

        for bid, item in enumerate(bucket_list):
            feat = item[self.in_key]
            frame_length = feat.shape[1]
            tensor_feat[bid, 0:frame_length] = torch.from_numpy(feat[0])
            if self.mask:
                tensor_mask[bid, 0:frame_length] = 1

        batch_out[self.out_key] = tensor_feat
        if self.mask:
            batch_out[self.mask_key] = tensor_mask


@PREPROCESS.register_module()
class WaveformCollateNoSplit:
    '''waveform collation, no split.'''

    def __init__(self, wav_key='waveform', out_key=None):
        '''init.'''
        self.wav_key = wav_key
        self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''do wavform collate.'''
        bsz = len(bucket_list)
        if sum(self.wav_key in item for item in bucket_list) != bsz:
            return

        # all sample_rate is same
        # wavform shape (1, -1)
        max_samples = max(item[self.wav_key].shape[-1] for item in bucket_list)
        wav_batch = torch.zeros([bsz, max_samples], dtype=torch.float32)
        for i, item in enumerate(bucket_list):
            wav = item[self.wav_key].reshape(-1)
            wav_batch[i, :] = torch.from_numpy(wav)
        batch_out[self.out_key] = wav_batch


@PREPROCESS.register_module()
class WaveformNarrowCollate(WaveformCollate):
    '''
    waveform collation with cat.
    Mainly to be compatible with the logic before wav2vec,
    which may be modified later.
    '''

    def __init__(
        self,
        wav_key='waveform',
        sample_rate_key='sample_rate',
        channel_key='channel_num',
        pad_zero_front=False,
        frame_chunk_size=FRAME_CHUNK_SIZE,
        frame_length=25,
        frame_shift=10,
        channel_num=1,
        max_sample_length=180000,
        max_sample_num=300,
    ):
        super().__init__(
            wav_key,
            sample_rate_key,
            channel_key,
            pad_zero_front,
            frame_chunk_size,
            frame_length,
            frame_shift,
            channel_num,
        )
        self.max_sample_length = max_sample_length
        self.max_sample_num = max_sample_num

    def __call__(self, bucket_list, batch_out):
        '''do wavform cat and collate.'''

        # cat waveform
        total_sample_length = sum(item[self.wav_key].shape[1] for item in bucket_list)
        bsz = min(self.max_sample_num, max(total_sample_length // self.max_sample_length, 1))
        samples = bsz * self.max_sample_length
        # wavform shape (1, -1)
        wavs = [item[self.wav_key] for item in bucket_list]
        buffer = np.concatenate(wavs, -1)
        batch = np.split(buffer[:, :samples], bsz, -1)
        missing = len(batch) - len(bucket_list)
        if missing > 0:
            bucket_list += [bucket_list[-1]] * missing
        for i in range(bsz):
            bucket_list[i][self.wav_key] = batch[i]
        bucket_list = bucket_list[:bsz]

        super().__call__(bucket_list, batch_out)


@PREPROCESS.register_module()
class StackCollate:
    """
    Aligned data collate
    """

    def __init__(self, key='src', out_key=None):
        '''init'''
        self.key = key
        self.out_key = key if out_key is None else out_key

    def __call__(self, input_list, batch_out):
        '''call'''
        data_list = [item[self.key] for item in input_list]
        batch_out[self.out_key] = torch.from_numpy(np.stack(data_list))


@PREPROCESS.register_module()
class OnehotLabelCollate:
    '''collate labels in onehot format'''

    def __init__(self, num_classes, key='label', onehot_key=None):
        '''init'''
        self.num_classes = num_classes
        self.key = key
        if onehot_key is None or onehot_key == key:
            self.onehot_key = "onehot_{}".format(key)
        else:
            self.onehot_key = onehot_key

    def __call__(self, bucket_list, batch_out):
        '''call'''
        bsz = len(bucket_list)

        labels = torch.zeros(bsz, dtype=torch.int64)
        onehot_labels = torch.zeros(bsz, self.num_classes, dtype=torch.float32)
        for bid, item in enumerate(bucket_list):
            labels[bid] = item[self.key][0]
            for cid in item[self.key]:
                onehot_labels[bid, int(cid)] = 1.0
        batch_out[self.key] = labels
        batch_out[self.onehot_key] = onehot_labels


@PREPROCESS.register_module()
class LengthRandomClipCollate:
    '''random clip the feature length'''

    def __init__(
        self, min_len, max_len, random_clip=True, same_start=True, key='fbank', out_key=None
    ):
        '''init'''
        self.min_len = min_len
        self.max_len = max_len
        self.random_clip = random_clip
        self.same_start = same_start if random_clip else True
        self.key = key
        self.out_key = key if out_key is None else out_key

    def __call__(self, bucket_list, batch_out):
        '''call'''
        if not bucket_list:
            return

        min_feat_len = np.min([item[self.key].shape[0] for item in bucket_list])
        min_len = min(min_feat_len, self.min_len)
        max_len = min(min_feat_len, self.max_len)
        crop_len = np.random.randint(min_len, max_len + 1)
        common_start = np.random.randint(0, min_feat_len - crop_len + 1) if self.random_clip else 0

        out = []
        for item in bucket_list:
            data = item[self.key]
            if self.same_start:
                item[self.key] = data[common_start : common_start + crop_len, ...]
            else:
                start = np.random.randint(0, data.shape[0] - crop_len + 1)
                item[self.key] = data[start : start + crop_len, ...]
            out.append(item[self.key])
        batch_out[self.out_key] = torch.from_numpy(np.stack(out, axis=0))


@PREPROCESS.register_module()
class MultiChannelWaveformCollate:
    '''waveform collation.'''

    def __init__(
        self,
        wav_key='waveform',
        mc_wav_key='mc_waveform',
        sample_rate_key='sample_rate',
        simu_channel_key='simu_channel_num',
        pad_zero_front=False,
        frame_chunk_size=FRAME_CHUNK_SIZE,
        frame_length=25,
        frame_shift=10,
    ):
        '''init.'''
        self.wav_key = wav_key
        self.mc_wav_key = mc_wav_key
        self.sample_rate_key = sample_rate_key
        self.pad_zero_front = pad_zero_front
        self.frame_chunk_size = frame_chunk_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.simu_channel_key = simu_channel_key

    def __call__(self, bucket_list, batch_out):
        '''do multichannel wavform collate.'''

        bsz = len(bucket_list)
        if sum(self.wav_key in item for item in bucket_list) != bsz:
            return

        # all sample_rate is same
        assert len(set(item[self.sample_rate_key] for item in bucket_list)) == 1

        sample_rate = bucket_list[0][self.sample_rate_key]
        window_shift = int(sample_rate * self.frame_shift * MILLISECONDS_TO_SECONDS)
        window_size = int(sample_rate * self.frame_length * MILLISECONDS_TO_SECONDS)

        simu_channel_num = bucket_list[0]['simu_channel_num']

        # wavform shape (1, -1)
        max_samples = max(item[self.wav_key].shape[1] for item in bucket_list)
        max_frames = (max_samples - window_size) // window_shift + 1
        # pad to self.frame_chunk_size
        max_frames = ceil(max_frames, self.frame_chunk_size)
        max_samples = max(max_samples, (max_frames - 1) * window_shift + window_size)

        frame_mask = torch.zeros([bsz, max_frames], dtype=torch.float32)
        wav_mask = torch.zeros([bsz, max_samples], dtype=torch.float32)
        wav_batch = torch.zeros([bsz, max_samples], dtype=torch.float32)
        mc_wav_batch = torch.zeros([bsz, simu_channel_num, max_samples], dtype=torch.float32)
        pad_frames, pad_samples = [], []
        for i, item in enumerate(bucket_list):
            wav = item[self.wav_key].reshape(-1)
            mc_wav = item[self.mc_wav_key].reshape(simu_channel_num, -1)
            samples = wav.shape[0]
            frames = (samples - window_size) // window_shift + 1
            if self.pad_zero_front:
                pad_frame = random.randint(0, max_samples - samples) // window_shift
            else:
                pad_frame = 0
            pad_sample = pad_frame * window_shift
            pad_samples.append(pad_sample)
            pad_frames.append(pad_frame)

            wav_batch[i, pad_sample : pad_sample + samples] = torch.from_numpy(wav)
            mc_wav_batch[i, :, pad_sample : pad_sample + samples] = torch.from_numpy(mc_wav)
            frame_mask[i, pad_frame : pad_frame + frames] = 1
            wav_mask[i, pad_sample : pad_sample + samples] = 1

        batch_out[self.wav_key] = wav_batch
        batch_out['wav_mask'] = wav_mask
        batch_out['src_mask'] = frame_mask
        batch_out[self.sample_rate_key] = sample_rate
        batch_out['pad_frames'] = pad_frames
        batch_out['pad_samples'] = pad_samples
        batch_out['channel_num'] = 1
        batch_out[self.simu_channel_key] = simu_channel_num
        batch_out[self.mc_wav_key] = mc_wav_batch.reshape(bsz, -1)


@PREPROCESS.register_module()
class MultiChannelWaveformCollateNoSplit:
    '''multichannel waveform collation, no split.'''

    def __init__(self, wav_key='waveform', out_key=None):
        '''init.'''
        self.wav_key = wav_key
        self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''do wavform collate.'''
        bsz = len(bucket_list)
        if sum(self.wav_key in item for item in bucket_list) != bsz:
            return
        # all sample_rate is same
        # wavform shape (nch, length)
        nch = bucket_list[0][self.wav_key].shape[0]
        if sum(item[self.wav_key].shape[0] == nch for item in bucket_list) != bsz:
            return
        max_samples = max(item[self.wav_key].shape[1] for item in bucket_list)
        wav_batch = torch.zeros([bsz, nch, max_samples], dtype=torch.float32)
        for i, item in enumerate(bucket_list):
            wav = item[self.wav_key].reshape(nch, -1)
            wav_batch[i, ...] = torch.from_numpy(wav)
        batch_out[self.out_key] = wav_batch


@PREPROCESS.register_module()
class FlagCollate:
    '''Collate of a single number'''

    def __init__(self, wav_key='waveform', out_key=None):
        '''init.'''
        self.wav_key = wav_key
        self.out_key = out_key

    def __call__(self, bucket_list, batch_out):
        '''do wavform collate.'''
        bsz = len(bucket_list)
        if sum(self.wav_key in item for item in bucket_list) != bsz:
            return
        # each element is a single number
        out = [item[self.wav_key] for item in bucket_list]
        out = np.array(out)
        batch_out[self.out_key] = torch.from_numpy(out)


@PREPROCESS.register_module()
class ContextMakePairsCollate:
    '''make contextual and non-contextual data-pairs for context-aware task.

    this Collate must be placed at the first position in batch_transform
    for extending the bucket_list
    '''

    def __init__(self, key='uttid', out_key=None, context_loss_scale=1.0):
        '''init list collate function, setup list key.'''
        self.key = key
        self.out_key = self.key
        if out_key is not None:
            self.out_key = out_key
        self.context_loss_scale = context_loss_scale
        self.loss_scale_key = 'item_loss_scales'

    def __call__(self, bucket_list, batch_out):
        '''
        do list collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if sum(self.key in item for item in bucket_list) != bsz:
            return
        new_items_list = []
        for item in bucket_list:
            # contextual loss scale
            item['ctx_loss_scale'] = self.context_loss_scale
            if self.context_loss_scale < 1:
                new_item = item.copy()
                # non-contextual loss scale
                new_item['ctx_loss_scale'] = 1 - self.context_loss_scale
                # discard the context_text
                new_item['context_text'] = np.zeros(0, dtype=np.int64)
                new_items_list.append(new_item)
        # extend the bucket_list to make data-pairs
        bucket_list.extend(new_items_list)
        batch_out[self.loss_scale_key] = torch.tensor(
            [item['ctx_loss_scale'] for item in bucket_list], dtype=torch.float
        )
        batch_out[self.out_key] = [item[self.key] for item in bucket_list]


@PREPROCESS.register_module()
class ComputeW2vMask:
    """compute wav2vec mask"""

    def __init__(
        self,
        mask_prob=0,
        mask_length=0,
        use_fbank=True,
        conv_feature_layers="",
        in_key="waveform",
        out_key="mask_idc",
        mask_type="static",
        mask_other=0.0,
        mask_minlen_type="random",
        min_masks=0,
        no_overlap=False,
        min_space=0,
        padding=True,
    ):
        '''init.'''
        self.use_fbank = use_fbank
        self.conv_feature_layers = conv_feature_layers
        self.mask_length = mask_length
        self.mask_prob = mask_prob
        self.in_key = in_key
        self.out_key = out_key
        self.mask_type = mask_type
        self.mask_other = mask_other
        self.mask_minlen_type = mask_minlen_type
        self.min_masks = min_masks
        self.no_overlap = no_overlap
        self.min_space = min_space
        self.padding = padding

    def __call__(self, bucket_list, batch_out):
        '''do compute w2v mask'''
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        input_data = batch_out[self.in_key]
        bsz = input_data.shape[0]
        all_sz = input_data.shape[1]
        if self.use_fbank:
            all_sz = int((int((all_sz + 1) / 2) + 1) / 2)
        else:
            conv_shapes = eval(self.conv_feature_layers)
            for conv_shape in conv_shapes:
                (_, kernel, stride) = conv_shape
                all_sz = int((all_sz - kernel) / stride + 1)

        if self.padding:
            padding_mask = (1 - batch_out['src_mask']).int().bool()
            extra = padding_mask.size(1) % all_sz
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), all_sz, -1)
            padding_mask = padding_mask.all(-1)
            padding_length = padding_mask.int().sum(-1).tolist()

        mask = np.full((bsz, all_sz), False)

        all_num_mask = int(
            # add a random number for probabilistic rounding
            self.mask_prob * all_sz / float(self.mask_length)
            + np.random.rand()
        )

        all_num_mask = max(self.min_masks, all_num_mask)

        mask_idcs = []
        for i in range(bsz):
            if self.padding:
                sz = all_sz - padding_length[i]
                num_mask = int(
                    # add a random number for probabilistic rounding
                    self.mask_prob * sz / float(self.mask_length)
                    + np.random.rand()
                )
                num_mask = max(self.min_masks, num_mask)
            else:
                sz = all_sz
                num_mask = all_num_mask

            if num_mask == 0:
                continue

            if self.mask_type == "static":
                lengths = np.full(num_mask, self.mask_length)
            elif self.mask_type == "uniform":
                lengths = np.random.randint(
                    self.mask_other, self.mask_length * 2 + 1, size=num_mask
                )
            elif self.mask_type == "normal":
                lengths = np.random.normal(self.mask_length, self.mask_other, size=num_mask)
                lengths = [max(1, int(round(x))) for x in lengths]
            elif self.mask_type == "poisson":
                lengths = np.random.poisson(self.mask_length, size=num_mask)
                lengths = [int(round(x)) for x in lengths]
            else:
                raise Exception("unknown mask selection " + self.mask_type)

            if sum(lengths) == 0:
                lengths[0] = min(self.mask_length, sz - 1)

            if self.no_overlap:
                mask_idc = []

                def arrange(s, e, length, keep_length):
                    span_start = np.random.randint(s, e - length)
                    # pylint: disable=cell-var-from-loop
                    mask_idc.extend(span_start + i for i in range(length))

                    new_parts = []
                    if span_start - s - self.min_space >= keep_length:
                        new_parts.append((s, span_start - self.min_space + 1))
                    if e - span_start - keep_length - self.min_space > keep_length:
                        new_parts.append((span_start + length + self.min_space, e))
                    return new_parts

                parts = [(0, sz)]
                min_length = min(lengths)
                for length in sorted(lengths, reverse=True):
                    lens = np.fromiter(
                        (e - s if e - s >= length + self.min_space else 0 for s, e in parts), np.int
                    )
                    l_sum = np.sum(lens)
                    if l_sum == 0:
                        break
                    probs = lens / np.sum(lens)
                    c = np.random.choice(len(parts), p=probs)
                    s, e = parts.pop(c)
                    parts.extend(arrange(s, e, length, min_length))
                mask_idc = np.asarray(mask_idc)
            else:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1

                mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

                mask_idc = np.asarray(
                    [
                        mask_idc[j] + offset
                        for j in range(len(mask_idc))
                        for offset in range(lengths[j])
                    ]
                )

            mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

        if len(mask_idcs) > 0:
            min_len = min(len(m) for m in mask_idcs)
            for i, mask_idc in enumerate(mask_idcs):
                if len(mask_idc) > min_len:
                    if self.mask_minlen_type == 'random':
                        mask_idc = np.random.choice(mask_idc, min_len, replace=False)
                    elif self.mask_minlen_type == 'crop_end':
                        mask_idc = mask_idc[:min_len]
                    elif self.mask_minlen_type == 'crop_both':
                        idx = np.random.choice(len(mask_idc) - min_len + 1)
                        mask_idc = mask_idc[idx : idx + min_len]
                    else:
                        raise Exception('unknown mask_minlen_type: ' + self.mask_minlen_type)

                mask[i, mask_idc] = True

        batch_out[self.out_key] = torch.from_numpy(mask)


@PREPROCESS.register_module()
class PhoneCollate:
    '''collate phone'''

    def __init__(
        self,
        key='phone',  # phone
        encode_key='undone_phone',
        out_key='phone_src',
        phone_mask_key='phone_length',
        out_encode_key='encode_src',
        encode_mask_key='encode_mask',
        frame_hunk_size=FRAME_CHUNK_SIZE,
        strict_mode=False,
    ):
        '''
        do init.
        Args:
            key: right phone key
            out_length_key: out phone length key
            undone_key: undone phone key, contains encode-list, pos, repeat
            phone_mask_key: phone length key
            out_encode_key: output encode list
            encode_mask_key: output encode phone mask key, contains length, bid, pos, repeat
            pos_key: append phone keys in raw phone tensor
            encode_key: need do prdict encode words list

        '''
        self.in_key = key
        self.encode_key = encode_key
        self.out_key = out_key
        self.phone_mask_key = phone_mask_key
        self.out_encode_key = out_encode_key
        self.encode_mask_key = encode_mask_key
        self.strict_mode = strict_mode
        self.chunk_size = frame_hunk_size

    def __call__(self, bucket_list, batch_out):
        '''
        do fbank collation.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        bsz = len(bucket_list)
        if self.strict_mode and sum(self.in_key in item for item in bucket_list) != bsz:
            return
        phone_length = []
        for item in bucket_list:
            if self.in_key in item:
                phone_length.append(item[self.in_key].shape[0])
            else:
                phone_length.append(self.chunk_size)

        # phone tensor: bsz * max_phone_len , int32
        max_phone_length = max(phone_length)
        phone_tensor = torch.zeros(bsz, max_phone_length, dtype=torch.int32)
        encode_phone_list = []
        encode_phone_mask = []
        for bid, item in enumerate(bucket_list):
            if self.in_key not in item:
                # set feat to all-zero for fake feature
                continue
            phone = item[self.in_key]
            phone_tensor[bid, 0 : phone_length[bid]] = torch.from_numpy(phone)
            encode_list = item[self.encode_key]
            if not encode_list:
                continue
            for encode_phone, steps, pos, repeat in encode_list:
                encode_phone_list += encode_phone
                encode_phone_mask.append([len(encode_phone), steps, bid, pos, repeat])
        batch_out[self.out_key] = phone_tensor
        batch_out[self.phone_mask_key] = phone_length  # list
        if not encode_phone_list:
            return
        batch_out[self.out_encode_key] = torch.tensor(encode_phone_list, dtype=torch.int32)
        batch_out[self.encode_mask_key] = encode_phone_mask  # encode mask, list


@PREPROCESS.register_module()
class MixBucketList:
    '''mix two utterances, but no collation.'''

    def __init__(
        self,
        tgt_dict=None,
        sc_token='$',
        mix_prob=0.0,
        mix_num=2,
        avg_overlap_dur=4.0,
        extra_start=0.5,
        extra_end=0.0,
        **_kwargs,
    ):
        '''
        init.
        '''
        self.sc_token = sc_token
        self.sc_index = tgt_dict.indices[sc_token]
        self.mix_prob = mix_prob
        self.mix_num = mix_num
        # kwargs for mixing waveform
        self.avg_overlap_dur = avg_overlap_dur
        self.extra_start = extra_start
        self.extra_end = extra_end

    def split_buckets_into_groups(self, bucket_list):
        '''
        split bucket_list into groups, item in the same group will be mixed next.
        '''
        indices = list(range(len(bucket_list)))
        random.shuffle(indices)
        groups = []
        for i in range(0, len(indices), self.mix_num):
            group = [bucket_list[idx] for idx in indices[i : i + self.mix_num]]
            groups.append(group)
        return groups

    def mix_waveform(self, waveforms, sample_rate=16000, spks=None):
        '''
        waveforms: list of float32 np.ndarray
        spks: None or list of int/string. len(waveforms) equals len(spks)
        '''
        # if spks is None, assume all speakers are different.
        if spks is None:
            spks = list(range(len(waveforms)))
        assert len(waveforms) == len(spks)

        es = round(sample_rate * self.extra_start)
        ee = round(sample_rate * self.extra_end)

        size = list(waveforms[0].shape)
        size[-1] = sum(waveform.shape[-1] for waveform in waveforms)
        new_waveform = np.zeros(size, dtype=waveforms[0].dtype)
        for i, (waveform, spk) in enumerate(zip(waveforms, spks)):
            wavlen = waveform.shape[-1]
            if i == 0:
                new_waveform[..., :wavlen] = waveform
                max_s = 0
                max_e = wavlen
                last_spk = spk
                continue

            if last_spk == spk:
                overlap = 0
            else:
                max_overlap = max(0, min(max_e - max_s - es, wavlen - ee))
                overlap = round(sample_rate * np.random.exponential(self.avg_overlap_dur))
                overlap = min(overlap, max_overlap)

            if overlap < wavlen:
                # waveform is partly overlapping with new_waveform
                offset = max_e - overlap
                new_waveform[..., offset : offset + wavlen] += waveform
                max_s = max_e
                max_e = offset + wavlen
                last_spk = spk
            else:
                # waveform is fully overlapped by new_waveform
                # could appear only if self.extra_end = 0
                offset = random.randint(max_s + es, max_e - wavlen)
                new_waveform[..., offset : offset + wavlen] += waveform
                max_s = offset + wavlen
        new_waveform = new_waveform[..., :max_e]
        return new_waveform

    def mix_label(self, labels):
        '''
        labels: list of list of str. eg: [['你', '好'], ...]
        '''
        new_label = labels[0]
        for label in labels[1:]:
            new_label.append(self.sc_token)
            new_label += label
        return new_label

    def mix_char(self, chars):
        '''
        chars: list of long tensor.
        '''
        new_char = [chars[0]]
        for char in chars[1:]:
            new_char.append(np.array([self.sc_index], dtype=char.dtype))
            new_char.append(char)
        new_char = np.concatenate(new_char)
        return new_char

    def __call__(self, bucket_list, batch_out):
        '''
        do mix data in bucket_list. No update for batch_out.
        Args:
            bucket_list(list): list of item in a bucket.
            batch_out(dict): output data.
        '''
        if random.random() > self.mix_prob:
            return

        groups = self.split_buckets_into_groups(bucket_list)
        bucket_list.clear()
        for group in groups:
            keys = group[0].keys()
            mix_item = {key: [item[key] for item in group] for key in keys}
            mix_item['sample_rate'] = mix_item['sample_rate'][0]
            mix_item['channel_num'] = mix_item['channel_num'][0]
            mix_item['uttid'] = self.sc_token.join(mix_item['uttid'])
            mix_item['waveform'] = self.mix_waveform(
                mix_item['waveform'],
                mix_item['sample_rate'],
            )
            mix_item['label'] = self.mix_label(mix_item['label'])
            mix_item['char'] = self.mix_char(mix_item['char'])
            bucket_list.append(mix_item)

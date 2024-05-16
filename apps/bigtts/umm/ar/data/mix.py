import itertools
import json
import logging
import math
import pickle
import random
import sys
from enum import IntEnum
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import numpy as np
import phonemizer
import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset.pipeline import DataPipeline

from apps.bigtts.umm.ar.data.collate import ARCollator
from apps.bigtts.umm.ar.data.phone_to_id import PhoneToId
from apps.bigtts.umm.ar.data.utils import normalize_text
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.lite.utils.frontend import sil_punc_symbols
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    FastNormalizeAudio,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self

logger = logging.getLogger(__name__)


class DatasetType(IntEnum):
    AUDIO = 1
    UMM_TOKEN = 2


class BaseTransforms:
    """Base class for all data transforms"""

    name = "BaseTransforms"

    def __init__(self, log_interval: int = 100):
        self.count = 0
        self.skipped = 0
        self.messages = {}
        self.log_interval = log_interval

    def _update_stats(self, skipped: bool, message: Optional[str] = None):
        self.count += 1
        if skipped:
            self.skipped += 1
        if message is not None:
            message = f"[{self.name}] {message}"
            if message not in self.messages:
                self.messages[message] = 0
            self.messages[message] += 1
        # Print
        if self.count > 0 and self.count % self.log_interval == 0:
            worker_id = torch.utils.data.get_worker_info()
            if worker_id is not None:
                worker_id = worker_id.id
            else:
                worker_id = "Undefined"
            print(
                f"[{worker_id}] "
                f"Skipped {self.skipped}/{self.count} items, "
                f"Messages: {self.messages}",
                file=sys.stderr,
                flush=True,
            )

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        raise NotImplementedError()


class BigTTSTransforms(BaseTransforms):
    name = "BigTTSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        umm_token_freq: int = 40,
        audio_key: str = "wav",
        target_token_key: str = "umm_token",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        phone2id=None,
        phone_tone_wordseg_dict=None,
        split_by_alignment: bool = False,
        ignore_code_switch: bool = False,
        frame_rate: int = 25,
        token_pretrain: bool = False,
        dropout_rate_zh_tone=None,
        whole_sentence_prob: int = 0.01,
        enable_contexutal: bool = False,
        use_text_cfg: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.umm_token_freq = umm_token_freq
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.target_token_key = target_token_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.phone2id = phone2id
        self.phone_tone_wordseg_dict = phone_tone_wordseg_dict
        self.split_by_alignment = split_by_alignment
        self.ignore_code_switch = ignore_code_switch
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        self.resampler = {}
        self.fast_normalizer = FastNormalizeAudio()
        self.normalize_audio = normalize_audio
        self.base_transform = Compose(base_transforms)
        self.lang2id = {"en": 0, "zh": 1, "zh_en": 2, "jp": 3}
        self.token_pretrain = token_pretrain
        self.dropout_rate_zh_tone = dropout_rate_zh_tone
        self.whole_sentence_prob = whole_sentence_prob
        self.enable_contextual = enable_contexutal
        self.use_text_cfg = use_text_cfg

    def phone_tone_wordseg_to_id(self, phone, tone, word_seg):
        text_id = (
            phone.astype(np.int64) * 1_000_000
            + tone.astype(np.int64) * 1_000
            + word_seg.astype(np.int64)
        )
        res = [0] * len(text_id)
        for i, t_id in enumerate(text_id):
            if t_id not in self.phone_tone_wordseg_dict:
                logger.info(f"===>>> phone_tone_wordseg_dict OOV")
                # np.save('tmp/{}.npy'.format(t_id), t_id)
                # key_idx = np.asarray(list(self.phone_tone_wordseg_dict.keys()))
                # min_ind = np.argmin(np.abs(key_idx - t_id))
                # res[i] = self.phone_tone_wordseg_dict[key_idx[min_ind]]
                return None
            else:
                res[i] = self.phone_tone_wordseg_dict[t_id]
        return np.asarray(res).astype(np.int32)

    def preprocess_meta(self, sample):
        if not self.enable_contextual:
            meta_obj = json.loads(sample["meta"])
            while not isinstance(meta_obj, dict):
                meta_obj = json.loads(meta_obj)
            item = {}
            item["labels"] = str(meta_obj.get("labels", ""))
            if self.split_by_alignment:
                item["alignment"] = str(meta_obj.get("alignment", ""))
            item["text"] = normalize_text(sample["text"])
            sample.update(item)
        else:
            text = []
            labels = []
            for meta in sample["contextual_meta_list"]:
                meta_obj = json.loads(meta)
                while not isinstance(meta_obj, dict):
                    meta_obj = json.loads(meta_obj)
                # handle text
                if "text" in meta_obj:
                    text.append(normalize_text(meta_obj["text"]))
                labels.append(str(meta_obj.get("labels", "")))
            sample.update(
                {
                    "text": text,
                    "labels": labels,
                    "speakers": sample["contextual_speaker_list"],
                    "uttid": sample["contextual_uttid_list"],
                }
            )
        return sample

    def process_audio(self, item):
        if self.enable_contextual:
            contextual_wavs = [
                self.base_transform(wav) for wav in item["contextual_wav_list"]
            ]
            audio_lengths = [
                wav.size(-1) * self.sample_rate / item["src_sample_rate"]
                for wav in contextual_wavs
            ]
            audio = torch.hstack(contextual_wavs)
        else:
            audio = self.base_transform(item[self.audio_key])
        audio = self.do_resample(item["src_sample_rate"], audio)
        if self.normalize_audio:
            audio = self.fast_normalizer(audio)

        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return None
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return None
        if self.enable_contextual:
            # split audio by audio_lengths
            audio = torch.hsplit(
                audio, np.cumsum(audio_lengths[:-1]).astype("int").tolist()
            )
        return audio

    def process_target_token(self, item):
        if not self.enable_contextual:
            try:
                target_token = torch.as_tensor(
                    pickle.loads(item[self.target_token_key]), dtype=torch.long
                )
                target_token_length = target_token.size(0)
            except:
                self._update_stats(skipped=True, message="load umm token fail")
                return None, None
        else:
            target_token = [
                torch.as_tensor(pickle.loads(raw_token), dtype=torch.long)
                for raw_token in item["contextual_umm_token_list"]
            ]
            # logger.info(f">>> #target_token: {len(target_token)}")
            target_token_length = [
                target_token_i.size(0) for target_token_i in target_token
            ]
            target_token = torch.cat(target_token)

        if target_token.size(-1) < self.min_duration * self.umm_token_freq:
            self._update_stats(skipped=True, message="umm token too short")
            return None, None
        if target_token.size(-1) > self.max_duration * self.umm_token_freq:
            self._update_stats(skipped=True, message="umm token too long")
            return None, None
        return target_token, target_token_length

    def process_sami_token(self, labels):
        if not self.enable_contextual:
            return self.sami_tokenize(labels)
        token_list = []
        phone_list = []
        tone_list = []
        wordseg_list = []
        token_length = []
        lang_id = None
        for sub_labels in labels:
            token, phone, tone, wordseg, lang_id, _ = self.sami_tokenize(sub_labels)
            if token is None:
                return None, None, None, None, None, None
            token_list.append(token)
            phone_list.append(phone)
            tone_list.append(tone)
            wordseg_list.append(wordseg)
            token_length.append(len(token))
        return (
            torch.cat(token_list),
            torch.cat(phone_list),
            torch.cat(tone_list),
            torch.cat(wordseg_list),
            lang_id,
            token_length,
        )

    def sami_tokenize(self, labels):
        labels = list(filter(lambda x: x != "", labels.split("\n")))
        if len(labels) < 2:
            self._update_stats(skipped=True, message="Label length is shorter than 2")
            return None, None, None, None, None, None
        if len(labels[-1].split("\t")) == 2:
            last_duration = labels[-1].split("\t")[-1]
            last_line = labels[-2]
            labels = labels[:-2]
            last_line = "\t".join(last_line.split("\t")[:-1] + [last_duration])
            labels.append(last_line)

        # Get language ID
        lang_key = self.get_lang(labels)
        lang_id = self.lang2id[lang_key]

        # Convert tacolabel to phone, tone, and wordseg ids
        text_id_phones_tones = self.phone2id.convert_tacolab_to_text_id(labels)
        if text_id_phones_tones is None:
            self._update_stats(
                skipped=True, message="convert_tacolab_to_text_id failed"
            )
            return None, None, None, None, None, None
        else:
            text_id, _, _, _, _ = text_id_phones_tones

        # Map phone, tone, and wordseg to id
        token = self.phone_tone_wordseg_to_id(
            text_id[0, :], text_id[1, :], text_id[2, :]
        )
        if token is None:
            self._update_stats(skipped=True, message="phone_tone_wordseg_to_id failed")
            # return None, None, None, None
            token = torch.from_numpy(text_id[0, :]).long()
        else:
            token = torch.from_numpy(token).long()
        # phone, tone, word_seg
        phone, tone, wordseg = text_id[0, :], text_id[1, :], text_id[2, :]
        phone = torch.from_numpy(phone).long()
        tone = torch.from_numpy(tone).long()
        wordseg = torch.from_numpy(wordseg).long()
        token_length = token.size(0)
        return token, phone, tone, wordseg, lang_id, token_length

    def do_resample(self, src_sample_rate, x):
        if src_sample_rate == self.sample_rate:
            return x

        if src_sample_rate not in self.resampler:
            self.resampler[src_sample_rate] = Resample(
                src_sample_rate, self.sample_rate
            )
        return self.resampler[src_sample_rate](x)

    def get_lang(self, tacolab):
        if len(tacolab[0].split("\t")) != 5:
            if (
                tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword"
                or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit"
                or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\talignment"
                or tacolab[0] == "phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment"
            ):
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split("\t")[0][:2] for x in tacolab]
        if "C0" in prefix_phn_list:
            if "E0" in prefix_phn_list:
                lang = "zh_en"
            else:
                lang = "zh"
        elif "JP" in prefix_phn_list:
            lang = "jp"
        else:
            lang = "en"
        return lang

    def __call__(self, buffer: Generator):
        for item in buffer:
            yield from self.item_transform(item)

    def item_transform(self, item: Dict[str, Any]) -> Generator:

        if self.audio_key in item:
            audio = self.process_audio(item)
            if audio is None:
                return

        if self.target_token_key in item:
            target_token, target_token_length = self.process_target_token(item)
            if target_token is None:
                return

        item = self.preprocess_meta(item)
        text = item["text"]
        labels = item["labels"]
        # labels = item["labels"].decode()

        if labels is None:
            self._update_stats(skipped=True, message="Label is None")
            return

        output_dict = {"text": text, "tag": "vocal"}
        if self.audio_key in item:
            output_dict[self.audio_key] = audio
        if self.target_token_key in item:
            output_dict[self.target_token_key] = target_token
            output_dict[f"{self.target_token_key}_length"] = target_token_length

        # token pretrain
        if not self.token_pretrain:

            if self.tokenizer is not None:
                #################################################################################
                ################################ process frontend ###############################
                #################################################################################
                if self.tokenizer == "sami":
                    try:
                        (
                            token,
                            phone,
                            tone,
                            wordseg,
                            lang_id,
                            token_length,
                        ) = self.process_sami_token(labels)
                        if token is None:
                            return
                        if self.dropout_rate_zh_tone is not None:
                            self.dropout_rate_zh_tone = float(self.dropout_rate_zh_tone)
                            lang_key = self.get_lang(labels)
                            if lang_key == "zh" or lang_key == "zh_en":
                                probability = (
                                    random.random()
                                )  # 生成一个在 [0.0, 1.0) 范围内的随机数
                                if (
                                    probability < self.dropout_rate_zh_tone
                                ):  # 如果概率小于 dropout_rate_zh_tone，则返回 True
                                    tone[tone != 2] = torch.zeros_like(
                                        tone[tone != 2]
                                    )  # en_tone: 12 or 14
                    except:
                        self._update_stats(
                            skipped=True, message="Tacolabel process failed"
                        )
                        return
                else:
                    encoded_text = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        # padding="longest",
                        return_tensors="pt",
                    )
                    token = encoded_text["input_ids"].squeeze(dim=0)

                #################################################################################
                ################################ filter outlier data ############################
                #################################################################################
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                else:
                    if self.audio_key in item and not self.enable_contextual:
                        frame_num = (
                            math.floor(audio.size(-1) / self.sample_rate)
                            * self.frame_rate
                        )
                    elif self.target_token_key in item:
                        frame_num = (
                            math.floor(target_token.size(-1) / self.umm_token_freq)
                            * self.frame_rate
                        )
                    else:
                        frame_num = 0
                    if token.size(-1) > frame_num:
                        self._update_stats(skipped=True, message="Token too long")
                        return

                if self.ignore_code_switch:
                    # filter zh_en tag
                    if lang_key not in ["zh", "en", "jp"]:
                        self._update_stats(
                            skipped=True, message="Code-switching detected"
                        )
                        return
                    # filter chinglish samples
                    if item["__dataset_name__"] in [
                        "tts_Lmand_Sxiaoyuzhou_F6.1_D0-10_P1",
                        "tts_Lmand_Sxiaoyuzhou_F6.1_D10-100_P1",
                        "tts_Lmand_Sximalaya_F6.1_D0-10_P1",
                        "tts_Lmand_Sximalaya_F6.1_D10-100_P1",
                        "tts_zh_Sbigspeech_P1",
                        "tts_Lmand_Sfanqie_F5.1_D0-10_P1",
                    ]:
                        if lang_key != "zh":
                            self._update_stats(
                                skipped=True, message="Code-switching detected"
                            )
                            return

                #################################################################################
                #################################### slpit data #################################
                #################################################################################
                if self.split_by_alignment:

                    #################   split by alignment    ############
                    if not self.enable_contextual:
                        start_sil_token = token[:1]
                        start_sil_phone = phone[:1]
                        start_sil_tone = tone[:1]
                        start_sil_wordseg = wordseg[:1]

                        token = token[1:]
                        phone = phone[1:]
                        tone = tone[1:]
                        wordseg = wordseg[1:]

                        # Map alignment to duration
                        if item["alignment"] == "":
                            # 回退
                            self._update_stats(
                                skipped=True, message="item['alignment'] is None"
                            )
                            return
                            # alignments = None
                            # candi_idx = []

                            ## Update 20240209： 如果 没有 alignment，则给整句，不丢数据
                            # logger.info(f"给整句，不丢数据")
                            # token = torch.cat([start_sil_token, token])
                            # phone = torch.cat([start_sil_phone, phone])
                            # tone = torch.cat([start_sil_tone, tone])
                            # wordseg = torch.cat([start_sil_wordseg, wordseg])
                            # target_token = target_token
                            # prompt_token = target_token[:0]  # 空

                        else:
                            # alignments = [float(i.split('\t')[1]) for i in item['alignment'].split('\n')]

                            new_alignments = []
                            for i, align in enumerate(item["alignment"].split("\n")):
                                if i == 0:
                                    new_alignments.append(align)
                                elif align.split("\t")[0] == "sil":
                                    continue
                                else:
                                    new_alignments.append(align)
                            alignments = new_alignments
                            alignments = alignments[1:]

                            if len(alignments) != len(token):
                                self._update_stats(
                                    skipped=True,
                                    message="len(alignments) != len(token)",
                                )
                                return

                            candi_idx = []
                            space_limit = 0.25
                            for i in range(2, len(alignments) - 2):
                                try:
                                    if (
                                        alignments[i].split("\t")[0] in sil_punc_symbols
                                        and float(alignments[i].split("\t")[1])
                                        - float(alignments[i - 1].split("\t")[1])
                                        >= space_limit
                                    ):
                                        # 标点且 space_limit / 2 以上静音，会视为分隔候选
                                        dur = (
                                            float(alignments[i].split("\t")[1])
                                            - space_limit / 2
                                        )
                                        candi_idx.append(
                                            (i + 1, int(self.umm_token_freq * dur))
                                        )
                                except Exception as e:
                                    self._update_stats(skipped=True, message=str(e))
                                    return

                            if (
                                len(candi_idx) == 0
                                or random.random() < self.whole_sentence_prob
                            ):  # 0.01的概率给整句
                                # logger.info(f"给整句概率：{self.whole_sentence_prob}")
                                token = torch.cat([start_sil_token, token])
                                phone = torch.cat([start_sil_phone, phone])
                                tone = torch.cat([start_sil_tone, tone])
                                wordseg = torch.cat([start_sil_wordseg, wordseg])
                                target_token = target_token
                                prompt_token = target_token[:0]  # 空
                            else:  # 随机挑个分隔段
                                is_training = self.whole_sentence_prob > 0
                                if is_training:
                                    split_idx, split_dur = random.choice(candi_idx)
                                else:
                                    split_idx, split_dur = candi_idx[0]
                                # if random.random() < split_dur/len(target_token):  # 更高概率让target更长些
                                # if split_dur * 2 > len(target_token):   # 先这样写省显存

                                # 走后续前 # (not is_training) or random.random() < 0.5:
                                if False:
                                    token = torch.cat(
                                        [start_sil_token, token[:split_idx]]
                                    )
                                    phone = torch.cat(
                                        [start_sil_phone, phone[:split_idx]]
                                    )
                                    tone = torch.cat([start_sil_tone, tone[:split_idx]])
                                    wordseg = torch.cat(
                                        [start_sil_wordseg, wordseg[:split_idx]]
                                    )
                                    prompt_token = target_token[split_dur:]
                                    target_token = target_token[:split_dur]
                                else:
                                    token = torch.cat(
                                        [start_sil_token, token[split_idx:]]
                                    )
                                    phone = torch.cat(
                                        [start_sil_phone, phone[split_idx:]]
                                    )
                                    tone = torch.cat([start_sil_tone, tone[split_idx:]])
                                    wordseg = torch.cat(
                                        [start_sil_wordseg, wordseg[split_idx:]]
                                    )
                                    prompt_token = target_token[:split_dur]
                                    target_token = target_token[split_dur:]

                        prompt_len = prompt_token.shape[0]
                        if prompt_len > 25 * 8:
                            prompt_len = 25 * 8  # 最多只用8s
                        if prompt_len > 25 * 2:  # 2s 以上做随机crop
                            prompt_len = random.randint(prompt_len // 2, prompt_len)
                        if (
                            prompt_len < prompt_token.shape[0]
                            and prompt_token.shape[0] > 0
                        ):
                            prompt_start = random.randint(
                                0, prompt_token.shape[0] - prompt_len
                            )
                            prompt_token = prompt_token[
                                prompt_start : prompt_start + prompt_len
                            ]
                        # alignments = item['alignment'].split('\n')
                        # 分隔标点：sil_punc_symbols

                        if self.target_token_key in item:
                            output_dict[
                                "prompt_" + self.target_token_key
                            ] = prompt_token
                            output_dict[
                                f"prompt_{self.target_token_key}_length"
                            ] = prompt_token.size(0)
                            output_dict[self.target_token_key] = target_token
                        output_dict[
                            f"{self.target_token_key}_length"
                        ] = target_token.size(0)

                    ###################  split by context  ##############
                    if self.enable_contextual:

                        if len(target_token_length) > 1:
                            # random split first or last utterance
                            if random.choice([True, False]):
                                # split label
                                phone = phone[token_length[0] :]
                                tone = tone[token_length[0] :]
                                wordseg = wordseg[token_length[0] :]
                                token = token[token_length[0] :]
                                token_length = token_length[1:]
                                # split token
                                output_dict[
                                    f"prompt_{self.target_token_key}"
                                ] = target_token[: target_token_length[0]]
                                output_dict[
                                    f"prompt_{self.target_token_key}_length"
                                ] = target_token_length[0]
                                output_dict[f"{self.target_token_key}"] = target_token[
                                    target_token_length[0] :
                                ]
                                output_dict[
                                    f"{self.target_token_key}_length"
                                ] = target_token_length[1:]
                            else:
                                # split label
                                phone = phone[: -token_length[-1]]
                                tone = tone[: -token_length[-1]]
                                wordseg = wordseg[: -token_length[-1]]
                                token = token[: -token_length[-1]]
                                token_length = token_length[:-1]
                                # split token
                                output_dict[
                                    f"prompt_{self.target_token_key}"
                                ] = target_token[-target_token_length[-1] :]
                                output_dict[
                                    f"prompt_{self.target_token_key}_length"
                                ] = target_token_length[-1]
                                output_dict[f"{self.target_token_key}"] = target_token[
                                    : -target_token_length[-1]
                                ]
                                output_dict[
                                    f"{self.target_token_key}_length"
                                ] = target_token_length[:-1]
                        else:
                            output_dict[
                                f"prompt_{self.target_token_key}"
                            ] = target_token[:0]
                            output_dict[
                                f"prompt_{self.target_token_key}_length"
                            ] = target_token[:0].size(0)

                        ######### merge context #########
                        try:
                            merge_token_length = []
                            merge_umm_token_length = []

                            for k, v in itertools.groupby(
                                zip(
                                    token_length,
                                    output_dict[f"{self.target_token_key}_length"],
                                ),
                                lambda x: x[0],
                            ):
                                v_list = list(v)
                                merge_token_length.append(sum(l[0] for l in v_list))
                                merge_umm_token_length.append(sum(l[1] for l in v_list))

                            token_length = merge_token_length
                            output_dict[
                                f"{self.target_token_key}_length"
                            ] = merge_umm_token_length

                            if len(token_length) != len(
                                output_dict[f"{self.target_token_key}_length"]
                            ):
                                self._update_stats(
                                    skipped=True, message="Round mismatch"
                                )
                                return

                        except:
                            self._update_stats(
                                skipped=True, message="Merging utterances failed"
                            )
                            return
                # if (self.use_text_cfg and random.random() < 0.1) or prompt_token.shape[
                #     0
                # ] == 0:
                #     # 取字典的最大值+1
                #     filled_token = filled_phone = (
                #         max(self.phone2id.phone_to_int.values()) + 1
                #     )
                #     filled_tone = max(self.phone2id.tone_to_int.values()) + 1
                #     filled_ws = max(self.phone2id.wordseg_to_int.values()) + 1
                #     token = torch.full((len(token),), filled_token, dtype=torch.long)
                #     phone = torch.full((len(phone),), filled_phone, dtype=torch.long)
                #     tone = torch.full((len(tone),), filled_tone, dtype=torch.long)
                #     wordseg = torch.full((len(wordseg),), filled_ws, dtype=torch.long)
                #     prompt_token = prompt_token[:0]

                assert len(token) == len(phone)
                output_dict.update(phone=phone)
                output_dict.update(tone=tone)
                output_dict.update(wordseg=wordseg)
                output_dict.update(token=token)
                output_dict.update(lang=lang_id)
                output_dict.update(token_length=token_length)
                output_dict.update(prompt_token=prompt_token)

        yield output_dict
        self._update_stats(skipped=False)


class BigTTSDataset(WebPipeline):
    name = "BigTTS"

    def __init__(
        self,
        data_id: int = None,
        url_pattern: str = None,
        sample_rate: int = 24000,
        umm_token_freq: int = 40,
        audio_key: str = "wav",
        target_token_key: str = "umm_token",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        phone2id=None,
        phone_tone_wordseg_dict=None,
        split_by_alignment=False,
        ignore_code_switch=False,
        frame_rate: int = 25,
        token_pretrain: bool = False,
        dropout_rate_zh_tone=None,
        whole_sentence_prob: int = 0.01,
        sample_config=None,
        enable_contexutal: bool = False,
        use_text_cfg: bool = False,
        **kwargs,
    ):
        logger.info(f"[{self.name}] [data_id: {data_id}] initializing...")
        dataset = ParquetDataset(
            data_id=data_id,
            data_urls=url_pattern,
            sample_config=sample_config,
            **kwargs,
        )
        transforms = BigTTSTransforms(
            sample_rate=sample_rate,
            umm_token_freq=umm_token_freq,
            audio_key=audio_key,
            target_token_key=target_token_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            phone2id=phone2id,
            phone_tone_wordseg_dict=phone_tone_wordseg_dict,
            split_by_alignment=split_by_alignment,
            ignore_code_switch=ignore_code_switch,
            frame_rate=frame_rate,
            token_pretrain=token_pretrain,
            dropout_rate_zh_tone=dropout_rate_zh_tone,
            whole_sentence_prob=whole_sentence_prob,
            enable_contexutal=enable_contexutal,
            use_text_cfg=use_text_cfg,
        )
        # preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        # pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        pipeline = [{"compose": [transforms]}]
        super().__init__(dataset, pipeline)
        logger.info(f"[{self.name}] initialized.")


class MixWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        umm_token_freq: int = 40,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn=ARCollator,
        weights: List[int] = [1, 1],
        tokenizer: str = None,
        frame_rate: int = 25,
        data_id: int = 2011,
        ctx_data_id: int = 2021,
        val_data_ids: List[int] = [2010],
        dataset_type: DatasetType = DatasetType.AUDIO,
        skip_validation: bool = False,
        target_token_key: str = "umm_token",
        target_audio_key: str = "wav",
        prefetch_factor: Union[int, None] = None,
        split_by_alignment: bool = False,
        ignore_code_switch: bool = False,
        token_pretrain: bool = False,
        dropout_rate_zh_tone: Union[float, None] = None,
        whole_sentence_prob: float = 0.01,
        sample_config=None,
        use_text_cfg: bool = False,
        replacement: bool = True,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn(
            target_audio_key, target_token_key, split_by_alignment
        )

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            self.phone2id = None
            self.phone_tone_wordseg_dict = None
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "sami":
            self.tokenizer = "sami"
            self.phone2id = PhoneToId()
            self.phone_tone_wordseg_dict = torch.load(
                "apps/bigtts/umm/ar/data/fronted_v3_2_dict.pyt"
            )
            logger.info(f"===>>> Using sami tokenizer")
            logger.info(
                f"===>>> Size of phone_tone_wordseg_dict: {len(self.phone_tone_wordseg_dict)}"
            )
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        if dataset_type == DatasetType.AUDIO:
            assert batch_size >= min_duration * sample_rate
        else:
            assert batch_size >= min_duration * umm_token_freq
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        logger.info(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        if dataset_type == DatasetType.AUDIO:
            buckets_samples = [x * sample_rate for x in buckets_samples]
        else:
            buckets_samples = [x * umm_token_freq for x in buckets_samples]

        logger.info(
            f"[MixWebDataModule] dataset_type={dataset_type} data_id={data_id} ctx_data_id={ctx_data_id}"
            f" val_data_ids={val_data_ids} target_token_key={target_token_key} {batch_size=}"
        )
        logger.info(f"weights: {weights}")
        logger.info(f"{buckets_samples=}")

        if dataset_type == DatasetType.AUDIO:
            length_fn = lambda x: x.get(target_audio_key).size(-1)
        else:
            length_fn = (
                lambda x: x.get(target_token_key).size(-1) + x.get("token").size(-1) + 3
            )  # bos + sep + prompt

        def bsz_evaluator(b, t):
            # 如果 t 太小，就不能填充太多数据，因为计算会变慢
            total_size = b * t
            if t < 128:
                total_size = int(total_size * 1.5)
            return total_size

        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=length_fn,
            bsz_evaluator=bsz_evaluator,
        )
        self.val_batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=length_fn,
            # bsz_evaluator=bsz_evaluator,
        )
        datasets = []
        if weights[0] > 0:
            bigtts_long_ctx = BigTTSDataset(
                data_id=ctx_data_id,
                audio_key=target_audio_key,
                target_token_key=target_token_key,
                sample_rate=sample_rate,
                umm_token_freq=umm_token_freq,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                phone2id=self.phone2id,
                phone_tone_wordseg_dict=self.phone_tone_wordseg_dict,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
                split_by_alignment=split_by_alignment,
                ignore_code_switch=ignore_code_switch,
                token_pretrain=token_pretrain,
                dropout_rate_zh_tone=dropout_rate_zh_tone,
                whole_sentence_prob=whole_sentence_prob,
                sample_config=sample_config,
                enable_contexutal=True,
                use_text_cfg=use_text_cfg,
                replacement=replacement,
            )
            datasets.append(bigtts_long_ctx)
        if weights[1] > 0:
            bigtts = BigTTSDataset(
                data_id=data_id,
                audio_key=target_audio_key,
                target_token_key=target_token_key,
                sample_rate=sample_rate,
                umm_token_freq=umm_token_freq,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                phone2id=self.phone2id,
                phone_tone_wordseg_dict=self.phone_tone_wordseg_dict,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
                split_by_alignment=split_by_alignment,
                ignore_code_switch=ignore_code_switch,
                token_pretrain=token_pretrain,
                dropout_rate_zh_tone=dropout_rate_zh_tone,
                whole_sentence_prob=whole_sentence_prob,
                sample_config=None,
                enable_contexutal=False,
                use_text_cfg=use_text_cfg,
                replacement=replacement,
            )
            datasets.append(bigtts)
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            wds.shuffle(self.shuffle_buffer_size),
            self.bucketize,
        )

        if skip_validation:
            self.validation_dataset = []
        else:
            self.validation_dataset = []
            for val_data_id in val_data_ids:
                val_dataset = DataPipeline(
                    BigTTSDataset(
                        data_id=val_data_id,
                        audio_key=target_audio_key,
                        target_token_key=target_token_key,
                        sample_rate=sample_rate,
                        umm_token_freq=umm_token_freq,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_num_crops=max_num_crops,
                        normalize_audio=normalize_audio,
                        tokenizer=self.tokenizer,
                        phone2id=self.phone2id,
                        phone_tone_wordseg_dict=self.phone_tone_wordseg_dict,
                        frame_rate=self.frame_rate,
                        resampled=False,
                        shardshuffle=False,
                        # use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                        split_by_alignment=split_by_alignment,
                        ignore_code_switch=ignore_code_switch,
                        token_pretrain=token_pretrain,
                        dropout_rate_zh_tone=dropout_rate_zh_tone,
                        whole_sentence_prob=0.0,
                        sample_config=None,
                        enable_contexutal=False,
                        nodesplitter=return_self,
                    ),
                    self.val_bucketize,
                )
                self.validation_dataset.append(val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=0,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

    def val_bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.val_batcher.collate_batch(item)
            if batch is not None:
                yield batch

        for batch in self.val_batcher.collect_last_batch():
            if batch:
                yield batch


if __name__ == "__main__":
    umm_token_freq = 25
    max_duration = 130
    batch_size = 5
    dm = MixWebDataModule(
        weights=[1, 1, 0],
        data_id=1768,
        ctx_data_id=1606,
        val_data_ids=[2307],
        umm_token_freq=25,
        target_token_key="umm_token",
        dataset_type=DatasetType.UMM_TOKEN,
        min_duration=1,
        max_duration=max_duration,
        batch_size=max_duration * umm_token_freq * batch_size,
        whole_sentence_prob=0.1,
        shuffle_buffer_size=10000,
        skip_validation=False,
        num_workers=8,
        sample_config={
            "name": "_ContexualParquetSample",
            "contextual_strategy": 5,
            "time_interval_threshold": 5,
            "max_num_speakers": 1,
            "max_duration": max_duration,
        },
        tokenizer="sami",
        pin_memory=True,
        prefetch_factor=32,
        split_by_alignment=True,
        token_pretrain=False,
        dropout_rate_zh_tone=None,
        normalize_audio=False,
        sample_rate=24000,
    )
    for item in dm.train_dataloader():
        logger.info(f"test {item.keys()=}")
        # logger.info("test", item["target_ids"].shape, item["target_ids_length"])
        break

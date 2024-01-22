import logging
import math
import pickle
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose


from .utils import normalize_text

from samantha.transforms.audio import (
    FastNormalizeAudio,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from samantha.dataio.lite.utils.parquet import get_meta_obj
from samantha.dataio.lite.utils.phone_to_id import PhoneToId
from samantha.dataio.lite.transform import ItemTransformBase
from samantha.dataio.lite.utils.stats import UpdateStatsMixin

logger = logging.getLogger(__name__)


class BigTTSTransforms(ItemTransformBase, UpdateStatsMixin):
    def __init__(
        self,
        sample_rate: int = 24000,
        umm_token_freq: int = 40,
        audio_key: str = "wav",
        token_key: str = "umm_token",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer: Union[str, Callable, None] = None,
        phone2id: Union[PhoneToId, None] = None,
        phone_tone_wordseg_dict: dict = None,
        frame_rate: int = 25,
        log_interval: int = 1000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.umm_token_freq = umm_token_freq
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.token_key = token_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.phone2id = phone2id
        self.phone_tone_wordseg_dict = phone_tone_wordseg_dict
        self.frame_rate = frame_rate
        self.log_interval = log_interval

        self.resampler = {}
        self.fast_normalizer = FastNormalizeAudio()
        self.normalize_audio = normalize_audio
        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )
        self.lang2id = {"en": 0, "zh": 1, "zh_en": 2}

    def phone_tone_wordseg_to_id(self, phone, tone, word_seg):
        text_id = (
            phone.astype(np.int64) * 1_000_000
            + tone.astype(np.int64) * 1_000
            + word_seg.astype(np.int64)
        )
        res = [0] * len(text_id)
        for i, t_id in enumerate(text_id):
            if t_id not in self.phone_tone_wordseg_dict:
                logger.warning(f"===>>> phone_tone_wordseg_dict OOV: {t_id}")
                # np.save('tmp/{}.npy'.format(t_id), t_id)
                # key_idx = np.asarray(list(self.phone_tone_wordseg_dict.keys()))
                # min_ind = np.argmin(np.abs(key_idx - t_id))
                # res[i] = self.phone_tone_wordseg_dict[key_idx[min_ind]]
                return None
            else:
                res[i] = self.phone_tone_wordseg_dict[t_id]
        return np.asarray(res).astype(np.int32)

    def preprocess_meta(self, sample: dict):
        meta_obj = get_meta_obj(sample)
        item = {}
        item["labels"] = str(meta_obj.get("labels", ""))
        sample.update(item)
        return sample

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
        else:
            lang = "en"
        return lang

    def __call__(self, item: Dict[str, Any]) -> Union[None, Dict[str, Any]]:
        if self.audio_key in item:
            audio = self.base_transform(item[self.audio_key])
            audio = self.do_resample(item["src_sample_rate"], audio)
            if self.normalize_audio:
                audio = self.fast_normalizer(audio)

            # Audio too short
            if audio.size(-1) < self.min_duration * self.sample_rate:
                self.update_stats(skipped=True, message="Audio too short")
                return
            # Audio too long
            if audio.size(-1) > self.max_duration * self.sample_rate:
                self.update_stats(skipped=True, message="Audio too long")
                return

        if self.token_key in item:
            target_token = torch.as_tensor(
                pickle.loads(item[self.token_key]), dtype=torch.long
            )
            # umm token too short
            if target_token.size(-1) < self.min_duration * self.umm_token_freq:
                self.update_stats(skipped=True, message=f"umm token too short")
                return
            # umm token too long
            if target_token.size(-1) > self.max_duration * self.umm_token_freq:
                self.update_stats(skipped=True, message=f"umm token too long")
                return

        text = normalize_text(item["text"])
        item = self.preprocess_meta(item)
        labels = item["labels"]
        # labels = item["labels"].decode()

        if labels is None:
            self.update_stats(skipped=True, message="Label is None")
            return

        output_dict = {"text": text, "tag": "vocal"}
        if self.audio_key in item:
            output_dict[self.audio_key] = audio
        if self.token_key in item:
            output_dict[self.token_key] = target_token

        if self.tokenizer is not None:
            if self.tokenizer == "sami":
                labels = list(filter(lambda x: x != "", labels.split("\n")))
                try:
                    if len(labels) < 2:
                        self.update_stats(
                            skipped=True, message="Label length is shorter than 2"
                        )
                        return
                    if len(labels[-1].split("\t")) == 2:
                        last_duration = labels[-1].split("\t")[-1]
                        last_line = labels[-2]
                        labels = labels[:-2]
                        last_line = "\t".join(
                            last_line.split("\t")[:-1] + [last_duration]
                        )
                        labels.append(last_line)

                    # Get language ID
                    lang_key = self.get_lang(labels)
                    lang_id = self.lang2id[lang_key]
                    output_dict.update(lang=lang_id)

                    # Convert tacolabel to phone, tone, and wordseg ids
                    text_id_phones_tones = self.phone2id.convert_tacolab_to_text_id(
                        labels
                    )
                    if text_id_phones_tones is None:
                        self.update_stats(
                            skipped=True, message="convert_tacolab_to_text_id failed"
                        )
                        return
                    else:
                        text_id, _, _, _, _ = text_id_phones_tones

                    # Map phone, tone, and wordseg to id
                    token = self.phone_tone_wordseg_to_id(
                        text_id[0, :], text_id[1, :], text_id[2, :]
                    )
                    if token is None:
                        self.update_stats(
                            skipped=True, message="phone_tone_wordseg_to_id failed"
                        )
                        return
                    else:
                        token = torch.from_numpy(token).long()

                    # phone, tone, word_seg
                    phone, tone, wordseg = text_id[0, :], text_id[1, :], text_id[2, :]
                    phone = torch.from_numpy(phone).long()
                    tone = torch.from_numpy(tone).long()
                    wordseg = torch.from_numpy(wordseg).long()
                    output_dict.update(phone=phone)
                    output_dict.update(tone=tone)
                    output_dict.update(wordseg=wordseg)

                except:
                    self.update_stats(skipped=True, message="Tacolabel process failed")
                    return
            else:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)

            if token.size(-1) == 0:
                self.update_stats(skipped=True, message="Token zero length")
                return
            else:
                if self.audio_key in item:
                    frame_num = (
                        math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
                    )
                elif self.token_key in item:
                    frame_num = (
                        math.floor(target_token.size(-1) / self.umm_token_freq)
                        * self.frame_rate
                    )
                else:
                    frame_num = 0
                if token.size(-1) > frame_num:
                    self.update_stats(skipped=True, message="Token too long")
                    return
            output_dict.update(token=token)
        self.update_stats(skipped=False)

        return output_dict

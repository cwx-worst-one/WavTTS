import random
from typing import Callable

import librosa
import numpy as np
import torch
import webdataset as wds
from transformers import T5Tokenizer, Wav2Vec2PhonemeCTCTokenizer

from apps.bigtts.umm.ar.data.phone_to_id import PhoneToId  # use master
from samantha.utils.cmu_phonemes import CMUPhonemeTokenizer
from samantha.utils.distributed import rank_zero_first
from samantha.utils.format_utils import normalize_text


def pad_crop(sequence, seq_len, dtype, padding_value=0):
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[: len(sequence)] = torch.as_tensor(sequence[:seq_len])
    return item_pad


class MCCInstrumentalBatchTransform:
    "Converts musiclm dataloader to work with bigmusic models"

    def __call__(self, item):
        wavs = item.pop("audio")
        if len(wavs.shape) == 3:
            wavs = wavs.squeeze(1)
        return {**item, "style_audio": wavs, "target_audio": wavs}


class MCCMetadataTextTransform:
    def _mcc_metadata_to_string(self, item, type="Vocal"):
        metadata = item["metadata"]
        mood = metadata.get("final_mood")
        genre = metadata.get("final_genre")
        gender = metadata.get("merge_aed")
        text = ""
        if type == "Vocal":
            text = "A "
            if mood is not None and mood != "nan":
                text += mood.lower() + " "
            if genre is not None and genre != "nan":
                text += genre.lower() + " "
            text += "song"
            if gender is not None and gender != "nan":
                if "Female" in gender:
                    text += " with female vocal"
                elif "Male" in gender:
                    text += " with male vocal"
            text += "."
        elif type == "Instrumental":
            text = ""
            if mood is not None and mood != "nan":
                text += mood.lower() + " "
            if genre is not None and genre != "nan":
                text += genre.lower() + " "
            text += "music."
        return text

    def __call__(self, item):
        metadata_string = self._mcc_metadata_to_string(item)
        return {**item, "style_text": metadata_string}


MCC_MOOD = [
    "nan",
    "Happy",
    "Chil",
    "Cute",
    "Sweet",
    "Romantic",
    "Excited",
    "Dynamic",
    "Lonely",
    "Sorrow",
    "Angry",
    "Tense",
]
MCC_GENRE = [
    "nan",
    "Rock",
    "Pop",
    "EDM",
    "R&B",
    "Country",
    "Jazz",
    "Reggae",
    "Blues",
    "Trap Rap",
    "Metal",
    "New Age",
]
MCC_VOICE = ["nan", "Female", "Male"]


class RandomGenreTextTransform(MCCMetadataTextTransform):
    "Randomly samples genre, mood, vocals. This is for non-MCC datasets where we don't have metadata"

    def __call__(self, item):
        metadata_item = {
            "metadata": {
                "final_mood": random.choice(MCC_MOOD),
                "final_genre": random.choice(MCC_GENRE),
                "merge_aed": random.choice(MCC_VOICE),
            }
        }
        metadata_string = self._mcc_metadata_to_string(metadata_item)
        return {**item, "style_text": metadata_string}


class MetadataT5Transform(MCCMetadataTextTransform):
    def __init__(self, max_seq_len: int = 50):
        self.text_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.max_seq_len = max_seq_len

    def __call__(self, item):
        if "style_text" in item:
            metadata_string = item["style_text"]
        elif "metadata" in item:
            metadata_string = self._mcc_metadata_to_string(item)
        else:
            # Could not encode metadata. Return original item
            return item
        style_tokens = torch.LongTensor(
            self.text_tokenizer.encode(
                metadata_string, padding="max_length", max_length=self.max_seq_len
            )
        )
        return {**item, "style_tokens": style_tokens, "style_text": metadata_string}


# Segment Transforms
class LyricsTokenTransform:
    def __init__(
        self,
        lyrics_tokenizer,
        pad_id,
        lyrics_max_seq_len: int,
        normalization_fn=normalize_text,
        truncate_long_lyrics: bool = False,
        handler: Callable = wds.ignore_and_continue,
    ):
        self.lyrics_tokenizer = lyrics_tokenizer
        self.lyrics_max_seq_len = lyrics_max_seq_len
        self.pad_id = pad_id
        self.truncate_long_lyrics = truncate_long_lyrics
        self.normalization_fn = normalization_fn
        self.handler = handler

    def __call__(self, item):
        # try:
        #     lyrics_text = item['lyrics']
        #     if self.normalization_fn and (not isinstance(self.lyrics_tokenizer, SAMITokenizer)):
        #         lyrics_text = self.normalization_fn(lyrics_text)
        #     lyrics_tokens = self.lyrics_tokenizer(lyrics_text)['input_ids']
        #     if not self.truncate_long_lyrics and (len(lyrics_tokens) > self.lyrics_max_seq_len):
        #         return None
        # except Exception as e:
        #     self.handler(e)
        #     print(f"Exception when calling lyric tokenizer: {e}")
        #     return None

        # debug
        lyrics_text = item["lyrics"]
        if self.normalization_fn and (
            not isinstance(self.lyrics_tokenizer, SAMITokenizer)
        ):
            lyrics_text = self.normalization_fn(lyrics_text)
        lyrics_tokens = self.lyrics_tokenizer(lyrics_text)["input_ids"]
        prompt_text_lens = self.lyrics_tokenizer(lyrics_text)["prompt_text_len"]

        # phone, tone, word_seg
        phones = self.lyrics_tokenizer(lyrics_text)["phone"]
        tones = self.lyrics_tokenizer(lyrics_text)["tone"]
        wordsegs = self.lyrics_tokenizer(lyrics_text)["wordseg"]

        # cfg_config
        phones_cfg = self.lyrics_tokenizer(lyrics_text)["phone_cfg"]
        tones_cfg = self.lyrics_tokenizer(lyrics_text)["tone_cfg"]
        wordsegs_cfg = self.lyrics_tokenizer(lyrics_text)["phone_cfg"]

        if not self.truncate_long_lyrics and (
            len(lyrics_tokens) > self.lyrics_max_seq_len
        ):
            return None

        # For inference only, batch size = 1
        # lyrics_tokens = pad_crop(torch.tensor(lyrics_tokens), self.lyrics_max_seq_len, torch.int, padding_value=self.pad_id)
        # return { **item, 'lyrics_tokens': lyrics_tokens, 'lyrics_normalized_text': lyrics_text }

        # Don't do padding
        # return { **item, 'lyrics_tokens': torch.tensor(lyrics_tokens), 'lyrics_normalized_text': lyrics_text }

        # Always add 10 padding
        padding_length = 0

        try:
            lyrics_tokens = pad_crop(
                torch.tensor(lyrics_tokens),
                padding_length + len(lyrics_tokens),
                torch.int,
                padding_value=self.pad_id,
            )
        except:
            lyrics_tokens = phones
            lyrics_tokens = pad_crop(
                torch.tensor(lyrics_tokens),
                padding_length + len(lyrics_tokens),
                torch.int,
                padding_value=self.pad_id,
            )
        phones = pad_crop(
            torch.tensor(phones),
            padding_length + len(phones),
            torch.int,
            padding_value=self.pad_id,
        )
        tones = pad_crop(
            torch.tensor(tones),
            padding_length + len(tones),
            torch.int,
            padding_value=self.pad_id,
        )
        wordsegs = pad_crop(
            torch.tensor(wordsegs),
            padding_length + len(wordsegs),
            torch.int,
            padding_value=self.pad_id,
        )
        prompt_text_lens = torch.tensor(prompt_text_lens)

        # cfg_config
        phones_cfg = pad_crop(
            torch.tensor(phones_cfg),
            padding_length + len(phones_cfg),
            torch.int,
            padding_value=self.pad_id,
        )
        tones_cfg = pad_crop(
            torch.tensor(tones_cfg),
            padding_length + len(tones_cfg),
            torch.int,
            padding_value=self.pad_id,
        )
        wordsegs_cfg = pad_crop(
            torch.tensor(wordsegs_cfg),
            padding_length + len(wordsegs_cfg),
            torch.int,
            padding_value=self.pad_id,
        )

        return {
            **item,
            "lyrics_tokens": lyrics_tokens,
            "lyrics_normalized_text": lyrics_text,
            "phones": phones,
            "tones": tones,
            "wordsegs": wordsegs,
            "prompt_text_lens": prompt_text_lens,
            "phones_cfg": phones_cfg,
            "tones_cfg": tones_cfg,
            "wordsegs_cfg": wordsegs_cfg,
        }

    @classmethod
    def init_cmu_tokenizer(cls, lyrics_max_seq_len, allow_unknown=False, **kwargs):
        cmu_tokenizer = CMUPhonemeTokenizer(allow_unknown=allow_unknown)
        return LyricsTokenTransform(
            cmu_tokenizer, cmu_tokenizer.pad_id, lyrics_max_seq_len, **kwargs
        )

    @classmethod
    def init_espeak_tokenizer(cls, lyrics_max_seq_len, **kwargs):
        with rank_zero_first():
            espeak_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            espeak_tokenizer._add_tokens(["<n>"])
        import logging

        import phonemizer

        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines"
        phonemizer.logger.get_logger().setLevel(logging.ERROR)
        return LyricsTokenTransform(
            espeak_tokenizer,
            espeak_tokenizer.pad_token_id,
            lyrics_max_seq_len,
            **kwargs,
        )

    @classmethod
    def init_sami_tokenizer(cls, lyrics_max_seq_len, test_wer, **kwargs):
        with rank_zero_first():
            phone2id = PhoneToId()
            phone_tone_wordseg_dict = torch.load(
                "apps/bigtts/umm/ar/data/fronted_v3_2_dict.pyt"
            )
            sami_tokenizer = SAMITokenizer(phone2id, phone_tone_wordseg_dict, test_wer)
        return LyricsTokenTransform(
            lyrics_tokenizer=sami_tokenizer,
            pad_id=0,
            lyrics_max_seq_len=lyrics_max_seq_len,
            **kwargs,
        )


class SAMITokenizer:
    def __init__(self, phone2id, phone_tone_wordseg_dict, test_wer):
        self.phone2id = phone2id
        self.phone_tone_wordseg_dict = phone_tone_wordseg_dict
        self.test_wer = test_wer

    def phone_tone_wordseg_to_id(self, phone, tone, word_seg):
        text_id = (
            phone.astype(np.int64) * 1_000_000
            + tone.astype(np.int64) * 1_000
            + word_seg.astype(np.int64)
        )
        res = [0] * len(text_id)
        for i, t_id in enumerate(text_id):
            if t_id not in self.phone_tone_wordseg_dict:
                print(f"===>>> phone_tone_wordseg_dict OOV")
                # np.save('tmp/{}.npy'.format(t_id), t_id)
                # key_idx = np.asarray(list(self.phone_tone_wordseg_dict.keys()))
                # min_ind = np.argmin(np.abs(key_idx - t_id))
                # res[i] = self.phone_tone_wordseg_dict[key_idx[min_ind]]
                return None
            else:
                res[i] = self.phone_tone_wordseg_dict[t_id]
        return np.asarray(res).astype(np.int32)

    def __call__(self, label_path):
        if "|" not in label_path:  # no prompt
            infer_lab = label_path
            infer_labels = infer_lab.split("\n")
            text_id_phones_tones_infer = self.phone2id.convert_tacolab_to_text_id_infer(
                infer_labels
            )
            text_id_infer, _, _, _, _ = text_id_phones_tones_infer
            phones, tones, word_segs = (
                text_id_infer[0],
                text_id_infer[1],
                text_id_infer[2],
            )
            ret_dict = {}
            ret_dict["input_ids"] = self.phone_tone_wordseg_to_id(
                phones, tones, word_segs
            )
            ret_dict["prompt_text_len"] = phones.shape[0]
            phone, tone, wordseg = phones, tones, word_segs
            ret_dict["phone"] = phone
            ret_dict["tone"] = tone
            ret_dict["wordseg"] = wordseg

            # cfg_config
            ret_dict["phone_cfg"] = np.full(
                len(phone), 1 + max(self.phone2id.phone_to_int.values())
            )
            ret_dict["tone_cfg"] = np.full(
                len(tone), 1 + max(self.phone2id.tone_to_int.values())
            )
            ret_dict["wordseg_cfg"] = np.full(
                len(wordseg), 1 + max(self.phone2id.wordseg_to_int.values())
            )
            return ret_dict

        prompt_lab, infer_lab = label_path.split("|")
        if not self.test_wer:
            with open(prompt_lab, "r") as f:
                prompt_labels = [l for l in f]
        else:
            prompt_labels = prompt_lab.split("\n")
        text_id_phones_tones_prompt = self.phone2id.convert_tacolab_to_text_id_infer(
            prompt_labels
        )
        text_id_prompt, _, _, _, _ = text_id_phones_tones_prompt
        phones, tones, word_segs = (
            text_id_prompt[0],
            text_id_prompt[1],
            text_id_prompt[2],
        )

        # with open(infer_lab, 'r') as f:
        #     infer_labels = [l for l in f]
        infer_labels = infer_lab.split("\n")
        text_id_phones_tones_infer = self.phone2id.convert_tacolab_to_text_id_infer(
            infer_labels
        )
        text_id_infer, _, _, _, _ = text_id_phones_tones_infer
        phones_infer, tones_infer, word_segs_infer = (
            text_id_infer[0],
            text_id_infer[1],
            text_id_infer[2],
        )

        ### 续写模式 zh2en 时，修改中文语调
        if (
            self.phone2id.get_lang(prompt_labels) == "zh"
            and self.phone2id.get_lang(infer_labels) == "en"
        ):
            tones[(tones != 2) & (tones != 12) & (tones != 14)] = 0

        ret_dict = {}
        ret_dict["prompt_text_len"] = phones.shape[0]

        phones = np.concatenate([phones, phones_infer[1:]], axis=0)
        tones = np.concatenate([tones, tones_infer[1:]], axis=0)
        word_segs = np.concatenate([word_segs, word_segs_infer[1:]], axis=0)

        ret_dict["input_ids"] = self.phone_tone_wordseg_to_id(phones, tones, word_segs)

        # phone, tone, word_seg
        phone, tone, wordseg = phones, tones, word_segs
        ret_dict["phone"] = phone
        ret_dict["tone"] = tone
        ret_dict["wordseg"] = wordseg
        # cfg_config
        ret_dict["phone_cfg"] = np.full(
            len(phone), max(self.phone2id.phone_to_int.values())
        )
        ret_dict["tone_cfg"] = np.full(
            len(tone), max(self.phone2id.tone_to_int.values())
        )
        ret_dict["wordseg_cfg"] = np.full(
            len(wordseg), max(self.phone2id.wordseg_to_int.values())
        )

        return ret_dict


class AddConditionsTransform:
    def __init__(self, conditions=""):
        self.conditions = conditions

    def __call__(self, item):
        return {**item, "conditions": self.conditions}


class AddMulanVocalTagTransform:
    # Mix mulan requires 'vocal' tag for vocal music generation
    def __call__(self, item):
        style_text = item["style_text"] + " vocal"
        return {**item, "style_text": style_text}


class VocalChromaTransform:
    "Transform for conditioning on vocal chromagram"

    def __init__(self, sample_rate=24_000, sample_duration=10):
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

    # 10sec = 59 timesteps. In general, multiple duration by 6 to get max_len
    @staticmethod
    def get_chromagram(y, sr, hop_length=2**12, n_fft=2**14, max_len=60):
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
        )
        silence = chroma.min(0) > 0.5
        notes = chroma.argmax(0)
        notes[silence] = 12
        notes_padded = np.full((max_len), 12)
        timesteps = min(max_len, len(notes))
        notes_padded[:timesteps] = notes[:timesteps]
        return notes_padded
        # return np.pad(notes, pad_width=((0, 0), (0, 60-notes.shape[-1])), mode='constant', constant_values=(12, 12))

    def __call__(self, item):
        if item is None or "vocal_audio" not in item:
            return item
        cropped_vocals = item["vocal_audio"]
        vocal_chroma = VocalChromaTransform.get_chromagram(
            cropped_vocals.numpy(),
            self.sample_rate,
            max_len=int(self.sample_duration * 6),
        )
        return {**item, "vocal_chroma": vocal_chroma}

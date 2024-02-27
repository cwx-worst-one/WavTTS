from array import array
import io
import random
from string import punctuation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Generator
import re
import pytorch_lightning as pl
import torch
import glob
import numpy as np
import webdataset as wds
import random
import json
import os
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline
import logging
import phonemizer

from recipes.bigmusic.datasets.transforms.leadsheet import (
    make_leadsheet_from_note_and_utterances,
    seq_offset,
    leadsheet_offset,
    split_leadsheet2note_and_phone,
    get_score_from_token_txt,
    concat_alignment,
    dump_leadsheet,
)
from recipes.bigmusic.datasets.transforms.prompt import Prompter
from recipes.bigmusic.datasets.tokenizers.speaker import spkr2id
from recipes.bigmusic.datasets.tokenizers.leadsheet import LeadSheetTokenizerV2
from recipes.bigmusic.datasets.transforms.lyrics import rewrite_metadata
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN
from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.datasets.mcc.mix import (
    WebDatasetBufferPreprocessor,
    BaseTransforms,
    DataModule
)
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id
from recipes.musiclm.utils.dist import local_zero_first
from recipes.musiclm.transforms.audio import (
    FastNormalizeAudio,
    LoudnessCheck,
    NormalizeAudio,
    ReadMP3,
)

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
    Pad,
)
from samantha.utils.webdataset import return_self
from transformers import Wav2Vec2PhonemeCTCTokenizer

NOTE_TOKEN_VOCAB_SIZE = 129  # extended MIDI range (128 note + rest)


def override_parameter(func, **kwargs):
    def wrapper(*args, **func_kwargs):
        func_kwargs.update(kwargs)
        return func(*args, **func_kwargs)
    return wrapper


def pad_crop(sequence, seq_len, dtype, padding_value=0):
    # in item_pad_idx, 0 indicates the values are padded.
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    if len(sequence) > seq_len and seq_len != 0:
        logging.warning(
            f"[pad_crop] input sequence len{len(sequence)} > max seq_len {seq_len}, might cause quality degradation.")
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    item_pad_idx = torch.full((seq_len,), fill_value=0, dtype=int)
    item_pad_idx[:len(sequence)] = torch.ones_like(
        torch.as_tensor(sequence[:seq_len]), dtype=int)
    return item_pad, item_pad_idx


def group_utterances(utterances, min_duration, max_duration, time_in_sec=False, include_intro=False, new_line_token=' <n> '):
    # Extract the start time, end time, raw lyrics, word sequence, note sequence for each utterance.
    new_utterances = []
    for i, u in enumerate(utterances):
        utt_start = u.get('start_time', None)
        if utt_start is None:
            continue
        utt_start = int(utt_start)
        utt_end = u.get('end_time', None)
        if utt_end is None:
            if i < (len(utterances) - 1):
                utt_end = max(int(utterances[i+1].get('start_time', 0)), utt_start)
            else:
                continue
        utt_end = int(utt_end)
        text = u.get('text', '')
        if "lyrics" in u:
            text = u["lyrics"]
        words = u.get('words', [("hello", "2.5", "3", 'he-l-lo')])
        phones = u.get('phoneme', '')
        if time_in_sec:
            new_utterances.append((utt_start, utt_end, text, words, phones))
        else:
            new_utterances.append(
                (int(utt_start/1000), int(utt_end/1000), text, words, phones))
    utterances = new_utterances
    if len(utterances) == 0:
        return []

    # Filter invalid utterances.
    if include_intro:
        u = utterances[0]
        if u[0] > 0:
            utterances.insert(0, (0, u[0], "", []))
    for i in range(len(utterances)-1):
        if utterances[i][1] > utterances[i+1][0]:
            logging.warning("utterances need to be non-overlapping")
            return []
    utterances = [u for u in utterances if u[1]-u[0] > 0]
    utterances = [u for u in utterances if u[3] is not None] # sometimes phone is none, (94, 121, '灿烂的笑脸', None, 'sil\t0')
    if len(utterances) == 0:
        return []

    # Group utterances to segments which duration fall between [min_duration, max_duration]
    segs = []
    i = 0
    s, cur_seg = i, []
    while i < len(utterances):
        if utterances[i][1] - utterances[s][0] < min_duration:
            cur_seg.append(utterances[i])
            i += 1
        else:
            j = i
            while j < len(utterances) and utterances[j][1] - utterances[s][0] <= max_duration:
                j += 1
            if j == i:
                i = s + 1
            else:
                k = random.randint(i+1, j)
                cur_seg.extend([utterances[k] for k in range(i, k)])
                utt_start_time = cur_seg[0][0]
                utt_end_time = cur_seg[-1][1]
                raw_lyrics = new_line_token.join([u[2] for u in cur_seg])

                words = []
                for u in cur_seg:
                    words.extend(u[3])
                timestamped_phoneme_sequence = []
                for word in words:
                    phone_start_time = float(word['start_time'])/1000.0
                    phone_end_time = float(word['end_time'])/1000.0
                    phoneme = word['phoneme']
                    if phoneme == None:
                        continue
                    phone_tone_ids, phones, tones = convert_labels_to_text_id(
                        phoneme.split("\n"))
                    data = {"start": phone_start_time, "end": phone_end_time,
                            "phone": phones, "phoneme_ids": phone_tone_ids[0]}
                    timestamped_phoneme_sequence.append(data)

                phonemes = []
                for u in cur_seg:
                    if isinstance(u[4], str):
                        phonemes.append(u[4])
                phonemes = "\n".join(phonemes)
                labels = list(
                    filter(
                        lambda x: x != "", phonemes.split("\n")
                    )
                )
                if labels:
                    phone_tone_ids, _, _ = convert_labels_to_text_id(labels)
                    phoneme_ids = phone_tone_ids[0]
                else:
                    phoneme_ids = None

                segs.append([utt_start_time,                     # start time
                             utt_end_time,                       # end time
                             raw_lyrics,                         # lyrics text
                             timestamped_phoneme_sequence,       # timestamped phonemes from words
                             phoneme_ids,                        # phoneme IDs
                             ])
                i = k
            # Move to the next vocal starting point
            while i < len(utterances):
                if len(utterances[i][2]) >= 2:
                    break
                i += 1
            if i < len(utterances):
                s, cur_seg = i, []
    return segs


def collate_fn(batch: List[torch.Tensor],
               conditions: str = "lyrics_tokens,leadsheet_tokens",
               sample_rate: int = 24000,
               semantic_frame_rate: int = 25,
               ) -> Dict[str, torch.Tensor]:
    PHONE_PAD_ID = 0
    LEADSHEET_PAD_ID = 0
    LEADSHEET_COFF_PAD_ID = 1
    # Batch {
    #     "audio": np.zeros((24000*20), dtype=int),
    #     "leadsheet_tokens": np.ones((200), dtype=int),
    #     "lyrics_tokens": np.ones((100), dtype=int),
    #     "normalized_text": "hello",
    #     "max_phone_len": 400,
    # }

    # OUTPUT
    audio = []
    audio_lengths = []
    style_audio = []
    index = []
    normalized_text = []
    style_text = []
    lyrics_tokens = []
    leadsheet_tokens = []
    leadsheet_tokens_coffs = []
    speaker_ids = []
    phoneme_tokens = []
    metadatas = []
    uttids = []

    # PAD
    max_phone_len = int(batch[0]["max_phone_len"])
    max_leadsheet_len = int(batch[0]["max_leadsheet_len"])
    max_length = max([x["audio"].shape[-1] for x in batch])
    max_style_audio_length = max([x["style_audio"].shape[-1] for x in batch])
    pad = Pad(n_samples=max_length)
    pad_style = Pad(n_samples=max_style_audio_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_leadsheet_token = torch.full(
        (max_leadsheet_len,), LEADSHEET_PAD_ID, dtype=torch.int)
    default_leadsheet_coff_token = torch.full(
        (max_leadsheet_len,), LEADSHEET_COFF_PAD_ID, dtype=torch.float)

    for idx in range(len(batch)):
        assert (batch[idx] != None)

        uttids.append(batch[idx]['uttid'])
        audio_lengths.append(batch[idx]["audio"].shape[1])
        audio_slice = pad(batch[idx]["audio"])
        audio.append(audio_slice)
        style_audio.append(pad_style(batch[idx]["style_audio"]))

        normalized_text.append(batch[idx]["normalized_text"]
                               if "normalized_text" in batch[idx] else "")
        style_text.append(batch[idx]["style_text"]
                          if "style_text" in batch[idx] else "")
        speaker_ids.append(batch[idx]["speaker_id"]
                           if "speaker_id" in batch[idx] else -1)
        index.append(batch[idx]["index"] if "index" in batch[idx] else None)

        phoneme_token = batch[idx]["phoneme_tokens"] if "phoneme_tokens" in batch[idx] else default_lyrics_token.detach(
        ).clone()
        phoneme_token, _ = pad_crop(phoneme_token,
                                    max_phone_len, torch.int, PHONE_PAD_ID)
        phoneme_tokens.append(phoneme_token)

        lyric_token = batch[idx]["lyrics_tokens"] if "lyrics_tokens" in batch[idx] else default_lyrics_token.detach(
        ).clone()
        lyric_token, _ = pad_crop(lyric_token,
                                  max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(lyric_token)

        leadsheet_token = batch[idx]["leadsheet_tokens"] if "leadsheet_tokens" in batch[idx] else default_leadsheet_token.detach(
        ).clone()
        leadsheet_token, _ = pad_crop(
            leadsheet_token,
            max_leadsheet_len, torch.int, LEADSHEET_PAD_ID)
        leadsheet_tokens.append(leadsheet_token)

        leadsheet_tokens_coff = batch[idx]["leadsheet_tokens_coff"] if "leadsheet_tokens_coff" in batch[
            idx] else default_leadsheet_coff_token.detach().clone()
        leadsheet_tokens_coff, _ = pad_crop(
            leadsheet_tokens_coff,
            max_leadsheet_len, torch.float, LEADSHEET_COFF_PAD_ID)
        leadsheet_tokens_coffs.append(leadsheet_tokens_coff)

    speaker_ids = torch.tensor(speaker_ids).view(-1, 1)
    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)
    stacked_style_audio = torch.stack(style_audio, dim=0)
    if stacked_style_audio.dim() == 3:
        stacked_style_audio = stacked_style_audio.squeeze(1)

    target_tokens_length = [int(x / sample_rate * semantic_frame_rate) for x in audio_lengths]

    return {
        "index": index,
        "uttids": uttids,
        "audio_lengths": torch.LongTensor(audio_lengths),
        "target_tokens_length": torch.LongTensor(target_tokens_length),
        "target_audio": stacked_audio,
        "spkr_ids": speaker_ids,
        "style_audio": stacked_style_audio,
        "vocal_audio": stacked_style_audio,
        "prefix_audio": stacked_style_audio,
        "lyrics": normalized_text,
        "style_text": style_text,
        "phoneme_tokens": torch.stack(phoneme_tokens),
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "leadsheet_tokens": torch.stack(leadsheet_tokens),
        "leadsheet_tokens_coff": torch.stack(leadsheet_tokens_coffs),
        "conditions": conditions,
        "metadata": metadatas,
    }


def rewrite_metadata_svs(metadata):
    # style text include speaker name, emotion, gender, style, and language etc.
    speaker_id = metadata.get('speaker_id')  # str 'merci_SVS'
    gender = metadata.get('gender')  # male, female
    language = metadata.get('language')  # ZH, EN
    mood = metadata.get('emotion')  # 'sad','firm','neutral','enthusiastic'
    if language == "ZH":
        language = "mandarin"
    elif language == "EN":
        language = "english"
    speaker_id = speaker_id.replace("_SVS", "")
    text = "A"
    if mood is not None and mood != 'nan' and mood.strip():
        text += " " + mood.lower()
    if language is not None and language != 'nan' and language.strip():
        text += " " + language.lower()
    text += " song"
    if gender is not None and gender != 'nan':
        if 'female' in gender:
            text += " with female vocal"
        elif 'male' in gender:
            text += " with male vocal"
    text += f". By a professional {language} singer {speaker_id}."
    return text


def speaker2id(speaker_name):
    speaker_name = speaker_name.replace("_SVS", "").replace("_svs", "")
    spkr_id = spkr2id[speaker_name]
    return spkr_id


class SVSPretrainTransforms(BaseTransforms):
    name = "SVSPretrainTranforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate=24000,
        audio_key: str = "vocal",
        index_key: str = "__index_data__",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        include_intro: bool = False,
        boundaries: List[float] = [1.0, 1.0],
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        leadsheet_align_mode: str = "concat",
        extra_audio_keys=None,
        **kwargs,
    ):
        self.genre_filtered = genre_filtered
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.boundaries = boundaries
        # metadata filtering
        self.lyrics_confidence = lyrics_confidence
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.include_intro = include_intro
        # segmentation
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.segment_max_leadsheet_len = segment_max_leadsheet_len
        self.max_seg_per_track = max_seg_per_track
        self.lyrics_tokenizer = lyrics_tokenizer
        self.leadsheet_align_mode = leadsheet_align_mode
        self.leadsheet_tokenizer = leadsheet_tokenizer
        if extra_audio_keys is None:
            extra_audio_keys = []
        self.extra_audio_keys = extra_audio_keys
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )

        assert len(boundaries) == 2
        assert self.lyrics_tokenizer
        assert self.leadsheet_tokenizer

        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        super().__init__()

    def is_confident_lyrics(self, utterance, threshold):
        conf, num_utt = 0., 0.
        for utt in utterance:
            if len(utt['text']) > 0:
                if "confidence" in utt:
                    conf += float(utt["confidence"])
                else:
                    conf += 1.0 # from force-alignment
                num_utt += 1
        if num_utt == 0:
            return False
        conf /= num_utt
        return True if conf > threshold else False

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        if self.genre_filtered and metadata.get("final_genre") == "Hip Hop/Rap":
            return False
        if self.aed_filtered and metadata.get("aed_filtered") == False:
            return False
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False
        return True

    def extract_metadata_and_utterances(self, index_data):
        metadata = index_data['metadata'] if 'metadata' in index_data else index_data
        try:
            utterances = metadata.get('lyrics').get(
                'utterances').get("result")[0].get("utterances")
        except Exception as e:
            utterances = None
            self.handler(e)
        metadata = {k: metadata[k] for k in
                    ('meta_song_id', 'meta_song_title', 'meta_song_author_name',
                     'final_genre', 'final_mood', 'merge_aed', 'label_id',
                     'audio_metrics', 'vad')}
        return metadata, utterances

    def __call__(self, item: Dict[str, Any]) -> Generator:
        assert item
        metadata = json.loads(item["meta"])
        # meta_song_id_str = str(metadata["meta_song_id"])

        if 'midi' not in metadata.keys() or metadata['midi'] == None or metadata['midi'] == []:
            # with open(f"vocal_test/bad_metadata_{meta_song_id_str}.json", "w") as f:
            #     json.dump(metadata, f)
            self._update_stats(skipped=True, message="No MIDI transription")
            return

        if not self.is_metadata_good(metadata):
            self._update_stats(skipped=True, message="bad metadata")
            return

        note_sequence = metadata['midi']
        if isinstance(note_sequence, dict):
            if 'vocal' not in note_sequence.keys() or note_sequence['vocal'] == None or note_sequence['vocal'] == []:
                self._update_stats(skipped=True, message="No MIDI transription")
                # print("No MIDI transription")
                return
            else:
                note_sequence = note_sequence['vocal']

        audio = self.base_transform(item[self.audio_key])
        extra_audio = [self.base_transform(item[k]) for k in self.extra_audio_keys]
        utterances = metadata["lyrics"]["result"][0]["utterances"]
        if not self.is_confident_lyrics(utterances, self.lyrics_confidence):
            self._update_stats(skipped=True, message="Low confidence lyrics")
            # print("Low confidence lyrics", meta['artist_name'])
            return
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            # print("No utterances", meta['artist_name'])
            return

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            extra_audio = [a.unsqueeze(0) for a in extra_audio]

        # Each segment is in the following format:
        # [start_sec_int, end_sec_int, lyrics_text, Seq({"start":start, "end":end, "phone":[phone1, phone2..]})]
        segments = group_utterances(utterances, self.min_duration, self.max_duration,
                                    time_in_sec=False, include_intro=self.include_intro)
        # Ensure segment is long enough for Mulan zero-shot tagging
        min_duration_sec = 10
        segments = [s for s in segments if s[1] - s[0] >= min_duration_sec]
        if len(segments) < 1:
            self._update_stats(skipped=True, message="No segments")
            return

        if self.segment_method == "first":
            segments = segments[:1]
        elif self.max_seg_per_track > 0:
            random.shuffle(segments)
            segments = segments[:self.max_seg_per_track]

        self._update_stats(skipped=False)
        for seg_i, segment in enumerate(segments):
            normalized_text = segment[2]
            timestamped_phones = segment[3]
            next_seg_i = seg_i + 1
            # if next_seg_i <= len(segments) - 1:
            #     # NOTE: Because ASR trends to ignore the run note at the end of utt
            #     # so we exapnd the time of this utt to the start time of next utt when possible
            #     next_segment = segments[next_seg_i]
            #     expanded_end_time = next_segment[3][0]['start']
            #     origin_end_time = timestamped_phones[-1]['end']
            #     if origin_end_time != expanded_end_time:
            #         print("1")
            #     expanded_end_time = min(origin_end_time + self.boundaries[1], expanded_end_time)
            #     timestamped_phones[-1]['end'] = expanded_end_time
            segment_phoneme_ids = segment[4]
            if len(normalized_text) == 0 or len(timestamped_phones) == 0:
                # print('normalized_text too short', len(normalized_text) ==0, len(timestamped_phones))
                continue

            if self.leadsheet_align_mode == "forced":
                # leadsheet is a sequence of tuples like (note, start, end, [phone_ids])
                leadsheet = make_leadsheet_from_note_and_utterances(note_sequence,
                                                                    phoneme_sequence=timestamped_phones,
                                                                    align_mode=self.leadsheet_align_mode)
                if len(leadsheet) == 0:
                    # print('len(leadsheet) too short', len(leadsheet))
                    continue
                leadsheet_tokens_dict = self.leadsheet_tokenizer(
                    leadsheet, return_dict=True)
                phoneme_tokens = leadsheet_tokens_dict['phoneme_tokens']
                leadsheet_tokens = leadsheet_tokens_dict['leadsheet_tokens']
                note_tokens = leadsheet_tokens_dict['note_tokens']

            elif self.leadsheet_align_mode == "concat":
                sliced_notes, sliced_phones = make_leadsheet_from_note_and_utterances(
                    note_sequence=note_sequence,
                    phoneme_sequence=timestamped_phones,
                    align_mode=self.leadsheet_align_mode,
                    boundaries=self.boundaries)
                if len(sliced_notes) == 0 or len(sliced_phones) == 0:
                    continue

                segment[0] = min(sliced_notes[0]['start'], sliced_phones[0]['start'])
                segment[1] = max(sliced_notes[-1]['end'], sliced_phones[-1]['end'])

                sliced_notes = seq_offset(sliced_notes, offset=-1*segment[0])
                sliced_phones = seq_offset(sliced_phones, offset=-1*segment[0])

                leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens = concat_alignment(
                    self.leadsheet_tokenizer, sliced_notes, sliced_phones)
            else:
                raise NotImplementedError

            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]
            extra_audio = [a[:, start:end] for a in extra_audio]
            # import torchaudio
            # torchaudio.save("clip.wav", clip, 24000)
            if clip.shape[-1] < self.sample_rate * self.min_duration // 2:
                # print('audio too short', clip.shape)
                continue
            if clip.shape[0] == 0 or clip.shape[1] == 0:
                # print('audio shape too short', clip.shape)
                continue

            if self.lyrics_tokenizer == "sami_tts_frontend_precompute":
                lyrics_tokens = segment_phoneme_ids
            else:
                lyrics_tokens = self.lyrics_tokenizer(
                    normalized_text,
                    add_special_tokens=False,
                    return_tensors="pt")["input_ids"].squeeze(dim=0)
            if lyrics_tokens is None or len(lyrics_tokens) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                continue
            # dataset output Example:
            # [
            #     {'audio': tensor([[1.1484,0.6641,  ...,0.0000]]),
            #       'style_audio': tensor([[1.1484,0.6641,  ...,0.0000]]),
            #       'style_text': 'placeholder',
            #       'speaker_id': 0,
            #       'note_tokens': tensor([549,400,550,552]),
            #       'leadsheet_tokens': tensor([549,400,550,552]),
            #       'normalized_text': '你眉头开了',
            #       'lyrics_tokens': tensor([66,37,3]),
            #       'max_phone_len': 400,
            #       'max_leadsheet_len': 800,
            #     }
            # ]

            # import torchaudio
            # torchaudio.save(f"dataset708/{seg_i}V1_{meta_song_id_str}.wav", clip.view(1, -1), 24000)
            # with open(f"dataset708/{seg_i}V1_{meta_song_id_str}.aligned_midi.txt", "w") as f:
            #     sliced_notes = '\n'.join(map(str, sliced_notes))
            #     f.write(sliced_notes)
            # with open(f"dataset708/{seg_i}V1_{meta_song_id_str}.aligned_phone.txt", "w") as f:
            #     sliced_phones = '\n'.join(map(str, sliced_phones))
            #     f.write(sliced_phones)
            # with open(f"dataset708/{seg_i}V1_{meta_song_id_str}.raw_timestamped_phones.txt", "w") as f:
            #     timestamped_phones = '\n'.join(map(str, timestamped_phones))
            #     f.write(timestamped_phones)
            yield {
                "uttid": item['uttid'],
                "audio": clip,
                "extra_audio": extra_audio,
                "style_audio": clip,
                "speaker_id": 0,
                "phoneme_tokens": phoneme_tokens,
                "lyrics_tokens": lyrics_tokens,
                "note_tokens": note_tokens,
                "leadsheet_tokens": leadsheet_tokens,
                "leadsheet_tokens_coff": leadsheet_tokens_coff,
                "normalized_text": normalized_text,
                "max_phone_len": self.segment_max_phone_len,
                "max_leadsheet_len": self.segment_max_leadsheet_len,
                "style_text": rewrite_metadata(metadata),
            }


class SVSPretrainDataset(WebPipeline):
    name = "SVSPretrain"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "",
        weights: int = 1,
        data_id: int = 0,
        url_pattern: str = None,
        sample_rate=24000,
        audio_key: str = "vocal",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        include_intro: bool = False,
        boundaries: List[float] = [1.0, 1.0],
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        handler=wds.warn_and_continue,
        extra_audio_keys=None,
        **kwargs,
    ):
        if extra_audio_keys is None:
            extra_audio_keys = []
        print(f"[{self.name}] initializing...")

        assert bool(url2index) != bool(data_id)
        if url2index:
            if isinstance(url2index, list):
                dataset = MultiIterableDataset(
                    datasets=[IndexedWebDataset(url2index=url, **kwargs)
                              for url in url2index],
                    weights=weights)
            else:
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        if data_id:
            dataset = ParquetDataset(data_id=data_id, data_urls=None, extra_fields_in_data=[
                                     'vocal', 'audio'], **kwargs)

        transforms = SVSPretrainTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            # audio filtering
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            # metadata filtering
            lyrics_confidence=lyrics_confidence,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            # segmentation
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            segment_max_leadsheet_len=segment_max_leadsheet_len,
            include_intro=include_intro,
            boundaries=boundaries,
            # lyrics tokenizer
            lyrics_tokenizer=lyrics_tokenizer,
            leadsheet_tokenizer=leadsheet_tokenizer,
            extra_audio_keys=extra_audio_keys
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if url2index:
            pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        if data_id:
            pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SVSEvalTransforms(BaseTransforms):
    name = "SVSEvalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate=24000,
        audio_key: str = "audio.npy",
        index_key: str = "__index_data__",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        leadsheet_align_mode: str = "concat",
        style_prompt_path: str = '',
        vocal_prompt_duration: float = 10.0,
        pitch_shift: int = 0,
        tmp_infer_example='',
        extra_audio_keys=None,
        language: List[str] = [],
        run_opts: dict = {},
        **kwargs,
    ):
        self.pitch_shift = pitch_shift
        self.leadsheet_align_mode = leadsheet_align_mode
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocal_prompt_duration = vocal_prompt_duration
        self.run_opts = run_opts
        # metadata filtering
        self.lyrics_confidence = lyrics_confidence
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.target_spkr_name = target_spkr_name
        self.use_empty_spkr_id = use_empty_spkr_id
        self.use_empty_style_audio = use_empty_style_audio
        self.language = language
        # segmentation
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.segment_max_leadsheet_len = segment_max_leadsheet_len
        self.max_seg_per_track = max_seg_per_track
        self.lyrics_tokenizer = lyrics_tokenizer
        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.style_prompt_path = style_prompt_path
        self.tmp_infer_example = tmp_infer_example
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        if extra_audio_keys is None:
            extra_audio_keys = []
        self.extra_audio_keys = extra_audio_keys
        assert self.lyrics_tokenizer
        self.dump_cache = {}

        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)

        self.prompter = Prompter(
            leadsheet_tokenizer, style_prompt_path, self.sample_rate, vocal_prompt_duration=vocal_prompt_duration)
        super().__init__()

    def __call__(self, item: Dict[str, Any]) -> Generator:
        assert item
        metadata = item['__index_data__']["metadata"]
        # metadata example:
        # {
        #     "uttid": "Todd-10001",
        #     "audio_fp": "/mnt/Todd-10001.wav",
        #     "musicxml": [],   # leadsheet in string, can be read with music21
        #     "interval": [],   # phoneme time boundary in second
        #     "note": [('Rest', 0.0, 0.5), ('Bb3', 0.5, 0.975), ('Db4', 0.975, 1.4), ('Bb3', 1.4, 1.8125)]
        #     "score": [('Rest', ['sil'], 0.0, 0.5), ('Bb3', [['C0d', 'C0eng']], 0.5, 0.975), ('Db4', [['C0g', 'C0uang']], 0.975, 1.4)]
        #     "phoneme": 'sil', 'C0d', 'C0eng', 'C0g', 'C0uang',
        #     "phoneme_timestamp": [('sil', 0.0, 0.5), ('C0d', 0.5, 0.57459), ('C0eng', 0.57459, 0.96891)]
        #     "lyric": "灯光也暗了，音乐低声了",
        #     "speaker_id": 'merci',
        #     "gender": 'female',
        #     "language": 'ZH',
        #     "emotion": 'enthusiastic',
        # }

        # dataset output Example:
        # [
        #     {'audio': tensor([[1.1484,0.6641,  ...,0.0000]]),
        #       'style_text': '',
        #       'leadsheet_tokens': tensor([549,400,550,552]),
        #       'normalized_text': '你眉头开了',
        #       'lyrics_tokens': tensor([66,37,3]),
        #       'max_phone_len': 400,
        #       'max_leadsheet_len': 800,
        #       'song_id': '/mnt/bn/merci-018001.wav',
        #       'metadata': {...}
        #     }
        # ]

        # musicxml and interval is ununsed currently, drop it for simplicity
        del metadata["musicxml"]
        del metadata["interval"]

        if metadata['language'] not in self.language:
            return

        meta_song_id_str = os.path.basename(metadata["audio_fp"])
        if self.target_spkr_name != "":
            spkr_name = self.target_spkr_name
        else:
            spkr_name = metadata['speaker_id'].replace("_SVS", "")
        if self.use_empty_spkr_id:
            # if use use_empty_spkr_id as condition
            speaker_id = 0
        else:
            speaker_id = speaker2id(spkr_name)

        style_text = rewrite_metadata_svs(metadata)
        audio = self.base_transform(item[self.audio_key])
        extra_audio = [self.base_transform(k) for k in self.extra_audio_keys]

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            extra_audio = [a.unsqueeze(0) for a in extra_audio]

        audio_len = int(self.min_duration * self.sample_rate)
        audio_transform = Compose([
            Pad(audio_len),
        ])
        audio = audio_transform(audio)
        extra_audio = [audio_transform(k) for k in self.extra_audio_keys]
        style_audio = audio

        lyric = metadata['lyric']
        normalized_text = normalize_text(lyric)

        leadsheet = metadata['score']
        # Pad audio and leadsheet to minimun duration
        leadsheet_duration = leadsheet[-1][3]
        if leadsheet[-1][3] < self.min_duration:
            leadsheet.append(("Rest", ["sil"], leadsheet_duration, self.min_duration))
        sliced_notes, sliced_phones = split_leadsheet2note_and_phone(leadsheet)

        # dump = True
        # if dump == True:
        #     import torchaudio
        #     meta_song_id_str = os.path.basename(metadata["audio_fp"])
        #     idx = int(meta_song_id_str.split("-")[1].replace(".wav", "")[-1])
        #     if idx == 1:
        #     # if speaker_id not in self.dump_cache:
        #     #     self.dump_cache[speaker_id] = 0
        #     # if self.dump_cache[speaker_id] < 10: # dump 10 example for every spkr would enough
        #         # self.dump_cache[speaker_id] += 1
        #         save_dir = os.path.join("assets/svs/svs_prompt", meta_song_id_str)
        #         logging.info("save_dir "+save_dir)
        #         os.makedirs(save_dir, exist_ok=True)
        #         torchaudio.save(f"{save_dir}/audio.wav", audio.view(1, -1), 24000)
        #         with open(f"{save_dir}/note.json", "w") as f:
        #             json.dump(sliced_notes, f, indent=4)
        #         with open(f"{save_dir}/phones.json", "w") as f:
        #             json.dump(sliced_phones, f, indent=4)
        # return 

        style_audio, sliced_notes, sliced_phones = self.prompter.warp_prompt(
            sliced_notes, sliced_phones, spkr_name, num_prompt=self.run_opts.get('vocal_prompt_number', 1))

        leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens = concat_alignment(
            self.leadsheet_tokenizer, sliced_notes, sliced_phones, pitch_shift=self.pitch_shift)
        if self.use_empty_style_audio:
            style_audio = torch.zeros_like(style_audio)

        if self.lyrics_tokenizer == "sami_tts_frontend_precompute":
            lyrics_tokens = phoneme_tokens
        else:
            lyrics_tokens = self.lyrics_tokenizer(
                normalized_text,
                add_special_tokens=False,
                return_tensors="pt")["input_ids"].squeeze(dim=0)

        metadata["pitch_shift"] = self.pitch_shift
        metadata["note_tokens"] = note_tokens.flatten().cpu().tolist()
        yield {
            "audio": audio,
            "extra_audio": extra_audio,
            "style_audio": style_audio,
            "style_text": style_text,
            "phoneme_tokens": phoneme_tokens,
            "note_tokens": note_tokens,
            "speaker_id": speaker_id,
            "leadsheet_tokens": leadsheet_tokens,
            "leadsheet_tokens_coff": leadsheet_tokens_coff,
            "normalized_text": normalized_text,
            "lyrics_tokens": lyrics_tokens,
            "max_phone_len": self.segment_max_phone_len,
            "max_leadsheet_len": self.segment_max_leadsheet_len,

            # DEBUG
            "uttid": meta_song_id_str,
            "index": meta_song_id_str,
        }


class SVSEvalDataset(WebPipeline):
    name = "SVSEval"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = [
            "hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
        weights: list = [1],
        data_id: int = 0,
        url_pattern: str = None,
        sample_rate=24000,
        audio_key: str = "audio.npy",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        include_intro: bool = False,
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        handler=wds.warn_and_continue,
        style_prompt_path: str = '',
        vocal_prompt_duration: float = 10.0,
        tmp_infer_example='',
        language: List[str] = [],
        extra_audio_keys=None,
        run_opts: dict = {},
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        self.handler = handler

        assert bool(url2index) != bool(data_id)
        if url2index:
            if isinstance(url2index, list):
                dataset = MultiIterableDataset(
                    datasets=[IndexedWebDataset(url2index=url, **kwargs)
                              for url in url2index],
                    weights=weights)
            else:
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        if data_id:
            dataset = ParquetDataset(data_id=data_id, data_urls=None, **kwargs)
        if extra_audio_keys is None:
            extra_audio_keys = []
        transforms = SVSEvalTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            style_prompt_path=style_prompt_path,
            vocal_prompt_duration=vocal_prompt_duration,
            tmp_infer_example=tmp_infer_example,
            run_opts=run_opts,
            # audio filtering
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            # metadata filtering
            lyrics_confidence=lyrics_confidence,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            target_spkr_name=target_spkr_name,
            use_empty_spkr_id=use_empty_spkr_id,
            use_empty_style_audio=use_empty_style_audio,
            language=language,
            # segmentation
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            segment_max_leadsheet_len=segment_max_leadsheet_len,
            # lyrics tokenizer
            lyrics_tokenizer=lyrics_tokenizer,
            leadsheet_tokenizer=leadsheet_tokenizer,
            extra_audio_keys=extra_audio_keys
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if url2index:
            pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        if data_id:
            pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SVSInferTransforms(BaseTransforms):
    name = "SVSInferTransforms"
    sample_rate = 24000

    def __init__(
        self,
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        leadsheet_align_mode: str = "concat",
        style_prompt_path: str = '',
        vocal_prompt_duration: float = 10.0,
        vocal_prompt_number: int = 1,
        pitch_shift: int = 0,
        segment_max_phone_len: int = 0,
        segment_max_leadsheet_len: int = 2500,
        target_spkr_name: str = '',
        time_format: str = "start,duration,merge_sil,merge_rest",
        min_duration: float = 10.0,
        **kwargs,
    ):
        self.pitch_shift = pitch_shift
        self.leadsheet_align_mode = leadsheet_align_mode
        self.vocal_prompt_duration = vocal_prompt_duration
        self.vocal_prompt_number = vocal_prompt_number
        self.target_spkr_name = target_spkr_name
        self.min_duration = min_duration

        self.lyrics_tokenizer = lyrics_tokenizer
        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.leadsheet_tokenizer.time_format = time_format
        self.style_prompt_path = style_prompt_path
        self.style_prompt_cache = {}

        self.segment_max_phone_len = segment_max_phone_len
        self.segment_max_leadsheet_len = segment_max_leadsheet_len

        self.prompter = Prompter(leadsheet_tokenizer, style_prompt_path,
                                 self.sample_rate, vocal_prompt_duration)
        super().__init__()

    def __call__(self, score_txt_path: Dict[str, Any]) -> dict:
        assert score_txt_path

        leadsheet = get_score_from_token_txt(score_txt_path)
        sliced_notes, sliced_phones = split_leadsheet2note_and_phone(leadsheet)

        style_audio, sliced_notes, sliced_phones = self.prompter.warp_prompt(
            sliced_notes, sliced_phones, self.target_spkr_name, num_prompt=self.vocal_prompt_number)

        # Pad leadsheet to minimun duration
        leadsheet_duration = leadsheet[-1][3]
        if leadsheet[-1][3] < self.min_duration:
            leadsheet.append(("Rest", ["sil"], leadsheet_duration, self.min_duration))
        sliced_notes, sliced_phones = split_leadsheet2note_and_phone(leadsheet)

        leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens = concat_alignment(
            self.leadsheet_tokenizer, sliced_notes, sliced_phones, pitch_shift=self.pitch_shift)

        normalized_text = "placeholder"
        if self.lyrics_tokenizer == "sami_tts_frontend_precompute":
            lyrics_tokens = phoneme_tokens
        else:
            lyrics_tokens = self.lyrics_tokenizer(
                normalized_text,
                add_special_tokens=False,
                return_tensors="pt")["input_ids"].squeeze(dim=0)

        uttid = index = os.path.basename(score_txt_path).split('.')[0]
        return {
            "uttid": uttid,
            "index": index,
            "audio": torch.zeros_like(style_audio),
            "style_audio": style_audio,
            "phoneme_tokens": phoneme_tokens,
            "note_tokens": note_tokens,
            "speaker_id": speaker2id(self.target_spkr_name),
            "leadsheet_tokens": leadsheet_tokens,
            "leadsheet_tokens_coff": leadsheet_tokens_coff,
            "max_phone_len": self.segment_max_phone_len,
            "max_leadsheet_len": self.segment_max_leadsheet_len,
        }


class SVSFinetuneTransforms(BaseTransforms):
    name = "SVSFinetuneTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate=24000,
        audio_key: str = "audio.npy",
        index_key: str = "__index_data__",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        is_eval: bool = False,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        leadsheet_align_mode: str = "concat",
        extra_audio_keys=None,
        language: List[str] = [],
        **kwargs,
    ):
        self.leadsheet_align_mode = leadsheet_align_mode
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.is_eval = is_eval
        # metadata filtering
        self.lyrics_confidence = lyrics_confidence
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.target_spkr_name = target_spkr_name
        self.use_empty_spkr_id = use_empty_spkr_id
        self.use_empty_style_audio = use_empty_style_audio
        self.language = language
        # segmentation
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.segment_max_leadsheet_len = segment_max_leadsheet_len
        self.max_seg_per_track = max_seg_per_track
        self.lyrics_tokenizer = lyrics_tokenizer
        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        if extra_audio_keys is None:
            extra_audio_keys = []
        self.extra_audio_keys = extra_audio_keys
        assert self.lyrics_tokenizer

        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        self.example_cache = {}
        super().__init__()

    def random_crop_sil(self, audio, leadsheet):
        # Trim leadsheet silence
        strip = random.randint(0, 3)
        if strip == 0:
            pass
        elif strip == 1:
            leadsheet = self.strip_leadsheet(leadsheet, lstrip=True)
        elif strip == 2:
            leadsheet = self.strip_leadsheet(leadsheet, rstrip=True)
        elif strip == 3:
            leadsheet = self.strip_leadsheet(leadsheet, lstrip=True, rstrip=True)
        else:
            raise NotImplementedError

        start_time = leadsheet[0][2]
        end_time = leadsheet[-1][3]
        audio = audio[:, int(start_time * self.sample_rate):int(end_time * self.sample_rate)]
        leadsheet = leadsheet_offset(leadsheet, offset=-1*start_time)

        # Crop head rest note
        first = leadsheet[0]
        if first[0] != 'Rest':
            return audio, leadsheet

        duration = first[3]
        random_start = random.uniform(0, duration)

        # crop audio
        start_idx = int(random_start * self.sample_rate)
        audio = audio[:, start_idx:]

        # crop leadsheet, overwrite first note
        if duration - random_start < 0.0125:
            # If we almost want to crop all beginning Rest
            leadsheet = leadsheet[1:]
            new_leadsheet = leadsheet_offset(leadsheet, offset=-1 * duration)
        else:
            new_leadsheet = leadsheet_offset(leadsheet, offset=-1 * random_start)
            note, phone, start, end = new_leadsheet[0]
            new_leadsheet[0] = (note, phone, 0.0, end)
        return audio, new_leadsheet

    def random_slice(self, audio, leadsheet):
        # Random slice leadsheet and audio accordingly
        # audio: torch.Tensor, size=(1, -1)
        # leadsheet: list of (note_str, phone_list, start_time_float, end_time_float)
        leadsheet_len = len(leadsheet)
        if leadsheet_len <= 5: 
            return audio, leadsheet

        end = random.choice(range(leadsheet_len))
        if end <= 5: 
            return audio, leadsheet

        start = random.choice(range(end))
        if end - start <= 1: 
            return audio, leadsheet

        start_time = leadsheet[start][2]
        end_time = leadsheet[end][3]
        start_idx = int(self.sample_rate*start_time)
        end_idx = int(self.sample_rate*end_time)
        audio = audio[:, start_idx:end_idx]
        leadsheet = leadsheet[start:end+1]
        leadsheet = leadsheet_offset(leadsheet, offset=-1*start_time)
        return audio, leadsheet

    def strip_leadsheet(self, leadsheet, lstrip=False, rstrip=False):
        first_note = leadsheet[0]
        last_note = leadsheet[-1]
        if len(leadsheet) <= 3:
            return leadsheet

        if lstrip == False and rstrip == False:
            return leadsheet

        # drop first note if is Rest
        if lstrip:
            if first_note[0] != "Rest":
                return leadsheet
            else:
                leadsheet = leadsheet[1:]

        # drop last note if is Rest
        if rstrip:
            if last_note[0] != "Rest":
                return leadsheet
            else:
                leadsheet = leadsheet[:-1]
        
        return self.strip_leadsheet(leadsheet, lstrip, rstrip)

    def random_pad_sil(self, audio, leadsheet, max_sil=10):
        # Random pad last note
        # audio: torch.Tensor, size=(1, -1)
        # leadsheet: list of (note_str, phone_list, start_time_float, end_time_float)

        audio_len = audio.shape[1]
        leadsheet_duration = leadsheet[-1][3]

        # new Rest note
        # long-tailed distution range from [0,max_sil)
        start_random_sil = random.betavariate(0.5, max_sil) * max_sil
        end_random_sil = random.betavariate(0.5, max_sil) * max_sil
        
        min_ms = int(self.sample_rate * 0.0125)
        start_random_idx = int(self.sample_rate * start_random_sil)
        end_random_idx = int(self.sample_rate * end_random_sil)
        if start_random_idx < min_ms or  end_random_idx < min_ms:
            return audio, leadsheet

        new_audio_len = start_random_idx + audio_len + end_random_idx
        padded_audio = torch.zeros([1, new_audio_len])
        padded_audio[:, start_random_idx:start_random_idx+audio_len] = audio
        audio = padded_audio

        start_note = [("Rest", ['sil'], 0, start_random_sil)]
        leadsheet_shift = leadsheet_offset(leadsheet, offset=start_random_sil)
        end_note = [("Rest", ['sil'], start_random_sil + leadsheet_duration, start_random_sil + leadsheet_duration + end_random_sil)]

        leadsheet = start_note + leadsheet_shift + end_note

        return audio, leadsheet

    def random_beak(self, audio, leadsheet, k=3):
        # Random insert Rest into leadsheet and audio accordingly
        # audio: torch.Tensor, size=(1, -1)
        # leadsheet: list of (note_str, phone_list, start_time_float, end_time_float)
        for i in range(k):
            # decide where to break
            random_sil = random.uniform(0.001, 2)
            leadsheet_len = len(leadsheet)
            breaking = random.choice(range(leadsheet_len))
            if breaking == leadsheet_len-1 or breaking == 0:
                continue
            insert_start_time = leadsheet[breaking][2]

            # break and insert leadsheet
            leadsheet_left, leadsheet_right = leadsheet[:breaking], leadsheet[breaking:]
            sil_note = [("Rest", ['sil'], insert_start_time, insert_start_time+random_sil)]
            leadsheet_right = leadsheet_offset(leadsheet_right, offset=random_sil)
            leadsheet = leadsheet_left + sil_note + leadsheet_right

            # break and insert audio
            sil_dur = int(self.sample_rate * breaking)
            sil_idx = int(self.sample_rate * insert_start_time)
            audio_left, audio_right = audio[:, :sil_idx], audio[:, sil_idx:]
            sil_audio = torch.zeros([1, sil_dur])
            audio = torch.cat([audio_left, sil_audio, audio_right], dim=1)
        return audio, leadsheet


    def random_repeat(self, audio, leadsheet, spkr_name, song_id, k=3):
        if spkr_name not in self.example_cache.keys():
            self.example_cache[spkr_name] = {}
        self.example_cache[spkr_name][song_id] = (audio, leadsheet)

        if len(self.example_cache[spkr_name]) <= k*k:
            return audio, leadsheet

        if random.randint(0, 1) == 0:
            return audio, leadsheet

        for i in range(k):
            extra_audio, extra_leadsheet = random.choice(
                list(self.example_cache[spkr_name].values()))
            if (audio.shape[1] + extra_audio.shape[1])/self.sample_rate > self.max_duration:
                continue

            mode = random.randint(0, 2)
            if mode == 0:
                continue
            elif mode == 1:
                extra_leadsheet = leadsheet_offset(
                    extra_leadsheet, offset=audio.shape[1]/self.sample_rate)
                audio = torch.cat([audio, extra_audio], dim=1)
                leadsheet = leadsheet + extra_leadsheet
            else:
                leadsheet = leadsheet_offset(
                    leadsheet, offset=extra_audio.shape[1]/self.sample_rate)
                audio = torch.cat([extra_audio, audio], dim=1)
                leadsheet = extra_leadsheet + leadsheet

        return audio, leadsheet

    def random_crop(self, audio, leadsheet):
        first = leadsheet[0]
        if first[0] != 'Rest':
            return audio, leadsheet

        duration = first[3]
        if duration < 0.050:
            return audio, leadsheet
        random_start = random.uniform(0.020, duration-0.020)
        start_idx = int(random_start * self.sample_rate)
        audio = audio[:, start_idx:]

        new_leadsheet = leadsheet_offset(leadsheet, offset=-1 * random_start)
        note, phone, start, end = new_leadsheet[0]
        new_leadsheet[0] = (note, phone, 0.0, end)
        return audio, new_leadsheet

    def __call__(self, item: Dict[str, Any]) -> Generator:
        assert item
        
        if '__index_data__' in item.keys():
            metadata = item['__index_data__']["metadata"]
            audio = item['audio.npy']
            index = metadata["audio_fp"]
        elif '__data_url__' in item.keys(): # parquet dataset
            metadata = json.loads(item["meta"])
            audio = item['wav']
            index = metadata["uttid"]
        else:
            logging.ERROR("[SVSFinetuneTransforms] unsupported item keys=", item.keys())
            raise NotImplementedError
        # metadata example:
        # {
        #     "uttid": "Todd-10001",
        #     "audio_fp": "/mnt/Todd-10001.wav",
        #     "musicxml": [],   # leadsheet in string, can be read with music21
        #     "interval": [],   # phoneme time boundary in second
        #     "note": [('Rest', 0.0, 0.5), ('Bb3', 0.5, 0.975), ('Db4', 0.975, 1.4), ('Bb3', 1.4, 1.8125)]
        #     "score": [('Rest', ['sil'], 0.0, 0.5), ('Bb3', [['C0d', 'C0eng']], 0.5, 0.975), ('Db4', [['C0g', 'C0uang']], 0.975, 1.4)]
        #     "phoneme": 'sil', 'C0d', 'C0eng', 'C0g', 'C0uang',
        #     "phoneme_timestamp": [('sil', 0.0, 0.5), ('C0d', 0.5, 0.57459), ('C0eng', 0.57459, 0.96891)]
        #     "lyric": "灯光也暗了，音乐低声了",
        #     "speaker_id": 'merci',
        #     "gender": 'female',
        #     "language": 'ZH',
        #     "emotion": 'enthusiastic',
        # }

        # dataset output Example:
        # [
        #     {'audio': tensor([[1.1484,0.6641,  ...,0.0000]]),
        #       'style_text': '',
        #       'leadsheet_tokens': tensor([549,400,550,552]),
        #       'normalized_text': '你眉头开了',
        #       'lyrics_tokens': tensor([66,37,3]),
        #       'max_phone_len': 400,
        #       'max_leadsheet_len': 800,
        #       'song_id': '/mnt/bn/merci-018001.wav',
        #       'metadata': {...}
        #     }
        # ]

        # musicxml and interval is ununsed currently, drop it for simplicity
        # del metadata["musicxml"]
        # del metadata["interval"]

        if metadata['language'] not in self.language:
            return

        song_id = metadata['uttid']
        # if use use_empty_spkr_id as condition
        spkr_name = metadata['speaker_id'].replace("_SVS", "")
        # if specific target_spkr_name for training
        if self.target_spkr_name != "" and spkr_name not in self.target_spkr_name:
            return

        if self.use_empty_spkr_id:
            speaker_id = 0
        else:
            speaker_id = speaker2id(spkr_name)
        # print(spkr_name, self.use_empty_spkr_id)

        style_text = rewrite_metadata_svs(metadata)
        audio = self.base_transform(audio)
        extra_audio = [self.base_transform(k) for k in self.extra_audio_keys]

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            extra_audio = [a.unsqueeze(0) for a in extra_audio]

        lyric = str(metadata['lyric'])
        normalized_text = normalize_text(lyric)


        leadsheet = metadata['score']
        import ast
        if isinstance(leadsheet, str):
            leadsheet = ast.literal_eval(leadsheet)
            leadsheet = list(map(list, leadsheet))
        
        if leadsheet[-1][0] == 'Rest':
            leadsheet[-1][3] = audio.shape[1] / self.sample_rate
        if self.is_eval == False:
            if len(extra_audio) > 0:
                raise NotImplementedError
            audio, leadsheet = self.random_crop(audio, leadsheet)
            audio, leadsheet = self.random_repeat(
                audio, leadsheet, spkr_name=spkr_name, song_id=song_id, k=10)

        # Pad audio and leadsheet to minimun duration
        leadsheet_duration = leadsheet[-1][3]
        if leadsheet[-1][3] < self.min_duration:
            leadsheet.append(("Rest", ["sil"], leadsheet_duration, self.min_duration))
        audio_len = int(self.min_duration * self.sample_rate)
        audio_transform = Compose([
            Pad(audio_len),
        ])
        audio = audio_transform(audio)
        extra_audio = [audio_transform(k) for k in self.extra_audio_keys]

        sliced_notes, sliced_phones = split_leadsheet2note_and_phone(leadsheet)
        leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens = concat_alignment(
            self.leadsheet_tokenizer, sliced_notes, sliced_phones)

        if self.lyrics_tokenizer == "sami_tts_frontend_precompute":
            lyrics_tokens = phoneme_tokens
        else:
            lyrics_tokens = self.lyrics_tokenizer(
                normalized_text,
                add_special_tokens=False,
                return_tensors="pt")["input_ids"].squeeze(dim=0)

        if self.use_empty_style_audio:
            style_audio = torch.zeros_like(audio)
        else:
            style_audio = audio

        # dump_leadsheet(index, audio, sliced_notes, sliced_phones)
        # seg_i=0
        # meta_song_id_str = os.path.basename(metadata["audio_fp"])
        # import torchaudio
        # torchaudio.save(f"/mnt/bn/bigmusic-lf/user/zhongyi/samantha/svssft/{seg_i}V1_{meta_song_id_str}.wav", audio.view(1, -1), 24000)
        # with open(f"/mnt/bn/bigmusic-lf/user/zhongyi/samantha/svssft/{seg_i}V1_{meta_song_id_str}.aligned_midi.txt", "w") as f:
        #     sliced_notes = '\n'.join(map(str, sliced_notes))
        #     f.write(sliced_notes)
        # with open(f"/mnt/bn/bigmusic-lf/user/zhongyi/samantha/svssft/{seg_i}V1_{meta_song_id_str}.aligned_phone.txt", "w") as f:
        #     sliced_phones = '\n'.join(map(str, sliced_phones))
        #     f.write(sliced_phones)

        yield {
            "audio": audio,
            "style_audio": style_audio,
            "style_text": style_text,
            "phoneme_tokens": phoneme_tokens,
            "note_tokens": note_tokens,
            "speaker_id": speaker_id,
            "leadsheet_tokens": leadsheet_tokens,
            "leadsheet_tokens_coff": leadsheet_tokens_coff,
            "normalized_text": normalized_text,
            "lyrics_tokens": lyrics_tokens,
            "max_phone_len": self.segment_max_phone_len,
            "max_leadsheet_len": self.segment_max_leadsheet_len,
            "extra_audio": extra_audio,

            # DEBUG
            "uttid": index,
            "index": index,
            "metadata": metadata,
        }


class SVSFinetuneDataset(WebPipeline):
    name = "SVSFinetune"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = [
            "hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
        weights: list = [1],
        data_id: int = 0,
        url_pattern: str = None,
        sample_rate=24000,
        audio_key: str = "audio.npy",
        min_duration: int = 20,
        max_duration: int = 30,
        normalize_audio: bool = True,
        is_eval: bool = False,
        # audio filtering
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        # metadata filtering
        lyrics_confidence: float = 0.8,
        aed_filtered: bool = False,
        genre_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        language: List[str] = [],
        # segmentation
        segment_method: str = "random",  # "first", "random"
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        include_intro: bool = False,
        # lyrics tokenizer
        lyrics_tokenizer=None,
        leadsheet_tokenizer=LeadSheetTokenizerV2(),
        handler=wds.warn_and_continue,
        extra_audio_keys=None,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...url2index={url2index}, data_id={data_id}")
        self.handler = handler

        assert bool(url2index) != bool(data_id)
        if url2index:
            if isinstance(url2index, list):
                dataset = MultiIterableDataset(
                    datasets=[IndexedWebDataset(url2index=url, **kwargs)
                              for url in url2index],
                    weights=weights)
            else:
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        if data_id:
            dataset = ParquetDataset(data_id=data_id, data_urls=None, **kwargs)
        if extra_audio_keys is None:
            extra_audio_keys = []
        transforms = SVSFinetuneTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            is_eval=is_eval,
            # audio filtering
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            # metadata filtering
            lyrics_confidence=lyrics_confidence,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            target_spkr_name=target_spkr_name,
            use_empty_spkr_id=use_empty_spkr_id,
            use_empty_style_audio=use_empty_style_audio,
            language=language,
            # segmentation
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            segment_max_leadsheet_len=segment_max_leadsheet_len,
            # lyrics tokenizer
            lyrics_tokenizer=lyrics_tokenizer,
            leadsheet_tokenizer=leadsheet_tokenizer,
            extra_audio_keys=extra_audio_keys,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if url2index:
            pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        if data_id:
            pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_buffer_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )
                for val in self.validation_dataset
            ]
        else:
            return DataLoader(
                self.validation_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self):
        if isinstance(self.predict_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )
                for val in self.predict_dataset
            ]
        else:
            return DataLoader(
                self.predict_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )


class SVSPretrainWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_pipe: bool = True,
        lyrics_tokenizer: str = "en_phoneme_espeak",
        leadsheet_tokenizer: Any = LeadSheetTokenizerV2(),
        normalize_audio: bool = False,
        wds_dataset_urls: List[str] = [],
        wds_validation_dataset_urls: List[str] = [
            "hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
        wds_dataset_weights: List[int] = [],
        parquet_dataset_ids: List[int] = [708],
        parquet_dataset_weights: List[int] = [1],
        extra_audio_keys=None,
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        audio_key: str = "vocal",
        conditions: str = "leadsheet_tokens",
        time_format: str = "start,duration",
        style_prompt_path: str = '',
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        vocal_prompt_duration: float = 10.0,
        lyrics_confidence: float = 0.7,
        boundaries: List[float] = [1.0, 1.0],
        tmp_infer_example: str = '',
        language: List[str] = [],
        run_opts: dict = {},
        semantic_frame_rate: int = 25,
        buckets_in_sec: List[int] = [
            10,
            20,
            25,
            30,
        ],
    ):
        if extra_audio_keys is None:
            extra_audio_keys = []
        collate_fn = override_parameter(
            collate_fn, 
            conditions=conditions,
            sample_rate=sample_rate,
            semantic_frame_rate=semantic_frame_rate,
            )
        min_duration = max(10.0, buckets_in_sec[0])

        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if lyrics_tokenizer == "zh_wordpiece":
            self.lyrics_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif lyrics_tokenizer == "sami_tts_frontend_precompute":
            self.lyrics_tokenizer = "sami_tts_frontend_precompute"
        elif lyrics_tokenizer == "en_phoneme_espeak":
            with local_zero_first():
                self.lyrics_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
            # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.lyrics_tokenizer = None
        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.leadsheet_tokenizer.time_format = time_format

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        self.wds_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_urls:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_datasets = [SVSPretrainDataset(
                url2index=wds_dataset_urls,
                weights=wds_dataset_weights,
                data_id=0,
                audio_key=audio_key,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                segment_max_leadsheet_len=segment_max_leadsheet_len,
                include_intro=include_intro,
                lyrics_tokenizer=self.lyrics_tokenizer,
                lyrics_confidence=lyrics_confidence,
                leadsheet_tokenizer=self.leadsheet_tokenizer,
                boundaries=boundaries,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
                extra_audio_keys=extra_audio_keys
                )]
        self.parquet_datasets = []
        if parquet_dataset_ids:
            for parquet_id in parquet_dataset_ids:
                self.parquet_datasets.append(
                    SVSPretrainDataset(
                        data_id=parquet_id,
                        url2index='',
                        weights=[],
                        audio_key=audio_key,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        segment_max_leadsheet_len=segment_max_leadsheet_len,
                        include_intro=include_intro,
                        lyrics_tokenizer=self.lyrics_tokenizer,
                        lyrics_confidence=lyrics_confidence,
                        leadsheet_tokenizer=self.leadsheet_tokenizer,
                        boundaries=boundaries,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                        extra_audio_keys=extra_audio_keys,
                        ))
        train_dataset = WebPipeline(
            MultiIterableDataset(datasets=self.wds_datasets + self.parquet_datasets,
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )
        validation_dataset = [WebPipeline(
            SVSFinetuneDataset(
                data_id=None,
                url2index=wds_validation_dataset_urls,
                min_duration=min_duration,
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                segment_max_leadsheet_len=segment_max_leadsheet_len,
                include_intro=include_intro,
                lyrics_tokenizer=self.lyrics_tokenizer,
                leadsheet_tokenizer=self.leadsheet_tokenizer,
                use_pipe=use_pipe,
                resampled=False,
                is_eval=True,
                language=language,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                extra_audio_keys=extra_audio_keys
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )]
        predict_dataset = WebPipeline(
            SVSEvalDataset(
                data_id=None,
                url2index=wds_validation_dataset_urls,
                min_duration=min_duration,
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                segment_max_leadsheet_len=segment_max_leadsheet_len,
                include_intro=include_intro,
                lyrics_tokenizer=self.lyrics_tokenizer,
                leadsheet_tokenizer=self.leadsheet_tokenizer,
                use_pipe=use_pipe,
                run_opts=run_opts,
                target_spkr_name=target_spkr_name,
                use_empty_spkr_id=use_empty_spkr_id,
                use_empty_style_audio=use_empty_style_audio,
                style_prompt_path=style_prompt_path,
                vocal_prompt_duration=vocal_prompt_duration,
                tmp_infer_example=tmp_infer_example,
                resampled=False,
                language=language,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                extra_audio_keys=extra_audio_keys
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

    def predict_dataloader(self):
        return map(self.collate_fn, self.predict_dataset)


class SVSFinetuneWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_pipe: bool = True,
        lyrics_tokenizer: str = "en_phoneme_espeak",
        leadsheet_tokenizer: Any = LeadSheetTokenizerV2(),
        normalize_audio: bool = False,
        wds_dataset_urls: List[str] = [
            "hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/train_v2/url2index.txt"],
        wds_validation_dataset_urls: List[str] = [
            "hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
        wds_dataset_weights: List[int] = [1],
        parquet_dataset_ids: List[int] = [],
        parquet_validation_dataset_ids: List[int] = [],
        parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        segment_max_leadsheet_len: int = 800,
        vocal_prompt_duration: float = 10.0,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        target_spkr_name: str = "",
        use_empty_spkr_id: bool = False,
        use_empty_style_audio: bool = False,
        time_format: str = "start,duration",
        conditions: str = "lyrics_tokens,leadsheet_tokens",
        style_prompt_path: str = '',
        language: List[str] = [],
        run_opts: dict = {},
        semantic_frame_rate: int = 25,
        buckets_in_sec: List[int] = [
            10,
            20,
            25,
            30,
        ],
    ):
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        min_duration = max(10.0, buckets_in_sec[0])

        collate_fn = override_parameter(
            collate_fn, 
            conditions=conditions,
            sample_rate=sample_rate,
            semantic_frame_rate=semantic_frame_rate,
            )
        self.collate_fn = collate_fn

        if lyrics_tokenizer == "zh_wordpiece":
            self.lyrics_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif lyrics_tokenizer == "sami_tts_frontend_precompute":
            self.lyrics_tokenizer = "sami_tts_frontend_precompute"
        elif lyrics_tokenizer == "en_phoneme_espeak":
            with local_zero_first():
                self.lyrics_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
            # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.lyrics_tokenizer = None

        self.leadsheet_tokenizer = leadsheet_tokenizer
        self.leadsheet_tokenizer.time_format = time_format

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )

        self.wds_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_urls:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_datasets = [SVSFinetuneDataset(
                url2index=wds_dataset_urls,
                weights=wds_dataset_weights,
                data_id=None,
                min_duration=min_duration,
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                segment_max_leadsheet_len=segment_max_leadsheet_len,
                include_intro=include_intro,
                lyrics_tokenizer=self.lyrics_tokenizer,
                leadsheet_tokenizer=self.leadsheet_tokenizer,
                target_spkr_name=target_spkr_name,
                use_empty_spkr_id=use_empty_spkr_id,
                use_empty_style_audio=use_empty_style_audio,
                language=language,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue
            )]
        self.parquet_datasets = []
        if parquet_dataset_ids:
            for parquet_id in parquet_dataset_ids:
                self.parquet_datasets.append(
                    SVSFinetuneDataset(
                        url2index=None,
                        data_id=parquet_id,
                        min_duration=min_duration,
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        segment_max_leadsheet_len=segment_max_leadsheet_len,
                        include_intro=include_intro,
                        lyrics_tokenizer=self.lyrics_tokenizer,
                        leadsheet_tokenizer=self.leadsheet_tokenizer,
                        target_spkr_name=target_spkr_name,
                        use_empty_spkr_id=use_empty_spkr_id,
                        use_empty_style_audio=use_empty_style_audio,
                        language=language,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue
                    ))
        train_dataset = WebPipeline(
            MultiIterableDataset(datasets=self.wds_datasets + self.parquet_datasets,
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )


        validation_dataset = []
        
        if wds_dataset_urls != []:
            validation_dataset.append(
                SVSFinetuneDataset(
                    data_id=None,
                    url2index=wds_validation_dataset_urls,
                    min_duration=min_duration,
                    max_duration=buckets_in_sec[-1],
                    normalize_audio=normalize_audio,
                    segment_method=segment_method,
                    max_seg_per_track=1,
                    segment_max_phone_len=segment_max_phone_len,
                    segment_max_leadsheet_len=segment_max_leadsheet_len,
                    include_intro=include_intro,
                    lyrics_tokenizer=self.lyrics_tokenizer,
                    leadsheet_tokenizer=self.leadsheet_tokenizer,
                    target_spkr_name=target_spkr_name,
                    use_empty_spkr_id=use_empty_spkr_id,
                    use_empty_style_audio=use_empty_style_audio,
                    language=language,
                    is_eval=True,
                    use_pipe=use_pipe,
                    resampled=False,
                    nodesplitter=return_self,
                    handler=wds.warn_and_continue
                ),
            )
        if parquet_validation_dataset_ids != []:
            for parquet_id in parquet_validation_dataset_ids:
                validation_dataset.append(
                SVSFinetuneDataset(
                    data_id=parquet_id,
                    url2index=None,
                    min_duration=min_duration,
                    max_duration=buckets_in_sec[-1],
                    normalize_audio=normalize_audio,
                    segment_method=segment_method,
                    max_seg_per_track=1,
                    segment_max_phone_len=segment_max_phone_len,
                    segment_max_leadsheet_len=segment_max_leadsheet_len,
                    include_intro=include_intro,
                    lyrics_tokenizer=self.lyrics_tokenizer,
                    leadsheet_tokenizer=self.leadsheet_tokenizer,
                    target_spkr_name=target_spkr_name,
                    use_empty_spkr_id=use_empty_spkr_id,
                    use_empty_style_audio=use_empty_style_audio,
                    language=language,
                    is_eval=True,
                    use_pipe=use_pipe,
                    resampled=False,
                    nodesplitter=return_self,
                    handler=wds.warn_and_continue
                ),
                )

        validation_dataset = WebPipeline(
            MultiIterableDataset(datasets=validation_dataset),
            pipeline=[{"compose": [self.bucketize]}],
        )

        predict_dataset = validation_dataset

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

    def validation_dataloader(self):
        return map(self.collate_fn, self.validation_dataset)

    def predict_dataloader(self):
        return map(self.collate_fn, self.predict_dataset)


if __name__ == "__main__":
    # dataset = SVSPretrainWebDataModule(
    #     wds_dataset_urls=[],
    #     lyrics_tokenizer="zh_wordpiece",
    #     wds_validation_dataset_urls=["hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
    #     parquet_dataset_ids=[1127],
    #     boundaries=[0,1.0],
    #     segment_max_phone_len= 0,
    #     segment_max_leadsheet_len= 1500,
    #     conditions= "style_audio,spkr_ids,leadsheet_tokens",
    #     time_format= "start,duration,merge_sil,merge_rest",
    #     language= ['ZH'],
    #     num_workers=1,
    #     batch_size=1,
    #     buckets_in_sec= [1, 3, 5, 10, 15, 20, 25, 30, 32, 35, 40, 45],
    #     ).train_dataloader().__iter__()
    # batch = next(dataset)
    # print("train", batch)
    dataset = SVSFinetuneWebDataModule(
        wds_dataset_urls=[],
        lyrics_tokenizer="zh_wordpiece",
        wds_validation_dataset_urls=["hdfs://haruna/home/byte_speech_sv/zhongyi.huang/svs_dataset/24000hz/test_v2/url2index.txt"],
        parquet_dataset_ids=[1127],
        boundaries=[0,1.0],
        segment_max_phone_len= 0,
        segment_max_leadsheet_len= 1500,
        conditions= "style_audio,spkr_ids,leadsheet_tokens",
        time_format= "start,duration,merge_sil,merge_rest",
        language= ['ZH'],
        num_workers=2,
        batch_size=2,
        style_prompt_path='assets/svs/svs_prompt',
        buckets_in_sec= [1, 3, 5, 10, 15, 20, 25, 30, 32, 35, 40, 45],
        )
    # batch = next(dataset.train_dataloader().__iter__())
    # print("Pretrain train", batch)
    # batch = next(dataset.predict_dataloader().__iter__())
    # print("Pretrain predict", batch)
    dataset = SVSFinetuneWebDataModule(
        wds_dataset_urls=[],
        wds_validation_dataset_urls=[],
        parquet_dataset_ids=[1272],
        parquet_validation_dataset_ids=[1273],
        batch_size=2, time_format='start,duration,merge_sil,merge_rest')
    batch = next(dataset.train_dataloader().__iter__())
    print("SFT train", batch)
    batch = next(dataset.validation_dataloader().__iter__())
    print("SFT val", batch)
    batch = next(dataset.predict_dataloader().__iter__())
    print("SFT predict", batch)
    # print("predict", batch)
    # import torchaudio
    # torchaudio.save("test.wav", batch['target_audio'].view(1, -1), 24000)
    # webdataset = SVSFinetuneWebDataModule(batch_size=1)
    # print(next(webdataset.predict_dataloader().__iter__()))

    # dataset = SVSPretrainWebDataModule(batch_size=16, target_spkr_name="Zach", style_prompt_path="svs_prompt/Zach-040001.wav_43", vocal_prompt_duration=6).predict_dataloader().__iter__()
    # working_example = 0
    # for i in dataset:
    #     working_example += 16
    #     print(i.keys())
    # batch = next(dataset)
    # print(batch)
    # import torchaudio
    # torchaudio.save("test.wav", batch['target_audio'].view(1, -1), 24000)

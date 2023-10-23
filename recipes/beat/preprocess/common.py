from abc import ABC
import os
import json
import numpy as np
import torch
import subprocess
import io
import soundfile as sf
from torchaudio.transforms import Resample

from samantha.dataio.preprocess import AudioLengthModifier
from torchaudio.transforms import Resample


if not os.path.exists('recipes/beat/conf/final_dict.json'):
    subprocess.run(f"hdfs dfs -get hdfs://harunava/home/byte_speech_sv/amy/log/final_dict.json recipes/beat/conf/final_dict.json", shell=True)
with open('recipes/beat/conf/final_dict.json', "r") as outfile:
    VALID_VOCAL = json.load(outfile)

def get_beats_labels(times, song_duration, dataset, hop_in_sec=0.2):
    hop_per_sec = 1 / hop_in_sec
    end_idx = np.ceil(song_duration * hop_per_sec).astype(np.int)
    beat_labels = np.zeros((end_idx, 2))

    if dataset in ["smc_beat", "simac_beat", "hjdb_beat"]:
        only_beat = True
    else:
        only_beat = False

    for time in times:
        t = np.round(np.array(time[0]) * hop_per_sec).astype(np.int)

        if t < end_idx:
            if not only_beat:
                if time[1] == 1:
                    beat_labels[t, 1] = 1
                else:
                    beat_labels[t, 0] = 1
            else:
                beat_labels[t, 0] = 1

    tempo = int(round((beat_labels.sum(0)[0] - 1) / (times[-1][0] - times[0][0]) * 60))
    return beat_labels, tempo


class MCCDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs):
        super().__init__()  # forwards all unused arguments
        default_sr = 44100
        self._audio_length_modifier = AudioLengthModifier(
            default_sr, target_duration_sec, is_random_crop=True
        )
        self._chunk_per_sample = chunk_per_sample
        self.resampler = Resample(44100, sampling_rate)

    def train_preprocess(self, x):
        np_audio, sr = sf.read(io.BytesIO(x["mp3"]))
        data_queue = []
        for i in range(self._chunk_per_sample):
            out = {}
            audio, _, _ = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            audio = self.resampler(audio.float())
            out["audio.npy"] = audio.squeeze(0)
            data_queue.append(out)

        return data_queue


class MCCMSSDatasetMixin(MCCDatasetMixin):
    def __init__(self, target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs):
        super().__init__(target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs)

    def train_preprocess(self, x):
        try:
            np_acc_audio, sr = sf.read(io.BytesIO(x["mss_acc"]))
            np_vocal_audio, sr = sf.read(io.BytesIO(x["mss_vocal"]))
            np_audio = np.concatenate((np_vocal_audio[None,], np_acc_audio[None]), 0)
            data_queue = []
            for i in range(self._chunk_per_sample):
                out = {}
                audio, _, _ = self._audio_length_modifier(
                    torch.from_numpy(np_audio)
                )
                audio = self.resampler(audio.float())

                out["audio.npy"] = audio
                data_queue.append(out)

            return data_queue
        except:
            return None


class HotGalaxyDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs):
        super().__init__()  # forwards all unused arguments
        default_sr = 16000
        self._audio_length_modifier = AudioLengthModifier(
            default_sr, target_duration_sec, is_random_crop=True
        )
        self._chunk_per_sample = chunk_per_sample
        #self.resampler = Resample(44100, sampling_rate)

    def train_preprocess(self, x):
        np_audio, sr = sf.read(io.BytesIO(x["mp3"]))
        #np_audio = np_audio.mean(-1)
        data_queue = []
        for i in range(self._chunk_per_sample):
            out = {}
            audio, _, _ = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            #audio = self.resampler(audio.float())
            out["audio.npy"] = audio.squeeze(0).float()
            data_queue.append(out)

        return data_queue


class HotGalaxyVocalDatasetMixin(MCCDatasetMixin):
    def __init__(self, target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs):
        super().__init__(target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs)
        self._audio_length_modifier = AudioLengthModifier(
            16000, target_duration_sec, is_random_crop=True
        )

    def train_preprocess(self, x):
        np_acc_audio, sr = sf.read(io.BytesIO(x["mss_acc"]))
        np_vocal_audio, sr = sf.read(io.BytesIO(x["mss_vocal"]))
        np_audio = np.concatenate((np_vocal_audio[None,], np_acc_audio[None]), 0)
        #np_audio = np_audio.mean(-1)
        data_queue = []
        for i in range(self._chunk_per_sample):
            out = {}
            audio, _, _ = self._audio_length_modifier(
                torch.from_numpy(np_audio)
            )
            #audio = self.resampler(audio.float())

            out["audio.npy"] = audio.float()
            data_queue.append(out)

        return data_queue


class NewHotGalaxyVocalDatasetMixin(MCCDatasetMixin):
    def __init__(self, target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs):
        super().__init__(target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs)
        self._audio_length_modifier = AudioLengthModifier(
            16000, target_duration_sec, is_random_crop=True
        )

    def train_preprocess(self, x):
        np_vocal_audio, sr = sf.read(io.BytesIO(x["mss_vocal"]))
        np_mix_audio, sr = sf.read(io.BytesIO(x["mss_mix"]))
        length = min(len(np_mix_audio), len(np_vocal_audio))
        np_acc_audio = np_mix_audio[:length] - np_vocal_audio[:length]
        np_audio = np.concatenate((np_vocal_audio[None, :length], np_acc_audio[None, :length]), 0)
        #np_audio = np_audio.mean(-1)
        data_queue = []
        for i in range(self._chunk_per_sample):
            out = {}
            audio, _, _ = self._audio_length_modifier(
                torch.from_numpy(np_audio)
            )
            #audio = self.resampler(audio.float())

            out["audio.npy"] = audio.float()
            data_queue.append(out)

        return data_queue


class BeatDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, label_hop, chunk_per_sample, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._label_hop = label_hop
        self._audio_length_modifier = AudioLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_hop = label_hop
        self._chunk_per_sample = chunk_per_sample

    def train_preprocess(self, x):
        beats = x["beats.pickle"]
        np_audio = x["audio.npy"]
        audio_duration_in_s = (
            np_audio.shape[0] / self._audio_length_modifier.sampling_rate
        )
        # convert beat to format for training
        beat_labels, tempo = get_beats_labels(
            beats, audio_duration_in_s, x["dataset.txt"], self._label_hop
        )
        label_len = int(
            self._audio_length_modifier.target_duration_sec / self._label_hop
        )
        data_queue = []
        for i in range(self._chunk_per_sample):
            out = x.copy()
            out["tempo_label"] = tempo
            np_audio, beats,  = self._sample_audio_and_beat(np_audio, beat_labels, label_len)
            out["audio.npy"] = torch.tensor(np_audio, dtype=torch.float32)
            out["beats.pickle"] = torch.tensor(beats)
            data_queue.append(out)
        return data_queue

    def val_preprocess(self, x):
        orig_beats = x["beats.pickle"]
        beat_label, tempo_label = get_beats_labels(
            x["beats.pickle"],
            len(x["audio.npy"]) / self._audio_length_modifier.sampling_rate,
            x["dataset.txt"],
            self._label_hop,
        )
        beat_label = beat_label[
            : int(
                len(x["audio.npy"])
                / self._audio_length_modifier.sampling_rate
                * (1 / self._label_hop)
            )
        ]

        pad_len = (
            self._audio_length_modifier.target_duration_sec * int(1 / self._label_hop)
            - beat_label.shape[0]
        )

        if pad_len > 0:
            beat_label = np.pad(beat_label, [(0, int(pad_len)), (0, 0)], "constant")
        x["orig_beats"] = orig_beats
        x["beats.pickle"] = torch.tensor(beat_label)
        x["tempo_label"] = torch.tensor(tempo_label)
        x["audio.npy"] = torch.tensor(x["audio.npy"])
        return x

    def _sample_audio_and_beat(self, np_audio, beat_labels, label_len):
        # sample from audio
        audio, _, start_idx = self._audio_length_modifier(
            torch.from_numpy(np_audio.reshape(1, -1))
        )

        # get beat labels in sampled audio
        sample_idx = int(
            round(
                (start_idx / self._audio_length_modifier.sampling_rate)
                * (1 / self._label_hop)
            )
        )
        beat_labels = beat_labels[sample_idx : sample_idx + label_len, :]
        pad_len = label_len - beat_labels.shape[0]
        if pad_len > 0:
            beat_labels = np.pad(beat_labels, [(0, pad_len), (0, 0)], "constant")

        sampled_np_audio = np.squeeze(audio.numpy())

        return sampled_np_audio, beat_labels


class BeatPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, label_hop, chunk_per_sample=1, *args, **kwargs):
        self._dataset_preprocessors = BeatDatasetMixin(
            target_duration_sec, sampling_rate, label_hop, chunk_per_sample, *args, **kwargs
        )
        self._mcc_preprocessors = MCCDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._mcc_mss_preprocessors = MCCMSSDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._hotgalaxy_preprocessors = HotGalaxyDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._hotgalaxy_vocal_preprocessors = HotGalaxyVocalDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._new_hotgalaxy_vocal_preprocessors = NewHotGalaxyVocalDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )

    def train_batch_preprocess(self, batch):
        try:
            for x in batch:
                if "license-based_mcc_mss_shard" in x["__url__"]:
                    if not ('VALID_VOCAL' in globals() and x['__key__'] in VALID_VOCAL):
                        continue
                    preprocessor = self._mcc_mss_preprocessors
                elif 'license-based_mcc' in x['__url__']:
                    preprocessor = self._mcc_preprocessors
                elif x["dataset.txt"] == 'galaxyark_mss':
                    preprocessor = self._new_hotgalaxy_vocal_preprocessors
                elif 'hot_galaxy/galaxyark_vocal' in x['__url__']:
                    preprocessor = self._hotgalaxy_vocal_preprocessors
                elif 'hot_galaxy' in x['__url__']:
                    preprocessor = self._hotgalaxy_preprocessors
                else:
                    preprocessor = self._dataset_preprocessors
                
                data_queue = preprocessor.train_preprocess(x)
                if data_queue is not None:
                    while len(data_queue) > 0:
                        data = data_queue.pop(0)
                        data['key'] = x["__key__"]
                        yield data
        except Exception as e:
            print(e)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)

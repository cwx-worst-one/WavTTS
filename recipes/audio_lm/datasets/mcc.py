import io
import pickle
import random

import librosa
import numpy as np
import torch
import webdataset as wds
from pydub import AudioSegment
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.audio_lm.datasets.utils as utils

MCC_GPT_GEN_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_gpt_generated/mcc_gpt_generated_{i:04d}.tar"  # noqa
    for i in range(3)  # 201
] + [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_gpt_generated/mcc_gpt_generated_{i:02d}.tar"  # noqa
    for i in range(2)  # 12
]
AED_GPT_GEN_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/aed700k_gpt_generated/aed700k_gpt_generated_{i:04d}.tar"  # noqa
    for i in range(274)
]
MCC_N2M_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/mcc_n2m/mcc_n2m_{i:04d}.tar"
    for i in range(6379)  # 6379
]


class MCCDataset(IterableDataset):
    def __init__(
        self, duration=10, sample_rate=24000, verbose=False, shardshuffle=True, **kwargs
    ):
        with open(
            "/mnt/bn/audio-diffusion/data/genre_balanced_mcc/genre_balanced_mcc_hdfs_paths_1704.csv",  # noqa
            "r",
        ) as fp:
            self.urls = [f"pipe:hdfs dfs -cat {line}" for line in fp.readlines()]
        self.dataset = (
            wds.WebDataset(self.urls, shardshuffle=shardshuffle, **kwargs)
            .decode()
            .map(self._process_audio)
        )
        with open("/mnt/bn/audio-diffusion/data/MCC_40M_metalist.txt", "r") as fp:
            self.metalist = [self.metalist2id(line.strip()) for line in fp.readlines()]

        self.duration = duration
        self.sample_rate = sample_rate
        self.segment_size = self.duration * self.sample_rate
        self.verbose = verbose

    def wds2id(self, url, key):
        """
        url: 'pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/genre_balanced_mcc_clean/classical.2.57/0.tar'  # noqa
        key: '6980478627137718274'
        return: 'classical.2.57/0/6980478627137718274'
        """
        sub_folder = url.split("/")[-2]
        tar_name = url.split("/")[-1].split(".")[0]
        return f"{sub_folder}/{tar_name}/{key}"

    def metalist2id(self, path):
        """
        path: '../genre_balanced_mcc_clean_tags/8-bit.0.54/0/0-7196752726452242434.json'
        return: '8-bit.0.54/0/7196752726452242434'
        """
        sub_folder = path.split("/")[-3]
        tar_name = path.split("/")[-2]
        id = path.split("/")[-1].split(".")[0].split("-")[-1]
        return f"{sub_folder}/{tar_name}/{id}"

    def get_active_frames(self, audio, threshold=0.05):
        window_size = int(self.sample_rate * 0.1)

        frames = librosa.util.frame(
            x=audio, frame_length=window_size, hop_length=window_size
        ).T
        energy = np.max(np.abs(frames), axis=-1)  # shape: (frames_num,)
        rate = np.sum(energy > threshold) / energy.shape[0]

        if rate < 1 / 10:
            return False
        return True

    # def mp3_read_f32(self, data: bytes) -> array:
    #     '''Reads and decodes the whole mp3 audio data. Resulting sample format is 32 bits float.'''  # noqa
    #     config = miniaudio.ffi.new('drmp3_config *')
    #     num_frames = miniaudio.ffi.new('drmp3_uint64 *')
    #     memory = miniaudio.lib.drmp3_open_memory_and_read_pcm_frames_f32(data, len(data), config, num_frames, miniaudio.ffi.NULL)  # noqa
    #     if not memory:
    #         raise miniaudio.DecodeError('cannot load/decode data')
    #     try:
    #         samples = array.array('f')
    #         buffer = miniaudio.ffi.buffer(memory, num_frames[0] * config.channels * 4)
    #         samples.frombytes(buffer)
    #         return samples, config.sampleRate, config.channels
    #     finally:
    #         miniaudio.lib.drmp3_free(memory, miniaudio.ffi.NULL)
    #         miniaudio.ffi.release(num_frames)

    def _process_audio(self, data):
        output = {"__skip__": False}
        if self.wds2id(data["__url__"], data["__key__"]) not in self.metalist:
            output["__skip__"] = f'-Filter- {data["__url__"]} {data["__key__"]}'
            return output

        clip_duration = float(data["metadata.json"]["clip_duration"])
        if not (
            (clip_duration >= (self.duration - 0.05) * 1000)
            and (clip_duration <= 360 * 1000)
        ):
            output[
                "__skip__"
            ] = f'-Duration- {data["__url__"]} {data["__key__"]} {clip_duration / 1000} seconds'  # noqa
            return output
        try:
            audio = data["mp3"]
            audio = AudioSegment.from_file(io.BytesIO(data["mp3"]), format="mp3")
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.sample_rate)
            wav = np.asarray(audio.get_array_of_samples())
        except Exception as e:
            output["__skip__"] = f'-Load- {data["__url__"]} {data["__key__"]} {e}'
            return output
        wav_len = wav.shape[0]

        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        wav = wav / np.max(np.abs(wav)) * 0.95

        if wav_len < self.segment_size:
            wav = np.pad(wav, ((0, 0), (0, self.segment_size - wav_len)))
        else:
            beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
            wav = wav[beg : beg + self.segment_size]
        if np.sqrt(np.mean(wav**2)) <= 1e-4 and self.get_active_frames(
            wav, threshold=0.05
        ):
            output[
                "__skip__"
            ] = f'-Silence- {data["__url__"]} {data["__key__"]} started at {beg}'
            return output
        output["wav"] = torch.from_numpy(wav)

        return output

    def __iter__(self):
        for item in self.dataset:
            if not item["__skip__"]:
                yield item["wav"]
            else:
                if self.verbose:
                    print(f"[SKIP] {item['__skip__']}")


class MCCGPTGenDataset(IterableDataset):
    def __init__(self, duration=10, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_GPT_GEN_URLS, **kwargs)
            .decode()
            .map(utils.process_audio(duration))
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        data["text"] = meta["gpt_generated_caption"]
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)


class AEDGPTGenDataset(IterableDataset):
    def __init__(self, duration=10, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(AED_GPT_GEN_URLS, **kwargs)
            .decode()
            .map(utils.process_audio(duration))
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_aed.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]
        # data["text"] = meta["gpt_generated_caption"]
        text = ""
        if str(meta["music_id"]) in self.chatgpt_g4:
            tags = self.chatgpt_g4[str(meta["music_id"])]
            # get all the items into list
            tags = [v for v in tags.values()]
            # randomize the order
            random.shuffle(tags)
            for tag in tags:
                if random.random() < 0.8:
                    text += tag
        data["text"] = text

        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)


class MCCN2MDataset(IterableDataset):
    def __init__(self, duration=10, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(MCC_N2M_URLS, **kwargs)
            .decode()
            .map(utils.process_audio(duration))
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )
        with open("assets/chatgpt_g4_mcc.pkl", "rb") as f:
            self.chatgpt_g4 = pickle.load(f)

    def _process_text(self, data):
        meta = data["meta.json"]
        # n2m_texts = meta["n2m"]
        # n2m is a list of 3 matched texts
        # data["text"] = random.choice(n2m_texts)
        text = ""
        if str(meta["music_id"]) in self.chatgpt_g4:
            tags = self.chatgpt_g4[str(meta["music_id"])]
            # get all the items into list
            tags = [v for v in tags.values()]
            # randomize the order
            random.shuffle(tags)
            for tag in tags:
                if random.random() < 0.8:
                    text += tag
        data["text"] = text
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]

        return data

    def __iter__(self):
        return iter(self.dataset)

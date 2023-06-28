import os
import librosa
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


def collate_fn(batches):
    max_16k_len = max([batch[0].shape[-1] for batch in batches])
    max_24k_len = max([batch[1].shape[-1] for batch in batches])
    max_text_len = max([batch[2].shape[-1] for batch in batches])

    b = len(batches)
    wavs_16k = np.zeros(shape=[b, max_16k_len])
    wavs_24k = np.zeros(shape=[b, max_24k_len])
    texts = np.zeros(shape=[b, max_text_len])

    wav_16k_lens = np.zeros(shape=[b,], dtype=np.int64)
    wav_24k_lens = np.zeros(shape=[b,], dtype=np.int64)
    text_lens = np.zeros(shape=[b,], dtype=np.int64)

    for i, batch in enumerate(batches):
        wav_16k, wav_24k, text = batch

        wavs_16k[i][0:wav_16k.shape[-1]] = wav_16k
        wavs_24k[i][0:wav_24k.shape[-1]] = wav_24k
        texts[i][0:text.shape[-1]] = text

        wav_16k_lens[i] = wav_16k.shape[-1]
        wav_24k_lens[i] = wav_24k.shape[-1]
        text_lens[i] = text.shape[-1]

    wavs_16k = (wavs_16k * 32768.0).astype(np.int32) # save as int32 to prevent deepspeed fp16 transform
    wavs_24k = (wavs_24k * 32768.0).astype(np.int32) # save as int32 to prevent deepspeed fp16 transform
    wavs_16k = torch.from_numpy(wavs_16k)
    wavs_24k = torch.from_numpy(wavs_24k)
    texts = torch.from_numpy(texts).long()

    wav_16k_lens = torch.from_numpy(wav_16k_lens).long()
    wav_24k_lens = torch.from_numpy(wav_24k_lens).long()
    text_lens = torch.from_numpy(text_lens).long()

    return wavs_16k, wavs_24k, texts, wav_16k_lens, wav_24k_lens, text_lens


class BarkDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        tokenizer_cls,
        min_dur=3,
        max_dur=30,
    ):
        self.metalist = []
        self.tokenizer = tokenizer_cls()
        self.max_dur = max_dur

        for meta in tqdm(meta_paths):
            # audio-path|text-path|tacolabel-path|duration|sample rate
            with open(meta, 'r') as f:
                lines = [l.strip() for l in f]
            for l in lines:
                _, _, _, dur, sr = l.split('|')
                sen_dur = float(dur) / float(sr)
                if sen_dur < max_dur and sen_dur > min_dur:
                    self.metalist.append(l)

    def __len__(self):
        return len(self.metalist)

    def __getitem__(self, idx):
        info = self.metalist[idx]
        audio_path, text_path, tacolabel_path, duration, sr = info.split('|')
        wav_24k, _ = librosa.load(audio_path, sr=24000)
        wav_24k = wav_24k / max(0.001, np.max(np.abs(wav_24k))) * 0.95
        wav_16k = librosa.resample(y=wav_24k, orig_sr=24000, target_sr=16000)
        with open(text_path, 'r') as f:
            text = f.read()
        text = np.asarray(self.tokenizer(text)["input_ids"])
        return wav_16k, wav_24k, text


class SoundStormDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        tokenizer_cls,
        min_dur=3,
        max_dur=30,
    ):
        self.metalist = []
        self.tokenizer = tokenizer_cls()
        self.max_dur = max_dur

        for meta in tqdm(meta_paths):
            with open(meta, 'r') as f:
                lines = [l.strip() for l in f]
            self.metalist += lines

    def __len__(self):
        return len(self.metalist)

    def __getitem__(self, idx):
        info = self.metalist[idx]
        audio_path = info.split('|')[0]
        wav_24k, _ = librosa.load(audio_path, sr=24000)
        #wav_24k = wav_24k / max(0.001, np.max(np.abs(wav_24k))) * 0.95

        wav_16k = librosa.resample(y=wav_24k, orig_sr=24000, target_sr=16000)

        text = np.random.randint(low=0, high=1024, size=[10,])

        return wav_16k, wav_24k, text

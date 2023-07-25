import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import random_formant_shift


class AudioDataset(Dataset):
    def __init__(
        self,
        wav_list,
        limit=None,
        sample_rate=24000,
        return_path=False,
        use_formant_shift=False,
    ):
        super().__init__()
        self.wav_list = wav_list
        self.limit = limit
        self.sample_rate = sample_rate
        self.return_path = return_path
        self.use_formant_shift = use_formant_shift
        self.hubert_tokens_dir = '/mnt/bd/huangzhiying-lq-valle-volume8/data/bytegen/s2s_hubert'

        # load dataset
        file_paths = []
        # load audio path list
        with open(wav_list, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                splits = line.split('|')
                file_path = splits[0]
                file_paths.append(file_path)
        if limit is not None:
            file_paths = file_paths[:limit]
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, batch_idx: list):
        x_lens, t_lens, bx, bp, bt = [], [], [], [], []
        for idx in batch_idx:
            data = self.get_idx(idx)
            if self.return_path:
                wav, tokens, path = data
                bp.append(path)
            else:
                wav, tokens = data
            bx.append(torch.FloatTensor(wav))
            x_lens.append(wav.shape[-1])
            if tokens is not None:
                bt.append(torch.LongTensor(tokens))  # use 0 for paddings
                t_lens.append(tokens.shape[-1])
        for i, x in enumerate(bx):
            if x.shape[-1] < max(x_lens):
                bx[i] = torch.cat([x, torch.zeros(max(x_lens) - x.shape[-1])], -1)
        for i, t in enumerate(bt):
            if t.shape[-1] < max(t_lens):
                bt[i] = torch.cat([t, 999 * torch.ones(max(t_lens) - t.shape[-1]).long()], -1)
        #if len(bt) != len(bx):
        #    print(len(bx), len(bt), flush=True)
        if self.return_path:
            return (torch.stack(bx), torch.LongTensor(x_lens)), None if len(bt) == 0 else (torch.stack(bt), torch.LongTensor(t_lens)), bp
        return (torch.stack(bx), torch.LongTensor(x_lens)), None if len(bt) == 0 else (torch.stack(bt), torch.LongTensor(t_lens))
    
    def get_idx(self, idx):
        path = self.file_paths[idx]
        try:
            if path.split('.')[-1] == 'npy':
                wav = np.load(path)
            else:
                wav, _ = librosa.load(path, sr=24000, mono=True)
        except FileNotFoundError as e:
            idx = np.random.choice(len(self.file_paths))
            print('File not found:', path, 'randomly select another and retry...')
            return self.get_idx(idx)

        wav = wav / wav.std() * 0.13
        if np.abs(wav).max() >= 0.99:
            wav = wav / (np.abs(wav).max() + 1e-2)
        
        if '/libri_light/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'libri_light/hubert_sami_v2.0', '/'.join(path.split('/')[-2:]).replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                print('Tokens path not found:', tokens_path)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/en_1400h/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'en_1400h/hubert_sami_v2.0', path.split('/')[-1].replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                print('Tokens path not found:', tokens_path)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/zh_TTS_xxh/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'zh_TTS_xxh/hubert_sami_v2.0', path.split('/')[-1].replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                print('Tokens path not found:', tokens_path)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/zh_Mandrain_8000h/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'zh_Mandrain_8000h/hubert_sami_v2.0', '/'.join(path.split('/')[-2:]).replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                print('Tokens path not found:', tokens_path)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        else:
            tokens = None
        
        if np.isnan(wav).any() or np.abs(wav).mean() < 1e-3:
            idx = np.random.choice(len(self.file_paths))
            print('NaN: Redraw!')
            return self.get_idx(idx)
        
        if self.use_formant_shift:
            wav = random_formant_shift(wav[None])[0]

        if len(wav) > 24000 * 10:
            wav = wav[:24000 * 10]
            if tokens is not None:
                tokens = tokens[:500]
            

        if self.return_path:
            return wav, tokens, path
        return wav, tokens

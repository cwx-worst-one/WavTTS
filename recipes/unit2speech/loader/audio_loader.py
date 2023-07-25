import os
import random

import librosa
import numpy as np
import torch
from recipes.unit2speech.loader.augmentation import random_formant_shift
from torch.utils.data import DataLoader, Dataset


def load_data(
    wav_list,
    batch_size,
    accelerator=None,
    limit=None,
    sample_rate=24000,
    return_path=False,
    use_formant_shift=False,
    deterministic=False,
    num_workers=20,
):
    dataset = AudioDataset(
        wav_list,
        batch_size,
        shuffle=not deterministic,
        limit=limit,
        accelerator=accelerator,
        sample_rate=sample_rate,
        return_path=return_path,
        use_formant_shift=use_formant_shift,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn,
        num_workers=0 if accelerator is None else num_workers
    )
    return loader


def collate_fn(batch):
    return batch[0]


class AudioDataset(Dataset):
    def __init__(
        self,
        wav_list,
        batch_size,
        shuffle=True,
        limit=None,
        accelerator=None,
        sample_rate=24000,
        return_path=False,
        use_formant_shift=False,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.wav_list = wav_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.limit = limit
        self.sample_rate = sample_rate
        self.return_path = return_path
        self.use_formant_shift = use_formant_shift
        self.hubert_tokens_dir = '/mnt/bd/huangzhiying-lq-valle-volume8/data/bytegen/s2s_hubert'
        
        shard = 0 if accelerator is None else accelerator.process_index
        num_shards = 1 if accelerator is None else accelerator.num_processes

        # load dataset
        file_paths = []
        # load audio path list
        with open(wav_list, "r") as f:
            lines = f.read().splitlines()
            #print(len(lines), "lines !", flush=True)
            for line in lines:
                splits = line.split('|')
                file_path = splits[0]
                file_paths.append(file_path)
        if limit is not None:
            file_paths = file_paths[:limit]
        self.file_paths = file_paths[shard:][::num_shards]
        if shuffle:
            random.shuffle(self.file_paths)

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __getitem__(self, batch_idx):
        x_lens, t_lens, bx, bp, bt = [], [], [], [], []
        for idx in range(self.batch_size):
            data = self.get_idx(batch_idx * self.batch_size + idx)
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
                wav, sr = librosa.load(path, sr=24000, mono=True)
        except FileNotFoundError as e:
            idx = np.random.choice(len(self.file_paths))
            self.accelerator.print('File not found:', path, flush=True)
            return self.get_idx(idx)

        wav_len = len(wav)
        wav = wav / wav.std() * 0.13
        if np.abs(wav).max() >= 0.99:
            wav = wav / (np.abs(wav).max() + 1e-2)
        
        if '/libri_light/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'libri_light/hubert_sami_v2.0', '/'.join(path.split('/')[-2:]).replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                self.accelerator.print('Tokens path not found:', tokens_path, flush=True)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/en_1400h/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'en_1400h/hubert_sami_v2.0', path.split('/')[-1].replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                self.accelerator.print('Tokens path not found:', tokens_path, flush=True)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/zh_TTS_xxh/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'zh_TTS_xxh/hubert_sami_v2.0', path.split('/')[-1].replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                self.accelerator.print('Tokens path not found:', tokens_path, flush=True)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        elif '/zh_Mandrain_8000h/' in path:
            tokens_path = os.path.join(self.hubert_tokens_dir,
                'zh_Mandrain_8000h/hubert_sami_v2.0', '/'.join(path.split('/')[-2:]).replace('.wav', '.npy'))
            if not os.path.isfile(tokens_path):
                self.accelerator.print('Tokens path not found:', tokens_path, flush=True)
                idx = np.random.choice(len(self.file_paths))
                return self.get_idx(idx)
            tokens = np.load(tokens_path)
        else:
            tokens = None
        
        if np.isnan(wav).any() or np.abs(wav).mean() < 1e-3:
            idx = np.random.choice(len(self.file_paths))
            self.accelerator.print('NaN: Redraw!', flush=True)
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


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    loader = load_data("../sound-stable-diffusion/stft_recons", 1) 
    limit = 10
    for batch in loader:
        #print(batch[-1][0])
        print(batch[0].shape, batch[0].max(), batch[0].min(), flush=True)
        #audio_writer.add_audio(batch[-1][0], batch[0][0].cpu().squeeze(0).squeeze(0).numpy(), limit - 1, 16000)
        limit -= 1
        if limit == 0:
            break

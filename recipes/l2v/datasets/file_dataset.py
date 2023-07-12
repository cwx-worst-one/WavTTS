from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from pathlib import Path
import numpy as np
import torch
import json
from typing import List
from tqdm import tqdm
import librosa
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import convert_text
from recipes.l2v.datasets.lyrics import pad_batch, pad_to_seq_length
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import phoneme_mixture_token, phoneme_vocal_token, phoneme_speech_token, phoneme_padding_value
from torchaudio_augmentations import Compose
from recipes.musiclm.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    NormalizeAudio
)

cache_dir = Path('/mnt/bn/ashaw-us/processed_data/file_list')
karaoke_dir = Path('/mnt/bn/ashaw-us/processed_data/karaoke')
karaoke_phoneme_dir = karaoke_dir/'karaoke_phoneme_10s'
karaoke_lyrics_dir = karaoke_dir/'karaoke_lyrics_10s'
karaoke_full_dir = karaoke_dir/'karaoke_full_10s'
karaoke_vocal_dir = karaoke_dir/'karaoke_vocal_10s'

libritts_dir = Path('/mnt/bn/ashaw-us/data/libritts/LibriTTS/train-clean-360')

# max sequence_length
# max phoneme_length
sr = 24_000
wav_max_len_seconds = 10
wav_max_seq_len = sr * wav_max_len_seconds
phoneme_max_seq_len = 150

audio_type2phoneme = {
    'vocal': phoneme_vocal_token,
    'mixture': phoneme_mixture_token,
    'speech': phoneme_speech_token
}

class KaraokeFileDataset(Dataset):
    def __init__(self, file_ids, karaoke_wav_dir, vocal_only, karaoke_phoneme_dir=karaoke_phoneme_dir, karaoke_lyrics_dir=karaoke_lyrics_dir):
        self.karaoke_wav_dir = Path(karaoke_wav_dir)
        self.karaoke_phoneme_dir = Path(karaoke_phoneme_dir)
        self.karaoke_lyrics_dir = Path(karaoke_lyrics_dir)
        self.file_ids = file_ids
        self.vocal_only = vocal_only

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [
                self.to_tensor, 
                self.audio_dim, 
                self.normalize_audio_fp32,
                self.normalize_audio
            ]
        )
        
    def load_file(self, file_id):
        karaoke_wav_fp = self.karaoke_wav_dir/f'{file_id}.npy'
        karaoke_wav_np = np.load(karaoke_wav_fp)
        karaoke_phoneme_fp = self.karaoke_phoneme_dir/f'{file_id}.npy'
        karaoke_phoneme_np = np.load(karaoke_phoneme_fp)
        karaoke_lyrics_fp = self.karaoke_lyrics_dir/f'{file_id}.txt'
        with open(karaoke_lyrics_fp, 'r') as f: 
            karaoke_lyrics = f.read()
        karaoke_wav_np = self.base_transform(karaoke_wav_np)[0]
        return {
            'audio': karaoke_wav_np, 
            'lyrics_tokens': karaoke_phoneme_np,
            'lyrics': karaoke_lyrics,
            'type': 'vocal' if self.vocal_only else 'mixture'
        }

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        return self.load_file(self.file_ids[idx])

class PhonemeCollator(object):
    def __init__(self, pad_to_max_seq_length=True, append_audio_type_token=True):
        self.pad_to_max_seq_length = pad_to_max_seq_length
        self.append_audio_type_token = append_audio_type_token

    def __call__(self, batches):
        return phoneme_collate(batches, pad_to_max_seq_length=self.pad_to_max_seq_length, append_audio_type_token=self.append_audio_type_token)


def phoneme_collate(batches, pad_to_max_seq_length=True, append_audio_type_token=True):
    item_pairs = [(item['wav'], item['phoneme'], item['lyrics'], item.get('type'), item.get('prompt')) for item in batches if item] # must filter out item as may have tokenization error
    wavs, phonemes, lyrics, audio_type, prompts = zip(*item_pairs)
    if audio_type[0] is None:
        append_audio_type_token = False
    else:
        audio_type_t = torch.tensor([audio_type2phoneme[t] for t in audio_type], dtype=torch.long)
    
    if pad_to_max_seq_length:
        wavs = pad_to_seq_length(wavs, wav_max_seq_len, dtype=torch.float)
        phonemes = pad_to_seq_length(phonemes, phoneme_max_seq_len, dtype=torch.long, padding_value=phoneme_padding_value)
        if append_audio_type_token:
            phonemes[:, -1] = audio_type_t
    else: 
        wavs = pad_batch(wavs),
        phonemes = pad_batch(phonemes, padding_value=phoneme_padding_value)
        if append_audio_type_token:
            phonemes = torch.concat([phonemes, audio_type_t[:, None]], dim=-1)
    result = {
        'audio': wavs,
        'phoneme': phonemes,
        'lyrics': lyrics,
        'type': audio_type,
        'prompts': prompts
    }
    return result

def create_karaoke_datasets(vocal_only):
    karaoke_wav_dir = karaoke_vocal_dir if vocal_only else karaoke_full_dir
    file_list = cache_dir/f'{karaoke_wav_dir.name}.json'
    if file_list.exists():
        with open(file_list, 'r') as f: file_ids = json.load(f)
    else:
        file_ids = [x.stem for x in karaoke_wav_dir.iterdir() if x.suffix == '.npy']
        with open(file_list, 'w') as f: json.dump(file_ids, f)

    train_file_ids, valid_file_ids = torch.utils.data.random_split(file_ids, [0.95, 0.05], generator=torch.Generator().manual_seed(42))

    train_ds = KaraokeFileDataset(train_file_ids, karaoke_wav_dir, vocal_only=vocal_only)
    valid_ds = KaraokeFileDataset(valid_file_ids, karaoke_wav_dir, vocal_only=vocal_only)
    return train_ds, valid_ds

# libritts

class LibrittsFileDataset(Dataset):
    def __init__(self, file_ids, libritts_dir=libritts_dir, debug=False):
        self.libritts_dir = Path(libritts_dir)
        self.file_ids = file_ids
        self.debug = debug
        
    def load_file(self, file_id):
        wav_fp = self.libritts_dir/f'{file_id}.wav'
        # sr, wav_np = wavfile.read(wav_fp)
        wav_np, sr = librosa.load(str(wav_fp))
        lyrics_fp = self.libritts_dir/f'{file_id}.normalized.txt'
        try:
            with open(lyrics_fp, 'r') as f: 
                lyrics = f.read()
            lyrics_tokens = convert_text(lyrics)
        except Exception as e:
            if self.debug: print('Could not convert:', file_id, e, lyrics)
            return {}
            
        return {
            'audio': wav_np, 
            'lyrics_tokens': lyrics_tokens,
            'lyrics': lyrics,
            'type': 'speech',
            'path': str(wav_fp)
        }

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        return self.load_file(self.file_ids[idx])
    
def get_clean_libritts_file_ids():
    file_list = cache_dir/f'{libritts_dir.name}.json'
    if file_list.exists():
        with open(file_list, 'r') as f: 
            file_ids = json.load(f)
        return file_ids

    lyrics_files = list(libritts_dir.glob('**/*.normalized.txt'))
    clean_ids = []
    for lyrics_fp in tqdm(lyrics_files):
        try:
            with open(lyrics_fp, 'r') as f: 
                lyrics = f.read()
            phonemes = convert_text(lyrics)
            file_id = str(lyrics_fp.relative_to(libritts_dir)).replace('.normalized.txt', '')
            clean_ids.append(file_id)
        except Exception as e:
            print(e)
    with open(file_list, 'w') as f: json.dump(clean_ids, f)
    return clean_ids

def create_libritts_datasets():
    file_ids = get_clean_libritts_file_ids()
    train_file_ids, valid_file_ids = torch.utils.data.random_split(file_ids, [0.98, 0.02], generator=torch.Generator().manual_seed(42))

    train_ds = LibrittsFileDataset(train_file_ids)
    valid_ds = LibrittsFileDataset(valid_file_ids)
    return train_ds, valid_ds


class PhonemeDataModule(LightningDataModule):
    @classmethod
    def init_karaoke(cls, batch_size, num_workers, vocal_only=True):
        train_ds, valid_ds = create_karaoke_datasets(vocal_only)
        return cls(train_ds, valid_ds, batch_size, num_workers)
    
    @classmethod
    def init_libritts(cls, batch_size, num_workers):
        train_ds, valid_ds = create_libritts_datasets()
        return cls(train_ds, valid_ds, batch_size, num_workers)
    
    @classmethod
    def init_karaoke_tts(cls, batch_size, num_workers, vocal_only=True):
        tts_train_ds, tts_valid_ds = create_libritts_datasets()
        k_train_ds, k_valid_ds = create_karaoke_datasets(vocal_only)
        train_ds = ConcatDataset([tts_train_ds, k_train_ds])
        valid_ds = ConcatDataset([tts_valid_ds, k_valid_ds])
        return cls(train_ds, valid_ds, batch_size, num_workers)

    def __init__(
        self,
        train_ds,
        valid_ds,
        batch_size,
        num_workers
    ):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=phoneme_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=phoneme_collate
        )



# Inference file ids
# file_ids = ['21423_0',
#  '28749_2',
#  '25336_1',
#  '29875_0',
#  '30867_0',
#  '10883_2',
#  '44410_2',
#  '64494_1']
file_ids = ['10883_2', '44410_2', '64494_1'] # original ids

file_ids = ['9495_0', '9493_1', '9460_1', '9581_0', '9600_0', '9614_0', '9645_0'] # tar valid ids

# 9536, 9564_0, 9594_0
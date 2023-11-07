import os
import re
import sys
import time
import random
import pickle

import tqdm
import torch
import torchaudio
import numpy as np 
from torch import nn
from pathlib import Path
from sox import Transformer
from torchaudio import load
from librosa.util import find_files
from joblib.parallel import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from torchaudio.sox_effects import apply_effects_file


EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

# Voxceleb 2 Speaker verification
class SpeakerVerifi_train(Dataset):
    def __init__(self, vad_config, key_list, file_path, meta_data, offline_root, layer, max_timestep=None, n_jobs=12):
        self.roots = file_path
        self.root_key = key_list
        self.max_timestep = max_timestep
        self.vad_c = vad_config 
        self.dataset = []
        self.all_speakers = []

        self.offline_root = os.path.join(offline_root, 'dev/wav')
        self.layer = layer

        for index in range(len(self.root_key)):
            cache_path = Path(os.path.dirname(__file__)) / '.wav_lengths' / f'{self.root_key[index]}_length.pt'
            cache_path.parent.mkdir(exist_ok=True)
            root = Path(self.roots[index])

            if not cache_path.is_file():
                def trimmed_length(path):
                    wav_sample = self._load_offline_feature(path)
                    length = wav_sample.shape[0]
                    return length

                wav_paths = find_files(root)
                wav_paths = [os.path.relpath(wav_path, root) for wav_path in wav_paths]
                wav_lengths = Parallel(n_jobs=n_jobs)(delayed(trimmed_length)(path) for path in tqdm.tqdm(wav_paths, desc="Preprocessing"))
                wav_tags = [Path(path).parts[-3:] for path in wav_paths]
                torch.save([wav_tags, wav_lengths], str(cache_path))
            else:
                wav_tags, wav_lengths = torch.load(str(cache_path))
                wav_paths = [root.joinpath(*tag) for tag in wav_tags]
                wav_paths = [os.path.relpath(wav_path, root) for wav_path in wav_paths]

            speaker_dirs = ([f.stem for f in root.iterdir() if f.is_dir()])
            self.all_speakers.extend(speaker_dirs)
            for path, length in zip(wav_paths, wav_lengths):
                if length > self.vad_c['min_sec']:
                    self.dataset.append(path)

        self.all_speakers.sort()
        self.speaker_num = len(self.all_speakers)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        try:
            wav = self._load_offline_feature(str(path))
        except Exception as e:
            print(e)
            return None
        length = wav.shape[0]
        
        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]

        tags = Path(path).parts[-3:]
        utterance_id = "-".join(tags).replace(".wav", "")
        label = self.all_speakers.index(tags[0])
        return wav, utterance_id, label
        
    def collate_fn(self, samples):
        samples = [x for x in samples if x is not None]
        return zip(*samples)

    def _load_offline_feature(self, wav_path):
        feats = np.load(os.path.join(self.offline_root, wav_path+'.npy'))
        # T = np.random.randint(50, 101)
        # feats = np.ones([T])
        if len(feats.shape) == 2 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 4 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 3 and self.layer is not None:
            feats = feats[self.layer]
        # print(feats, '  ', feats.shape, '  ', os.path.join(self.offline_root, wav_path+'.npy'))
        return feats


class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, file_path, meta_data, offline_root, layer):
        self.root = file_path
        self.meta_data = meta_data

        self.offline_root = offline_root
        self.layer = layer


        self.necessary_dict = self.processing()
        self.vad_c = vad_config 
        self.dataset = self.necessary_dict['spk_paths']
        self.pair_table = self.necessary_dict['pair_table']

        
    def processing(self):
        pair_table = []
        spk_paths = set()
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(os.path.join(self.offline_root, 'dev/wav'), list_pair[1])
            pair_2= os.path.join(os.path.join(self.offline_root, 'dev/wav'), list_pair[2])
            spk_paths.add(pair_1)
            spk_paths.add(pair_2)
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {
            "spk_paths": list(spk_paths),
            "total_spk_num": None,
            "pair_table": pair_table
        }

    def __len__(self):
        return len(self.necessary_dict['spk_paths'])

    def __getitem__(self, idx):
        x_path = self.dataset[idx]

        x_name = x_path

        try:
            wav = self._load_offline_feature(x_path)
        except Exception as e:
            print(e)
            return None, None

        return wav, x_name

    def collate_fn(self, data_sample):
        # data_sample[0][0] = [x for x in data_sample[0][0] if x is not None]
        # data_sample[0][1] = [x for x in data_sample[0][1] if x is not None]
        data_sample = [(x,y) for x,y in data_sample if x is not None and y is not None]
        wavs, x_names = zip(*data_sample)
        return wavs, x_names

    def _load_offline_feature(self, wav_path):
        feats = np.load(os.path.join(self.offline_root, wav_path+'.npy'))
        # T = np.random.randint(50, 101)
        # feats = np.ones([T])
        if len(feats.shape) == 2 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 4 and feats.shape[0] == 1:
            feats = feats[0]
        if len(feats.shape) == 3 and self.layer is not None:
            feats = feats[self.layer]
        # print(feats, '  ', feats.shape, '  ', os.path.join(self.offline_root, wav_path+'.npy'))
        return feats

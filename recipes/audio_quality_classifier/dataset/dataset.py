import os
import glob
import random
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def audio_loader(path, source):
    if source == 'mos':
        audio = torch.from_numpy(sf.read(path)[0]).float()[None, ]
        if len(audio.shape) == 3:
            audio = audio.mean(-1, keepdim=False)

    elif source == 'hq':
        audio = torch.from_numpy(sf.read(path)[0].T).mean(0, True).float()

    elif source == 'lq':
        audio = torch.tensor((np.load(path) / 32768.0).astype("float32"))

    return audio

class RewardDataset(Dataset):
    def __init__(
        self, 
        asset_path
    ):
        self._prepare_data(asset_path)

    def _prepare_data(self, asset_path):
        # read prompt from txt
        data_dict = {}
        with open(f'{asset_path}/keys.txt', 'r') as f:
            for line in f:
                data_dict[line.strip()] = None
        
        # read reward from pt
        remove_keys = []
        for prompt in data_dict.keys():
            data = None
            try:
                data = torch.load(f'{asset_path}/{prompt[:50]}.pt')
            except:
                remove_keys.append(prompt)

            if data is not None and len(data) == 1:
                remove_keys.append(prompt)
            else:
                data_dict[prompt] = data

        for key in remove_keys:
            data_dict.pop(key)

        self.data_dict = data_dict
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        prompt, audio = self.data[idx]
        N = len(audio) - 1
        pairs = []
        for i in range(N + 1):
            for j in range(i + 1, N + 1):
                pairs.append(torch.tensor([i, j]))

        pairs = torch.stack(pairs)

        samples = {
            'text': [prompt for _ in range(len(audio))],
            'audio': audio,
            'pairs': pairs
        }
        return samples

class RewardDataset2(Dataset):
    def __init__(
        self, 
        asset_path,
        data_slice=[0, 1]
    ):
        self._prepare_data(asset_path, data_slice=data_slice)

    def _prepare_data(self, asset_path, data_slice):
        # read prompt from txt
        data = [] 
        
        audio_paths = glob.glob(f'{asset_path}/*.wav')
        random.seed(0)
        random.shuffle(audio_paths)
        start = int(len(audio_paths)*data_slice[0])
        end = int(len(audio_paths)*data_slice[1])

        data_dict = {}
        for audio_path in audio_paths[start:end]:
            _, score = os.path.basename(audio_path)[:-4].split('_')
            if score == 'nan':
                continue
            # audio = torch.from_numpy(torch.load(audio_path)).float()
            audio = torch.from_numpy(sf.read(audio_path)[0]).float()
            # data.append((audio, float(score)))
            if float(score) not in data_dict:
                data_dict[float(score)] = [audio]
            else:
                data_dict[float(score)].append(audio)


        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.num_data = end - start
        self.cnt = 0

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        # random choose audio from key
        score_idx = self.cnt % len(self.keys)
        score = self.keys[score_idx]
        audio_idx = torch.randint(0, len(self.data_dict[score]), (1,)).item()
        self.cnt += 1

        return self.data_dict[score][audio_idx], score

class RewardDataset3(Dataset):
    def __init__(
        self, 
        asset_path,
        data_slice=[0, 1]
    ):
        self._prepare_data(asset_path, data_slice=data_slice)

    def _prepare_data(self, asset_path,  data_slice):
        # read prompt from txt
        data = [] 
        
        h_audio_paths = glob.glob(f'{asset_path}/high_quality/*.wav')
        l_audio_paths = glob.glob(f'{asset_path}/low_quality/*.npy')
        df = pd.read_csv(f'{asset_path}/dirty_kaggle.csv')
        music_ids = df['music_id'].to_list()
        l_audio_paths = [i for i in l_audio_paths if os.path.basename(i)[:-4] in music_ids]

        random.seed(0)
        random.shuffle(h_audio_paths)
        h_start = int(len(h_audio_paths)*data_slice[0])
        h_end = int(len(h_audio_paths)*data_slice[1])

        h_audio_paths = h_audio_paths[h_start:h_end]

        random.seed(0)
        random.shuffle(l_audio_paths)
        l_start = int(len(l_audio_paths)*data_slice[0])
        l_end = int(len(l_audio_paths)*data_slice[1])
        l_audio_paths = l_audio_paths[l_start:l_end]
    

        self.h_audio_paths = h_audio_paths
        self.l_audio_paths = l_audio_paths
        self.num_data = l_end - l_start

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):

        # random choose audio 
        h_audio_idx = torch.randint(0, len(self.h_audio_paths), (1,)).item()
        l_audio_idx = torch.randint(0, len(self.l_audio_paths), (1,)).item()

        h_audio = torch.from_numpy(sf.read(self.h_audio_paths[h_audio_idx])[0].T).mean(0, True).float()
        l_audio = torch.tensor((np.load(self.l_audio_paths[l_audio_idx]) / 32768.0).astype("float32"))

        # random sample 4 second each
        start = torch.randint(0, h_audio.shape[-1] - 4*24000, (1,)).item()
        h_audio = h_audio[:, start:start+3*24000]
        start = torch.randint(0, l_audio.shape[-1] - 4*24000, (1,)).item()
        l_audio = l_audio[:, start:start+3*24000]

        return h_audio, l_audio

class RewardDataset4(Dataset):
    def __init__(
        self, 
        asset_path,
        audio_duration=2,
        data_slice=[0, 1]
    ):
        self.audio_duration = audio_duration
        self._prepare_data(asset_path, data_slice=data_slice)

    def _prepare_data(self, asset_path, data_slice):
        # read prompt from txt
        data = [] 
        
        audio_paths = glob.glob(f'{asset_path}/*.wav')
        random.seed(0)
        random.shuffle(audio_paths)
        start = int(len(audio_paths)*data_slice[0])
        end = int(len(audio_paths)*data_slice[1])

        data_dict = {}
        for audio_path in audio_paths[start:end]:
            _, score = os.path.basename(audio_path)[:-4].split('_')
            if score == 'nan':
                continue
            if float(score) not in data_dict:
                data_dict[float(score)] = [('mos', audio_path)]
            else:
                data_dict[float(score)].append(('mos', audio_path))
        
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.num_data = end - start
        self.cnt = 0

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # random choose audio from key
        score_idx = self.cnt % len(self.keys)
        score = self.keys[score_idx]

        sampling = True
        while sampling:
            audio_idx = torch.randint(0, len(self.data_dict[score]), (1,)).item()
            source, path = self.data_dict[score][audio_idx]
            audio = audio_loader(path, source)
            
            if audio.shape[-1] <= 0.5 * self.audio_duration*24000:
                continue
            else:
                if audio.shape[-1] < self.audio_duration*24000:
                    audio = torch.nn.functional.pad(audio, (0, self.audio_duration*24000 - audio.shape[-1]))
                start = torch.randint(0, audio.shape[-1] - self.audio_duration*24000 + 1, (1,)).item()
                audio = audio[:, start:start+self.audio_duration*24000]
                sampling = False
        self.cnt += 1

        return audio, score

def collate(batch):

    return batch

def collate2(batch):
    data_dict = {}
    for sample in batch: 
        if sample is not None:  
            data_dict[sample[1]] = sample[0]

    # sort data dict based on key value
    data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[0], reverse=True)}
    audio_tensor = torch.stack(list(data_dict.values()))

    N = len(audio_tensor) - 1
    pairs = []
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            pairs.append(torch.tensor([i, j]))
    try:
        pairs = torch.stack(pairs)
    except:
        print(data_dict.keys(), len(batch))
    return {
        'text': ['audio' for _ in range(len(audio_tensor))],
        'audio': audio_tensor,
        'pairs': pairs
    }

def collate3(batch):

    audio = []
    pairs = []
    for sample in batch:    
        audio.append(sample[0])
        audio.append(sample[1])
        pairs.append(torch.tensor([len(audio) - 2, len(audio) - 1]))
    
    audio_tensor = torch.stack(audio)

    pairs = torch.stack(pairs)
    return {
        'text': ['audio' for _ in range(len(audio_tensor))],
        'audio': audio_tensor,
        'pairs': pairs
    }

if __name__ == '__main__':
    dataset = RewardDataset4(asset_path='aq_mos')
    # for i in range(10):
    #     dataset.__getitem__(i)
    # assert 1==2


    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=5, num_workers=1, shuffle=False, collate_fn=collate2)

    for i, d in enumerate(loader):
        print(d['audio'].shape)
        print(d['pairs'])
        assert 1==2

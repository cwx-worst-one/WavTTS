import os
import librosa
import numpy as np
import torch
import random
import colorednoise as cn
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
from recipes.umm_062.models.rmvpe.constants import *
from recipes.umm_062.models.rmvpe.spec import MelSpectrogram

class MIR1K(Dataset):
    def __init__(self, path, hop_length, groups=None, whole_audio=False, use_aug=True):
        self.path = path
        self.HOP_LENGTH = hop_length
        self.num_class = N_CLASS
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.paths = []
        self.data_buffer = {}
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.load(*input_files)

    def __getitem__(self, index):
        audio_path = self.paths[index]
        data_buffer = self.data_buffer[audio_path]
        
        start_frame = 0 if self.whole_audio else random.randint(1, data_buffer['len'] - 128)
        end_frame = data_buffer['len'] if self.whole_audio else start_frame + 128
        
        if self.use_aug:
            key_shift = random.uniform(0, 5)
        else:
            key_shift = 0
        factor = 2 ** (key_shift / 12)
        win_length_new = int(np.round(WINDOW_LENGTH * factor))
        start_id = WINDOW_LENGTH + start_frame * self.HOP_LENGTH - win_length_new // 2
        end_id = WINDOW_LENGTH + (end_frame - 1) * self.HOP_LENGTH + (win_length_new + 1) // 2
        
        audio = data_buffer['audio'][start_id : end_id]
        if self.use_aug:
            if data_buffer['noise'] is None:
                noise = cn.powerlaw_psd_gaussian(random.uniform(0, 2), len(audio))
                noise = torch.from_numpy(noise).float() * (10 ** random.uniform(-6, -1))
            else:
                noise = random.uniform(-1, 1) * data_buffer['noise'][start_id : end_id]
            audio_aug = audio + noise
            max_amp = float(torch.max(torch.abs(audio_aug))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            audio_aug = audio_aug * (10 ** log10_vol_shift)
        else:
            if data_buffer['noise'] is None:
                noise = 0
            else:
                noise = data_buffer['noise'][start_id : end_id]
            audio_aug = audio + noise
            
        mel = self.mel(audio_aug.unsqueeze(0), keyshift = key_shift, center=False).squeeze(0)
        cent = data_buffer['cent'][start_frame : end_frame] + 1200 * np.log2(win_length_new / WINDOW_LENGTH)
        voice = data_buffer['voice'][start_frame : end_frame]
        
        index = (cent - CONST) / 20
        pitch_label = torch.exp(-(torch.arange(N_CLASS).expand(end_frame - start_frame, -1) - index.unsqueeze(-1)) ** 2 / 2 / 1.25 ** 2)
        pitch_label = pitch_label * voice.unsqueeze(-1)
        return dict(mel=mel, pitch=pitch_label, file=audio_path)

    def __len__(self):
        return len(self.data_buffer)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.pv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
        if len(wav.shape) > 1: 
            noise = wav[0]
            noise = np.pad(noise, (WINDOW_LENGTH, WINDOW_LENGTH), mode='reflect')
            noise = torch.from_numpy(noise).float()
            audio = wav[1]
        else:
            noise = None
            audio = wav

        n_frames = len(audio) // self.HOP_LENGTH + 1
        audio = np.pad(audio, (WINDOW_LENGTH, WINDOW_LENGTH), mode='reflect')
        audio = torch.from_numpy(audio).float()
        
        cent = torch.zeros(n_frames, dtype=torch.float)
        voice = torch.zeros(n_frames, dtype=torch.float)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                i += 1
                if float(line) != 0 and i < n_frames:
                    freq = 440 * (2.0 ** ((float(line) - 69.0) / 12.0))
                    cent[i] = 1200 * np.log2(freq / 10)
                    voice[i] = 1
        self.paths.append(audio_path)
        self.data_buffer[audio_path] = {'len': n_frames, 'audio': audio, 'noise': noise, 'cent': cent, 'voice': voice}


class MIR_ST500(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.tsv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)

        midi = np.loadtxt(label_path, delimiter='\t', skiprows=1)
        for onset, offset, note in midi:
            left = int(round(onset * SAMPLE_RATE / self.HOP_LENGTH))
            right = int(round(offset * SAMPLE_RATE / self.HOP_LENGTH)) + 1
            freq = 440 * (2.0 ** ((float(note) - 69.0) / 12.0))
            cent = 1200 * np.log2(freq / 10)
            index = int(round((cent - CONST) / 20))
            pitch_label[left:right, index] = 1
            voice_label[left:right] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH + 1
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len + WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step],
                                 voice=voice_label[begin_step:end_step], file=audio_path))
            data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:],
                             voice=voice_label[-n_steps:], file=audio_path))
        else:
            data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
        return data


class MDB(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.csv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)

        df_label = pd.read_csv(label_path)
        for i in range(len(df_label)):
            if float(df_label['midi'][i]):
                freq = 440 * (2.0 ** ((float(df_label['midi'][i]) - 69.0) / 12.0))
                cent = 1200 * np.log2(freq / 10)
                index = int(round((cent - CONST) / 20))
                pitch_label[i][index] = 1
                voice_label[i] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH + 1
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len + WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step],
                                 voice=voice_label[begin_step:end_step], file=audio_path))
            data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:],
                             voice=voice_label[-n_steps:], file=audio_path))
        else:
            data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
        return data

import os

import librosa
import numpy as np
from pydub import AudioSegment
from torch.utils.data import Dataset
from tqdm import tqdm 
from transformers import PreTrainedTokenizerFast
from transformers import Wav2Vec2FeatureExtractor



def get_active_frames(audio, threshold=0.05, sample_rate=24000):
    window_size = int(sample_rate * 0.1)

    frames = librosa.util.frame(
        x=audio, frame_length=window_size, hop_length=window_size
    ).T
    energy = np.max(np.abs(frames), axis=-1)  # shape: (frames_num,)
    rate = np.sum(energy > threshold) / energy.shape[0]

    if rate < 1 / 10:
        return False
    return True


class AudioLMDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        sample_rate=16000,
        sample_duration=10,
        minn_duration=3,
        use_cache=False,
        cache_path=None,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.minn_duration = minn_duration
        self.use_cache = use_cache
        self.segment_size = sample_duration * self.sample_rate
        self.audio_paths = []

        for meta_path in meta_paths:
            self.audio_paths += self.get_meta_data(meta_path)
        if use_cache:
            with open(cache_path, "r") as f:
                lines = [line.strip() for line in f]
            self.cache_dict = {"w2v": {}, "mulan": {}}
            for line in lines:
                line = line.split("|")
                self.cache_dict["w2v"].update({line[0]: line[1]})

    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            # if (length >= self.segment_size - 0.05 * self.sample_rate) and (
            #     length <= self.sample_rate * 600
            # ):
            if (length >= self.minn_duration * self.sample_rate) and (
                length <= self.sample_rate * 600
            ):
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if path.endswith(".npy"):
                    wav = np.load(path)
                else:
                    audio = AudioSegment.from_file(path)
                    audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
                    wav = np.asarray(audio.get_array_of_samples())
                if wav.dtype == np.int16:
                    wav = wav / 32768.0
                elif wav.dtype == np.int32:
                    wav = wav / 2_147_483_648.0
                if len(wav.shape) >= 2:
                    wav = wav[0]

                # librosa.core.resample(a, 24000, 16000)

                scale = np.max(np.abs(wav))
                wav = wav.astype(np.float32) 
                # random slice
                wav_len = wav.shape[0]
                # prevent data not long enough
                if wav_len < self.minn_duration * self.sample_rate:
                    raise Exception
                if wav_len < self.segment_size:
                    wav = np.pad(wav, (0, self.segment_size - wav_len))
                    rand_slice = wav
                else:
                    beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
                    rand_slice = wav[beg : beg + self.segment_size]
                    wav_len = rand_slice.shape[0]
                # prevent silence
                if np.sqrt(np.mean(rand_slice**2)) > 1e-2 and get_active_frames(
                    rand_slice, threshold=0.05, sample_rate=self.sample_rate
                ):
                    rand_slice = rand_slice / scale * 0.95
                    break
                else:
                    path = np.random.choice(self.audio_paths)
            except Exception as e:
                print("File {} cant load, choose another one".format(path))
                print(e)
                path = np.random.choice(self.audio_paths)
        return rand_slice, wav_len



class AudioLMHuggingfaceDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        ssl_model_name,
        sample_rate=16000,
        sample_duration=10,
        minn_duration=3,
        use_cache=False,
        cache_path=None,
    ):
        self.sample_rate = sample_rate
        self.durations = sample_duration
        self.minn_duration = minn_duration
        self.use_cache = use_cache
        self.segment_size = sample_duration * self.sample_rate
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)
        self.audio_paths = []

        for meta_path in meta_paths:
            self.audio_paths += self.get_meta_data(meta_path)
        if use_cache:
            with open(cache_path, "r") as f:
                lines = [line.strip() for line in f]
            self.cache_dict = {"w2v": {}, "mulan": {}}
            for line in lines:
                line = line.split("|")
                self.cache_dict["w2v"].update({line[0]: line[1]})

    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            # if (length >= self.segment_size - 0.05 * self.sample_rate) and (
            #     length <= self.sample_rate * 600
            # ):
            if (length >= self.minn_duration * self.sample_rate) and (
                length <= self.sample_rate * 600
            ):
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if path.endswith(".npy"):
                    wav = np.load(path)
                else:
                    audio = AudioSegment.from_file(path)
                    audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
                    wav = np.asarray(audio.get_array_of_samples())
                if wav.dtype == np.int16:
                    wav = wav / 32768.0
                elif wav.dtype == np.int32:
                    wav = wav / 2_147_483_648.0
                if len(wav.shape) >= 2:
                    wav = wav[0]

                # librosa.core.resample(a, 24000, 16000)

                scale = np.max(np.abs(wav))
                wav = wav.astype(np.float32) 
                # random slice
                wav_len = wav.shape[0]
                # prevent data not long enough
                if wav_len < self.minn_duration * self.sample_rate:
                    raise Exception
                if wav_len < self.segment_size:
                    wav = np.pad(wav, (0, self.segment_size - wav_len))
                    rand_slice = wav
                else:
                    beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
                    rand_slice = wav[beg : beg + self.segment_size]
                    wav_len = rand_slice.shape[0]
                # prevent silence
                if np.sqrt(np.mean(rand_slice**2)) > 1e-2 and get_active_frames(
                    rand_slice, threshold=0.05, sample_rate=self.sample_rate
                ):
                    rand_slice = rand_slice / scale * 0.95
                    break
                else:
                    path = np.random.choice(self.audio_paths)
            except Exception as e:
                print("File {} cant load, choose another one".format(path))
                print(e)
                path = np.random.choice(self.audio_paths)

        rand_slice = self.processor(rand_slice, return_tensors="pt", sampling_rate=self.sample_rate).input_values
        rand_slice = rand_slice.squeeze(0)
        return rand_slice, wav_len



class OfflineTokensDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        maxn_tokens_num=500, # 50hz * 10s
        minn_tokens_num=250, # 50hz * 5s
        use_cache=False,
        cache_path=None,
    ):
        self.maxn_tokens_num = maxn_tokens_num
        self.minn_tokens_num = minn_tokens_num
        self.use_cache = use_cache
        self.token_paths = []

        for meta_path in meta_paths:
            self.token_paths += self.get_meta_data(meta_path)


    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            # if (length >= self.segment_size - 0.05 * self.sample_rate) and (
            #     length <= self.sample_rate * 600
            # ):
            if length >= self.minn_tokens_num:
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.token_paths)


    def __getitem__(self, idx):
        path = self.token_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if not path.endswith(".npy"):
                    raise Exception
                    
                token = np.load(path)
                token_len = token.shape[0]

                # prevent data not long enough
                if token_len < self.minn_tokens_num:
                    raise Exception
                if token_len < self.maxn_tokens_num:
                    token = np.pad(token, (0, self.maxn_tokens_num - token_len))
                    rand_slice = token
                else:
                    beg = np.random.randint(low=0, high=token_len - self.maxn_tokens_num + 1)
                    rand_slice = token[beg : beg + self.maxn_tokens_num]
                    token_len = rand_slice.shape[0]

                break

            except Exception as e:
                print("File {} cant load, choose another one".format(path))
                print(e)
                path = np.random.choice(self.token_paths)
        return rand_slice, token_len



class OfflineMergeTokensDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        maxn_tokens_num=500, # 50hz * 10s
        minn_tokens_num=250, # 50hz * 5s
        use_cache=False,
        cache_path=None,
    ):
        self.maxn_tokens_num = maxn_tokens_num
        self.minn_tokens_num = minn_tokens_num
        self.use_cache = use_cache
        self.token_paths = []

        for meta_path in meta_paths:
            self.token_paths += self.get_meta_data(meta_path)


    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            # if (length >= self.segment_size - 0.05 * self.sample_rate) and (
            #     length <= self.sample_rate * 600
            # ):
            if length >= self.minn_tokens_num:
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.token_paths)


    def __getitem__(self, idx):
        path = self.token_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if not path.endswith(".npy"):
                    raise Exception
                    
                token = np.load(path)
                token_len = token.shape[0]

                # prevent data not long enough
                if token_len < self.minn_tokens_num:
                    raise Exception

                merged_token = [int(token[0])]
                orilen_list = []
                curlen = 1
                for it in (list(token[1:]) + [-1]):
                    if it == merged_token[-1]:
                        curlen +=1 
                    else:
                        if it != -1:
                            merged_token.append(it)
                        orilen_list.append(curlen)
                        curlen = 1

                assert len(merged_token) == len(orilen_list)

                merged_token_len = len(merged_token)
                merged_token = np.array(merged_token)
                orilen_list = np.array(orilen_list)


                if merged_token_len < self.maxn_tokens_num:
                    merged_token = np.pad(merged_token, (0, self.maxn_tokens_num - merged_token_len))
                    orilen_list =  np.pad(orilen_list, (0, self.maxn_tokens_num - merged_token_len))
                    rand_slice = merged_token
                    rand_slice_orilen = orilen_list
                    rand_slice_len = merged_token_len
                else:
                    beg = np.random.randint(low=0, high=merged_token_len - self.maxn_tokens_num + 1)
                    rand_slice = merged_token[beg : beg + self.maxn_tokens_num]
                    rand_slice_orilen = orilen_list[beg : beg + self.maxn_tokens_num]
                    rand_slice_len = rand_slice.shape[0]
                break

            except Exception as e:
                print("File {} cant load, choose another one".format(path))
                print(e)
                path = np.random.choice(self.token_paths)
        return rand_slice, rand_slice_orilen, rand_slice_len



class OfflineTokensTextDataset(Dataset):
    def __init__(
        self,
        text_paths,
        meta_paths,
        bpe_tokenizer_file, 
        audio_tokens_num,
        minn_audio_split=25,
        maxn_tokens_num=500, # 50hz * 10s
        minn_tokens_num=250, # 50hz * 5s
        use_cache=False,
        cache_path=None,
    ):
        self.minn_audio_split = minn_audio_split
        self.maxn_tokens_num = maxn_tokens_num
        self.minn_tokens_num = minn_tokens_num
        self.use_cache = use_cache
        self.token_paths = []
        self.buffer = 60

        self.tokenizer = BPETokenizerWithAudioTokens(
            bpe_tokenizer_file, audio_tokens_num=audio_tokens_num)

        self.sep_u2t = self.tokenizer.sep_u2t
        self.sep_t2u = self.tokenizer.sep_t2u
        self.eos = self.tokenizer.eos

        print('loading text dict ...')
        self.text_dict = {}
        for text_path in text_paths:
            with open(text_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    k, v = line.split('\t')
                    # assert k not in self.text_dict, "%s repeated.."%str(k)
                    self.text_dict[k] = v

        for meta_path in meta_paths:
            self.token_paths += self.get_meta_data(meta_path)


    def get_meta_data(self, meta_path):
        with open(meta_path, "r") as f:
            lines = [line.strip() for line in f]
        paths = []
        for line in lines:
            l_res = line.split("|")
            path, length = l_res[0:2]
            length = float(length)
            # if (length >= self.segment_size - 0.05 * self.sample_rate) and (
            #     length <= self.sample_rate * 600
            # ):
            if length >= self.minn_tokens_num and length <= (self.maxn_tokens_num - self.buffer):
                paths.append(path)
        paths = [
            os.path.abspath(os.path.join(os.path.dirname(meta_path, p)))
            if p[0] != "/"
            else p
            for p in paths
        ]
        return paths

    def __len__(self):
        return len(self.token_paths)


    def __getitem__(self, idx):
        path = self.token_paths[idx]
        while True:
            """
            file_size = os.path.getsize(path)
            if file_size / 1024 / 1024 > 60: # skip > 60M audio
                path = np.random.choice(self.audio_paths)
                continue
            """
            try:
                if not path.endswith(".npy"):
                    raise Exception
                    
                audio_token = np.load(path)
                audio_token_len = audio_token.shape[0]

                file_key = os.path.basename(path).replace('.npy', '').strip()
                text = self.text_dict[file_key].strip()
                text_token = self.tokenizer.tokenize(text)
                text_token_len = len(text_token)

                split_point = np.random.randint(low=self.minn_audio_split, high=audio_token_len - self.minn_audio_split)

                token = np.concatenate([audio_token[:split_point], [self.sep_u2t], 
                            text_token, [self.sep_t2u], audio_token[split_point:], [self.eos]], axis=0)
                token_len = token.shape[0]

                assert token_len == (audio_token_len+text_token_len+3)

                # prevent data not long enough
                if token_len < self.minn_tokens_num:
                    raise Exception
                if token_len < self.maxn_tokens_num:
                    token = np.pad(token, (0, self.maxn_tokens_num - token_len))
                else:
                    raise Exception

                break

            except Exception as e:
                print("File {} cant load, choose another one".format(path))
                # print(e)
                path = np.random.choice(self.token_paths)
        return token, token_len, split_point, text_token_len, audio_token_len



class BPETokenizerWithAudioTokens():
    def __init__(self, bpe_tokenizer_file, audio_tokens_num=1024):
        self.audio_tokens_num = audio_tokens_num
        self.bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file=bpe_tokenizer_file)
        self.vocab_size = self.bpe_tokenizer.vocab_size + audio_tokens_num + 3 + 50 # <sep> <s> </s> <pad> and 50 placeholders
        print("Total vocab size: %d" % self.vocab_size)  # 2077
        self.sep_u2t = self.vocab_size - 3
        self.sep_t2u = self.vocab_size - 2
        self.eos = self.vocab_size - 1
        self.pad = 0

    def tokenize(self, sentence):
        return [item + self.audio_tokens_num + 1 for item in self.bpe_tokenizer.encode(sentence)] # pad and audio tokens



class AudioPathDataset(Dataset):
    def __init__(self, dir, cache_dir=None, filter=["wav", "mp3", "mp4", "m4a", "npy"]):
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        local_path = (
            f"{cache_dir}/{os.path.basename(dir)}"
            if cache_dir
            else os.path.basename(dir)
        )

        from ..utils.hdfs_tools import hdfs_cp

        if dir != local_path:
            hdfs_cp(dir, local_path, True)

        self.filter = filter
        self.list = [
            os.path.join(root, f)
            for root, _, files in os.walk(local_path)
            for f in files
            if self.is_audio(f)
        ]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

    def is_audio(self, fname):
        fname = fname.split(".")[-1]
        if fname in self.filter:
            return True
        return False




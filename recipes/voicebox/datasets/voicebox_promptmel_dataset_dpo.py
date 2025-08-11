import json
import logging
import pickle
import numpy as np
import torch
import pandas as pd
import io
import tqdm
import glob
import hashlib
import os
import soundfile as sf
from torch.utils.data import IterableDataset
from torchaudio.transforms import Resample
from tqdm.contrib.concurrent import thread_map, process_map

from recipes.voicebox.datasets.utils import (
    collate_1d,
    collate_2d,
)
from samantha.dataio.batching import BucketBatcher
from samantha.utils.distributed import get_local_rank, rank_zero_first

import librosa
import subprocess
import json
import random
import pyloudnorm as pyln
import requests
import functools
import argparse

logger = logging.getLogger(__name__)

def get_audio_channels(file_path):
    try:
        # 直接调用 ffprobe 命令
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=channels',
            '-of', 'json',
            file_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe 错误: {result.stderr}")
        
        # 解析 JSON 输出
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            return info['streams'][0]['channels']
        else:
            raise ValueError("未找到音频流")
            
    except Exception as e:
        print(f"错误: {e}")
        return None


def read_wav_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        audio_bytes = io.BytesIO(response.content)
        audio_data, sample_rate = sf.read(audio_bytes)
        
        return audio_data, sample_rate
    
    except requests.exceptions.RequestException as e:
        print(f"HTTP请求错误: {e}")
        return None, None
    except sf.LibsndfileError as e:
        print(f"音频解码错误: {e}")
        return None, None
    except Exception as e:
        print(f"未知错误: {e}")
        return None, None   

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

class PerferrenceDataset(object):
    def __init__(self, dir_path):
        self.cache_dir = dir_path
        pair_dataset_urls = []
        num_of_vaild_pair = 0
        for csv_path in glob.glob(f"{dir_path}/*.csv"):
            df = pd.read_csv(csv_path)
            if "AB" in csv_path:
                # A-B comparison
                for idx, row in df.iterrows():
                    compare_urls = []
                    left_url = None
                    right_url = None
                    _score = None
                    for j, colname in enumerate(df.columns):
                        if '左' in colname:
                            if isinstance(row[j], str) and len(row[j]) > 10:
                                left_url = row[j].lstrip('"').rstrip('"')
                        if '右' in colname:
                            if isinstance(row[j], str) and len(row[j]) > 10:
                                right_url = row[j].lstrip('"').rstrip('"')
                        if '音质' in colname or '整体听感' in colname or 'sound quality' in colname:
                            _score = row[j]
                    if _score is None or left_url is None or right_url is None:
                        print(csv_path, df.columns)
                        continue
                    if not isinstance(_score, str):
                        continue
                    if '左好' in _score:
                        compare_urls.append((left_url, 1))
                        compare_urls.append((right_url, 0))
                    elif '右好' in _score:
                        compare_urls.append((left_url, 0))
                        compare_urls.append((right_url, 1))
                    else:
                        pass
                    if len(compare_urls) > 1:
                        pair_dataset_urls.append(compare_urls)
                        n = len(compare_urls)
                        num_of_vaild_pair += (n * (n - 1) // 2)
            else:
                # Multiple models in a row
                for _, row in df.iterrows():
                    compare_urls = []
                    scores = []
                    entered = False
                    for j, colname in enumerate(df.columns):
                        if 'model' in colname:
                            entered = True
                            if isinstance(row[j], str) and len(row[j]) > 10:
                                audio_url = row[j]
                                score = row[j + 1]
                                try:
                                    score = float(score)
                                    compare_urls.append((audio_url, score))
                                    scores.append(score)
                                except:
                                    print(f"error score:\n{csv_path=}\n{colname=}\n{score=}\n")
                                    continue
                    # avoid case where all model get same score
                    if len(compare_urls) > 1 and len(set(scores)) > 1:
                        pair_dataset_urls.append(compare_urls)
                        n = len(compare_urls)
                        num_of_vaild_pair += (n * (n - 1) // 2)
                    if not entered:
                        print(f"Read File Error:{csv_path=}\n{df.columns=}")

        num_valid_rows = len(pair_dataset_urls)
        num_unique_audio = len(set([url for urls in pair_dataset_urls for url, _ in urls]))
        
        print(f"Number of compared rows: {num_valid_rows}")
        print(f"Number of unique audio: {num_unique_audio}")
        print(f"Number of valid pairs: {num_of_vaild_pair}")

        all_urls_and_scores = [item for sublist in pair_dataset_urls for item in sublist]

        results = thread_map(self._read_audio_wrapper, all_urls_and_scores, desc="Loading audio", chunksize=1)

        pair_dataset = []
        idx = 0
        for compare_urls in pair_dataset_urls:
            compare = []
            for _ in compare_urls:
                audio_info = results[idx]
                if audio_info is not None:
                    compare.append(audio_info)
                idx += 1
            if compare:
                pair_dataset.append(compare)

        self.pair_dataset = pair_dataset
        seed = get_local_rank()
        random.seed(42+seed)
        print(f"Load dataset done, n={len(pair_dataset)}")

    def _read_audio_wrapper(self, url_and_score):
        audio_url, score = url_and_score
        md5 = hashlib.md5(audio_url.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{md5}.wav")
        if not os.path.exists(cache_path):
            audio_data, sample_rate = read_wav_url(audio_url)
            if audio_data is None:
                return None
            sf.write(cache_path, audio_data, sample_rate)

        return {
            "cache_path": cache_path,
            "score": score,
        }

    def sample(self):
        compare = None
        while compare is None:
            idx = random.randint(0, len(self.pair_dataset) - 1)
            utt_id = str(idx)
            compare = self.pair_dataset[idx]

        win_info = None
        lose_info = None
        
        while win_info is None or lose_info is None:
            left = random.choice(compare)
            right = random.choice(compare)
            if left["score"] == right["score"]:
                win_info = None
                lose_info = None
            elif left["score"] > right["score"]:
                win_info = left
                lose_info = right
            else:
                win_info = right
                lose_info = left

        return utt_id, win_info, lose_info

def load_check_audio(file):
    groundtruth_file = file.replace(".recon.wav", ".wav")
    if os.path.exists(groundtruth_file):
        if get_audio_channels(groundtruth_file) == 2 and librosa.get_duration(filename=groundtruth_file) > 10:
            return file, groundtruth_file
    return None, None

class ReconPairDataset(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        index_file = os.path.join(dir_path, "index.txt")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                lines = f.readlines()
                recon_files = [line.strip() for line in lines]
            groundtruth_files = [file.replace(".recon.wav", ".wav") for file in recon_files]
        else:
            recon_files = []
            groundtruth_files = []
            print(f"Loading from {dir_path}/**/*.recon.wav")
            audios = glob.glob(f"{dir_path}/**/*.recon.wav", recursive=True)
            print(f"Get {len(audios)} files")
            for file, groundtruth_file in thread_map(load_check_audio, audios, desc="Loading audio", chunksize=1):
                if file is not None:
                    recon_files.append(file)
                    groundtruth_files.append(groundtruth_file)
            open(index_file, "w").write("\n".join(recon_files))
        print(f"Number of valid pair_dataset: {len(recon_files)}")
        self.recon_files = recon_files
        self.groundtruth_files = groundtruth_files

    def sample(self):
        idx = random.randint(0, len(self.recon_files) - 1)
        utt_id = str(idx)
        win_info = {
            "cache_path": self.groundtruth_files[idx],
            "score": 1,
        }
        lose_info = {
            "cache_path": self.recon_files[idx],
            "score": 0,
        }
        return utt_id, win_info, lose_info




class VoiceBoxParquetDataset(IterableDataset):
    def __init__(self,
        data_id=None,
        bn_audio_freq = 44100,
        token_audio_freq = 24000,
        umm_frame_rate=25,
        audio_norm=False,
        batcher_config=None,
        drop_last=False,
        dataset_cls="PerferrenceDataset", # or ReconPairDataset
        **kwargs,
    ):
        self.data_id = data_id
        self.dataset = eval(dataset_cls)(data_id)

        self.umm_frame_rate = umm_frame_rate
        self.bn_audio_freq = bn_audio_freq
        self.pipelines = []

        if audio_norm:
            self.meter = pyln.Meter(bn_audio_freq)
        else:
            self.meter = None

        self.audio_resampler = Resample(orig_freq=bn_audio_freq, new_freq=token_audio_freq)
        self.batcher = BucketBatcher(**batcher_config)
        self.drop_last = drop_last

    def get_meta_obj(self, sample):
        meta_obj = json.loads(sample["meta"])
        while not isinstance(meta_obj, dict):
            meta_obj = json.loads(meta_obj)
        return meta_obj

    def get_bn(self, sample, key="bns"):
        bn = pickle.loads(sample[key])
        bn = torch.from_numpy(bn)
        bn = torch.transpose(bn, 0, 1)
        return bn

    def process(self, utt_id, audio_info, slice_start_ratio, slice_end_ratio):
        # load wav
        wav, sr = sf.read(audio_info["cache_path"])
        def log_sample(sample, wav):
            try:
                print('Logging sample wav / dataset / url', wav.shape, sample['__dataset_name__'], sample['__data_url__'])
            except: pass

        data_dict = dict()

        if wav.shape[0] == 0:
            print(f'Empty wav duration {wav.shape}')
            return None

        if len(wav.shape) != 2 or wav.shape[-1] != 2:
            print(f'Invalid wav src sample shape. Perhaps mono? {wav.shape}')
            return None

        if sr != self.bn_audio_freq:
            print('Invalid src sample rate', sr, self.bn_audio_freq)
            # raise Exception("Src sample rate does not equal")
            return None

        wav = wav.astype(np.float32)
        wav = torch.FloatTensor(wav)
        wav = wav.transpose(0, 1) # L, CH -> CH, L
        sr = self.bn_audio_freq
        umm_hz = self.umm_frame_rate # UMM token frame rate. 

        dataset_name = f"DPO_{self.data_id}"

        # slice wav base on slice ratio
        slice_start = int(slice_start_ratio * wav.shape[-1])
        slice_end = int(slice_end_ratio * wav.shape[-1])
        wav = wav[:, slice_start:slice_end]

        # For now, truncate to nearest 25hz divisible
        max_wav_len = int(int(wav.shape[-1] / sr * umm_hz) * sr / umm_hz)
        wav = wav[:, :max_wav_len]
        data_dict['wav'] = wav

        try:
            wav_24k = self.audio_resampler(wav.mean(0))
        except Exception as e:
            print(f'Error resampling wav {wav.shape}', e)
            return None
        data_dict['wav_24k'] = wav_24k
        data_dict['utt_id'] = utt_id
        data_dict['score'] = audio_info["score"]

        return data_dict

    def __iter__(self):
        while True:
            utt_id, win_info, lose_info = self.dataset.sample()
            slice_start_ratio = random.uniform(0, 0.8)
            slice_end_ratio = random.uniform(slice_start_ratio+0.1, 1.0)
            win_data = self.process(utt_id, win_info, slice_start_ratio, slice_end_ratio)
            lose_data = self.process(utt_id, lose_info, slice_start_ratio, slice_end_ratio)
            if win_data is None or lose_data is None:
                continue

            data_dict = dict()
            data_dict['win_wav'] = win_data['wav']
            data_dict['lose_wav'] = lose_data['wav']
            if "wav_24k" in win_data:
                data_dict['win_wav_24k'] = win_data['wav_24k']
                data_dict['lose_wav_24k'] = lose_data['wav_24k']
            data_dict['win_utt_id'] = win_data['utt_id']
            data_dict['lose_utt_id'] = lose_data['utt_id']
            data_dict['win_score'] = win_data['score']
            data_dict['lose_score'] = lose_data['score']

            batch = self.batcher.collate_batch(data_dict)
            if batch:
                yield batch

        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch

class VoiceBoxCollator(object):
    def __init__(self):
        self.tokenizer_pad = 16384

    def __call__(self, batches):
        results = []
        for item in batches:
            if item is not None:
                results.append(item)
        if len(results) == 0:
            return None
        ret_dict = {}

        wav_lens = [b['win_wav'].shape[-1] for b in batches] + [b['lose_wav'].shape[-1] for b in batches]
        wav_lens = torch.from_numpy(np.array(wav_lens))
        max_wav_len = wav_lens.max().item()
        wav = collate_2d(
            [b["win_wav"] for b in batches] + [b["lose_wav"] for b in batches],
            pad_idx=0.0,
            max_len=max_wav_len
        )
        ret_dict['wav'] = wav # B x CH x L
        ret_dict['wav_lens'] = wav_lens
        ret_dict["wav_mask"] = sequence_mask(wav_lens,
                  max_len=ret_dict["wav"].shape[-1])

        if 'win_wav_24k' in batches[0]:
            wav_24k_lens = [b['win_wav_24k'].shape[-1] for b in batches] + [b['lose_wav_24k'].shape[-1] for b in batches]
            wav_24k = collate_1d(
                [b["win_wav_24k"] for b in batches] + [b["lose_wav_24k"] for b in batches],
                pad_idx=0.0,
                max_len=max(wav_24k_lens)
            )
            ret_dict['wav_24k'] = wav_24k
        
        utt_ids = [b['win_utt_id'] for b in batches] + [b['lose_utt_id'] for b in batches]
        
        ret_dict["utt_id"] = utt_ids

        return ret_dict

def test_PerferrenceDataset(data_id):
    dataset = PerferrenceDataset(data_id)
    for i in range(10):
        utt_id, win_info, lose_info = dataset.sample()
        print(utt_id, win_info["cache_path"], lose_info["cache_path"], win_info["score"], lose_info["score"])

def test_ReconPairDataset(data_id = "/mnt/bn/lab-speechsv-zhongyi-lq/vocoder/ar_augment_stereov3/"):
    dataset = ReconPairDataset(data_id)
    for i in range(10):
        utt_id, win_info, lose_info = dataset.sample()
        print(utt_id, win_info["cache_path"], lose_info["cache_path"], win_info["score"], lose_info["score"])

def test_VoiceBoxParquetDataset(data_id = "/mnt/bn/lab-speechsv-zhongyi-lq/vocoder/ar_augment_stereov3/"):
    bn_hop_size = 900
    buckets = list(range(49, 44100000, 49))
    bn_dim = 64
    bn_audio_freq = 44100
    maximum_bucket_size = 441000

    batcher_config = { 
        "buckets": buckets,  
        "dynamic_batch": True,
        "maximum_bucket_size": maximum_bucket_size,
        "length_fn": lambda x: x["win_wav"].shape[-1]*2 if "win_wav" in x else 0
    }

    collate_fn = VoiceBoxCollator()

    dataset = VoiceBoxParquetDataset(data_id=data_id, 
                              max_length=60,
                              drop_last=False,
                              umm_frame_rate=25,
                              bn_audio_freq=bn_audio_freq,
                              dataset_cls="ReconPairDataset",
                              batcher_config=batcher_config
                              )
    

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=None,
        collate_fn=collate_fn,
        pin_memory=True
    )

    import time
    prev = time.time()
    for idx, item in enumerate(dataloader):
        cur = time.time()
        print(f"==== {idx} {cur-prev}")
        prev = cur
        for key in item:
            if torch.is_tensor(item[key]):
                print(f"\t {key} -> shape={item[key].shape} is_contiguous={item[key].is_contiguous()}")
            else:
                print(f"\t {key} -> {item[key]}")
        if idx > 10:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, default="/mnt/bn/lab-speechsv-zhongyi-lq/vocoder/ar_augment_stereov3/")
    args = parser.parse_args()
    with rank_zero_first():
        # test_ReconPairDataset(args.data_id)
        test_PerferrenceDataset(args.data_id)
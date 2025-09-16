import json
import logging
import pickle
import math
import copy
import numpy as np
import torch
import librosa
from typing import Optional
import io
import soundfile as sf
from torch.utils.data import IterableDataset
from torchaudio.transforms import Resample
from torchaudio.functional import resample

from recipes.voicebox.datasets.utils import (
    collate_1d,
    collate_2d,
)
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.ra_wds import WebDataset

import random
import torchaudio
import pyloudnorm as pyln

def read_wav_sf(sample):
    if "audio_44k" in sample: # special case for data_id=16944
        byte_stream = io.BytesIO(sample["audio_44k"])
    else:
        byte_stream = io.BytesIO(sample["wav"])
    with sf.SoundFile(byte_stream) as wav_file:
        # sample_rate = wav_file.samplerate
        # num_channels = wav_file.channels
        audio_data = wav_file.read()

    return audio_data

def read_wav2(sample):
    wav = np.frombuffer(sample["wav"][44:], dtype=np.int16)
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    elif wav.dtype in [np.float32, np.float64]:
        wav = wav
    else:
        raise Exception("Not support data type: {}".format(wav.dtype))
    return wav

def read_wav_torch(item) -> tuple[torch.Tensor, int]:
    if "wav_path" in item: # support meta.lst from local path
        wav_npy, sr = torchaudio.load(item["wav_path"])
    elif "audio_44k" in item: # special case for data_id=16944
        wav_npy, sr = torchaudio.load(io.BytesIO(item["audio_44k"]))
    else:
        wav_npy, sr = torchaudio.load(io.BytesIO(item["wav"]))
    return wav_npy, sr

def read_wav_librosa(sample, sr: Optional[int] = None, mono: bool = False) -> np.ndarray:
    byte_stream = io.BytesIO(sample["wav"])
    wav, sr = librosa.load(byte_stream, sr=sr, mono=mono)
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav



logger = logging.getLogger(__name__)

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask

class VoiceBoxParquetDataset(IterableDataset):
    def __init__(self,
        data_id=None,
        drop_last=False,
        batcher_config=None,
        umm_frame_rate=25,
        bn_config=None,
        token_audio_freq = 24000,
        bn_audio_freq = 44100,
        shuffle_buffer_size=100,
        allow_mono=False,
        allow_resample=False
    ):
        self.wds = (
            ParquetDataset(data_id=data_id, resampled=True, extra_fields_in_data=["audio_44k"])
            .shuffle(shuffle_buffer_size)
            .map(self.process)
        )

        self.bn_audio_freq = bn_audio_freq
        self.token_audio_freq = token_audio_freq

        self.pipelines = []
        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)
        self.umm_frame_rate = umm_frame_rate
        self.max_audio_samples = max(batcher_config["buckets"])
        self.min_audio_samples = min(batcher_config["buckets"])

        self.bn_config = bn_config

        self.audio_resampler = Resample(orig_freq=bn_audio_freq, new_freq=token_audio_freq)
        self.allow_mono = allow_mono
        self.allow_resample = allow_resample
        

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

    def process(self, sample):
        # load wav
        # meta_obj = self.get_meta_obj(sample)
        def log_sample(sample, wav):
            try:
                print('Logging sample wav / dataset / url', wav.shape, sample['__dataset_name__'], sample['__data_url__'])
            except: pass

        data_dict = dict()

        try:
            wav, sr = read_wav_torch(sample) # CH, L
        except Exception as e:
            print("Failed to load audio")
            return None

        if wav.shape[-1] == 0:
            print(f'Empty wav duration {wav.shape}')
            del sample
            return None

        if len(wav.shape) != 2 or wav.shape[0] != 2:
            if self.allow_mono and len(wav.squeeze().shape)==1: 
                wav = torch.stack([wav.squeeze(), wav.squeeze()], dim=0)
            else:
                print(f'Invalid wav src sample shape. Perhaps mono? {wav.shape}')
                log_sample(sample, wav)
                del sample
                return None

        if sr != self.bn_audio_freq:
            if self.allow_resample:
                wav = resample(wav, sr, self.bn_audio_freq)
            else:
                print('Invalid src sample rate', sr, self.bn_audio_freq)
                log_sample(sample, wav)
                # raise Exception("Src sample rate does not equal")
                del sample
                return None

        sr = self.bn_audio_freq
        umm_hz = self.umm_frame_rate # UMM token frame rate. 

        # truncate short audio. mulan cannot handle under 10
        if wav.shape[-1] < self.min_audio_samples:
            print(f'Wav too short {wav.shape}')
            return None
        
        ## handle long wavs here. truncate to max audio - 60 seconds
        max_audio_samples = self.max_audio_samples
        if wav.shape[-1] > max_audio_samples:
            if "umm_token" in sample: # umm already extracted. do not do random croping for now
                # TODO: enable random cropping
                wav = wav[..., :max_audio_samples]
            else:
                start = random.randint(0, wav.shape[-1] - max_audio_samples + 1)
                # min_audio_samples = self.min_audio_samples
                wav_duration = wav.shape[-1] / sr
                min_audio_samples = (wav_duration // 60 * 60) * sr # truncate to minute
                min_audio_samples = min(min_audio_samples, max_audio_samples)
                end = random.randint(start + min_audio_samples, start + max_audio_samples)
                wav = wav[..., start:end]

        ## UMM pre-extracted features
        if "umm_token" in sample:
            # WARNING: may be a mismatch in UMM token alignment. UMM is a half token short. Wav length is usually x.5 umm token length
            umm_token = torch.as_tensor(pickle.loads(sample["umm_token"]), dtype=torch.long)
            if start is not None:
                umm_token = umm_token[:, start // sr * umm_hz : end // sr * umm_hz]

            umm_token_len = min(int(wav.shape[-1] / sr * umm_hz), umm_token.shape[-1]) # get minimum token length
            max_wav_len = int(umm_token_len * sr / umm_hz)
            
            # print('Umm duration', umm_token.shape, wav.shape, umm_hz, umm_token_len, max_wav_len)

            data_dict["token"] = umm_token[..., :umm_token_len]
            data_dict['wav'] = wav[:, :max_wav_len].clone()
        else: # Reasample to wav_24k for OTF umm extraction
            # For now, truncate to nearest 25hz divisible
            # max_wav_len = int(int(wav.shape[-1] / sr * umm_hz) * sr / umm_hz)
            max_wav_len = int(wav.shape[-1] / sr) * sr
            wav = wav[:, :max_wav_len]
            data_dict['wav'] = wav.clone()

            try:
                wav_24k = self.audio_resampler(wav.mean(0))
            except Exception as e:
                print(f'Error resampling wav {wav.shape}', e)
                log_sample(sample, wav)
                del sample
                del wav
                return None
            data_dict['wav_24k'] = wav_24k

        utt_id = sample["__key__"]
        data_dict['utt_id'] = utt_id
        del sample
        del wav

        return data_dict

    def __iter__(self):
        self.batcher.clear()
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
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

        wav_lens = [b['wav'].shape[-1] for b in batches]
        wav_lens = torch.from_numpy(np.array(wav_lens))
        max_wav_len = wav_lens.max().item()
        wav = collate_2d(
            [b["wav"] for b in batches],
            pad_idx=0.0,
            max_len=max_wav_len
        )
        ret_dict['wav'] = wav # B x CH x L
        ret_dict['wav_lens'] = wav_lens
        ret_dict["wav_mask"] = sequence_mask(wav_lens,
                  max_len=ret_dict["wav"].shape[-1])

        
        if 'wav_24k' in batches[0]:
            wav_24k_lens = [b['wav_24k'].shape[-1] for b in batches]
            wav_24k = collate_1d(
                [b["wav_24k"] for b in batches],
                pad_idx=0.0,
                max_len=max(wav_24k_lens)
            )
            ret_dict['wav_24k'] = wav_24k
        if 'token' in batches[0]:
            token_lens = [b['token'].shape[0] for b in batches]
            max_token_len = max(token_lens)
            token = collate_1d(
                [b["token"] for b in batches],
                pad_idx=self.tokenizer_pad,
                max_len=max_token_len
            )
            ret_dict['token'] = token
        
        utt_ids = [b['utt_id'] for b in batches]
        
        ret_dict["utt_id"] = utt_ids

        return ret_dict


if __name__ == "__main__":

    # umm 25hz + vocoder 49hz 
    data_id = 2225
    bn_hop_size = 900
    buckets = list(range(49, 3100, 49))
    bn_dim = 64
    bn_audio_freq = 44100


    maximum_bucket_size = 30000

    batcher_config = { 
        "buckets": buckets,  
        "dynamic_batch": True,
        "maximum_bucket_size": maximum_bucket_size,
        "length_fn": lambda x: x["bn"].shape[1]
    }


    bn_config = {
        "hop_size": bn_hop_size,
        "bn_dim": bn_dim,
        "bn_norm_std": 1,
        "bn_norm_mean": 0,
        "silence_bn_path": "recipes/voicebox/vocoder/silence_wvae/v3_40hz/silence_wvae.npy"
    }

    collate_fn = VoiceBoxCollator()

    dataset = VoiceBoxParquetDataset(data_id=data_id, 
                              batcher_config=batcher_config, 
                              max_length=60,
                              drop_last=False,
                              bn_config=bn_config,
                              umm_frame_rate=25,
                              bn_audio_freq=bn_audio_freq,
                              )
    

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=dataset,
        num_workers=15,
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
        # break

import json
import logging
import os
import pickle
import random
import sys
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchaudio.transforms import Resample

from recipes.voicebox.datasets.utils import (
    Masking,
    PhoneToId,
    collate_1d,
    collate_2d,
    get_duration_frames_wds,
)
from recipes.voicebox.utils.infer_utils import save_wav
from recipes.voicebox.vocoder.BigVGAN.meldataset import mel_spectrogram, mel_spectrogram_unimelgan
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.ra_wds import WebDataset
import librosa

logger = logging.getLogger(__name__)

def trim_silence(wav):
    """
    Trim leading and trailing silence
    """
    # These params are separate and tunable per dataset.
    
    origin_wav_len = len(wav)
    wav = np.pad(wav, (5400, 5400))

    unused_trimed, index = librosa.effects.trim(
        wav, top_db=30, frame_length=512, hop_length=128
    )
    # num_sil_samples = int(8 * 300)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - 3200, 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + 5400, len(wav))

    trimmed = wav[start_idx:stop_idx]

    head_trim_nums = 5400 - start_idx
    tail_trim_nums = stop_idx - 5400 - origin_wav_len
    
    return trimmed, head_trim_nums, tail_trim_nums

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


class VoiceBoxDataset(IterableDataset):
    def __init__(self,
        wds_urls,
        drop_last=False,
        batcher_config=None,
        max_length=30,
        mel_config=None,
        mel_norm_mean=-5.8843,
        mel_norm_std=2.2615,
        mel_padding=-2,
        mask_p_drop_x=0.3,
        p_drop_audio_frames=(0.7, 1.0),
        mask_use_alignment=True,
        use_f0=False,
        wav_amp_aug=None
        ):

        self.wds = (
            WebDataset(
                urls=wds_urls,
                resampled=True,
                skip_instance_cache=True,
            )
            .decode()
            .shuffle(2048)
            .map(self.process)
        )

        self.max_wav_len = int(max_length * 24000)
        self.phone2id = PhoneToId()

        self.pipelines = []
        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)
        self.mel_config = mel_config
        self.mel_norm_mean = mel_norm_mean
        self.mel_norm_std = mel_norm_std
        self.use_f0 = use_f0

        self.wav_amp_aug = wav_amp_aug

        self.utt2duration = {}
        self.shot = 0
        self.total = 0

        if mel_config['sampling_rate'] != 24000: # audio in web-dataset is 24k
            self.audio_resampler = Resample(orig_freq=16000, new_freq=mel_config['sampling_rate'])
        else:
            self.audio_resampler = None

        self.hop_ms = mel_config['hop_size'] / mel_config['sampling_rate']

        self.mask_use_alignment = mask_use_alignment
        self.masking = Masking(p_drop_x=mask_p_drop_x, 
                p_drop_audio_frames=p_drop_audio_frames,
                padding_value=mel_padding)

    
    def get_duration_wds(self, utt_id, alignments, phonemes, mel_len):
        if alignments is not None and len(alignments) > 0:
            duration_frames = get_duration_frames_wds(alignments, phonemes, self.hop_ms)
            if duration_frames is not None and abs(sum(duration_frames) - mel_len) > 2:
                duration_frames = None
            if duration_frames is not None:
                pad_len = sum(duration_frames) - mel_len
                duration_frames[-1] = duration_frames[-1] - pad_len
                if duration_frames[-1] < 0:
                    duration_frames = None
        else:
            duration_frames = None
        self.utt2duration[utt_id] = duration_frames
        self.total += 1
        if duration_frames is not None:
            self.shot += 1
        if self.total % 1000 == 0:
            shot_rate = round(float(self.shot) / float(self.total) * 100, 2)
            logging.info(f"{shot_rate}%")

        return duration_frames


    def process(self, sample):
        # load wav
        wav = np.frombuffer(sample["wav"][44:], dtype=np.int16)
        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        elif wav.dtype in [np.float32, np.float64]:
            wav = wav
        else:
            raise Exception("Not support data type: {}".format(wav.dtype))
        if len(wav.shape) >= 2:
            wav = wav[0]
        wav = wav.astype(np.float32)
        wav = torch.FloatTensor(wav)
        if self.audio_resampler is not None:
            wav = self.audio_resampler(wav)

        if len(wav) > self.max_wav_len:
            wav_start = np.random.randint(len(wav) - self.max_wav_len)
            wav = wav[wav_start:wav_start+self.max_wav_len]

        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        if self.wav_amp_aug is not None:
            wav_aug_scale = np.random.rand() * (self.wav_amp_aug["max"] - self.wav_amp_aug["min"]) + self.wav_amp_aug["min"]
            wav = wav * wav_aug_scale

        mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)
        mel = (mel - self.mel_norm_mean) / self.mel_norm_std

        data_dict = dict()

        # load phone seq & duration
        text = sample["text"]
        url = sample["__url__"]
        lab = sample["labels"]
        utt_id = sample["__key__"]

        labels = lab.decode()
        if labels is None:
            return None
        labels = list(filter(lambda x: x != "", labels.split('\n')))

        if len(labels) < 2:
            return None
        if len(labels[-1].split('\t')) == 2:
            last_duration = labels[-1].split('\t')[-1]
            last_line = labels[-2]
            labels = labels[:-2]
            last_line = '\t'.join(last_line.split('\t')[:-1] + [last_duration])
            labels.append(last_line)

        try:
            text_id_phones_tones = self.phone2id.convert_tacolab_to_text_id(labels)
        except Exception as e:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id exception: {e}")
            print(f"{utt_id} convert_tacolab_to_text_id exception: {e}")
            return None
        
        if text_id_phones_tones is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
            return None
        else:
            text_id, phones, tones, word_segs, alignments = text_id_phones_tones

        duration = self.get_duration_wds(utt_id, alignments, phones, mel.shape[1])

        if duration is None:
            return None
        try:
            text_id = np.repeat(text_id, duration, axis=1)
            assert text_id.shape[1] == mel.shape[1], f"sum duration {sum(duration)} vs {mel.shape[1]}"
        except Exception as e:
            print(e)
            return None
        data_dict['duration_ori'] = duration # for masking methods
        data_dict["duration"] = duration

        data_dict['utt_id'] = utt_id
        data_dict['mel'] = mel.transpose(1, 0)
        data_dict["lab"] = lab

        # masking.
        data_dict = self.masking.masking(data_dict) # update ctx & ctx_mask
        data_dict['mel_ctx'] = data_dict['mel_ctx'].transpose(0, 1)
        data_dict['mel'] = data_dict['mel'].transpose(0, 1)
        data_dict['wav'] = wav
        data_dict["text"] = text
        #if data_dict["mel"].shape[1] > 1000:
        #    print(data_dict["mel"].shape)

        return data_dict

    def __iter__(self):
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch


class VoiceBoxParquetDataset(IterableDataset):
    def __init__(self,
        data_id=None,
        drop_last=False,
        batcher_config=None,
        max_length=30,
        mel_config=None,
        mel_norm_mean=-5.8843,
        mel_norm_std=2.2615,
        mel_padding=-2,
        umm_hop_size=600,
        mask_p_drop_x=0.3,
        p_drop_audio_frames=(0.7, 1.0),
        mask_use_alignment=True,
        wav_divide=2400,
        use_text=False,
        text_drop_rate=0,
        wav_amp_aug=None,
        token_spec_aug=None,
        use_bn=False,
        bn_config=None,
        mel_vocoder_type="hifigan",
        use_phone_lang=False,
        adapt_bigmusic_data=True,
    ):
        self.wds = (
            ParquetDataset(data_id=data_id, resampled=True)
            .shuffle(2048)
            .map(self.process)
        )

        self.max_length = max_length
        self.max_wav_len = int(max_length * 24000)
        self.phone2id = PhoneToId()

        self.wav_amp_aug = wav_amp_aug
        self.token_spec_aug = token_spec_aug

        self.pipelines = []
        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)
        self.mel_config = mel_config
        self.mel_norm_mean = mel_norm_mean
        self.mel_norm_std = mel_norm_std
        self.umm_hop_size = umm_hop_size
        self.umm_hz = 24000 // umm_hop_size

        self.use_text = use_text
        self.text_drop_rate = text_drop_rate
        self.wav_divide = wav_divide
        self.use_bn = use_bn
        self.bn_config = bn_config
        self.mel_vocoder_type = mel_vocoder_type

        self.use_phone_lang = use_phone_lang

        self.shot = 0
        self.total = 0

        self.audio_resampler = {}

        self.hop_ms = mel_config['hop_size'] / mel_config['sampling_rate']
        self.masking = Masking(p_drop_x=mask_p_drop_x, 
                p_drop_audio_frames=p_drop_audio_frames,
                padding_value=mel_padding if not self.use_bn else bn_config["bn_padding"])
        self.mask_use_alignment = mask_use_alignment
        
        if self.use_bn:
            self.bn_hz = 24000 // bn_config["hop_size"]
            lcm_umm_bn = math.lcm(self.umm_hz, self.bn_hz)
            self.lcm_umm_factor = lcm_umm_bn // self.umm_hz
            self.lcm_bn_factor = lcm_umm_bn // self.bn_hz
        
        self.adapt_bigmusic_data = adapt_bigmusic_data

    def get_duration_wds(self, utt_id, alignments, phonemes, mel_len):
        if alignments is not None and len(alignments) > 0:
            duration_frames = get_duration_frames_wds(alignments, phonemes, self.hop_ms)
            if duration_frames is not None and abs(sum(duration_frames) - mel_len) > 2:
                # print(f"{sum(duration_frames)}, {mel_len=}")
                duration_frames = None
            if duration_frames is not None:
                pad_len = sum(duration_frames) - mel_len
                duration_frames[-1] = duration_frames[-1] - pad_len
                if duration_frames[-1] < 0:
                    duration_frames = None
        else:
            duration_frames = None
        self.total += 1
        if duration_frames is not None:
            self.shot += 1
        if self.total % 1000 == 0:
            shot_rate = round(float(self.shot) / float(self.total) * 100, 2)
            logging.info(f"{shot_rate}%")

        return duration_frames

    def get_meta_obj(self, sample):
        meta_obj = json.loads(sample["meta"])
        while not isinstance(meta_obj, dict):
            meta_obj = json.loads(meta_obj)
        return meta_obj

    def get_bn(self, sample, key="bns"):
        bn = pickle.loads(sample[key])
        bn = torch.from_numpy(bn)
        out_dim = bn.shape[1] // 2
        m, logs = torch.split(bn, out_dim, dim=-1)
        bn = m + torch.randn_like(m) * torch.exp(logs)
        return bn

    def process(self, sample):
        # load wav
        meta_obj = self.get_meta_obj(sample)
        data_dict = dict()

        # for key in sample:
        #     if torch.is_tensor(sample[key]):
        #         print(f"\t {key} -> shape={sample[key].shape}")
        #     else:
        #         print(f"\t {key} -> {type(sample[key])}")

        if "vc_bns" in sample:
            vocal_included = True
        else:
            vocal_included = False
        
        if self.use_bn:
            data_dict["token"] = torch.as_tensor(pickle.loads(sample["umm_token"]), dtype=torch.long)

            bn = self.get_bn(sample)
            assert bn.shape[1] == 64
            if "vc_bns" in sample:
                vc_bn = self.get_bn(sample, "vc_bns")
                assert vc_bn.shape[1] == 64

            if self.bn_hz == self.umm_hz:
                max_bn_len = min(bn.shape[0], data_dict["token"].shape[0])
                max_umm_len = max_bn_len
            else:
               # align BN with UMM
                if (bn.shape[0] / self.bn_hz) < (data_dict["token"].shape[0] / self.umm_hz):
                    max_lcm_len = int(bn.shape[0] * self.lcm_bn_factor) 
                else:
                    max_lcm_len = int(data_dict["token"].shape[0] * self.lcm_umm_factor) 
                max_lcm_len = max_lcm_len - (max_lcm_len % (self.lcm_umm_factor*self.lcm_bn_factor))

                max_bn_len = max_lcm_len // self.lcm_bn_factor
                max_umm_len = max_lcm_len // self.lcm_umm_factor

            if "vc_bns" in sample:
                # vc_bn might have shorter length than bn because leading and trailing silence will be trimmed in feature extraction
                data_dict["vc_bn"] = (vc_bn - self.bn_config['bn_norm_mean']) / self.bn_config['bn_norm_std']
            data_dict["bn"] = bn[:max_bn_len, :]
            data_dict["bn"] = (data_dict["bn"] - self.bn_config['bn_norm_mean']) / self.bn_config['bn_norm_std']
        else:
            wav = np.frombuffer(sample["wav"][44:], dtype=np.int16)
            if wav.dtype == np.int16:
                wav = wav / 32768.0
            elif wav.dtype == np.int32:
                wav = wav / 2_147_483_648.0
            elif wav.dtype in [np.float32, np.float64]:
                wav = wav
            else:
                raise Exception("Not support data type: {}".format(wav.dtype))
            if len(wav.shape) >= 2:
                wav = wav[0]

            wav = wav.astype(np.float32)
            wav = torch.FloatTensor(wav)

            if sample["src_sample_rate"] != self.mel_config['sampling_rate']:
                src_sr = sample["src_sample_rate"]
                if src_sr not in self.audio_resampler.keys():
                    self.audio_resampler[src_sr] = Resample(orig_freq=src_sr, new_freq=self.mel_config['sampling_rate'])
                wav = self.audio_resampler[src_sr](wav)

            if wav.shape[0] > self.max_wav_len or wav.shape[0] < 12000: 
                return None

            scale = max(0.001, torch.max(torch.abs(wav)))
            wav = wav / scale * 0.95
            if self.wav_amp_aug is not None and self.wav_amp_aug["use"]:
                wav_aug_scale = np.random.rand() * (self.wav_amp_aug["max"] - self.wav_amp_aug["min"]) + self.wav_amp_aug["min"]
                wav = wav * wav_aug_scale

            # handle umm & mel length
            crop_wav_len = wav.shape[0] % self.wav_divide
            max_wav_len = wav.shape[0] - crop_wav_len 
            max_mel_len = max_wav_len // self.mel_config["hop_size"]
            max_umm_len = max_wav_len // self.umm_hop_size

            if self.mel_vocoder_type == "unimelgan":
                mel = mel_spectrogram_unimelgan(wav.unsqueeze(0), **self.mel_config).squeeze(0)
            else:
                mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)

            mel = (mel - self.mel_norm_mean) / self.mel_norm_std
            data_dict['mel'] = mel.transpose(1, 0)

        # load phone seq & duration
        text = sample["text"]
        lab = meta_obj.get("labels", "")
        utt_id = sample["__key__"]

        labels = lab # .decode()
        if not self.adapt_bigmusic_data:
            if labels is None:
                return None
            labels = list(filter(lambda x: x != "", labels.split('\n')))
            if len(labels) < 2:
                return None
            if len(labels[-1].split('\t')) == 2:
                last_duration = labels[-1].split('\t')[-1]
                last_line = labels[-2]
                labels = labels[:-2]
                last_line = '\t'.join(last_line.split('\t')[:-1] + [last_duration])
                labels.append(last_line)
            try:
                text_id_phones_tones = self.phone2id.convert_tacolab_to_text_id(labels)
            except Exception as e:
                logger.warning(f"{utt_id} convert_tacolab_to_text_id exception: {e}")
                return None
            if text_id_phones_tones is None:
                logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
                return None
            else:
                text_id, phones, tones, word_segs, alignments = text_id_phones_tones

            if self.mask_use_alignment:
                duration = self.get_duration_wds(utt_id, alignments, phones, mel.shape[1])
                if duration is None:
                    return None
                data_dict['duration_ori'] = duration # for masking methods
                data_dict["duration"] = duration
        
        if self.use_text:
            text_id = np.concatenate([text_id, np.ones([text_id.shape[0], 1])], axis=-1)
            if self.text_drop_rate > 0:
                if np.random.rand() <= self.text_drop_rate:
                    text_id[:, :] = 1
            
            data_dict["phone"] = text_id[0, :]
            data_dict["tone"] = text_id[1, :]
            data_dict["word_seg"] = text_id[2, :]
            if self.use_phone_lang:
                data_dict["lang"] = text_id[3, :]

        data_dict['utt_id'] = utt_id
        data_dict["lab"] = labels
        data_dict["text"] = text
        data_dict["token"] = data_dict["token"][:max_umm_len]

        # masking.
        if self.use_bn:
            # mask bn.
            data_dict = self.masking.masking(data_dict, "bn") # update ctx & ctx_mask
            data_dict['bn'] = data_dict["bn"].transpose(0, 1)
            data_dict['bn_ctx'] = data_dict["bn_ctx"].transpose(0, 1)
            data_dict['bn_ctx_mask'] = data_dict["ctx_mask"]
            if "vc_bns" in sample:
                data_dict['vc_bn'] = data_dict["vc_bn"].transpose(0, 1)
        else:
            data_dict = self.masking.masking(data_dict, "mel") # update ctx & ctx_mask
            data_dict['mel_ctx'] = data_dict['mel_ctx'].transpose(0, 1)[:, :max_mel_len]
            data_dict['mel'] = data_dict['mel'].transpose(0, 1)[:, :max_mel_len]
            data_dict['mel_ctx_mask'] = data_dict['ctx_mask'][:max_mel_len]

        return data_dict

    def __iter__(self):
        for item in self.wds:
            batch = self.batcher.collate_batch(item)
            if batch:
                yield batch
        if not self.drop_last:
            for batch in self.batcher.collect_last_batch():
                yield batch


class VoiceBoxCollator(object):
    def __init__(self, 
            tokenizer_pad, 
            mel_config=None, 
            mel_padding=-2,
            max_crop_second=30.0,
            bn_config=None,
            bn_padding=-5,
            use_bn=True
            ):
        self.tokenizer_pad = tokenizer_pad

        self.use_bn = use_bn
        if use_bn:
            self.feat_hz = mel_config["sampling_rate"] // bn_config["hop_size"]
            self.feat_padding = bn_padding
        else:
            self.feat_hz = mel_config["sampling_rate"] // mel_config["hop_size"]
            self.feat_padding = mel_padding

        self.max_crop_len = int(max_crop_second * self.feat_hz)

    def __call__(self, batches):
        results = []
        for item in batches:
            if item is not None:
                results.append(item)
        if len(results) == 0:
            return None
        ret_dict = {}

        token_lens = [b['token'].shape[0] for b in batches]
        max_token_len = max(token_lens)
        token = collate_1d(
            [b["token"] for b in batches],
            pad_idx=self.tokenizer_pad,
            max_len=max_token_len
        )
        ret_dict['token'] = token
        ret_dict["token_mask"] = sequence_mask(torch.from_numpy(np.array(token_lens)),
                  max_len=ret_dict["token"].shape[1])
        
        # pad bn/mel
        if self.use_bn:
            feat_name = "bn"
        else:
            feat_name = "mel"

        feat_lens = [b[feat_name].shape[1] for b in batches]
        max_feat_len = max(feat_lens)
        min_feat_len = min(feat_lens)
        
        feats = collate_2d(
            [b[feat_name] for b in batches], 
            pad_idx=self.feat_padding,
            max_len=max_feat_len)
        
        feat_ctx = collate_2d(
            [b[f'{feat_name}_ctx'] for b in batches], 
            pad_idx=self.feat_padding,
            max_len=max_feat_len)
        
        if "vc_bn" in batches[0]:
            vc_bn_lens = [b["vc_bn"].shape[1] for b in batches]
            max_vc_bn_lens = max(vc_bn_lens)
            min_vc_bn_lens = min(vc_bn_lens)
            vc_bn = collate_2d(
                [b["vc_bn"] for b in batches], 
                pad_idx=self.feat_padding,
                max_len=max_vc_bn_lens)
        
        feat_ctx_mask = collate_1d(
            [b[f'{feat_name}_ctx_mask'] for b in batches], 
            pad_idx=0.0, 
            max_len=max_feat_len)

        ret_dict[feat_name] = feats.transpose(1, 2)  # [B, T, C]
        ret_dict[f"{feat_name}_ctx"] = feat_ctx.transpose(1, 2)
        ret_dict[f"{feat_name}_lens"] = torch.from_numpy(np.array(feat_lens))
        ret_dict[f"{feat_name}_mask"] = sequence_mask(
            seq_lens=ret_dict[f"{feat_name}_lens"], 
            max_len=max_feat_len)
        ret_dict[f"{feat_name}_ctx_mask"] = feat_ctx_mask

        utt_ids = [b['utt_id'] for b in batches]
        
        if "phone" in batches[0]:
            text_lens = [b['phone'].shape[0] for b in batches]
            max_text_len = max(text_lens)
            phones = collate_1d([torch.LongTensor(b['phone']) for b in batches], pad_idx=0, max_len=max_text_len)
            tones = collate_1d([torch.LongTensor(b['tone']) for b in batches], pad_idx=0, max_len=max_text_len)
            word_segs = collate_1d([torch.LongTensor(b['word_seg']) for b in batches], pad_idx=0, max_len=max_text_len)
            ret_dict["phone"] = phones  
            ret_dict["tone"] = tones
            ret_dict['word_seg'] = word_segs
            ret_dict['text_lens'] = torch.from_numpy(np.array(text_lens))
            ret_dict["text_mask"] = sequence_mask(ret_dict["text_lens"], 
                                    max_len=max_text_len)
            ret_dict["text_mel_mask"] = sequence_mask(
                ret_dict[f"{feat_name}_lens"] + ret_dict['text_lens'], 
                max_len=torch.max(ret_dict[f"{feat_name}_lens"] + ret_dict['text_lens']))
            if "lang" in batches[0]:
                ret_dict["lang"] = collate_1d([torch.LongTensor(b['lang']) for b in batches], pad_idx=0, max_len=max_text_len)

        if "vc_bn" in batches[0]:
            crop_len = max(self.feat_hz, np.random.rand() * min_vc_bn_lens)
            crop_len = int(min(self.max_crop_len , crop_len))
            start_point = random.randint(0,  max(min_vc_bn_lens - crop_len - 1, 0))
            ret_dict[f"prompt_{feat_name}"] = vc_bn[:, :, start_point:start_point+crop_len]
        else:
            crop_len = max(self.feat_hz, np.random.rand() * min_feat_len)
            crop_len = int(min(self.max_crop_len , crop_len))
            start_point = random.randint(0,  max(min_feat_len - crop_len - 1, 0))
            ret_dict[f"prompt_{feat_name}"] = ret_dict[feat_name][:, start_point:start_point+crop_len, :].transpose(1, 2)
        
        ret_dict["utt_id"] = utt_ids

        return ret_dict


if __name__ == "__main__":

    batcher_config = { 
        "buckets": list(range(40, 2500, 60)),  
        "dynamic_batch": True,
        "maximum_bucket_size": 17500,
        "length_fn": lambda x: x["bn"].shape[1]
    }

    # batch_size = 40
    # sample_per_batch = 60
    # cache_size = batch_size*sample_per_batch
    # batcher_config = { 
    #     "dynamic_batch": False,
    #     "batch_size": batch_size,
    #     # "length_fn": lambda x: x["mel"].shape[1]
    #     "length_fn": length_fn_list,
    # }

    mel_config = {
        "n_fft": 1200,
        "num_mels": 160,
        "sampling_rate": 24000,
        "hop_size": 240,
        "win_size": 1200,
        "fmin": 0,
        "fmax": 12000
    }

    bn_config = {
        "hop_size": 600,
        "bn_norm_std": 15,
        "bn_norm_mean": 0,
        "bn_padding": -2,
        "silence_bn_path": "recipes/voicebox/vocoder/silence_wvae/v3_40hz/silence_wvae.npy"
    }

    data_id = 1097

    collate_fn = VoiceBoxCollator(
        tokenizer_pad=16384,
        mel_config=mel_config,
        mel_padding=-5,
        max_crop_second=30,
        bn_config=bn_config,
        bn_padding=bn_config['bn_padding'],
        use_bn=True    
    )

    dataset = VoiceBoxParquetDataset(data_id=data_id, 
                              batcher_config=batcher_config, 
                              mel_config=mel_config, 
                              max_length=60,
                              drop_last=False,
                              use_text=False,
                              use_bn=True,
                              mask_use_alignment=False,
                              bn_config=bn_config,
                              mel_norm_mean=0,
                              mel_norm_std=1,
                              umm_hop_size=960)
    

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

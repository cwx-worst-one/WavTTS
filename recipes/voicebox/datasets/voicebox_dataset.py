from ctypes import alignment
import logging
import sys
import os 
import numpy as np
import random
from torch.utils.data import IterableDataset
from torchaudio.transforms import Resample
import torch
from scipy.io.wavfile import write

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.webdataset.ra_wds import WebDataset
from recipes.voicebox.datasets.utils import PhoneToId, collate_2d, collate_1d, get_duration_frames_wds, Masking
from recipes.voicebox.vocoder.BigVGAN.meldataset import mel_spectrogram
from recipes.voicebox.modules.speaker_encoder.utils import load_config
from recipes.voicebox.modules.speaker_encoder.processor import AudioProcessor


logger = logging.getLogger(__name__)


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class VoiceBoxDataset(IterableDataset):
    def __init__(self,
        wds_urls,
        drop_last=False,
        batcher_config=None,
        max_length=4096,
        duration_dir="/mnt/bn/ttsnas001/data/res",
        mel_config=None,
        mel_norm_mean=-5.8843,
        mel_norm_std=2.2615,
        use_speaker_encoder=False,
        speaker_encoder_config=None,
        max_crop_second=10.0
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

        self.max_length = max_length
        self.phone2id = PhoneToId()

        self.pipelines = []
        self.drop_last = drop_last
        self.batcher = BucketBatcher(**batcher_config)
        self.mel_config = mel_config
        self.mel_norm_mean = mel_norm_mean
        self.mel_norm_std = mel_norm_std

        self.utt2duration = {}
        self.duration_dir = duration_dir
        self.shot = 0
        self.total = 0

        if mel_config['sampling_rate'] != 24000: # audio in web-dataset is 24k
            self.audio_resampler = Resample(orig_freq=24000, new_freq=mel_config['sampling_rate'])
        else:
            self.audio_resampler = None

        self.use_speaker_encoder = use_speaker_encoder
        if self.use_speaker_encoder:
            self.encoder_config = load_config(speaker_encoder_config)
            self.spk_enc_ap = AudioProcessor(**self.encoder_config.audio)
            self.audio_resampler_16k = Resample(orig_freq=mel_config['sampling_rate'], new_freq=16000)
        else:
            self.audio_resampler_16k = None
        
        self.hop_ms = mel_config['hop_size'] / mel_config['sampling_rate']
        self.masking = Masking(padding_value=-2)
        self.max_crop_second = max_crop_second

    
    def get_duration_wds(self, utt_id, alignments, phonemes, mel_len):
        if alignments is not None and len(alignments) > 0:
            duration_frames = get_duration_frames_wds(alignments, phonemes, self.hop_ms)
            if duration_frames is not None and abs(sum(duration_frames) - mel_len) > 2:
                duration_frames = None
            if duration_frames is not None:
                pad_len = sum(duration_frames) - mel_len
                duration_frames[-1] = duration_frames[-1] - pad_len
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
        spkenc_wav = wav

        scale = max(0.001, torch.max(torch.abs(wav)))
        wav = wav / scale * 0.95
        mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)
        mel = (mel - self.mel_norm_mean) / self.mel_norm_std

        #wav *= 0
        #mel = mel_spectrogram(wav.unsqueeze(0), **self.mel_config).squeeze(0)
        #mel = (mel - self.mel_norm_mean) / self.mel_norm_std

        # load phone seq & duration
        text = sample["text"]
        url = sample["__url__"]
        lab = sample["labels"]
        utt_id = sample["__key__"]

        data_dict = dict()
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
            return None
        
        if text_id_phones_tones is None:
            logger.warning(f"{utt_id} convert_tacolab_to_text_id failed ...")
            return None
        else:
            text_id, phones, tones, word_segs, alignments = text_id_phones_tones

        # # expand textid using duration
        # duration = self.get_duration(utt_id, phones, mel.shape[1])

        duration = self.get_duration_wds(utt_id, alignments, phones, mel.shape[1])

        if duration is None:
            return None
        try:
            text_id = np.repeat(text_id, duration, axis=1)
            assert text_id.shape[1] == mel.shape[1], f"sum duration {sum(duration)} vs {mel.shape[1]}"
        except Exception as e:
            print(e)
            return None

        data_dict["phone"] = text_id[0, :]
        data_dict["tone"] = text_id[1, :]
        data_dict["word_seg"] = text_id[2, :]
        data_dict['utt_id'] = utt_id
        data_dict['raw_phone'] = phones
        data_dict['raw_tones'] = tones
        data_dict['mel'] = mel.transpose(1, 0)
        data_dict['duration_ori'] = duration # for masking methods
        data_dict["duration"] = duration
        data_dict["lab"] = lab

        # masking.
        data_dict = self.masking.masking(data_dict) # update ctx & ctx_mask
        data_dict['mel_ctx'] = data_dict['mel_ctx'].transpose(0, 1)
        data_dict['mel'] = data_dict['mel'].transpose(0, 1)
        data_dict['wav'] = wav
        data_dict["text"] = text

        if self.use_speaker_encoder:
            crop_len = max(24000, np.random.rand() * (len(spkenc_wav) * 0.75))
            crop_len = min(self.max_crop_second * 24000, crop_len)
            crop_len = int(crop_len)
            start_point = random.randint(0,  max(spkenc_wav.shape[0] - crop_len - 1, 0))
            spkenc_wav = torch.nn.functional.pad(spkenc_wav, (0, crop_len))[start_point: start_point+crop_len]

            spkenc_wav_16k = self.audio_resampler_16k(spkenc_wav)
            scale = max(0.001, torch.max(torch.abs(spkenc_wav_16k)))
            spkenc_wav_16k = spkenc_wav_16k / scale * 0.95
            spkenc_wav_16k = self.spk_enc_ap.rms_volume_norm(spkenc_wav_16k.numpy(), self.spk_enc_ap.db_level)
            spkenc_wav_16k = torch.FloatTensor(spkenc_wav_16k)

            data_dict["spkenc_wav_16k"] = spkenc_wav_16k
            data_dict["prompt_len"] = crop_len // self.mel_config["hop_size"]

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
    def __init__(self, tokenizer_pad, block_sparse=False, use_speaker_encoder=False):
        self.pad = tokenizer_pad
        self.block_sparse = block_sparse
        self.use_speaker_encoder = use_speaker_encoder

    def __call__(self, batches):
        # fake
        results = []
        for item in batches:
            if item is not None:
                results.append(item)
        if len(results) == 0:
            return None

        mel_lens = [b['mel'].shape[1] for b in batches]
        max_mel_len = max(mel_lens)

        min_wav_len = min([b['wav'].shape[0] for b in batches])

        mels = collate_2d(
            [b['mel'] for b in batches], 
            pad_idx=-2,
            max_len=max_mel_len)
        
        mel_ctx = collate_2d(
            [b['mel_ctx'] for b in batches], 
            pad_idx=-2,
            max_len=max_mel_len)

        mel_ctx_mask = collate_1d(
            [b['mel_ctx_mask'] for b in batches], 
            pad_idx=0.0, 
            max_len=max_mel_len)

        
        text_lens = [b['phone'].shape[0] for b in batches]
        max_text_len = max(text_lens)
        phones = collate_1d([torch.LongTensor(b['phone']) for b in batches], pad_idx=self.pad, max_len=max_text_len)
        tones = collate_1d([torch.LongTensor(b['tone']) for b in batches], pad_idx=self.pad, max_len=max_text_len)
        word_segs = collate_1d([torch.LongTensor(b['word_seg']) for b in batches], pad_idx=self.pad, max_len=max_text_len)
        phoneme_durations = collate_1d([torch.LongTensor(b['duration']) for b in batches], pad_idx=0)
        prompt_lens = [b['prompt_len'] for b in batches]

        utt_ids = [b['utt_id'] for b in batches]
        ret_dict = {}
        ret_dict["mel"] = mels.transpose(1, 2)  # [B, T, C]
        ret_dict["mel_lens"] = torch.from_numpy(np.array(mel_lens))
        ret_dict["mel_mask"] = sequence_mask(ret_dict["mel_lens"], 
                max_len=ret_dict["mel"].shape[1])

        ret_dict["phone"] = phones  
        ret_dict["tone"] = tones
        ret_dict['word_seg'] = word_segs
        ret_dict['utt_ids'] = utt_ids
        ret_dict['text_lens'] = torch.from_numpy(np.array(text_lens))
        ret_dict["prompt_lens"] = torch.from_numpy(np.array(prompt_lens))
        ret_dict["phone_durations"] = phoneme_durations
        ret_dict["prompt_mel_mask"] = sequence_mask(ret_dict["prompt_lens"], 
                max_len=ret_dict["mel"].shape[1])
        ret_dict['mel_ctx'] = mel_ctx.transpose(1, 2)  # [B, T, C]
        ret_dict['mel_ctx_mask'] = mel_ctx_mask

        if self.use_speaker_encoder:
            spkenc_wav_16k_lens = [b['spkenc_wav_16k'].shape[0] for b in batches]
            max_spkenc_wav_16k_len = max(spkenc_wav_16k_lens)
            spkenc_wav_16k = collate_1d(
                [b['spkenc_wav_16k'] for b in batches], 
                pad_idx=0.0, 
                max_len=max_spkenc_wav_16k_len)
            ret_dict['spkenc_wav_16k'] = spkenc_wav_16k

        return ret_dict

if __name__ == "__main__":
    urls = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/BigTTS/librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_5_10s/package/wav_1.0_web_dataset_1/data/*/*.tar"
    from samantha.dataio.utils import parse_data_urls
    wds_urls = parse_data_urls(data_urls=urls)

    batcher_config = { 
        "buckets": list(range(0, 6000, 100)),  # [0, 100, 200 ... 4000] 4000以上的可以先丢掉
        "dynamic_batch": True,
        "maximum_bucket_size": 17500,
        "length_fn": lambda x: x["mel"].shape[1] # seq.shape
    }
    mel_config = {
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 22050,
        "hop_size": 256,
        "win_size": 1024,
        "fmin": 0,
        "fmax": 8000,
        "center": False
    }

    dataset = VoiceBoxDataset(wds_urls, batcher_config=batcher_config, mel_config=mel_config, drop_last=False)
    # collector = VoiceBoxCollator(tokenizer_pad=0)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=None, collate_fn=collector)
    import tqdm
    output_dir="/mnt/bn/jcong5/tmp/voicebox_testset/"
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    metaout = open(os.path.join(output_dir, "meta.out"), "w")
    for item in tqdm.tqdm(dataset):
        if len(item[0]['lab'].decode().split("\n")) == len(item[0]['duration']):
            text = item[0]["text"]
            uttid = item[9]["utt_id"]
            outlab_path = os.path.join(output_dir, f"{uttid}.lab")
            fout = open(outlab_path, "w")
            duration = item[0]['duration']
            for i, line in enumerate(item[0]['lab'].decode().split("\n")):
                fout.write(f"{line}\t{duration[i]}\n")
            out_wavpath = os.path.join(output_dir, f"{uttid}.wav")
            save_wav(item[0]['wav'].cpu().numpy(), out_wavpath, sr=22050)
            fout.close()
            metaout.write(f"{uttid}|{text}|{outlab_path}|{out_wavpath}\n")
            count = count + 1
            if count > 20:
                metaout.close()
                break
        # for uttid, mel in zip(item['utt_ids'], item['mel']):
        #     np.save(os.path.join(output_dir, uttid), mel.numpy())
        # break

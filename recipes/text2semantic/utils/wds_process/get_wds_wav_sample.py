import torch
import numpy as np
import webdataset as wds
import os
import ffmpeg
import time
import json
import sys
from functools import lru_cache
from torchdata.datapipes.iter import IterableWrapper, Mapper
from tqdm import tqdm
from scipy.signal import resample
import random

from torch.utils.data import IterableDataset
from recipes.text2semantic.datasets.sami_tacolabel import enc_taco_label_no_bytes
from recipes.text2semantic.utils.remote_io import load_json
from samantha.utils.hparams import DotDict
from scipy.io.wavfile import write

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def save_wav_int16(audio, output_file, sr=24000):
    write(output_file, sr, audio)
    return

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    st = time.time()
    try:
        seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    # print(f"ffmpeg time cost: {time.time() - st:.4f}s")
        return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)
    except Exception as e:
        return None

def sample_len(sample):
    '''
    估算 token_len
    '''
    # TODO: remove hard-code
    wav_len = int(sample[4] / 300) # 300 = 0.0125 * 24000
    text_len = sample[5]
    return wav_len + text_len

def dynamic_bucketizer(dataset, max_token_count=1000, min_len=10, max_len=1492, buffer_size=10):
    dataset = dataset.max_token_bucketize(
        buffer_size=buffer_size,
        max_token_count=max_token_count, 
        len_fn=sample_len,
        min_len=min_len,
        max_len=max_len,
        include_padding=True) # max_len must equal to max_token_count, or may return empty batch when there is a sample longer than max_token_count
    return dataset

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

class WDSDataset(IterableWrapper):
    def __init__(self, wds_lst, meta_lst, metaid2textid, hp, 
                 resampled=True, out_dir=None):
        # self.meta_dir = meta_dir
        self.hp = DotDict(hp)
        self.sample_rate = self.hp.sample_rate
        self.text_converter_dict = load_json(metaid2textid)
        self.meta_dict = load_json(meta_lst)
        urls = self._load_wds_lst(wds_lst)

        self.dataset = (
            wds.WebDataset(urls=urls, 
                        resampled=resampled, 
                        shardshuffle=True,
                        nodesplitter=wds.split_by_node)
            .decode() # 根据已知的后缀名或类型进行解码
            .compose(self._split_wav) # 应用自定义处理函数
        )
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    
    def __getstate__(self):
        pass

    @lru_cache(maxsize=5)
    def cache_meta(self, key):
        res = {}
        try:
            for fn in self.meta_dict[key]:
                res.update(load_json(fn))
            return res
        except Exception as e:
            return None

    def quality_check(self, sent):
        # confidence > 0.6、rms_max > -13、snr > 6、speaker_similarity > 0.6, mos > 2.8
        try:
            rms_max = sent['rms_stats']['rms_max']
            snr = sent['snr']
            speaker_similarity_min = sent['speaker_similarity']['min']
            mos = sent['mos']
            if rms_max >= -13 and snr >= 6 and speaker_similarity_min >= 0.6 and mos >= 2.8:
                return True
            else:
                return False
        except Exception as e:
            return False

    def norm_pad(self, wav):
        wav = wav * 1.0 / max(0.01, np.max(np.abs(wav)))
        sil = np.zeros(2400)
        wav = np.concatenate((sil, wav, sil))
        return wav

    def _load_wds_lst(self, fn):
        urls = []
        with open(fn, "r", encoding="utf-8") as fi:
            for line in fi:
                wds_path = line.strip()
                if not wds_path:
                    continue
                if wds_path.startswith("hdfs://"):
                    url = f"pipe: hdfs dfs -cat {wds_path}"
                else:
                    url = f"pipe: cat {wds_path}"
                urls.append(url)
        return urls

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def _split_wav(self, samples, sr=24000):
        for item in samples: # item表示一个20mins的片段: __key__, __url__, bin
            if 'bin' not in item.keys():
                # process wav numpy
                wav = item['npy']
                source_sample_rate = 16000
                if wav.dtype == np.int16:
                    wav = wav / 32768.0
                if source_sample_rate != sr:
                    new_len = round(wav.shape[-1] * sr / source_sample_rate)
                    wav = resample(wav, int(new_len))
            else:
                # process wav bin
                audio_bin = item['bin']
                wav = ffmpeg_read_audio(audio_bin, sr)
                if wav is None:
                    continue
            wav = wav.astype(np.float32)

            # split wav
            key = item['__key__']
            tar_id_uniq = '/'.join(item['__url__'].split('/')[-3:])
            if tar_id_uniq[-5:] == '.orig':
                wds_key = tar_id_uniq[:-9]
                tar_id = wds_key.split('/')[-1]
            else:
                wds_key = tar_id_uniq[:-4]
                tar_id = wds_key.split('/')[-1]
            # tar_id = os.path.splitext(os.path.basename(item['__url__']))[0]
            # if os.path.basename(item['__url__'])[-5:] == '.orig':
            #     tar_id = os.path.basename(item['__url__'])[:-9] # '.tar.orig'
            meta_data = self.cache_meta(wds_key)
            if meta_data is None or key not in meta_data:
                # print(f"Warning: {tar_id}/{key} has no meta json!")
                continue

            g = torch.Generator()
            g.manual_seed(1111)
            idxs = torch.randperm(len(meta_data[key]['sent']), generator=g).tolist()
            for idx in idxs[:2]:
                data = meta_data[key]['sent'][idx]
                if self.quality_check(data):
                    try:
                        st, et = data['start_time'], data['end_time']
                        cur_wav = wav[int(st * sr): int(et * sr)]
                        text_id = self.convert_tacolab_to_text_id(data['labels'])
                        if cur_wav.shape[0] / text_id.shape[0] < 1600 or cur_wav.shape[0] / text_id.shape[0] > 3800:
                            continue
                        cur_wav = self.norm_pad(cur_wav)
                    except Exception as e:
                        # print("Warning:", e)
                        continue
                    yield tar_id, idx, cur_wav, text_id, cur_wav.shape[0], text_id.shape[0], data['text'], key
       
    def convert_tacolab_to_text_id(self, tacolab):
        tacolab = list(filter(lambda x: x != "", tacolab.split('\n')))
        with HiddenPrints():
            metas = enc_taco_label_no_bytes(
                None, 
                tacolab, 
                {"use_prsdword": False, "forced_refix": True})
        text_id =  metas[0].astype(np.int64) * 1_000_000_000 + \
                    metas[1].astype(np.int64) * 1_000_000 + \
                    metas[2].astype(np.int64) * 1_000 + \
                    metas[3].astype(np.int64)
        text_id = np.asarray([self.text_converter_dict[str(x)] for x in text_id]).astype(np.int64)
        return text_id

# class DummyCollator(object):
#     def __init__(self):
#         pass

#     def __call__(self, batches):
#         utt_ids = []
#         wavs = []
#         text_ids = []
#         wav_lens = []
#         text_lens = []
#         batches = batches[0] # in dynamic bucketing, batch size must be 1
#         max_wav_len = max([batch[4] for batch in batches])
#         max_text_len = max([batch[5] for batch in batches])
#         for batch in batches:
#             assert len(batch) == 6, batch
#             tar_id, idx, wav, text_id, wav_len, text_len = batch
#             utt_ids.append((tar_id, idx))
#             text_id = np.pad(
#                 text_id,
#                 (0, max_text_len - text_len),
#                 mode="constant",
#                 constant_values=0,
#             )
#             text_ids.append(text_id)
#             wav_lens.append(wav_len)
#             text_lens.append(text_len)
#             wav = np.pad(
#                 wav,
#                 (0, max_wav_len - wav_len),
#                 mode="constant",
#                 constant_values=0.,
#             )
#             wavs.append(wav)

#         # to numpy
#         wavs = np.asarray(wavs)
#         wav_lens = np.asarray(wav_lens)
#         text_ids = np.asarray(text_ids)
#         text_lens = np.asarray(text_lens)
    
#         wavs = torch.from_numpy(wavs)
#         text_ids = torch.from_numpy(text_ids)

#         return wavs, text_ids, wav_lens, text_lens, utt_ids

if __name__ == "__main__":
    from tqdm import tqdm
    from recipes.soundstream.utils.utils import get_config_from_file
    hp = get_config_from_file('/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/soundstream_config.yaml').hparams

    wds_lst = sys.argv[1]
    meta_lst = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    lines = open(wds_lst).readlines()

    for line in lines:
        tar_id_uniq = '/'.join(line.strip().split('/')[-3:])
        if tar_id_uniq[-5:] == '.orig':
            wds_key = tar_id_uniq[:-9]
            tar_id = wds_key.split('/')[-1]
        else:
            wds_key = tar_id_uniq[:-4]
            tar_id = wds_key.split('/')[-1]

        os.makedirs(out_dir + '/' + tar_id, exist_ok=True)

        f_w = open(wds_lst + '.' + tar_id, 'w')
        f_w.write(line)
        f_w.close()

        f = open(meta_lst)
        wds2meta = json.load(f)
        cur_wds2meta = dict()
        cur_wds2meta[wds_key] = wds2meta[wds_key]
        f_w = open(meta_lst + '.' + tar_id, 'w')
        json.dump(cur_wds2meta, f_w, ensure_ascii=False, indent=2)
        f_w.close()


        dataset = WDSDataset(
            wds_lst=wds_lst + '.' + tar_id,
            meta_lst=meta_lst + '.' + tar_id,
            hp=hp,
            metaid2textid="/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/datasets/dict/metaid_to_textid.json",
            out_dir=out_dir + '/' + tar_id
        )

        st = time.time()
        i = 0
        for batch in dynamic_bucketizer(dataset, max_token_count=1500, min_len=10, max_len=1492, buffer_size=1):
            for x in batch:
                tar_id, idx, cur_wav, text_id, cur_wav_len, text_id_len, text, key = x
                uttname = tar_id + "_" + key + "_" + str(idx)
                print("ratio: ", uttname, cur_wav_len/text_id_len, cur_wav_len / 24000)

                # uttname = tar_id + "_" + key + "_" + str(idx)
                # if cur_wav_len/text_id_len < 1600 or cur_wav_len/text_id_len > 3800:
                f_w = open(out_dir + '/' + tar_id + '/' + uttname + '.txt', 'w')
                f_w.write(text + '\n')
                f_w.close()

                save_wav_int16(cur_wav, out_dir + '/' + tar_id + '/' + uttname + '.wav')

            print(f"{time.time() - st}s")
            st = time.time()
            if i == 10:
                break
            i += 1


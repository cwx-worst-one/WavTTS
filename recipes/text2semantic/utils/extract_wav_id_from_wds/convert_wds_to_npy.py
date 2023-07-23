import torch
import numpy as np
import webdataset as wds
import os
# import ffmpeg
import time
import json
import sys
# from functools import lru_cache
# from torchdata.datapipes.iter import IterableWrapper, Mapper
from tqdm import tqdm
# from scipy.signal import resample

from torch.utils.data import IterableDataset
# from recipes.text2semantic.datasets.sami_tacolabel import enc_taco_label_no_bytes
# from recipes.text2semantic.utils.remote_io import load_json
# from samantha.utils.hparams import DotDict
from scipy.io.wavfile import write

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return

def save_wav_int16(audio, output_file, sr=24000):
    write(output_file, sr, audio)
    return

# class HiddenPrints:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout

# def ffmpeg_read_audio(audio_bin, sample_rate=24000):
#     st = time.time()
#     seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
#     # print(f"ffmpeg time cost: {time.time() - st:.4f}s")
#     return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)

# def sample_len(sample):
#     '''
#     估算 token_len
#     '''
#     # TODO: remove hard-code
#     wav_len = int(sample[4] / 300) # 300 = 0.0125 * 24000
#     text_len = sample[5]
#     return wav_len + text_len

# def dynamic_bucketizer(dataset, max_token_count=1000, min_len=10, max_len=1492):
#     dataset = dataset.max_token_bucketize(
#         buffer_size=10,
#         max_token_count=max_token_count, 
#         len_fn=sample_len,
#         min_len=min_len,
#         max_len=max_len,
#         include_padding=True) # max_len must equal to max_token_count, or may return empty batch when there is a sample longer than max_token_count
#     return dataset

# def nodesplitter(src, group=None):
#     if torch.distributed.is_initialized():
#         if group is None:
#             group = torch.distributed.group.WORLD
#         rank = torch.distributed.get_rank(group=group)
#         size = torch.distributed.get_world_size(group=group)
#         print(f"nodesplitter: rank={rank} size={size}")
#         count = 0
#         for i, item in enumerate(src):
#             if i % size == rank:
#                 yield item
#                 count += 1
#         print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
#     else:
#         yield from src

class WDSDataset(IterableDataset):
    def __init__(self, wds_lst, out_dir):
        urls = self._load_wds_lst(wds_lst)

        self.dataset = (
            wds.WebDataset(urls=urls)
            .decode() # 根据已知的后缀名或类型进行解码
        )
        self.out_dir = out_dir

    
    def __getstate__(self):
        pass

    # @lru_cache(maxsize=5)
    # def cache_meta(self, key):
    #     res = {}
    #     try:
    #         for fn in self.meta_dict[key]:
    #             res.update(load_json(fn))
    #         return res
    #     except Exception as e:
    #         return None

    def quality_check(self, sent):
        # confidence > 0.6、rms_max > -13、snr > 6、speaker_similarity > 0.6, mos > 2.8
        try:
            rms_max = sent['rms_stats']['rms_max']
            snr = sent['snr']
            speaker_similarity_min = sent['speaker_similarity']['min']
            mos = sent['mos']
            if rms_max > -13 and snr > 6 and speaker_similarity_min > 0.6 and mos > 2.8:
                return True
            else:
                return False
        except Exception as e:
            return False

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
    
 
    # def convert_tacolab_to_text_id(self, tacolab):
    #     tacolab = list(filter(lambda x: x != "", tacolab.split('\n')))
    #     with HiddenPrints():
    #         metas = enc_taco_label_no_bytes(
    #             None, 
    #             tacolab, 
    #             {"use_prsdword": False, "forced_refix": True})
    #     text_id =  metas[0].astype(np.int64) * 1_000_000_000 + \
    #                 metas[1].astype(np.int64) * 1_000_000 + \
    #                 metas[2].astype(np.int64) * 1_000 + \
    #                 metas[3].astype(np.int64)
    #     text_id = np.asarray([self.text_converter_dict[str(x)] for x in text_id]).astype(np.int64)
    #     return text_id

if __name__ == "__main__":
    from tqdm import tqdm

    wds_lst = sys.argv[1]
    out_dir = sys.argv[2]

    dataset = WDSDataset(wds_lst=wds_lst, out_dir=out_dir)

    st = time.time()
    i = 0
    for x in tqdm(dataset):
        uttname = x['__key__']
        shardname = x['__url__'].split('/')[-1].split('.')[0]

        tacolab_dir = os.path.join(out_dir, shardname, 'tacofrontend')
        os.makedirs(tacolab_dir, exist_ok=True)
        tacolab_path = os.path.join(tacolab_dir, uttname + '.lab')
        tacolab = x['labels'].decode()
        f_w = open(tacolab_path, 'w')
        f_w.write(tacolab)
        f_w.close()

        wav_id_dir = os.path.join(out_dir, shardname, 'wav_id')
        os.makedirs(wav_id_dir, exist_ok=True)
        wav_id_path = os.path.join(wav_id_dir, uttname + '.npy')
        wav_id = x['wav_id']
        wav_id = np.frombuffer(wav_id, dtype=np.int64).reshape(-1, 6)
        np.save(wav_id_path, wav_id)

        # text_dir = os.path.join(out_dir, shardname, 'text')
        # os.makedirs(text_dir, exist_ok=True)
        # text_path = os.path.join(text_dir, uttname + '.txt')
        # text = x['text']
        # f_w = open(text_path, 'w')
        # f_w.write(text)
        # f_w.close()

        other_metainfo_dir = os.path.join(out_dir, shardname, 'other_metainfo')
        os.makedirs(other_metainfo_dir, exist_ok=True)
        other_metainfo_path = os.path.join(other_metainfo_dir, uttname + '.json')
        other_metainfo = dict()
        other_metainfo['text'] = x['text']
        other_metainfo['mos'] = x['mos'].decode()
        other_metainfo['rms_stats_rms_max'] = x['rms_stats_rms_max'].decode()
        other_metainfo['snr'] = x['snr'].decode()
        other_metainfo['speaker_similarity_min'] = x['speaker_similarity_min'].decode()
        f_w = open(other_metainfo_path, 'w')
        json.dump(other_metainfo, f_w, ensure_ascii=False, indent=2)
        f_w.close()
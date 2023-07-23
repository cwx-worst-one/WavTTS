import torch
import numpy as np
import webdataset as wds
import os
import ffmpeg
import time
import json
import sys
from functools import lru_cache
from scipy.signal import resample

from torch.utils.data import IterableDataset
from sami_tacolabel import enc_taco_label_no_bytes
from remote_io import load_json
from hparams import DotDict
from scipy.io.wavfile import write
import hdfs_helper as hh
from webdataset import TarWriter
from utils import get_config_from_file
from tqdm import tqdm

np.set_printoptions(threshold=1000000)

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

class WDSDataset(IterableDataset):
    def __init__(self, wds_lst, meta_lst, metaid2textid, hp, 
                 resampled=True, longform=False):
        # self.meta_dir = meta_dir
        self.hp = DotDict(hp)
        self.longform = longform
        self.sample_rate = self.hp.sample_rate
        self.meta_dict = load_json(meta_lst)
        urls = self._load_wds_lst(wds_lst)

        self.dataset = (
            wds.WebDataset(urls=urls, 
                        resampled=resampled, 
                        shardshuffle=True,
                        nodesplitter=wds.split_by_node)
            .decode() # 根据已知的后缀名或类型进行解码
            .compose(self._split_wav)
            # .batched(1) # 应用自定义处理函数
        )
        self.text_converter_dict = load_json(metaid2textid)

    
    def __getstate__(self):
        pass

    @lru_cache(maxsize=100)
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
            start_time = sent['start_time']
            end_time = sent['end_time']
            if rms_max >= -13 and snr >= 7 and speaker_similarity_min >= 0.6 and mos >= 4.2 and end_time - start_time < 25:
                return True
            else:
                # print("quality check failed.")
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
        print("split wav...")
        for item in samples: # item表示一个20mins的片段: __key__, __url__, bin
            # split wav
            key = item['__key__']
            wds_key = item['__url__'].split(' ')[-1]

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
                    print("warning: wav read error")
                    continue
            wav = wav.astype(np.float32)
           
            # tar_id_uniq = '/'.join(item['__url__'].split('/')[-3:])
            # if tar_id_uniq[-5:] == '.orig':
            #     wds_key = tar_id_uniq[:-9]
            #     tar_id = wds_key.split('/')[-1]
            # else:
            #     wds_key = tar_id_uniq[:-4]
            #     tar_id = wds_key.split('/')[-1]
            # print("key: ", key)
            # print("wds_key: ", wds_key)

            meta_data = self.cache_meta(wds_key)
            if meta_data is None or key not in meta_data:
                print(f"Warning: {key} has no meta json!")
                continue

            for idx, data in enumerate(meta_data[key]['sent']): # data表示一段长wav
                if self.quality_check(data) or self.longform:
                    try:
                        st, et = data['start_time'], data['end_time']
                        cur_wav = wav[int(st * sr): int(et * sr)]
                        cur_wav = cur_wav * 1.0 / max(0.01, np.max(np.abs(cur_wav)))
                        # text_id = self.convert_tacolab_to_text_id(data['labels'])
                        text = data['text']
                        # if cur_wav.shape[0] / text_id.shape[0] < 1600 or cur_wav.shape[0] / text_id.shape[0] > 3800:
                        #     continue

                        uttname = f"{key}_{idx:0>4}"
                        labels = data['labels']
                        if self.longform: # fake data for longform
                            snr = 10
                            mos = 5.0
                            rms_stats_rms_max = 0.0
                            speaker_similarity_min = 1.0
                        else:
                            snr = data['snr']
                            mos = data['mos']
                            rms_stats_rms_max = data['rms_stats']['rms_max']
                            speaker_similarity_min = data['speaker_similarity']['min']
                        # save_wav(cur_wav, os.path.join(uttname + '.wav'))
                    except Exception as e:
                        print("Warning:", e)
                        continue
                    if (uttname is None) or (cur_wav is None) or (text is None) or (labels is None) or (snr is None) or (mos is None) or (rms_stats_rms_max is None) or (speaker_similarity_min is None):
                        print("Warning: Some value is None, skip")
                        continue
                    yield uttname, cur_wav, text, labels, snr, mos, rms_stats_rms_max, speaker_similarity_min

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

def get_codec_codes(requires, x):
    output = requires["ss"](x)[2]
    output = torch.stack(output, dim=2) # [b, t, n_code]
    return output

def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_sound_stream_encoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_encoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_encoder_{local_rank}.pt"

    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_path):
            if not hh.get(h_ss, local_path):
                raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        return {"ss_enc": load_torch_script_module(local_path, device)}
    else:
        return {"ss_enc": load_torch_script_module(h_ss, device)}

def init_sound_stream_decoder(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_decoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_decoder_{local_rank}.pt"

    if not os.path.exists(local_path):
        if not hh.get(h_ss, local_path):
            raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
    return {"ss_dec": load_torch_script_module(local_path, device)}

def init_sound_stream(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    h_ss = f"{hpath}/ss_encoder_{local_rank}.pt"
    local_path = f"{cache_dir}/ss_encoder_{local_rank}.pt"

    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_path):
            if not hh.get(h_ss, local_path):
                raise ConnectionError(f"Cannot retrieve file from {h_ss}.")
        return {"ss": load_torch_script_module(local_path, device)}
    else:
        return {"ss": load_torch_script_module(h_ss, device)}

class DummyCollator(object):
    def __init__(self):
        pass

    def __call__(self, batches):
        # batch: uttname, cur_wav, text, labels, snr, mos, rms_stats_rms_max, speaker_similarity_min
        uttnames = []
        cur_wavs = []
        wav_lens = []
        texts = []
        labelses = []
        snrs = []
        moses = []
        rms_stats_rms_maxs = []
        speaker_similarity_mins = []

        max_wav_len = max([batch[1].shape[0] for batch in batches])
        for batch in batches:
            assert len(batch) == 8, batch
            uttname, cur_wav, text, labels, snr, mos, rms_stats_rms_max, speaker_similarity_min = batch
            uttnames.append(uttname)

            wav_len = cur_wav.shape[0]
            wav_lens.append(wav_len)

            cur_wav = np.pad(
                cur_wav,
                (0, max_wav_len - wav_len),
                mode="constant",
                constant_values=0.,
            )
            cur_wavs.append(cur_wav)

            texts.append(text)
            labelses.append(labels)
            snrs.append(snr)
            moses.append(mos)
            rms_stats_rms_maxs.append(rms_stats_rms_max)
            speaker_similarity_mins.append(speaker_similarity_min)

        # to numpy
        cur_wavs = np.asarray(cur_wavs)
        wav_lens = np.asarray(wav_lens)

        cur_wavs = torch.from_numpy(cur_wavs)
        wav_lens = torch.from_numpy(wav_lens)

        return uttnames, cur_wavs, wav_lens, texts, labelses, snrs, moses, rms_stats_rms_maxs, speaker_similarity_mins
  
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta",
        default="",
        dest="meta", help="")
    parser.add_argument("--wds",
        default="",
        dest="wds", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    parser.add_argument("--rank",
        default=0,
        type=int,
        dest="rank", help="")
    parser.add_argument("--data_workers",
        default=16,
        type=int,
        dest="data_workers", help="")
    parser.add_argument("--longform",
        default=False, const=True, nargs="?",
        dest="longform", help="") 
    args = parser.parse_args()

    wds_lst = args.wds
    meta_lst = args.meta
    out_dir = args.outputs
    local_rank = args.rank

    hp = get_config_from_file('soundstream_config.yaml').hparams
    os.makedirs(out_dir, exist_ok=True)

    lines = open(wds_lst).readlines()

    # requires = init_sound_stream("hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank=local_rank, cache_dir=".module_cache/llm_debug/")
    # requires = init_sound_stream("hdfs://harunava/home/byte_speech_sv/litang/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank=local_rank, cache_dir=".module_cache/llm_debug/")

    vqgan_model_encoder = init_sound_stream_encoder("hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank, cache_dir=".cache_dir")["ss_enc"]
    vqgan_model_encoder.eval()

    vqgan_model_decoder = init_sound_stream_decoder("hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export", local_rank, cache_dir=".cache_dir")["ss_dec"]
    vqgan_model_decoder.eval()

    ### for each tar file
    for line in tqdm(lines):
        # get wds_key and tar_id
        # tar_id_uniq = '/'.join(line.strip().split('/')[-3:])
        tar_id_uniq = line.strip()
        wds_key = tar_id_uniq
        if tar_id_uniq[-5:] == '.orig':
            tar_id = wds_key.split('/')[-1][:-9]
        else:
            tar_id = wds_key.split('/')[-1][:-4]

        # out wds path
        out_wds_path = os.path.join(out_dir, tar_id + '.tar')

        # gen sub_wds_lst
        f_w = open(wds_lst + '.' + tar_id, 'w')
        f_w.write(line)
        f_w.close()

        # gen sub_meta_lst
        f = open(meta_lst)
        wds2meta = json.load(f)
        cur_wds2meta = dict()
        cur_wds2meta[wds_key] = wds2meta[wds_key]
        f_w = open(meta_lst + '.' + tar_id, 'w')
        json.dump(cur_wds2meta, f_w, ensure_ascii=False, indent=2)
        f_w.close()

        # wds dataset
        dataset = WDSDataset(
            wds_lst=wds_lst + '.' + tar_id,
            meta_lst=meta_lst + '.' + tar_id,
            hp=hp,
            longform=args.longform,
            metaid2textid="metaid_to_textid.json",
        )

        writer = TarWriter(open(out_wds_path, "wb"))
        print("out_wds_path: ", out_wds_path)
        for x in tqdm(torch.utils.data.DataLoader(dataset, num_workers=args.data_workers, batch_size=2, collate_fn=DummyCollator())):
            # uttname, cur_wav, text, labels, snr, mos, rms_stats_rms_max, speaker_similarity_min = x
            uttnames, cur_wavs, wav_lens, texts, labelses, snrs, moses, rms_stats_rms_maxs, speaker_similarity_mins = x
            # print("uttnames: ", uttnames)
            # print("cur_wavs: ", cur_wavs.shape)
            # print("wav_lens: ", wav_lens)
            # print("texts: ", texts)
            # print("labelses: ", labelses)
            # print("snrs: ", snrs)
            # print("moses: ", moses)
            # print("rms_stats_rms_maxs: ", rms_stats_rms_maxs)
            # print("speaker_similarity_mins: ", speaker_similarity_mins)

            # if args.longform:
            #     uttname = f"lf_{uttname[0]}"
            # else:
            #     uttname = uttname[0]

            cur_wavs = cur_wavs.to(f"cuda:{local_rank}")
            wav_ids_ori = torch.stack(vqgan_model_encoder(cur_wavs)[2], dim=2)
            # wav_ids_ori = get_codec_codes(requires, cur_wavs)
            wav_ids = wav_ids_ori.cpu().numpy()
            wav_id_lens = wav_lens // 300 
            batch_size = len(cur_wavs)
            for i in range(batch_size):
                # print("wav_ids_ori: ", wav_ids_ori.shape)
                # wav_id_ori = wav_ids_ori[i:i+1, :, :]
                # print("wav_id_ori1: ", wav_id_ori.shape)
                # wav_id_ori = wav_id_ori[:, :wav_id_lens[i], :].transpose(1, 2)
                # print("wav_id_ori2: ", wav_id_ori.shape)
                # wav2 = vqgan_model_decoder(wav_id_ori)
                # wav2 = wav2.detach().cpu().squeeze(1).squeeze(0).numpy()
                # print("wav2: ", wav2.shape)
                # save_wav(wav2, os.path.join(uttnames[i] + '_rec.wav'))
                writer.write({
                    "__key__": uttnames[i],
                    "wav_id": wav_ids[i:i+1, :wav_id_lens[i], :].tobytes(),
                    "text": texts[i],
                    "labels": labelses[i],
                    "snr": str(snrs[i]),
                    "mos": str(moses[i]),
                    "rms_stats_rms_max": str(rms_stats_rms_maxs[i]),
                    "speaker_similarity_min": str(speaker_similarity_mins[i]),
                })

        writer.close()

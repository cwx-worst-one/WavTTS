import os
import random
import numpy as np
import webdataset as wds
import librosa

from pydub import AudioSegment
from tqdm import tqdm

from samantha.dataio.webdataset.writer import ShardWriter


def process_data(path, wav_path):
    wav, _ = librosa.load(wav_path, sr=24000)
    wav_id_path, text_id_path, _ = path.split('|')
    wav_id = np.load(wav_id_path)
    text_id = np.load(text_id_path)
    return wav_id, text_id, (wav * 32768.0).astype('int16')


def path_to_key(path):
    path = path.split('|')[0] # wav_id_path, text_id_path, len
    path = path.split('/')[-1]
    path = path.split('.')
    if len(path) >= 2:
        path = path[0:-1]
    path = '.'.join(path)
    return path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--world_size', type=int, default=1, help='world_size')
    args = parser.parse_args()

    lines = []
    with open('datas/meta_list_all.txt.filter.train.add_metalen', 'r') as f:
        lines += [l.strip() for l in f]

    random.seed(1996)
    random.shuffle(lines)
    lines = lines[args.rank::args.world_size]
    with open("/mnt/bd/huangzhiying-lq-valle-volume3/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list", 'r') as f:
        wav_list = [l.strip() for l in f]
    wav_dict = {path_to_key(w): w for w in wav_list}

    output_pattern = "hdfs://haruna/home/byte_speech_sv/user/litang/bark_data/libri_light_ver_{}/%05d.tar".format(args.rank)
    maxcount = 2000   # maximum number of samplers per shard
    maxsize = 1 << 32 # 4GiB, maximum size of each shard
    paths = lines

    with ShardWriter(output_pattern, maxcount=maxcount, maxsize=maxsize) as f:
        cnt = 0
        for path in tqdm(paths):
            try:
                if '/1400h/' in path:
                    continue
                else:
                    cnt += 1
                key = path_to_key(path).replace('.', '-_-')
                wav_path = wav_dict[path_to_key(path)]
                wav_id, text_id, wav = process_data(path, wav_path)
                item = {
                    "__key__": key,
                    "wav.npy": wav,
                    "wav_id.npy": wav_id,
                    "text_id.npy": text_id,
                }
                f.write(item)
            except Exception as e:
                print(path)
                print(e)

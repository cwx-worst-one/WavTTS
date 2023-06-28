import os
import random
import numpy as np
import webdataset as wds
import librosa

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
    with open('/mnt/bd/litang-lq-fairseq/samantha/datas/podcast_data/podcast_metalist_training.txt', 'r') as f:
        lines += [l.strip() for l in f]
    with open('/mnt/bd/litang-lq-fairseq/samantha/datas/podcast_data/podcast_metalist_dev.txt', 'r') as f:
        lines += [l.strip() for l in f]
    lines = [l.replace('/opt/tiger/liuzhengxi/podcast_data/', '/mnt/bd/litang-lq-fairseq/samantha/datas/podcast_data/') for l in lines]

    random.seed(1996)
    random.shuffle(lines)
    lines = lines[args.rank::args.world_size]
    with open("/mnt/bd/litang-lq-fairseq/datas/podcast_20230511/podcast_150h.txt", 'r') as f:
        wav_list = [l.strip().split('|')[0] for l in f]
    with open("/mnt/bd/litang-lq-fairseq/datas/podcast_20230511/podcast_150h.txt", 'r') as f:
        text_list = [l.strip().split('|')[1] for l in f]
    wav_dict = {path_to_key(w): w for w in wav_list}
    text_dict = {path_to_key(w): w for w in text_list}

    output_pattern = "hdfs://haruna/home/byte_speech_sv/user/litang/bark_data/podcast_150h/%05d.tar"
    maxcount = 2000   # maximum number of samplers per shard
    maxsize = 1 << 32 # 4GiB, maximum size of each shard
    paths = lines

    with ShardWriter(output_pattern, maxcount=maxcount, maxsize=maxsize) as f:
        cnt = 0
        for path in tqdm(paths):
            cnt += 1
            key = path_to_key(path).replace('.', '-_-')
            wav_path = wav_dict[path_to_key(path)]
            txt_path = text_dict[path_to_key(path)]
            with open(txt_path, 'r') as f_txt:
                text = f_txt.read().strip()
            wav_id, text_id, wav = process_data(path, wav_path)
            item = {
                "__key__": key,
                "wav.npy": wav,
                "wav_id.npy": wav_id,
                "text_id.npy": text_id,
                "text": text
            }
            f.write(item)

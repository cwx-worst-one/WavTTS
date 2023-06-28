import os
import ffmpeg
import numpy as np
import torch
import librosa
import webdataset as wds

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from samantha.dataio.webdataset.writer import ShardWriter
from recipes.clvp.requires.model_initializer import init_sound_stream


def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    return (np.frombuffer(seg_bin, dtype="int16"))


class CustomerDataset(Dataset):

    def __init__(self, wav_list, text_dict):
        self.wav_list = wav_list
        self.text_dict = text_dict

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path = self.wav_list[idx]
        key = '.'.join(wav_path.split('/')[-1].split('.')[0:-1])
        text = self.text_dict[key]

        audio, _ = librosa.load(wav_path, sr=24000)
        audio = audio / max(0.01, np.max(np.abs(audio))) * 0.95
        audio = torch.from_numpy(audio).float()

        return audio, text, key


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--world_size', type=int, default=1, help='world_size')
    args = parser.parse_args()

    with open('/mnt/bd/huangzhiying-lq-valle-volume1/data/original_data/ASR/en/GigaSpeech/data/wav.list.train', 'r') as f:
        wav_list = [l.strip() for l in f]

    with open('/mnt/bd/huangzhiying-lq-valle-volume1/data/original_data/ASR/en/GigaSpeech/data/text.train', 'r') as f:
        text_list = [l.strip() for l in f]
    text_dict = {t.split('\t')[0]: t.split('\t')[1] for t in text_list}

    cuda_rank = args.rank % 8
    device = 'cuda:{}'.format(cuda_rank)
    ss_encoder = init_sound_stream(
        hpath="hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export",
        local_rank=cuda_rank,
        cache_dir='ttt',
    )["ss"]

    # prepare dataset
    wav_list = wav_list[args.rank::args.world_size]
    trainset = CustomerDataset(wav_list, text_dict)
    train_loader = DataLoader(trainset,
                              num_workers=8,
                              shuffle=False,
                              sampler=None,
                              batch_size=1,
                              pin_memory=True,
                              drop_last=False)

    with torch.no_grad():
        output_pattern = "hdfs://haruna/home/byte_speech_sv/user/litang/clvp/data/ss_x300/giga-chunk-{}-of-{}/%05d.tar".format(args.rank, args.world_size)
        maxcount = 10000   # maximum number of samplers per shard
        maxsize = 1 << 32 # 4GiB, maximum size of each shard
        with ShardWriter(output_pattern, maxcount=maxcount, maxsize=maxsize) as wds_f:
            for i, loaded_data in enumerate(tqdm(train_loader)):
                audio, text, key = loaded_data
                audio = audio.to(device)
                wav_id = ss_encoder(audio)[2]
                wav_id = torch.stack(wav_id, dim=2) # [b=1, t, n_code]
                wav_id = wav_id.detach().cpu().squeeze(0).numpy()
                item = {
                    "__key__": "giga_speech/" + key[0],
                    "wav_id.npy": wav_id,
                    "text": text[0],
                }
                wds_f.write(item)

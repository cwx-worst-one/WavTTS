import json
import multiprocessing
import os
import json
import numpy as np
from tqdm import tqdm
import librosa
from pathlib import Path
import shutil
import webdataset
from samantha.dataio.webdataset import WebDataset

# Jupyter Notebook
# sanity_check('10000_0')

tar_files = 'hdfs://harunava/home/byte_speech_sv/data/karaoke_for_singsong/shards-0131.tar'

karaoke_full_dir = Path('/mnt/bn/audio-diffusion/data/karaoke_preview/full') # TAG_full.mp3
karaoke_lyrics_dir = Path('/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/lyrics/lyrics_en_only') # TAG_lyrics.json
karaoke_vocal_dir = Path('/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio') # TAG_voca.mp3

# process_lyrics()
import sys
def create_tar(input_hdfs_url):
    source_webdataset = WebDataset(input_hdfs_url)
    tar_name = Path(input_hdfs_url).name

    output_tar_path = Path('/mnt/bn/ashaw-us/processed_data/karaoke/webdataset')/tar_name
    output_tar_path.parent.mkdir(exist_ok=True, parents=True)
    lyrics_index_output_fp = f'/mnt/bn/ashaw-us/processed_data/karaoke/webdataset/{tar_name}.index'
    sink = webdataset.TarWriter(str(output_tar_path))
    lyrics_writer = open(lyrics_index_output_fp, 'w')
    for index, item in tqdm(enumerate(source_webdataset)):
        file_id = item['__key__']
        vocal_fp = karaoke_vocal_dir/f'{file_id}_voca.mp3'
        acc_fp = karaoke_vocal_dir/f'{file_id}_acco.mp3'
        lyrics_fp = karaoke_lyrics_dir/f'{file_id}_lyrics.json'
        full_fp = karaoke_full_dir/f'{file_id}_full.mp3'

        with open(vocal_fp, 'rb') as f:
            vocal_data = f.read()
        with open(acc_fp, 'rb') as f:
            acc_data = f.read()
        with open(full_fp, 'rb') as f:
            full_data = f.read()
        sink.write({
            "__key__": file_id,
            'full.mp3': full_data,
            'vocal.mp3': vocal_data,
            'acc.mp3': acc_data
        })
        
        if lyrics_fp.exists():
            with open(lyrics_fp, 'r') as f: lyrics_json = json.load(f)
            if lyrics_json is not None:
                lyrics_str = json.dumps({ 'lyrics': lyrics_json })
                lyrics_writer.write(f'{file_id}\t{lyrics_str}\n')
        
    print('Tarred files to output url:', index, output_tar_path, lyrics_index_output_fp)
    sink.close()
    lyrics_writer.close()

tar_files = [f'hdfs://harunava/home/byte_speech_sv/data/karaoke_for_singsong/shards-{idx:04d}.tar' for idx in range(132)]

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(8) as pool:
    pool.map(create_tar, tar_files)
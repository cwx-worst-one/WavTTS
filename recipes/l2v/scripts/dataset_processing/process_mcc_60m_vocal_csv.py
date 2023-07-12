import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import os
from subprocess import Popen, PIPE

data_dir = Path('/mnt/bn/audio-diffusion/data/vocal_mcc')
input_index_dir = data_dir/'asr_index_230616'
output_index_dir = data_dir/'asr_index_230616_en_only'
csv_list_fp = data_dir / 'mcc_60m_partial_csv_list_final.txt'
index_list_fp = data_dir / 'mcc_60m_partial_index_list_final.txt'

def process_csv(input_csv_fp, valid_langs=('en',)):
    input_csv_fp = Path(input_csv_fp)
    assert input_csv_fp.suffix == '.csv', f'Invalid path {input_csv_fp}'
    df = pd.read_csv(input_csv_fp)
    relative_csv_fp = input_csv_fp.relative_to(input_index_dir)
    output_index_fp = output_index_dir / relative_csv_fp.with_suffix('.index')
    if output_index_fp.exists(): return output_index_fp
    output_index_fp.parent.mkdir(parents=True, exist_ok=True)
    en_lyrics_count = 0
    with open(output_index_fp, 'w') as index_file:
        for idx, row in df.iterrows():
            lyrics_json = json.loads(row.utterances)
            metadata_json = json.loads(row.metadata)

            if len(lyrics_json) and row.lang.strip() in valid_langs:
                song_id = row.song_id
                # Only write transcriptions if there are lyrics
                json_str = json.dumps({ 'lyrics': { 'utterances': lyrics_json }, 'metadata': metadata_json })
                index_file.write(f'{song_id}\t{json_str}\n')
                en_lyrics_count += 1
        return output_index_fp

import argparse
import concurrent.futures

def csv_to_hdfs_path(csv_fp):
    hdfs_base_path = 'hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/230605_mcc/dataset' # seed path
#     hdfs_base_path = 'hdfs://harunava/home/byte_speech_sv/data/230605_mcc/dataset/'
    csv_fp = Path(csv_fp)
    splits = csv_fp.parent.name.split('.')
    paths = '.'.join(splits[:-3]), '.'.join(splits[-3:-1]), (splits[-1] + '.tar')
    paths = '/'.join(paths)
    return hdfs_base_path + '/' + paths

if not Path(csv_list_fp).exists():
    cmd = f'find {input_index_dir} -type f -name "*.csv" > {csv_list_fp}'
    print(cmd)
    os.system(cmd)

# Load csv list
with open(csv_list_fp, 'r') as f:
    csv_list = f.read().splitlines()

output_tsv = data_dir / 'mcc_60m_url2index_final.tsv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Start url index")
    parser.add_argument("--end", default=70_000, type=int, help="End url index") # 63_333 total

    args = parser.parse_args()

    with concurrent.futures.ProcessPoolExecutor(32) as executor:
        process_results = list(tqdm(executor.map(process_csv, csv_list[args.start:args.end])))
        
    with open('/mnt/bn/audio-diffusion/data/vocal_mcc/mcc60m_vocal_tar_list.txt', 'r') as f:
        tar_list = f.read().splitlines()

    if not Path(index_list_fp).exists():
        cmd = f'find {output_index_dir} -type f -name "*.index" > {index_list_fp}'
        print(cmd)
        os.system(cmd)

    # Load csv list
    with open(index_list_fp, 'r') as f:
        index_list = f.read().splitlines()


    min_num = 200
    with open(output_tsv, 'w') as f:
        for index_fp in index_list:
            if not Path(index_fp).exists(): continue
            with open(index_fp, 'r') as url_f:
                lines = url_f.read().splitlines()
            if len(lines) < min_num:
                print('Skipping', len(lines))
                continue
            hdfs_path = csv_to_hdfs_path(index_fp)
            if hdfs_path in tar_list:
                f.write(f'{hdfs_path}\t{index_fp}\n')
            else:
                print('Could not find', hdfs_path)

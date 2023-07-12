import webdataset
import json
from tqdm import tqdm
from subprocess import Popen, PIPE
from pathlib import Path
import os
import shutil

def format_key(key):
    if key.startswith('./'):
        key = key[2:]
    else:
        print('Error formatting key', key)
    return key

def get_lyrics_json(key, lyrics_unzip_fp):
    lyrics_txt_fp = Path(lyrics_unzip_fp)/'output'/f'{key}.txt'
    try:
        with open(lyrics_txt_fp, 'r') as f:
            jout = json.load(f)
        jout = jout['resp']['results'][0]
        if len(jout['utterances']) == 0:
            print('No utterances')
            return None
        
        # # Remove extra spaces to save space
        # words = jout['utterances'][0]['words']
        # jout['utterances'][0]['words'] = [row for row in words if row['text'].strip()]
        
    except Exception as e:
        # print(e)
        return None
    return jout

def upload_to_hdfs(local_tar, hdfs_path):
    # Upload to hdfs
    cmd = ['hdfs', 'dfs', '-put', 
     '-f', str(local_tar), 
     hdfs_path]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (lines, err) = proc.communicate()
    if proc.returncode != 0:
        print(err.decode("utf-8"))
    elif err:
        print('stderr:\n' + err.decode("utf-8"))

def tar_exists(output_url):
    cmd = ['hdfs', 'dfs', '-ls', str(output_url)]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    (lines, err) = proc.communicate()

    tar_exists = False
    if proc.returncode != 0:
        err = err.decode("utf-8")
#         print(err)
        if 'No such file' in err:
            tar_exists = False
    else:
        lines = lines.decode("utf-8")
        if str(output_url) in lines:
            tar_exists = True
#         print(lines)
    return tar_exists


def process_fs_hdfs_url(item):
    idx, fs_hdfs_url = item

    if 'full_song_107.tar' not in fs_hdfs_url: return
    # File paths
    tar_name = Path(fs_hdfs_url).name
    file_id = int(Path(tar_name).stem.split('_')[-1])
    fs_tar_local_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/input_tars/{tar_name}'
    fs_tar_output_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/output_tars/{tar_name}'
    fs_hdfs_output_url = fs_hdfs_url.replace('9M_en_fullsong/full_song', '9M_en_fullsong/full_song_tars')
    
    
    lyrics_hdfs_base_url = 'hdfs://harunava/home/byte_speech_sv/data/9M_en_fullsong/aligned_wordlevel'
    lyrics_hdfs_url = lyrics_hdfs_base_url + '/' + f'aligned_{file_id*10000}_{file_id*10000+10000}_wordlevel.tar'
    lyrics_tar_name = f'lyrics_{file_id}.tar'
    lyrics_tar_local_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/input_tars/{lyrics_tar_name}'
    lyrics_unzip_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/lyrics_unzip/{lyrics_tar_name[:-4]}'
    Path(lyrics_unzip_fp).mkdir(exist_ok=True, parents=True)
    lyrics_tar_output_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/output_tars/{lyrics_tar_name}'
    lyrics_hdfs_output_url = f'hdfs://harunava/home/byte_speech_sv/data/9M_en_fullsong/aligned_wordlevel_tars/{lyrics_tar_name}'
    lyrics_index_output_fp = f'/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/indexes/{tar_name}.index'
    
    # if tar_exists(lyrics_hdfs_output_url):
    #     print('HDFS tar exists already:', lyrics_hdfs_output_url)
    #     return
    
    print('Processing url:', idx, fs_hdfs_url)
    # Copy tars to local
    if not Path(fs_tar_local_fp).exists():
        cmd = f'hdfs dfs -copyToLocal {fs_hdfs_url} {fs_tar_local_fp}'
        os.system(cmd)
    # copy lyrics to local
    if not Path(lyrics_tar_local_fp).exists():
        # copy local
        cmd = f'hdfs dfs -copyToLocal {lyrics_hdfs_url} {lyrics_tar_local_fp}'
        os.system(cmd)
        # unzip lyrics
        cmd  = f'tar -xf {lyrics_tar_local_fp} -C {lyrics_unzip_fp}'
        out = os.system(cmd)

    # Process tars
    lyrics_writer = open(lyrics_index_output_fp, 'w')
    fs_sink = webdataset.TarWriter(str(fs_tar_output_fp))
    lyrics_sink = webdataset.TarWriter(str(lyrics_tar_output_fp))

    dataset = webdataset.WebDataset(fs_tar_local_fp).decode()
    it = iter(dataset)
    for index, item in tqdm(enumerate(it), position=idx):
        key = format_key(item['__key__'])
        lyrics_json = get_lyrics_json(key, lyrics_unzip_fp)

        if lyrics_json is not None:
            lyrics_str = json.dumps({ 'lyrics': lyrics_json })
            lyrics_writer.write(f'{key}\t{lyrics_str}\n')

        fs_sink.write({
            "__key__": key,
            "mp3": item['mp3']
        })
        lyrics_sink.write({
            "__key__": key,
            "lyrics.json": lyrics_json
        })

    fs_sink.close()
    lyrics_sink.close()
    lyrics_writer.close()
    
    upload_to_hdfs(fs_tar_output_fp, fs_hdfs_output_url)
    upload_to_hdfs(lyrics_tar_output_fp, lyrics_hdfs_output_url)
    print('Tarred files complete:', fs_hdfs_url, fs_tar_output_fp, fs_hdfs_output_url)
    print('Indexes', fs_hdfs_output_url, lyrics_index_output_fp)
    Path(fs_tar_output_fp).unlink()
    Path(lyrics_tar_output_fp).unlink()
    Path(fs_tar_local_fp).unlink()
    Path(lyrics_tar_local_fp).unlink()
    shutil.rmtree(lyrics_unzip_fp)


# 9M fullsong
full_song_tars_hdfs = 'hdfs://harunava/home/byte_speech_sv/data/9M_en_fullsong/full_song/'

# 9M lyrics data
lyrics_tars_hdfs = 'hdfs://harunava/home/byte_speech_sv/data/9M_en_fullsong/aligned_wordlevel/'

lyrics_tar_list_fp = '/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/9M_en_fullsong_aligned_wordlevel_tar_files.txt'
full_song_tar_list_fp = '/mnt/bn/ashaw-us/data/lyrics_transcription/mcc9m/9M_en_fullsong_tar_files.txt'

if not Path(lyrics_tar_list_fp).exists():
    cmd = f"hdfs dfs -ls {lyrics_tars_hdfs}* | awk '{{print $8}}' > {lyrics_tar_list_fp}"
    os.system(cmd)

if not Path(full_song_tar_list_fp).exists():
    cmd = f"hdfs dfs -ls {full_song_tars_hdfs}* | awk '{{print $8}}' > {full_song_tar_list_fp}"
    os.system(cmd)

with open(full_song_tar_list_fp, 'r') as f:
    full_song_tar_list = f.read().splitlines()


import argparse
import concurrent.futures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int, help="Start url index")
    parser.add_argument("--end", default=140, type=int, help="End url index")

    args = parser.parse_args()

    with concurrent.futures.ProcessPoolExecutor(32) as executor:
        process_results = list(tqdm(executor.map(process_fs_hdfs_url, enumerate(full_song_tar_list[args.start:args.end]))))


# mlx worker launch --gpu 0 --cpu 16 --memory 128 -- python3 /mnt/bn/ashaw-us/repos/samantha/recipes/l2v/tokenizers/process_mcc.py --start 20 --end 40
# from samantha.dataio.webdataset.ra_wds import tar_file_iterator
from samantha.dataio.webdataset import WebDataset
from pathlib import Path
import json
# import tqdm
import concurrent.futures
from io import BytesIO
from recipes.musiclm.transforms.audio import (
    ReadMP3
)
lyrics_json_dir = Path('/mnt/bn/ashaw-us/data/lyrics_transcription/resso/lyrics/json')
output_dir = Path('/mnt/bn/ashaw-us/data/lyrics_transcription/resso/mss_filtered/indexed_webdataset/indexes')
output_dir.mkdir(exist_ok=True, parents=True)
index_list_fp = output_dir.parent/'resso.tar_to_index.tsv'

shard_list_original_fp = '/mnt/bn/ashaw-us/data/lyrics_transcription/resso/mss/resso_1.3m_mss_shard_list.txt'

with open(shard_list_original_fp, 'r') as f: 
    shard_list_original = f.read().splitlines()

def process_tar(shard_fp):
    output_index_fp = output_dir/(Path(shard_fp).stem+'.index')
    mp3_reader = ReadMP3(sample_rate=24000)
    print('Output Index:', output_index_fp)
    try:
        it = WebDataset(shard_fp)
    except Exception as e:
        print('Exception', e)
        return (None, None, 0, 10)
        
    with open(output_index_fp, 'w') as index_file:

        lyrics_count = 0
        file_count = 0

        for idx, item in enumerate(it):
            file_count += 1
            file_id = item['__key__']
            json_fp = lyrics_json_dir/f'{file_id}.json'
            if not json_fp.exists(): continue
            with open(json_fp, 'r') as j:
                j = json.load(j)
            # Only write transcriptions if there are lyrics
            if 'utterances' not in j or not j['utterances']:
                continue

            try:
                mp3_reader(BytesIO(item['mss_vocal']))
                mp3_reader(BytesIO(item['mss_acc']))
            except Exception as e:
                print('Error reading mp3', e)
                continue

            json_str = json.dumps({ 'lyrics': j })
            index_file.write(f'{file_id}\t{json_str}\n')
            lyrics_count += 1
        print('Wrote', lyrics_count, file_count, output_index_fp)
        return shard_fp, output_index_fp, lyrics_count, file_count

# with concurrent.futures.ProcessPoolExecutor(64) as pool:
#     results = pool.map(process_tar, shard_list_original)

# with open(index_list_fp, 'w') as index_list_file:
#     for shard_fp, output_index_fp, lyrics_count, file_count in results:
#         if lyrics_count / file_count < 0.3 or file_count < 100:
#             print('Skipping', lyrics_count, '/', file_count)
#         else:
#             index_list_file.write(f'{shard_fp}\t{str(output_index_fp)}\n')
        

min_num = 120
with open(index_list_fp, 'w') as f:
    for s in shard_list_original:
        tar_id = Path(s).stem
        index_fp = output_dir/f'{tar_id}.index'
        if not index_fp.exists(): continue
        with open(index_fp, 'r') as url_f:
            lines = url_f.read().splitlines()
        if len(lines) < min_num:
            print('Skipping', len(lines))
            continue
        f.write(f'{s}\t{str(index_fp)}\n')
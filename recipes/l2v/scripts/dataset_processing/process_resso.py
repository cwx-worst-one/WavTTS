from samantha.dataio.webdataset.ra_wds import tar_file_iterator
from samantha.dataio.webdataset import WebDataset
from pathlib import Path
import json
import concurrent.futures

shard_dir = Path('/mnt/bn/ashaw-us/data/lyrics_transcription/resso/lyrics/shard')
output_dir = Path('/mnt/bn/ashaw-us/data/lyrics_transcription/resso/indexed_webdataset/indexes')
output_dir.mkdir(exist_ok=True)
index_list_fp = output_dir.parent/'resso.tar_to_index.tsv'

shard_list_original_fp = '/mnt/bn/audio-diffusion/data/resso_original_shard_list.txt'

with open(shard_list_original_fp, 'r') as f: 
    shard_list_original = f.read().splitlines()

def process_tar(original_shard_fp):
    lyrics_shard_fp = shard_dir/Path(original_shard_fp).name
    output_index_fp = output_dir/(Path(lyrics_shard_fp).stem+'.index')
    print('Output Index:', output_index_fp)
    try:
        it = tar_file_iterator(open(lyrics_shard_fp, 'rb'))
        next(it) # skip empty files
        it = tar_file_iterator(open(lyrics_shard_fp, 'rb'))
    except Exception as e:
        print('Exception', e)
        return (None, None, None, 0, 10)
        
    with open(output_index_fp, 'w') as index_file:

        lyrics_count = 0
        file_count = 0
        for idx, item in enumerate(it):
            file_count += 1

            name = item['fname'].replace('.json', '')
            data = item['data'].decode()
            j = json.loads(data)
            # Only write transcriptions if there are lyrics
            if 'utterances' in j and j['utterances']:
                json_str = json.dumps({ 'lyrics': j })
                index_file.write(f'{name}\t{json_str}\n')
                lyrics_count += 1
        print('Wrote', lyrics_count, file_count, output_index_fp)
        return original_shard_fp, lyrics_shard_fp, output_index_fp, lyrics_count, file_count

with concurrent.futures.ProcessPoolExecutor(16) as pool:
    results = list(pool.map(process_tar, shard_list_original))

# with open(index_list_fp, 'w') as index_list_file:
#     for original_shard_fp, lyrics_shard_fp, output_index_fp, lyrics_count, file_count in results:
#         if lyrics_count / file_count < 0.3:
#             print('Skipping', lyrics_count, '/', file_count)
#         else:
#             index_list_file.write(f'{original_shard_fp}\t{str(output_index_fp)}\n')

min_num = 150
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
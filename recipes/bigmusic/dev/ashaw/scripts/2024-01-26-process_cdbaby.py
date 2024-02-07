import pandas as pd
import json
from tqdm import tqdm
from samantha.dataio.parquet.writer import ShardWriter, ParquetWriter
from pathlib import Path
import simplejson as json
import functools
import os
os.chdir('/mnt/bn/ashaw-lq/repos/samantha')
dump_fn = functools.partial(json.dumps, ensure_ascii=False, ignore_nan=True)

## Process jsonlines to index map
# df = pd.read_csv('/mnt/bn/ashaw-lq/data/cd_baby/scrape/all.jsonl', sep='\t', header=None)
# id2meta = {}
# for meta_str in tqdm(df[df.columns[1]].values):
#     meta = json.loads(meta_str)
#     if 'isrc' in meta:
#         id2meta[meta['isrc'].upper()] = meta
# with open('/mnt/bn/ashaw-lq/data/cd_baby/scrape/all.json', 'w') as f:
#     json.dump(id2meta, f)


with open('/mnt/bn/ashaw-lq/data/cd_baby/scrape/all.json', 'r') as f:
    id2meta = json.load(f)
# index_parquet_fp = '/mnt/bn/ashaw-lq/data/cd_baby/index_files/index_4/shard_00000.index_4.parquet'

output_root = '/mnt/bn/ashaw-lq/data/cd_baby/index_files'
filename_pattern="shard-%05d.parquet"
idx_version=6
# partition_path = "/".join(partitions)
partition_path = ""
idx_pattern = os.path.join(
    output_root, f"index_{idx_version}", partition_path, filename_pattern
)
index_parquet_fps = list(Path('/mnt/bn/ashaw-lq/data/cd_baby/index_files/index_4').glob('*.parquet'))
# meta_dict = j[0]
for index_parquet_fp in tqdm(index_parquet_fps):
    df_index = pd.read_parquet(index_parquet_fp)
    index_records = df_index.to_dict(orient='records')
    
    index_fp = Path(f'/mnt/bn/ashaw-lq/data/cd_baby/index_files/index_{idx_version}/')/index_parquet_fp.name
    index_fp = str(index_fp).replace('index_4', f'index_{idx_version}')
    Path(index_fp).parent.mkdir(exist_ok=True)
    idx_writer = ParquetWriter(index_fp, row_group_size=32000, need_row_group_no=False)
    
    for meta_dict in index_records:
        meta_dict = {**meta_dict}
        meta = json.loads(meta_dict.pop('meta'))
        isrc = meta['meta_song_isrc']
        if isrc not in id2meta:
            continue
        meta['meta_cd_baby'] = id2meta[isrc]
        meta_dict['meta'] = dump_fn(meta)
        idx_writer.write(meta_dict)
    idx_writer.close()

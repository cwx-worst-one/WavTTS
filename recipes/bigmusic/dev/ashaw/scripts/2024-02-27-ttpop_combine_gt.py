import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm

spot_dir = Path('/mnt/bn/ashaw-lq/data/tt_pop/BigMusic_groupA_28k_newGenres')
spot_index_fps = list(spot_dir.glob('**/*.parquet'))
# df_spot = pd.read_parquet(spot_index_fps[0])
# json.loads(df_spot.meta[0])

gt_dir = Path('/mnt/bn/ashaw-lq/data/tt_pop/music_Smcc_Mvocal_PA_Tttpopularity_N666k')
gt_index_fps = list(gt_dir.glob('**/*.parquet'))
# df_gt = pd.read_parquet(gt_index_fps[0])

isrc2meta = {}
# duplicates = defaultdict(list)
for gt_index_fp in tqdm(gt_index_fps):
    df_gt = pd.read_parquet(gt_index_fp)
    for meta_string in df_gt.meta.values:
        meta = json.loads(meta_string)
        isrc = meta['meta_song_isrc']
#         if isrc in isrc2meta:
# #             print('Duplicate isrc found', isrc, meta['meta_song_title'], meta['full_duration'])
#             duplicates[isrc].append(isrc2meta[isrc])
#             duplicates[isrc].append(meta)
        isrc2meta[isrc] = meta

# with open('/mnt/bn/ashaw-lq/data/tt_pop/isrc2gt_map.json', 'w') as f:
#     json.dump(isrc2meta, f)


###### PART 2 #####

import pandas as pd
import json
from tqdm import tqdm
from samantha.dataio.parquet.writer import ShardWriter, ParquetWriter
from pathlib import Path
import simplejson as json
import functools
import os

dump_fn = functools.partial(json.dumps, ensure_ascii=False, ignore_nan=True)

# with open('/mnt/bn/ashaw-lq/data/tt_pop/isrc2gt_map.json', 'r') as f:
#     isrc2meta = json.load(f)

output_root = '/mnt/bn/ashaw-lq/data/tt_pop/BigMusic_groupA_28k_newGenres'
filename_pattern="shard-%05d.parquet"
idx_version_prev=3
idx_version=4
partition_path = ""
idx_pattern = os.path.join(
    output_root, f"index_{idx_version}", partition_path, filename_pattern
)
index_parquet_fps = list(Path(f'/mnt/bn/ashaw-lq/data/tt_pop/BigMusic_groupA_28k_newGenres/index_{idx_version_prev}').glob('*.parquet'))

len(index_parquet_fps)

for index_parquet_fp in tqdm(index_parquet_fps):
    df_index = pd.read_parquet(index_parquet_fp)
    index_records = df_index.to_dict(orient='records')
    
    index_fp = Path(f'/mnt/bn/ashaw-lq/data/tt_pop/BigMusic_groupA_28k_newGenres/index_{idx_version}/')/index_parquet_fp.name
    index_fp = str(index_fp).replace(f'index_{idx_version_prev}', f'index_{idx_version}')
    Path(index_fp).parent.mkdir(exist_ok=True)
    idx_writer = ParquetWriter(index_fp, row_group_size=32000, need_row_group_no=False)
    
    for meta_dict in index_records:
        meta_dict = {**meta_dict}
        meta = json.loads(meta_dict.pop('meta'))
        isrc = meta['meta_song_isrc']
        if isrc in isrc2meta:
            meta = { **isrc2meta[isrc], **meta }
        meta_dict['meta'] = dump_fn(meta)
        idx_writer.write(meta_dict)
    idx_writer.close()

import json
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import subprocess
import multiprocessing as mp
import pandas as pd
import sys
import json
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
from samantha.dataio.utils import resolve_data_urls
import pandas as pd

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def export_meta(idx_csv_pair):
    idx_pq, csv_file = idx_csv_pair        
    df = pd.read_parquet(idx_pq)
    meta_list = []
    for row in df.to_dict(orient="record"):        
        meta = json.loads(row["meta"])                
        tagging = meta['music_tagging']
        export_meta = {
            "deepchorus": meta['deepchorus']['segments'], 
            "timestamped_lyrics": meta['lyrics'],
            "genre": tagging['Genre20']['result'],
            "mood": tagging['Mood']['result'],
            "is_sinking": True if tagging['MusicLowQuality']['Sinking'] > 0.4 else False,
            "song_name": meta["song_name"],
            "artist_name": meta["artist_name"]}
        meta_list.append(export_meta)
    write_jsonl(meta_list, csv_file)

urls = resolve_data_urls(data_id=1526, data_urls=None)
idx_pqs = [i["index"] for i in urls]
csv_dir = 'music_zh_copyright_deepchorus_SATag/'
csv_files = [csv_dir+'.'.join(Path(idx_pq).name.split('.')[:-1] + ['jsonl']) for idx_pq in idx_pqs]


with mp.Pool(48) as pool:
    work = pool.imap_unordered(export_meta, zip(idx_pqs, csv_files))
    for _ in tqdm(work, total=len(idx_pqs)):
        pass

import os
import json

def merge_jsonl_files(input_dir, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith('.jsonl'):
                with open(os.path.join(input_dir, filename), 'r') as infile:
                    for line in infile:
                        outfile.write(line)

# Example usage
input_directory = 'music_zh_copyright_deepchorus_SATag'  # Replace this with the path to your directory
output_file = 'merged_lyrics_deepchorus_SATag.jsonl'  # Name of the output file

merge_jsonl_files(input_directory, output_file)
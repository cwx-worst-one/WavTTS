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
from webdataset import WebDataset
import subprocess
import multiprocessing as mp


import sys


def slice_idx(tar_idx):
    # label ID 1:WMG, 3:SMG , 4:UMG
    _, idx_file = tar_idx
    idx_file_name = Path(idx_file).name
    idx_names = ["groupA", "groupB", "1_WMG", "3_SMG", "4_UMG"]
    all_done = True
    for idx_name in idx_names:
        index_output_fp = f'/mnt/bn/audio-diffusion/data/mcc60_slices/{idx_name}/{idx_file_name}'
        if not os.path.exists(index_output_fp):
            all_done = False 
    if not all_done:            
        groupA_idx = []
        groupB_idx = []
        label1_idx = []
        label3_idx = []
        label4_idx = []
        with open(idx_file, 'r') as f:
            idx_list = f.read().splitlines()
            for fileid_idx in idx_list:
                xx = fileid_idx.split('\t', 1)
                if len(xx) != 2:
                    # print("fileid_idx pair length !=2")
                    continue
                _, idx = xx                            
                try:
                    idx_dict = json.loads(idx)
                except:
                    # print(idx)
                    continue
                label_id = int(idx_dict.get('label_id', 0))
                for license in idx_dict.get("license_types", []):
                    if license == 'A':
                        groupA_idx.append(fileid_idx)
                        if label_id == 1:
                            label1_idx.append(fileid_idx)                        
                        elif label_id == 3:
                            label3_idx.append(fileid_idx)                        
                        elif label_id == 4:
                            label4_idx.append(fileid_idx)                        
                    if license == 'B':                    
                        groupB_idx.append(fileid_idx)           
        idx_file_name = Path(idx_file).name
        idx_lists = [groupA_idx, groupB_idx, label1_idx, label3_idx, label4_idx]
        for idx_list, idx_name in zip(idx_lists, idx_names):
            index_output_fp = f'/mnt/bn/audio-diffusion/data/mcc60_slices/{idx_name}/{idx_file_name}'
            index_writer = open(index_output_fp, 'w')
            for idx in idx_list:
                index_writer.write(idx+'\n')


if __name__ == "__main__":
    # write url2inx file for slices
    idx_names = ["groupA", "groupB", "1_WMG", "3_SMG", "4_UMG"]
    for idx_name in idx_names:
        os.makedirs(f'/mnt/bn/audio-diffusion/data/mcc60_slices/{idx_name}', exist_ok=True)

    with open('/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt', 'r') as f:
        tar_idx_list = f.read().splitlines()
        tar_idx_list = [x.split('\t') for x in tar_idx_list]

    with mp.Pool(64) as pool:
        work = pool.imap_unordered(slice_idx, tar_idx_list)
        for _ in tqdm(work, total=len(tar_idx_list)):
            pass


    for idx_name in idx_names:
        url2inx_fp = f'/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_{idx_name}.txt'        
        url2inx_writer = open(url2inx_fp, 'w')
        for tar_idx in tar_idx_list:
            tar_file, idx_file = tar_idx
            idx_file_name = Path(idx_file).name
            new_idx_file_name = f'/mnt/bn/audio-diffusion/data/mcc60_slices/{idx_name}/{idx_file_name}'
            url2inx_writer.write(f'{tar_file}\t{new_idx_file_name}\n')
        url2inx_writer.close()
    
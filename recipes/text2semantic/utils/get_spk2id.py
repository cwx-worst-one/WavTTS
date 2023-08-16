import json
import re
import subprocess

hdfs_root = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/data/bigtts/WFVAE_v2_labv3_punc'
out_json_path = '/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/datasets/dict/sft_spk_0812.json'
spk_dict = {}

def get_hdfs_ls(hdfs_path):
    command = 'hdfs dfs -ls %s' % hdfs_path
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    pattern = hdfs_root + r'[^ \n]*'
    matches = re.findall(pattern, output)
    return matches

matches = get_hdfs_ls(hdfs_root)
count = 0
for dataset_path in matches:
    dataset_name = dataset_path.split('/')[-1]
    spk_matches = get_hdfs_ls(dataset_path)
    for spk_path in spk_matches:
        spk_name = spk_path.split('/')[-1]
        key = '/'.join([dataset_name, spk_name])
        if key not in spk_dict:
            spk_dict[key] = count
            count += 1

print(spk_dict)
with open(out_json_path, "w") as json_file:
    json.dump(spk_dict, json_file)

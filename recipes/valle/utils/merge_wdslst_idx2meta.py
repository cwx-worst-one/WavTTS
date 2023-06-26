from genericpath import exists
import sys, os
import json

in_root_dir = sys.argv[1]
in_dirname_list = sys.argv[2]
out_dir = sys.argv[3]

os.makedirs(out_dir, exist_ok=True)

in_dirnames = in_dirname_list.split(',')
train_wds_lsts = []
valid_wds_lsts = []
wds2meta_jsons = []

for in_dirname in in_dirnames:
    in_dir = os.path.join(in_root_dir, in_dirname)
    train_wds_lst = os.path.join(in_dir, 'train_wds.lst')
    valid_wds_lst = os.path.join(in_dir, 'valid_wds.lst')
    wds2meta_json = os.path.join(in_dir, 'wds2meta.json')
    if not os.path.exists(train_wds_lst) or not os.path.exists(valid_wds_lst) or not os.path.exists(wds2meta_json):
        print("some info not exists, exit()", in_dir)
        exit()
    train_wds_lsts.append(train_wds_lst)
    valid_wds_lsts.append(valid_wds_lst)
    wds2meta_jsons.append(wds2meta_json)

out_train_wds_lst_path = os.path.join(out_dir, 'train_wds.lst')
f_w = open(out_train_wds_lst_path, 'w')
for train_wds_lst in train_wds_lsts:
    lines = open(train_wds_lst).readlines()
    for line in lines:
        f_w.write(line)
f_w.close()

out_valid_wds_lst_path = os.path.join(out_dir, 'valid_wds.lst')
f_w = open(out_valid_wds_lst_path, 'w')
for valid_wds_lst in valid_wds_lsts:
    lines = open(valid_wds_lst).readlines()
    for line in lines:
        f_w.write(line)
f_w.close()

out_wds2meta_json_path = os.path.join(out_dir, 'wds2meta.json')
out_wds2meta_json = dict()
for wds2meta_json in wds2meta_jsons:
    f = open(wds2meta_json)
    data = json.load(f)
    out_wds2meta_json.update(data)

with open(out_wds2meta_json_path, "w", encoding="utf-8") as fo:
    json.dump(out_wds2meta_json, fo, ensure_ascii=False, indent=2)
import sys, os
import json
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def convert_v3_to_v1(tacolab, lab_path):
    tacolab_v1 = []
    if len(tacolab) == 0:
        print("Empty tacolab", lab_path)
        return None
    if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit':
        tacolab = tacolab[1:]
    for x in tacolab:
        x_split = x.split('\t')
        if len(x_split) == 7:
            phone, tone, ws, pw, stype, word, _ = x_split
        elif len(x_split) == 6:
            phone, tone, ws, pw, stype, word = x_split
        else:
            print("Wrong tacolab_version", lab_path)
            return None
        tacolab_v1.append('\t'.join([phone, tone, '0.0 0.0 0.0 1.0', ws, pw]))
    return tacolab_v1

def sub_process(labdir):
    lab_list = []
    lab_names = os.listdir(labdir)
    for lab_name in tqdm(lab_names):
        if lab_name[-4:] != '.lab':
            continue
        lab_path = os.path.join(labdir, lab_name)
        f = open(lab_path)
        lines = f.readlines()
        f.close()
        lines = convert_v3_to_v1([line.strip('\n') for line in lines], lab_path)
        if lines is None:
            continue

        for i in range(len(lines)):
            line = lines[i]
            if i != 0 and line.split('\t')[0] == 'sil':
                continue
            phone, tone, _, ws, pw = line.split('\t')
            lab = '_'.join([phone, tone, ws, pw])
            if lab not in lab_list:
                lab_list.append(lab)
    return (lab_list)

labdir_list = sys.argv[1]
lab2id_path = sys.argv[2]

f = open(labdir_list)
lines = f.readlines()
f.close()

executor = ProcessPoolExecutor(max_workers=32)
futures = []
for line in tqdm(lines):
    labdir = line.strip()
    futures.append(executor.submit(partial(sub_process, labdir)))
results = [ future.result() for future in tqdm(futures)]

lab_list = []
for result in results:
    lab_list = lab_list + result
lab_list = list(set(lab_list))
lab_list.sort()

lab2id = dict()
for i in range(len(lab_list)):
    lab2id[lab_list[i]] = i

with open(lab2id_path, "w") as f:
    json.dump(lab2id, f, ensure_ascii=False, indent=2)
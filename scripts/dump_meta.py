''' dump meta file '''

import sys
import os
import json
import numpy as np
import pickle
from core.dataset import ScpDictionary
from falconpai import FalconReader

def write_dict_to_file(data, filename):
    if data is None or not filename:
        return
    with open(filename, 'w') as fp:
        items = []
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, ScpDictionary):
            items = data.indices.items()
        for k, v in items:
            fp.write("{}\t{}\n".format(k, v))

def main():
    if len(sys.argv) < 3:
        print("Usage:\n\t python3 %s hdfs_meta_path local_meta_dir\n" % sys.argv[0])
        exit(1)

    meta_path = sys.argv[1]
    out_dir = sys.argv[2]

    reader = FalconReader(meta_path)

    keys = reader.list_keys()
    vals = reader.read_many(list(range(len(keys))))
    vals = sum(vals, [])

    vals = [pickle.loads(t) for t in vals]

    os.makedirs(out_dir, exist_ok=True)

    vocab_size = 0
    # keys in meta: ['cmvn_mean', 'cmvn_var', 'reorder_dict_map', 'reorder_tgt_dict', 'tgt_dict', 'total.code']
    print('keys in meta:', keys)
    for key, val in zip(keys, vals):
        if key in ('cmvn_mean', 'cmvn_var'):
            with open("%s/%s" % (out_dir, key), 'w') as fp:
                np.savetxt(fp, val)
            print(key + ' vec:\n', val)
        elif key == 'total.code':
            with open("%s/%s.txt" % (out_dir, key), 'w') as fp:
                fp.write(val)
        else:
            if key == 'reorder_tgt_dict':
                vocab_size = len(val)
            out_dict = "%s/%s.txt" % (out_dir, key)
            write_dict_to_file(val, out_dict)

    if vocab_size > 0:
        print('vocab_size:', len(val))


if __name__ == '__main__':
    main()

import sys, os
import json
from samantha.dataio.utils import get_dataset_collection_info, __expand_paths
from tqdm import tqdm

data_id = int(sys.argv[1])
url2tag_path = sys.argv[2]
wds2tag_path = sys.argv[3]

# urls from data_id
os.environ["DatasetID"] = str(data_id)
path_list = get_dataset_collection_info(data_id)
paths = [v["data"].replace("\n", " ") for v in path_list]

# url2tag
with open(url2tag_path) as f:
    url2tag = json.load(f)

# check
assert len(paths) == len(url2tag.keys()), (len(paths), len(url2tag.keys()))
for path in paths:
    assert path in url2tag.keys(), (path)

# wds2tag
wds2tag = dict()
for path in tqdm(paths):
    cur_paths = [path]
    cur_tag = url2tag[path]

    cur_wdses = __expand_paths(cur_paths)
    assert len(cur_wdses) != 0, (cur_paths)
    for cur_wds in cur_wdses:
        wds2tag[cur_wds] = cur_tag

with open(wds2tag_path, "w") as f_w:
    json.dump(wds2tag, f_w, ensure_ascii=False, indent=2)
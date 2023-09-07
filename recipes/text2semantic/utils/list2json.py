import sys
import json

list_path = sys.argv[1]
json_path = sys.argv[2]

f = open(list_path)
lines = f.readlines()
f.close()

lab_list = [line.strip() for line in lines]
lab_list.sort()

lab2id = dict()
for i in range(len(lab_list)):
    lab2id[lab_list[i]] = i

with open(json_path, "w") as f:
    json.dump(lab2id, f, ensure_ascii=False, indent=2)
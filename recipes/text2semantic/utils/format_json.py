import sys
import json

in_json_path = sys.argv[1]
out_json_path = sys.argv[2]

f = open(in_json_path)
data = json.load(f)

with open(out_json_path, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
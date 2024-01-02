import json


def get_meta_obj(sample):
    meta_obj = json.loads(sample["meta"])
    while not isinstance(meta_obj, dict):
        meta_obj = json.loads(meta_obj)
    return meta_obj

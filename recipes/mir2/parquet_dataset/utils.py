
def combine_tag_types(datasets: list=[]):
    combined_tag_types = {}
    for dataset in datasets:
        for k, v in dataset.items():
            combined_tag_types[k] = v
    return combined_tag_types

def get_tag_type_to_dataset(datasets: list=[], dataset_ids: list=[]):
    tag_type_to_dataset = {}
    for dataset, dataset_id in zip(datasets, dataset_ids):
        for tag_type in dataset:
            tag_type_to_dataset[tag_type] = dataset_id
    return tag_type_to_dataset

def get_tag_type_to_id(tag_types: dict={}):
    tag_type_to_id = {}
    idx = 0
    for tag_type in tag_types:
        tag_type_to_id[tag_type] = idx
        idx += 1
    return tag_type_to_id
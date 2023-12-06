from bytedance import easycycle
from lightning_fabric.utilities.cloud_io import get_filesystem


def find_all_parts(root_path, filesystem=None):
    if filesystem is None:
        filesystem = get_filesystem(root_path)
    _, parts, _ = next(filesystem.walk(root_path, maxdepth=1))
    return parts


def get_dataset_name(dataset_id):
    return easycycle.get_dataset_detail(dataset_id)["dataset"]["name"]


def get_partition(path, fs):
    partitions, suffix = [], None
    path = f"{path}/data"
    while path is not None:
        for item in fs.listdir(path):
            if item["type"] == "directory":
                basename = item["name"].split("/")[-1]
                if "=" in basename:
                    partitions.append(basename.split("=")[0])
                path = item["name"]
                break
            elif not item["name"].endswith("SUCCESS"):
                path = None
                suffix = item["name"].split(".")[-1]
                break
    return partitions, suffix


if __name__ == "__main__":
    parts = find_all_parts(
        "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/tts_Lmand_Smagicdata_P1/data"
    )
    print(parts)

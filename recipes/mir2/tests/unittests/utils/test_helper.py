from recipes.mir2.utils.helper import HdfsFileWrapper


def test_hdfs_file_wrapper():
    import torch

    file_path = "hdfs://haruna/home/byte_speech_sv/haonanchen/logs/musicfm/FM7.pt"
    local_cache_dir = "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/"
    hdfs_wrapper = HdfsFileWrapper(file_path, local_cache_dir)
    model = torch.load(hdfs_wrapper)
    with open(hdfs_wrapper, "rb") as f:
        content = f.read()

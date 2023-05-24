import time

from webdataset import WebDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset


tar_path = "pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/genre_balanced_mcc_clean/classical.2.57/0.tar"


def time_it(dataset):
    start = time.time()
    for _ in dataset:
        pass
    end = time.time()
    return round(end - start, 2)


def time_it_dataset(tar_path, index_file=None, repeat=1):
    if index_file is None:
        # Iterate through entire tar
        dataset = WebDataset(tar_path).decode()
        times = []
        times += [time_it(dataset) for _ in range(repeat)]
    else:
        # Iterate through a few items by indexed file
        dataset = IndexedWebDataset({tar_path: index_file}).decode()
        times = []
        times += [time_it(dataset) for _ in range(repeat)]
    return times


if __name__ == "__main__":
    # Test full tar: 100s
    full = time_it_dataset(tar_path, repeat=3)
    print(f"Full tar: {full}")

    # Index with all keys: 15s
    all_index = "hdfs://harunava/home/byte_speech_sv/duc.le1/index_test/classical.2.57-0.index"
    all_time = time_it_dataset(tar_path, index_file=all_index, repeat=3)
    print(f"All Random Access: {all_time}")

    # Index at beginning random access: 3s
    head_index = "hdfs://harunava/home/byte_speech_sv/duc.le1/index_test/classical.2.57-0.index_head"
    head = time_it_dataset(tar_path, index_file=head_index, repeat=5)
    print(f"Head Random Access: {head}")

    # Index at middle random access: 5s
    middle_index = "hdfs://harunava/home/byte_speech_sv/duc.le1/index_test/classical.2.57-0.index_middle"
    middle = time_it_dataset(tar_path, index_file=middle_index, repeat=5)
    print(f"Middle Random Access: {middle}")

    # Index at end random access: 7s
    tail_index = "hdfs://harunava/home/byte_speech_sv/duc.le1/index_test/classical.2.57-0.index_tail"
    tail = time_it_dataset(tar_path, index_file=tail_index, repeat=5)
    print(f"Tail Random Access: {tail}")

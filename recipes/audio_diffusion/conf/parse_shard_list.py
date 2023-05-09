import random


def get_lines(fname):
    lines = []
    with open(fname, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def get_train_shards(fnames):
    shard_lists = []
    for i, fname in enumerate(fnames):
        shard_list = get_lines(fname)[1:]
        random.shuffle(shard_list)
        print(f"Shard list {i}: {shard_list[:10]}")
        shard_lists.append(
            [
                f"pipe:{'hdfs dfs -cat' if tar.startswith('hdfs') else 'cat'} {tar}"
                for tar in shard_list
            ]
        )
    return shard_lists


def get_validation_shards(fnames):
    shard = get_lines(fnames[0])[0]
    print(f"validation shards: {shard}")
    return f"pipe:{'hdfs dfs -cat' if shard.startswith('hdfs') else 'cat'} {shard}"

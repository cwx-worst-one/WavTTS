def get_mcc_shardlist(fname):
    with open(fname, "r") as fp:
        urls = [f"pipe:hdfs dfs -cat {line}" for line in fp.readlines()]
    return urls

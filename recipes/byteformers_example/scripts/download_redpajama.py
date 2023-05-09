"""This scripts downloads the RedPajama dataset, which contains
~1.2 trillion language tokens of the following datasets:

Commoncrawl:    878 Billion
C4:             175 Billion
GitHub:         59 Billion
Books:          26 Billion
ArXiv:          28 Billion
Wikipedia:      24 Billion
StackExchange:  20 Billion
Total:          1.2 Trillion
"""

import os
import subprocess
from collections import defaultdict
from multiprocessing import Pool

import requests


def check_hash(h):
    hash, target_fp = h.split()
    assert os.path.exists(target_fp)
    p = subprocess.Popen(["sha256sum", target_fp], stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    expected_hash, expected_target_fp = stdout.decode().split()
    assert hash == expected_hash and target_fp == expected_target_fp
    print(f"File downloaded and verified: {target_fp}")


def check_sha256(source: str, n_processes: int):
    with open(os.path.join("sha256", f"{source}_SHA256SUMS.txt")) as f:
        hash_map = f.read().splitlines()

    with Pool(n_processes) as p:
        p.map(check_hash, hash_map)


if __name__ == "__main__":
    verify_sha256 = True
    max_conn = 64
    sources = {
        "arxiv": 100,
        "book": 1,
        "c4": 1024,
        "common_crawl": 859,
        "github": 98,
        "stackexchange": 1,
        "wikipedia": 1,
    }
    base_url = "https://data.together.xyz/redpajama-data-1T/v1.0.0/"

    urls = requests.get(os.path.join(base_url, "urls.txt"))
    urls = urls.content.decode().split("\n")
    for s in sources:
        source_urls = [u for u in urls if f"v1.0.0/{s}" in u]
        n_files = len(source_urls)
        assert n_files == sources[s]
        print(f"Num files: {len(source_urls)}")

        group_subdirs = defaultdict(list)
        for url in source_urls:
            subdir = os.path.dirname(url.replace(base_url, ""))
            os.makedirs(subdir, exist_ok=True)
            group_subdirs[subdir].append(url)

        for subdir, u in group_subdirs.items():
            c = subprocess.Popen(
                ["parfive", "--max-conn", str(max_conn), "--directory", subdir, *u]
            )
            c.wait()

        if verify_sha256:
            check_sha256(s, n_processes=max_conn)

        print(f"Completed download/verification for {s}")

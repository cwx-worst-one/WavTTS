import argparse
import logging
import tarfile
import time
from dataclasses import dataclass

from pyarrow.fs import FileSystem


@dataclass
class HDFSPath:
    # The HDFS file system path, e.g., "hdfs://harunava"
    fs: str
    # The file path within the file system, e.g., "/home/byte_speech_sv/test.txt"
    file: str


def parse_hdfs_path(path: str):
    parts = path.replace("hdfs://", "", 1).split("/")
    if len(parts) == 0:
        return None
    return HDFSPath(fs=f"hdfs://{parts[0]}", file="/" + "/".join(parts[1:]))


logger = logging.getLogger(__name__)


# def get_audioset_urls():
#     hdfs_base = "hdfs://harunava/home/byte_speech_sv/mulan/audioset_uio"
#     result_urls = {}
#     shards_by_parts = [40, 470, 469, 468, 507]
#     for field in ["audio", "text"]:
#         urls = [
#             f"{hdfs_base}/{field}_shards/shards_{i}_{j:03d}.tar"
#             for i, shards in enumerate(shards_by_parts)
#             for j in range(shards)
#         ]
#         result_urls[field] = urls
#     return result_urls


def get_ecals_urls():
    hdfs_base = "hdfs://harunava/home/byte_speech_sv/mulan/ecals_uio"
    result_urls = {}
    shards = 445
    for field in ["audio", "text"]:
        urls = [f"{hdfs_base}/{field}_shards/shards_{j:06d}.tar" for j in range(shards)]
        result_urls[field] = urls
    return result_urls


def get_karaoke_urls():
    hdfs_base = "hdfs://harunava/home/byte_speech_sv/mulan/karaoke_uio"
    result_urls = {}
    shards = 485
    for field in ["audio", "text"]:
        urls = [f"{hdfs_base}/{field}_shards/shards_{j:06d}.tar" for j in range(shards)]
        result_urls[field] = urls
    return result_urls


def get_output_mapping():
    hdfs_base = "hdfs://harunava/home/byte_speech_sv/jingsong.gao/ecals"
    urls = get_ecals_urls()
    output_mapping = []
    for i in range(len(urls["audio"])):
        mapping = []
        for field in ["audio", "text"]:
            mapping.append(f"{urls[field][i]}")
        mapping.append(f"{hdfs_base}/ecals_{i:04d}.tar")
        output_mapping.append("\t".join(mapping))
    return output_mapping


def main(start, end):
    filesystems = {}
    for line in get_output_mapping()[start:end]:
        s = time.time()
        input1, input2, output = map(parse_hdfs_path, line.split("\t"))

        # Initialize HDFS file system object(s) if necessary
        for fs in [input1.fs, input2.fs, output.fs]:
            if fs not in filesystems:
                logger.info("Initializing file system: %s", fs)
                filesystems[fs] = FileSystem.from_uri(fs)[0]

        # Will error out if file path doesn't exist
        instream1 = filesystems[input1.fs].open_input_stream(input1.file)
        instream2 = filesystems[input2.fs].open_input_stream(input2.file)
        # Will override any existing file, be careful!
        outstream = filesystems[output.fs].open_output_stream(output.file)

        with tarfile.open(fileobj=instream1, mode="r|") as tar1, tarfile.open(
            fileobj=instream2, mode="r|"
        ) as tar2, tarfile.open(fileobj=outstream, mode="w|") as tarout:
            for m1, m2 in zip(tar1, tar2):
                tarout.addfile(m1, tar1.extractfile(m1))
                tarout.addfile(m2, tar2.extractfile(m2))

            e = time.time()
            print(
                "Read from %s & %s, wrote to %s, duration: %.2f s"
                % (input1.file, input2.file, output.file, e - s)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "start", type=int, help="The starting index of the output tar files"
    )
    parser.add_argument(
        "end", type=int, help="The ending index of the output tar files"
    )
    args = parser.parse_args()
    main(start=args.start, end=args.end)

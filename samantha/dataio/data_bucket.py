import os
from typing import Dict

from samantha.utils.hdfs_tools import ARNOLD_REGION, hdfs_get


class DataBucketBase:
    _root: Dict[str, str] = {}

    @property
    def region_root(self):
        if ARNOLD_REGION is None or ARNOLD_REGION not in self._root.keys():
            raise Exception(
                f"ARNOLD_REGION must be specified as either: {self._root.keys()}"
            )
        return self._root[ARNOLD_REGION]


class ByteNASDataBucket(DataBucketBase):
    _root: Dict[str, str] = {"US": ""}  # TODO

    def __call__(self, fp: str):
        return os.path.join(self.region_root, fp)


class HDFSDataBucket(DataBucketBase):
    _root: Dict[str, str] = {
        "US": "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js",
        "CN": "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/js",
    }

    @staticmethod
    def _local_fp(hdfs_path: str):
        if hdfs_path.startswith("hdfs://"):
            local_path = hdfs_path.replace("hdfs://", "/tmp/_hdfs/")
        else:
            local_path = hdfs_path

        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hdfs_get(hdfs_path, local_path)
        return local_path

    def __call__(self, fp: str, cache: bool = False):
        fp = os.path.join(self.region_root, fp)
        if cache:
            fp = self._local_fp(fp)
        return fp


# TODO: we can define various buckets
data_bucket = HDFSDataBucket()

import gc
import io
import json
import logging
import re
import sys
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import librosa
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from webdataset import warn_and_continue

from samantha.dataio.utils import parquet_reader

from .contextual.iterator import get_iterator

DATASET_NAME_KEY = "__dataset_name__"

logger = logging.getLogger(__name__)


def setup_sampler(
    handler: Callable[[Exception], bool] = warn_and_continue,
    sample_limit_per_file: Union[int, float] = None,
    extra_fields_in_data: Optional[List[str]] = None,
    config: dict = {},
):
    sampler_name = (
        "_ParquetSample" if config is None else config.get("name", "_ParquetSample")
    )
    sampler = globals()[sampler_name]
    return sampler(handler, sample_limit_per_file, extra_fields_in_data, config)


class _BaseSample:
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
    ):
        self.handler = handler
        self.sample_limit_per_file = sample_limit_per_file
        self.extra_fields_in_data = extra_fields_in_data
        self.meta = {}

    def get_src_url(self, src: Dict):
        src_url = {f"__{k}_url__": v for k, v in src.items()}
        # parse dataset name from path
        dataset_name = re.findall(r"/([^/]+)/index_\d+/", src["index"])
        src_url[DATASET_NAME_KEY] = dataset_name[0] if dataset_name else None
        return src_url

    def get_parent_uttid(self, uttid):
        parts = uttid.rpartition("-")
        if parts[2].isdigit() and parts[0] != "":
            return parts[0]
        parts = uttid.rpartition("_")
        if parts[2].isdigit() and parts[0] != "":
            return parts[0]
        return uttid

    def add_index_intermediate_info(self, item):
        if "meta" not in item:
            return
        try:
            meta = json.loads(item["meta"])
            if "start_time" not in item:
                item["start_time"] = self.get_index_start_time(meta)
            if "end_time" not in item:
                item["end_time"] = self.get_index_end_time(meta)
            if "speaker_id" not in item:
                item["speaker_id"] = self.get_meta_speaker_id(meta)
        except Exception:
            return

    def get_meta_speaker_id(self, meta):
        unknown_speaker = f"unknown_speaker_{uuid.uuid4().hex[:8]}"
        if meta is None:
            return unknown_speaker
        speaker_id = meta.get("speaker_id", None)
        if speaker_id:
            return speaker_id
        if meta is not None and "sents" in meta and len(meta["sents"]) > 0:
            speaker_id = meta["sents"][0].get("speaker_id", None)
        if not speaker_id:
            return unknown_speaker
        return speaker_id

    def get_index_start_time(self, meta):
        if meta is None:
            return -1
        if "start_time" in meta:
            return meta["start_time"]
        if "sents" not in meta or len(meta["sents"]) <= 0:
            return -1
        return meta["sents"][0].get("start_time", -1)

    def get_index_end_time(self, meta):
        if meta is None:
            return -1
        if "end_time" in meta:
            return meta["end_time"]
        if "sents" not in meta or len(meta["sents"]) <= 0:
            return -1
        return meta["sents"][-1].get("end_time", -1)

    def get_mem_usage(self, cache):
        size = 0
        for values in cache.values():
            for value in values.values():
                for v in value.values():
                    size += sys.getsizeof(v)
        return size

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        raise NotImplementedError


@dataclass
class BasicItemInfo:
    uttid: str
    parent_uttid: str
    group_no: int


@dataclass
class ItemCluster:
    item_infos: field(default_factory=list)
    start_row_group: int = -1
    end_row_group: int = -1


# format like:
#    {
#       "parent_uttid":"xxxx",
#       "features":{"index": ItemCluster, "data": ItemCluster, "umm_token": ItemCluster}
#    }
@dataclass
class MultiFeatureCluster:
    parent_uttid: str = ""
    features: dict = None


class _ContexualParquetSample(_BaseSample):
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
    ):
        super().__init__(handler, sample_limit_per_file, extra_fields_in_data)
        self.iterator = get_iterator(sample_config)

    def compose_clusters(self, clusters, sorted_parent_uttid):
        composed_parent_uttids = None
        composed_uttids = None
        for _, member in clusters.items():
            if composed_parent_uttids is None:
                composed_parent_uttids = set(member.keys())
            else:
                composed_parent_uttids.intersection_update(member.keys())
            uttids = set()
            for _, item_cluster in member.items():
                for item_info in item_cluster.item_infos:
                    uttids.add(item_info.uttid)
            if composed_uttids is None:
                composed_uttids = uttids
            else:
                composed_uttids.intersection_update(uttids)
        fixed_clusters = {}
        for name, member in clusters.items():
            fixed_clusters[name] = {}
            for parent_uttid, item_cluster in member.items():
                if parent_uttid not in composed_parent_uttids:
                    continue
                if name not in fixed_clusters:
                    fixed_clusters[name] = {}
                fixed_item_infos = [
                    item_info.uttid
                    for item_info in item_cluster.item_infos
                    if item_info.uttid in composed_uttids
                ]
                if fixed_item_infos:
                    fixed_clusters[name][parent_uttid] = ItemCluster(
                        item_infos=fixed_item_infos,
                        start_row_group=item_cluster.start_row_group,
                        end_row_group=item_cluster.end_row_group,
                    )
        return [
            MultiFeatureCluster(
                parent_uttid,
                {name: member[parent_uttid] for name, member in clusters.items()},
            )
            for parent_uttid in sorted_parent_uttid
            if parent_uttid in composed_parent_uttids
        ]

    def get_clusters(self, src, filesystem):
        # bridge_clusters ===>
        # format: {"index": {"parent_uttid": ItemCluster},
        #          "data": {"parent_uttid": ItemCluster}}
        # name is type of feature, index/data/utt_token
        bridge_clusters = {}
        sorted_parent_uttids = []
        for name, url in src.items():
            item_infos = [
                BasicItemInfo(
                    uttid=e["uttid"],
                    group_no=group_no,
                    parent_uttid=self.get_parent_uttid(e["uttid"]),
                )
                for group_no, e in parquet_reader(
                    url, columns=["uttid"], fs=filesystem, meta=self.meta
                )
            ]
            current_cluster = ItemCluster([])
            current_parent_uttid = ""
            for i, item_info in enumerate(item_infos):
                if i > 0 and item_info.parent_uttid != item_infos[i - 1].parent_uttid:
                    if name not in bridge_clusters:
                        bridge_clusters[name] = {}
                    bridge_clusters[name][current_parent_uttid] = current_cluster
                    if name == "index":  # base on index
                        sorted_parent_uttids.append(current_parent_uttid)
                    current_cluster = ItemCluster([])
                current_parent_uttid = item_info.parent_uttid
                current_cluster.item_infos.append(item_info)
                current_cluster.start_row_group = (
                    item_info.group_no
                    if current_cluster.start_row_group == -1
                    else min(item_info.group_no, current_cluster.start_row_group)
                )
                current_cluster.end_row_group = (
                    item_info.group_no
                    if current_cluster.end_row_group == -1
                    else max(item_info.group_no, current_cluster.end_row_group)
                )

            if name not in bridge_clusters:
                bridge_clusters[name] = {}
            bridge_clusters[name][current_parent_uttid] = current_cluster
            if name == "index":  # base on index
                sorted_parent_uttids.append(current_parent_uttid)

        # convert bridge_clusters to final_clusters
        return self.compose_clusters(bridge_clusters, sorted_parent_uttids)

    # output format:
    # {"index":{uttid:item}, "data":{uttid:item}}
    def get_all_items(self, cache, readers, cluster):
        items = {}
        for name, reader in readers.items():
            item_cluster = cluster.features[name]
            items[name] = {}
            if name not in cache:
                cache[name] = {}
            # check row groups need set to parquet reader
            for i in range(
                item_cluster.start_row_group, item_cluster.end_row_group + 1
            ):
                if name in cache and i in cache[name]:
                    for item in cache[name][i]:
                        if "uttid" in item:
                            items[name][item["uttid"]] = item
                else:
                    cache[name][i] = []
                    reader.set_row_groups([i])
                    try:
                        for _, item in iter(reader):
                            if "uttid" in item:
                                self.add_index_intermediate_info(item)
                                items[name][item["uttid"]] = item
                                cache[name][i].append(item)
                    except StopIteration:
                        continue
            # only keep the last row group in the cache
            for i in range(item_cluster.start_row_group, item_cluster.end_row_group):
                cache[name].pop(i, None)
        return items

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        handler = self.handler
        for src in sources:
            src_url = self.get_src_url(src)
            readers = None
            try:
                filesystem = get_filesystem(src["index"])
                # [MultiFeatureCluster]
                clusters = self.get_clusters(src, filesystem)

                # create readers based on row_groups for each parquet
                readers = {
                    name: _ParquetReader(url, fs=filesystem)
                    for name, url in src.items()
                }

                # cache format: {"index":dict{row_group:items},
                #                "data":dict{row_group:items}}
                cache = {name: {} for name in readers}

                # read data and combine to sample
                for cluster in clusters:
                    if cluster.features is None or "index" not in cluster.features:
                        continue
                    try:
                        items = self.get_all_items(cache, readers, cluster)
                        sample_template = deepcopy(src_url)
                        yield from self.iterator(items, cluster, sample_template)
                    except Exception as exn:
                        # raise exn
                        if hasattr(exn, "args"):
                            exn.args = exn.args + (
                                json.dumps(src_url),
                                f"parent_uttid={cluster.parent_uttid}",
                            )
                        handler(exn)
                        break  # drop this file incase infinite loop
            except Exception as exn:  # pragma: no cover
                exn.args = exn.args + (json.dumps(src_url),)
                if handler(exn):
                    continue
                else:
                    break
            finally:
                if readers is None:
                    continue
                for reader in readers.values():
                    reader.close()


class _ParquetSample(_BaseSample):
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
    ):
        super().__init__(handler, sample_limit_per_file, extra_fields_in_data)

    def get_all_utt(self, src, filesystem):
        """This function first builds a dictionary of utterance id's and group id's.

        - Utterance id's are unique identifiers within both the data- and index shards.
        - Group id's are the 'locations' of the data within the data shards.

        The utterance id's within the index shards, are used to retrieve the group id's
        in the data shards. The group id's are a kind of iterator ('random access') on the data shard.

        - `common_utt` is a set of utterances within the shard urls. The first iterator makes sure that
            there are no duplicates.

        - `utt_sampler` is then used to order the `common_utt` according to the `index` shard.
            This requires another iteration on the index (that was already done), but is negligible due to
            the index shard being of a smaller file size.

        Args:
            src (dict): dictionary of data/index shard url's.
            filesystem (str): filesystem type

        Returns:
            utt_sampler: De-duplicated, ordered list of utterance id's (according to the index shard)
            utt2group_no: Dictionary that maps utterance id's to group id's
        """

        utt2group_no = {}
        common_utt = None
        for name, url in src.items():
            utt2group_no[name] = {}
            cur_utt = set()
            for group_no, e in parquet_reader(
                url, columns=["uttid"], fs=filesystem, meta=self.meta
            ):
                uttid = e["uttid"]
                cur_utt.add(uttid)
                utt2group_no[name][uttid] = group_no
            if common_utt is None:
                common_utt = cur_utt
            else:
                common_utt.intersection_update(cur_utt)

        # get ordered uttid from index parquet
        utt_sampler = parquet_reader(
            src["index"],
            columns=["uttid"],
            fs=filesystem,
            sample_limit=self.sample_limit_per_file,
            meta=self.meta,
        )
        ordered_utt = [e["uttid"] for _, e in utt_sampler if e["uttid"] in common_utt]
        return ordered_utt, utt2group_no

    def construct_sample(self, sample, cache, name, reader_iter, common_utt):
        # TODO: this is definitely not efficient if the data shard is shuffled differently than the index shard
        utt = sample["uttid"]
        cur_sample = cache[name].pop(utt, None)
        if cur_sample is None:
            try:
                _, cur_sample = next(reader_iter)
                while cur_sample["uttid"] != utt:
                    cur_utt = cur_sample["uttid"]
                    if cur_utt in common_utt:
                        cache[name][cur_utt] = deepcopy(cur_sample)
                    _, cur_sample = next(reader_iter)
            except StopIteration:
                pass

        if name == "data":
            audio_bin = cur_sample["audio"]
            extra_fields = {}
            if self.extra_fields_in_data:
                for extra_field in self.extra_fields_in_data:
                    if extra_field in cur_sample:
                        extra_fields[extra_field] = cur_sample.get(extra_field, None)

            try:
                src_sample_rate = librosa.get_samplerate(io.BytesIO(audio_bin))
            except Exception as e:
                src_sample_rate = None
                logger.warn(e)

            cur_sample = {
                "wav": audio_bin,
                "src_sample_rate": src_sample_rate,
                # extra fields in data, possible vocal/acc for mss
                **extra_fields,
            }
        elif name == "index":
            cur_sample.pop("row_group_no", None)
            cur_sample.pop("data_file", None)
        sample.update(cur_sample)
        return sample

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        handler = self.handler
        for src in sources:
            src_url = self.get_src_url(src)
            readers = None
            try:
                filesystem = get_filesystem(src["index"])
                # get common uttid from all parquet(data/index/feat)
                ordered_utt, utt2group_no = self.get_all_utt(src, filesystem)

                # limit common uttid, compute row groups for each parquet
                common_utt = set(ordered_utt)  # NOTE: this is already a set

                # this is a dictionary of the shard name ('data', 'index') and the group id's
                # now, group id's used to be locations of each datapoint (i.e., [0, 0, 1, 1, 1, 2, 2])
                # this line actually evaluates it to [0, 1, 2], so the point of storing the locations of
                # each group id is lost..?
                row_groups = {
                    name: sorted({utt2group_no[name][uttid] for uttid in common_utt})
                    for name in src
                }
                # create readers based on row_groups for each parquet
                readers = {
                    name: _ParquetReader(
                        url, fs=filesystem, row_groups=row_groups[name]
                    )
                    for name, url in src.items()
                }

                # this turns each ParquetReader into an iterable
                reader_iters = {name: iter(reader) for name, reader in readers.items()}

                # NOTE: re-evaluate this...!
                cache = {name: {} for name in readers}

                # read data and combine to sample
                for utt in ordered_utt:
                    if utt not in common_utt:
                        continue
                    try:
                        # this copies the dictionary of `src_url` to create a sample dictionary
                        sample = deepcopy(src_url)
                        sample.update({"__key__": utt, "uttid": utt})

                        # this updates the sample dictionary for each data/index shard:
                        for name, rit in reader_iters.items():
                            # the sample is constructed by querying the `rit` (iterator) on the `utt` variable
                            # from the `ordered_utt`. This means that we're kindof doing a random access here??
                            sample = self.construct_sample(
                                sample, cache, name, rit, common_utt
                            )
                        yield sample

                    except Exception as exn:
                        if hasattr(exn, "args"):
                            exn.args = exn.args + (json.dumps(src_url), f"{utt=}")
                        handler(exn)
                        break  # drop this file incase infinite loop
                    common_utt.remove(utt)
            except Exception as exn:  # pragma: no cover
                exn.args = exn.args + (json.dumps(src_url),)
                if handler(exn):
                    continue
                else:
                    break
            finally:
                if readers is None:
                    continue
                for reader in readers.values():
                    reader.close()


class _ParquetSampleFast(_BaseSample):
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
    ):
        super().__init__(handler, sample_limit_per_file, extra_fields_in_data)

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        for src in sources:
            index_reader = None
            data_reader = None
            index = {}

            try:
                src_url = self.get_src_url(src)

                # first, define the index shard reader and build a dictionary of samples.
                # this is fast, because the indexes are usually much smaller than the shards containing
                # the binary data
                index_reader = _ParquetReader(src["index"], return_all=True)
                for (
                    row_group,
                    last_row_group,
                    group_index,
                    last_group,
                    item,
                ) in index_reader:
                    uttid = item["uttid"]

                    # build index sample:
                    sample = deepcopy(src_url)
                    sample["__key__"] = uttid
                    sample["meta"] = item["meta"]
                    sample["text"] = item["text"]
                    index[uttid] = sample

                index_reader.close()

                # second, define the data shard reader that will be actually used to yield samples
                # back downstream (i.e., the iterator).
                data_reader = _ParquetReader(src["data"], return_all=True)

                for (
                    row_group,
                    last_row_group,
                    group_index,
                    last_group,
                    item,
                ) in data_reader:
                    uttid = item["uttid"]

                    # NOTE: this can be slightly more optimized
                    if uttid not in index:
                        continue

                    sample = index[uttid]

                    extra_fields = {}
                    if self.extra_fields_in_data:
                        for extra_field in self.extra_fields_in_data:
                            if extra_field in item:
                                extra_fields[extra_field] = item.get(extra_field, None)

                    # add the audio binaries:
                    audio_keys = list([k for k in item.keys() if "audio" in k])
                    for k in audio_keys:
                        sample[k] = item[k]

                    # rename "audio" to "audio.wav" (default extension)
                    if "audio" in sample:
                        sample["audio.wav"] = sample.pop("audio")

                    sample.update(
                        {
                            "uttid": uttid,
                            "src_sample_rate": None,
                            "row_group": row_group,
                            "last_row_group": last_row_group,
                            "last_group": last_group,
                            "group_index": group_index,
                            **extra_fields,
                        }
                    )
                    yield sample

            except Exception as e:
                logger.error(e)

            if index_reader is not None:
                index_reader.close()

            if data_reader is not None:
                data_reader.close()

            del index
            del index_reader
            del data_reader
            gc.collect()
            logger.debug("[gc] Cleared parquet readers")


class _ParquetSampleIndex(_BaseSample):
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
    ):
        super().__init__(handler, sample_limit_per_file, extra_fields_in_data)

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        for src in sources:
            index_reader = None
            try:
                src_url = self.get_src_url(src)
                index_reader = _ParquetReader(src["index"], return_all=True)
                for (
                    row_group,
                    last_row_group,
                    group_index,
                    last_group,
                    item,
                ) in index_reader:
                    uttid = item["uttid"]

                    # build index sample:
                    sample = deepcopy(src_url)
                    sample["__key__"] = uttid
                    sample["uttid"] = uttid
                    sample["meta"] = item["meta"]
                    sample["text"] = item["text"]
                    yield sample

            except Exception as e:
                logger.error(e)

            if index_reader is not None:
                index_reader.close()

            del index_reader
            gc.collect()
            logger.debug("[gc] Cleared parquet readers")


class _ParquetReader:
    def __init__(
        self, url, fs=None, columns=None, row_groups=None, return_all: bool = False
    ):
        if fs is None:
            fs = get_filesystem(url)
        self.return_all = return_all
        self.stream = fs.open(url, skip_instance_cache=True)
        self.parquet_file = ParquetFile(self.stream)
        self.columns = columns
        self.row_groups = row_groups or list(range(self.parquet_file.num_row_groups))
        self.last_row_group = self.row_groups[-1]

    @property
    def metadata(self) -> "pyarrow._parquet.FileMetaData":
        return self.parquet_file.metadata

    def count_rows(self):
        row_count = 0
        for row_group in self.row_groups:
            group_data = self.parquet_file.read_row_group(
                row_group, columns=self.columns
            )
            row_count += len(group_data)
        return row_count

    def __iter__(self):
        for row_group in self.row_groups:
            group_data = self.parquet_file.read_row_group(
                row_group, columns=self.columns
            )
            group_datas = group_data.to_pandas()
            if not self.return_all:
                for row in group_datas.iterrows():
                    item = row[1].to_dict()
                    yield row_group, item
            else:
                last_group_index = len(group_datas) - 1
                for group_index, row in group_datas.iterrows():
                    yield row_group, self.last_row_group, group_index, last_group_index, row.to_dict()

    def set_row_groups(self, row_groups=None):
        self.row_groups = row_groups

    def close(self):
        self.parquet_file.close()
        self.stream.close()

import io
import json
import logging
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import librosa
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from webdataset import warn_and_continue

from samantha.dataio.utils import parquet_reader

DATASET_NAME_KEY = "__dataset_name__"

logger = logging.getLogger(__name__)

feature_common_keys = {"uttid", "dataset_name"}


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
                item["speaker_id"] = meta.get("speaker_id", "")
        except Exception:
            return

    def get_index_start_time(self, meta):
        if meta is None or "sents" not in meta or len(meta["sents"]) <= 0:
            return -1
        return meta["sents"][0].get("start_time", -1)

    def get_index_end_time(self, meta):
        if meta is None or "sents" not in meta or len(meta["sents"]) <= 0:
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


class ContextualStrategy(IntEnum):
    CONTINUOUS_CONTEXT = 1
    SAME_SPEAKER = 2


class _ContexualParquetSample(_BaseSample):
    def __init__(
        self,
        handler: Callable[[Exception], bool] = warn_and_continue,
        sample_limit_per_file: Union[int, float] = None,
        extra_fields_in_data: Optional[List[str]] = None,
        sample_config: Optional[Any] = None,
    ):
        super().__init__(handler, sample_limit_per_file, extra_fields_in_data)
        self.meta = {}
        if sample_config is None:
            sample_config = {}
        self.min_context_num = sample_config.get("min_context_num", 1)
        self.max_context_num = sample_config.get("max_context_num", 1)
        self.time_interval_threshold = sample_config.get("time_interval_threshold", 1)
        self.contextual_strategy = sample_config.get(
            "contextual_strategy", ContextualStrategy.CONTINUOUS_CONTEXT
        )
        if self.contextual_strategy > ContextualStrategy.SAME_SPEAKER:
            raise ValueError(f"invalid contextual strategy: {self.contextual_strategy}")

    def generate_contextual_key(self, feature_name):
        return f"contextual_{feature_name}_list"

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

    def merge_contextual_sample(self, sample_items):
        uttid = sample_items["index"][-1]["uttid"]
        sample = {"__key__": uttid, "uttid": uttid}
        # construct current features
        for name, items in sample_items.items():
            cur_item = items[-1]
            if name == "data":
                audio_bin = cur_item["audio"]
                cur_item = {
                    "wav": audio_bin,
                    "src_sample_rate": librosa.get_samplerate(io.BytesIO(audio_bin)),
                }
            elif name == "index":
                cur_item.pop("row_group_no", None)
                cur_item.pop("data_file", None)
            sample |= cur_item

        # construct contextual features
        for name, items in sample_items.items():
            for cur_item in items:
                if name == "data":
                    audio_bin = cur_item["audio"]
                    sample["contextual_wav_list"] = sample.get(
                        "contextual_wav_list", []
                    )
                    sample["contextual_wav_list"].append(audio_bin)
                elif name == "index":
                    sample["contextual_uttid_list"] = sample.get(
                        "contextual_uttid_list", []
                    )
                    sample["contextual_uttid_list"].append(cur_item["uttid"])
                    sample["contextual_meta_list"] = sample.get(
                        "contextual_meta_list", []
                    )
                    sample["contextual_meta_list"].append(cur_item["meta"])
                else:
                    for key in cur_item.keys():
                        if key in feature_common_keys:
                            continue
                        new_key = self.generate_contextual_key(key)
                        sample[new_key] = sample[new_key] if new_key in sample else []
                        sample[new_key].append(cur_item[key])
        return sample

    def iter_contexual_sample(self, items, cluster, sample_template):
        basis_item_cluster = cluster.features["index"]
        sorted_uttids = [
            item_info.uttid
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
        ]
        item_cnt = 0
        for i in range(len(sorted_uttids)):
            context_len = random.randint(self.min_context_num, self.max_context_num)
            uttid = sorted_uttids[i]
            sample_items = {}
            is_valid = True
            for name, sub_items in items.items():
                if uttid not in sub_items:
                    is_valid = False
                    break
                else:
                    sample_items[name] = [sub_items[uttid]]
            if not is_valid:
                continue
            # the context can be accepted must fit these conditions:
            # condition1: the time interval of continous 2 items
            #             should < time_interval_threshold
            # condition2: max num of context should < context_len
            # find context
            last_start_time = sample_items["index"][0].get("start_time", -1)
            for j in range(min(context_len, i)):
                context_uttid = sorted_uttids[i - j - 1]
                context_index_item = items["index"][context_uttid]
                context_end_time = context_index_item.get("end_time", -1)
                # check time interval
                if (
                    context_end_time < 0
                    or last_start_time < 0
                    or last_start_time - context_end_time > self.time_interval_threshold
                ):
                    break
                is_context_valid = all(
                    context_uttid in sub_items for _, sub_items in items.items()
                )
                if not is_context_valid:
                    continue
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name] = [sub_items[context_uttid]] + sample_items[name]
                last_start_time = sample_items["index"][0].get("start_time", -1)

            # construct sample
            sample = self.merge_contextual_sample(sample_items)
            item_cnt += 1
            if sample is not None:
                sample.update(sample_template)
                yield sample
        if item_cnt != len(sorted_uttids):
            logger.warning(
                f"total {len(sorted_uttids)} items, but yield {item_cnt} items"
            )

    def iter_same_speaker_sample(self, items, cluster, sample_template):
        # sourcery skip: low-code-quality
        basis_item_cluster = cluster.features["index"]
        basis_items = items["index"]
        item_speakers = {
            item_info.uttid: basis_items[item_info.uttid]["speaker_id"]
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
        }
        raw_sorted_uttids = [
            item_info.uttid
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
            and item_info.uttid in item_speakers
        ]
        sorted_uttids = sorted(
            raw_sorted_uttids,
            key=lambda x: (item_speakers[x], raw_sorted_uttids.index(x)),
        )
        start_pos = 0
        item_cnt = 0
        while start_pos < len(sorted_uttids):
            context_len = random.randint(self.min_context_num, self.max_context_num)
            uttid = sorted_uttids[start_pos]
            sample_items = {
                name: [sub_items[uttid]] for name, sub_items in items.items()
            }
            last_end_time = sample_items["index"][0].get("end_time", -1)
            basis_speaker = sample_items["index"][0].get("speaker_id", "")

            # the context can be accepted must fit these conditions:
            # condition1: all item should have the same speaker
            # condition2: the time interval of continous 2 items
            #             should < time_interval_threshold
            # condition3: max num of context should < context_len
            next_start_pos = start_pos + 1
            for j in range(context_len):
                context_pos = start_pos + j + 1
                if context_pos >= len(sorted_uttids):
                    next_start_pos = context_pos
                    break

                context_uttid = sorted_uttids[context_pos]
                context_index_item = items["index"][context_uttid]
                context_start_time = context_index_item.get("start_time", -1)
                context_speaker = context_index_item.get("speaker_id", "")
                # check time interval & speaker
                if (
                    context_start_time < 0
                    or last_end_time < 0
                    or context_start_time - last_end_time > self.time_interval_threshold
                ) or (
                    basis_speaker == ""
                    or context_speaker == ""
                    or basis_speaker != context_speaker
                ):
                    next_start_pos = context_pos
                    break
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name].append(sub_items[context_uttid])
                last_end_time = sample_items["index"][-1].get("end_time", -1)
                next_start_pos = context_pos + 1

            start_pos = next_start_pos
            # construct sample
            sample = self.merge_contextual_sample(sample_items)
            item_cnt += len(sample_items["index"])
            if sample is not None:
                sample.update(sample_template)
                yield sample
        if item_cnt != len(sorted_uttids):
            logger.warning(
                f"total {len(sorted_uttids)} items, but yield {item_cnt} items"
            )

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
                        if (
                            self.contextual_strategy
                            == ContextualStrategy.CONTINUOUS_CONTEXT
                        ):
                            yield from self.iter_contexual_sample(
                                items, cluster, sample_template
                            )
                        if self.contextual_strategy == ContextualStrategy.SAME_SPEAKER:
                            yield from self.iter_same_speaker_sample(
                                items, cluster, sample_template
                            )
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
        self.meta = {}

    def get_all_utt(self, src, filesystem):
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
            cur_sample = {
                "wav": audio_bin,
                "src_sample_rate": librosa.get_samplerate(io.BytesIO(audio_bin)),
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
                common_utt = set(ordered_utt)
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
                reader_iters = {name: iter(reader) for name, reader in readers.items()}
                cache = {name: {} for name in readers}

                # read data and combine to sample
                for utt in ordered_utt:
                    if utt not in common_utt:
                        continue
                    try:
                        sample = deepcopy(src_url)
                        sample.update({"__key__": utt, "uttid": utt})

                        for name, rit in reader_iters.items():
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


class _ParquetReader:
    def __init__(self, url, fs=None, columns=None, row_groups=None):
        if fs is None:
            fs = get_filesystem(url)
        self.stream = fs.open(url, skip_instance_cache=True)
        self.parquet_file = ParquetFile(self.stream)
        self.columns = columns
        self.row_groups = row_groups or list(range(self.parquet_file.num_row_groups))

    def __iter__(self):
        for row_group in self.row_groups:
            group_data = self.parquet_file.read_row_group(
                row_group, columns=self.columns
            )
            group_datas = group_data.to_pandas()
            for row in group_datas.iterrows():
                item = row[1].to_dict()
                yield row_group, item

    def set_row_groups(self, row_groups=None):
        self.row_groups = row_groups

    def close(self):
        self.parquet_file.close()
        self.stream.close()

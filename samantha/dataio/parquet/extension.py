import io
import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import librosa
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from webdataset import warn_and_continue

from samantha.dataio.utils import parquet_reader

DATASET_NAME_KEY = "__dataset_name__"


class _ParquetSample:
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

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        handler = self.handler
        for src in sources:
            src_url = {f"__{k}_url__": v for k, v in src.items()}
            # parse dataset name from path
            dataset_name = re.findall(r"/([^/]+)/index_\d+/", src["index"])
            if not dataset_name:
                src_url[DATASET_NAME_KEY] = None
            else:
                src_url[DATASET_NAME_KEY] = dataset_name[0]

            readers = None
            try:
                filesystem = get_filesystem(src["index"])
                # get common uttid from all parquet(data/index/feat)
                common_utt = None
                utt2group_no = {}
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
                ordered_utt = [
                    e["uttid"] for _, e in utt_sampler if e["uttid"] in common_utt
                ]

                # limit common uttid, compute row groups for each parquet
                common_utt = set(ordered_utt)
                row_groups = {}
                for name in src:
                    row_groups[name] = sorted(
                        set(utt2group_no[name][uttid] for uttid in common_utt)
                    )

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
                            cur_sample = cache[name].pop(utt, None)
                            if cur_sample is None:
                                try:
                                    _, cur_sample = next(rit)
                                    while cur_sample["uttid"] != utt:
                                        cur_utt = cur_sample["uttid"]
                                        if cur_utt in common_utt:
                                            cache[name][cur_utt] = deepcopy(cur_sample)
                                        _, cur_sample = next(rit)
                                except StopIteration:
                                    pass

                            if name == "data":
                                audio_bin = cur_sample["audio"]
                                extra_fields = {}
                                if self.extra_fields_in_data:
                                    for field in self.extra_fields_in_data:
                                        if field in cur_sample:
                                            extra_fields[field] = cur_sample.get(
                                                field, None
                                            )
                                cur_sample = {
                                    "wav": audio_bin,
                                    "src_sample_rate": librosa.get_samplerate(
                                        io.BytesIO(audio_bin)
                                    ),
                                    # extra fields in data, possible vocal/acc for mss
                                    **extra_fields,
                                }
                            elif name == "index":
                                cur_sample.pop("row_group_no", None)
                                cur_sample.pop("data_file", None)
                            sample.update(cur_sample)
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
        if row_groups:
            self.row_groups = row_groups
        else:
            self.row_groups = list(range(self.parquet_file.num_row_groups))

    def __iter__(self):
        for row_group in self.row_groups:
            group_data = self.parquet_file.read_row_group(
                row_group, columns=self.columns
            )
            group_datas = group_data.to_pandas()
            for row in group_datas.iterrows():
                item = row[1].to_dict()
                yield row_group, item

    def close(self):
        self.parquet_file.close()
        self.stream.close()

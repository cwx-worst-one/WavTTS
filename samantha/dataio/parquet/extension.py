import io
import json
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable

import librosa
from lightning_fabric.utilities.cloud_io import get_filesystem
from pyarrow.parquet import ParquetFile
from webdataset import warn_and_continue

from samantha.dataio.utils import parquet_reader

DATASET_NAME_KEY = "__dataset_name__"


class _ParquetSample:
    def __init__(
        self, filesystem=None, handler: Callable[[Exception], bool] = warn_and_continue
    ):
        self.filesystem = filesystem
        self.handler = handler

    def __call__(self, sources: Iterable[Dict[str, Any]]):
        handler = self.handler
        filesystem = self.filesystem
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
                readers = {
                    name: _ParquetReader(url, fs=filesystem)
                    for name, url in src.items()
                }
                reader_iters = {name: iter(reader) for name, reader in readers.items()}
                common_utt = set(
                    e["uttid"]
                    for v in src.values()
                    for _, e in parquet_reader(v, columns=["uttid"], fs=filesystem)
                )

                ordered_utt = []
                for _, e in parquet_reader(
                    src["index"], columns=["uttid"], fs=filesystem
                ):
                    if e["uttid"] in common_utt:
                        ordered_utt.append(e["uttid"])
                cache = {name: {} for name in readers}
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
                                cur_sample = {
                                    "wav": audio_bin,
                                    "src_sample_rate": librosa.get_samplerate(
                                        io.BytesIO(audio_bin)
                                    ),
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
    def __init__(self, url, fs=None, columns=None):
        if fs is None:
            fs = get_filesystem(url)
        self.stream = fs.open(url, skip_instance_cache=True)
        self.parquet_file = ParquetFile(self.stream)
        self.columns = columns

    def __iter__(self):
        row_group_num = self.parquet_file.num_row_groups
        row_groups = list(range(row_group_num))
        for group_no, row_group in enumerate(row_groups):
            group_data = self.parquet_file.read_row_group(
                row_group, columns=self.columns
            )
            group_datas = group_data.to_pandas()
            for row in group_datas.iterrows():
                item = row[1].to_dict()
                yield group_no, item

    def close(self):
        self.parquet_file.close()
        self.stream.close()

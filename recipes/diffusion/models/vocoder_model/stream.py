# Copyright 2024 ByteDance Inc. All Rights Reserved.
# Author: Feihu La (lafeihu@bytedance.com)

from math import prod
import torch
import numpy as np


def default_item_fn(item):
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item
    if isinstance(item, dict):
        return item.get("data")
    raise ValueError(f"unsupported item:{item}")


class StreamRectifier(object):
    def __init__(self, stream, dim=-1, overlap=0, item_fn=default_item_fn) -> None:
        self.buffer: torch.Tensor = None
        self.stream = stream
        self.dim = dim
        self.overlap = overlap
        self.item_fn = item_fn
        self._done = False

    def __getitem__(self, idx):
        self.wait_until_ready(idx)
        return self.narrow(idx.start, idx.stop)

    def wait_until_ready(self, idx):
        while not self.ready(idx.stop):
            try:
                self.next()
            except StopIteration:
                self._done = True
                break

    def next(self):
        chunk = next(self.stream)
        chunk = self.item_fn(chunk)
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk)

        if self.buffer is None:
            self.buffer = chunk
        else:
            buffer = self.narrow(None, None if self.overlap == 0 else -self.overlap)
            self.buffer = torch.cat([buffer, chunk], dim=self.dim)

    def ready(self, stop):
        return isinstance(stop, int) and self.size() >= stop

    def size(self):
        if self.buffer is None:
            return 0

        return self.buffer.size(self.dim)

    def done(self):
        return self._done

    def narrow(self, start, stop):
        buffer = self.buffer.transpose(self.dim, -1)
        sz = list(buffer.size())
        buffer = buffer.view(prod(sz[:-1]), -1)
        chunk = buffer[:, start:stop].view(*(sz[:-1] + [-1])).transpose(-1, self.dim)
        return chunk


def islice(gen, step=49, overlap=None, dim=-1, upstream_overlap=0, debug=False):
    def _check_overlap(overlap=None):
        if isinstance(overlap, int):
            overlap = max(0, overlap)
            return (overlap, overlap)
        if isinstance(overlap, (list, tuple)) and len(overlap) == 2:
            left, right = max(0, overlap[0]), max(0, overlap[1])
            return (left, right)
        return (0, 0)

    def _iterator(rectifier, overlap):
        idx, offset = 0, 0
        while True:
            start = max(0, offset - overlap[0])
            end = offset + step + overlap[1]

            chunk = rectifier[start:end]
            valid_offset = (offset - start, min(offset - start + step, chunk.size(dim)))
            if valid_offset[1] - valid_offset[0] <= 0:
                break

            if debug:
                print(
                    f"{idx=}, pos=[{start}:{end}], {chunk.shape=}, {valid_offset=}"
                )

            yield {
                "idx": idx,
                "data": chunk,
                "offset": valid_offset,
            }
            idx += 1
            offset += step

    def _buffered(iter, size=1):
        buffer = []
        for item in iter:
            item["is_last"] = False
            if len(buffer) < size:
                buffer.append(item)
                continue
            buffer.append(item)
            yield buffer.pop(0)

        if len(buffer) > 0:
            buffer[-1]["is_last"] = True
        for item in buffer:
            yield item

    rectifier = StreamRectifier(gen, dim=dim, overlap=upstream_overlap)
    iter0 = _iterator(rectifier, _check_overlap(overlap))

    return _buffered(iter0, size=1)
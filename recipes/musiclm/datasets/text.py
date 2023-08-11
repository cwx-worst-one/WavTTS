import json
import random

from torch.utils.data import IterableDataset


def read_prompts(fname):
    prompts = []
    with open(fname, "r") as f:
        skipped_header = False
        for line in f:
            if not skipped_header:
                skipped_header = True
                continue
            ary = line.strip().split(",")
            if len(ary) < 2:
                print(f"Invalid prompt line: {line.strip()}")
                continue
            prompts.append(",".join(ary[1:]))
    return prompts


class PromptsDataset(IterableDataset):
    def __init__(self, prompts_fnames):
        self.epoch = -1
        self.prompts = []
        for fname in prompts_fnames:
            prompts = read_prompts(fname)
            print(f"Read {len(prompts)} prompts from {fname}")
            self.prompts.extend(prompts)
        assert len(self.prompts) > 0

    def __iter__(self):
        self.epoch += 1
        random.shuffle(self.prompts)
        for prompt in self.prompts:
            yield {"text": prompt}
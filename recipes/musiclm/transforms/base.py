import sys
from typing import Dict, Generator

import torch


class TransformBase:
    """Base class for all data transforms"""

    def __init__(self):
        self.count = 0
        self.skipped = 0

    def _update_stats(self, skipped: bool):
        self.count += 1
        if skipped:
            self.skipped += 1
            if self.skipped % 100 == 0:
                print(
                    f"Skipped {self.skipped}/{self.count} items",
                    file=sys.stderr,
                    flush=True,
                )

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        raise NotImplementedError()

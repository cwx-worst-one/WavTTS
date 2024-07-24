from typing import List

import torch
import torch.nn as nn

from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)

class FeatureNormalizerIdentity(nn.Module):

    def inverse(self, normalized_feature: torch.Tensor) -> torch.Tensor:
        return normalized_feature

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature

class FeatureNormalizer(nn.Module):

    def __init__(self, stats_fp: str):
        super().__init__()
        stats = torch.load(stats_fp)
        self.register_buffer("mean", stats["mean"])  # [1]
        self.register_buffer("std", stats["std"])  # [1]
        self.n_items = stats["n_items"]

    def __repr__(self):
        return f"{self.mean=}, {self.std=}, {self.n_items=}"

    def inverse(self, normalized_feature: torch.Tensor) -> torch.Tensor:
        normalized_feature = normalized_feature.float()
        with torch.cuda.amp.autocast(enabled=False):
            mel = normalized_feature * self.std - self.mean
        return mel

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.float()
        with torch.cuda.amp.autocast(enabled=False):
            normalized_feature = (feature + self.mean) / self.std
        return normalized_feature

    @classmethod
    def write(
        self,
        fp: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        var: torch.Tensor,
        n_items: int,
    ):
        stats_dict = {"mean": mean, "std": std, "var": var, "n_items": n_items}
        torch.save(stats_dict, fp)
        logger.info(stats_dict)

    @classmethod
    def merge(self, fps: List[str], f: str):
        stats_dict = {}
        for fp in fps:
            for k, v in torch.load(fp).items():
                if k in stats_dict:
                    stats_dict[k] += v
                else:
                    stats_dict[k] = v

        self.write(
            f,
            stats_dict["mean"] / len(fps),
            stats_dict["std"] / len(fps),
            stats_dict["var"] / len(fps),
            stats_dict["n_items"],
        )

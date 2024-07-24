from typing import Optional

from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning import seed_everything as pl_seed_everything

from samantha.utils.logger import RankedLogger

logger = RankedLogger()


def seed_everything(seed: Optional[int] = None, workers: bool = False):
    global_rank = rank_zero_only.rank
    logger.info(
        f"GLOBAL RANK: {global_rank} | BASE SEED: {seed} | RANK SEED: {seed + global_rank}"
    )
    seed = seed + global_rank
    pl_seed_everything(seed, workers)

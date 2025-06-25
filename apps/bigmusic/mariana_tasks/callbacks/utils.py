import time
from pathlib import Path

import pytorch_lightning as pl


def sync_all_ranks(
    trainer: "pl.Trainer",
    output_dir: Path,
    operation_name: str,
) -> bool:
    """rank synchronization
    return
        True: should execute
        False: should not execute
    """
    if trainer is None:  # for debug
        return True

    output_dir = Path(output_dir)

    # Signal this rank is done
    (output_dir / f"{operation_name}.{trainer.global_rank}.SUCCESS").touch()

    if trainer.is_global_zero:
        ts = time.time()
        while not all(
            [
                (output_dir / f"{operation_name}.{rank}.SUCCESS").exists()
                for rank in range(trainer.world_size)
            ]
        ):
            time.sleep(10)
            print(
                f"[{operation_name}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)"
            )
        return True
    return False
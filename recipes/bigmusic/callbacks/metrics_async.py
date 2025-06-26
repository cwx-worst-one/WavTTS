import pytorch_lightning as pl
from recipes.bigmusic.scripts.exp_utils.collect_results_test import run_metrics_batch
import argparse
from pathlib import Path
import time

class MetricsAsyncCallback(pl.Callback):
    def __init__(self, max_retry=5, metrics=['all']) -> None:
        super().__init__()
        self.max_retry = max_retry
        self.metrics = metrics

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None) -> None:

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if trainer is None:
            run_metrics_batch([output_dir], self.max_retry, self.metrics)
            return None

        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()

        if trainer.is_global_zero:
            ts = time.time()
            while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                time.sleep(10)
                print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")
            
            run_metrics_batch([output_dir], self.max_retry, self.metrics)

        

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    MetricsAsyncCallback().on_predict_end(None, None, args.output_dir)
from pathlib import Path
from typing import Any
import glob
import time

import pytorch_lightning as pl
import torch

from recipes.bigmusic.utils.format_utils import update_json


class LatentMeanStdCallback(pl.Callback):
    '''
    Diffusion mean and std might be not exactly 0 and 1,
    When reference dataset available, use this callback to calculate the mean and std of generated latent,
    It returns a corrected mean and std to denorm diffusion output.
    '''
    def __init__(self) -> None:
        super().__init__()

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        (
            Path(output_dir)
            / f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS"
        ).touch()

        if trainer.is_global_zero:
            ts = time.time()
            while not all(
                [
                    (
                        Path(output_dir) / f"{self.__class__.__name__}.{rank}.SUCCESS"
                    ).exists()
                    for rank in range(trainer.world_size)
                ]
            ):
                time.sleep(10)
                print(
                    f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)"
                )

            generated_output_fps = list(
                Path(output_dir).glob("**/*.diffusion_output.pt")
            )

            vocoder_mean = pl_module.bn_norm.mean
            vocoder_std = pl_module.bn_norm.std
            mean, std, outlier_ratio = run_latent_meanstd(generated_output_fps, vocoder_mean, vocoder_std)
            metrics_fp = Path(output_dir) / "metrics.json"
            update_json(
                metrics_fp,
                {
                    "diffusion_metrics": {
                        "bn_norm_mean": mean,
                        "bn_norm_std": std,
                        "outlier_ratio": outlier_ratio,
                    }
                },
            )


def run_latent_meanstd(generated_output_fps, vocoder_mean=0, vocoder_std=1):
    all_latent = []
    for f in generated_output_fps:
        latent = torch.load(f, map_location="cpu")
    all_latent.append(latent.flatten())

    if len(all_latent) == 0:
        print("No valid latent files found.")
        return 0, 0
    all_latent = torch.cat(all_latent, dim=0)
    mean = all_latent.mean().item()
    std = all_latent.std().item()

    denorm_mean = vocoder_mean - (mean * vocoder_std / std)
    denorm_std = vocoder_std / std
    denorm_latent = (all_latent * denorm_mean) + denorm_std
    # calcultate outlier ratio outside of 3 * vocoder_std
    outlier_ratio = (denorm_latent.abs() > 3 * vocoder_std).sum().item() / len(denorm_latent)
    print(
        f"############################\n"
        f"Diffusion latent denorm statistics:\n"
        f"bn_norm_mean={denorm_mean:.3f}\n"
        f"bn_norm_std={denorm_std:.3f}\n"
        f"outlier_ratio={outlier_ratio:.3f}\n"
    )
    return mean, std, outlier_ratio


if __name__ == "__main__":
    run_latent_meanstd(
        glob.glob("./recon_results/recondebug_cfg1.6_paths_nfe10/*.diffusion_output.pt")
    )

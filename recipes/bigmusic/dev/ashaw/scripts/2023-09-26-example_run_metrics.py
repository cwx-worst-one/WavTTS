from argparse import ArgumentParser
from recipes.bigmusic.callbacks.wer_metrics import run_wer_metrics
from recipes.bigmusic.callbacks.mcs_metrics import run_mcs_metrics
from recipes.musiclm.requires.mulan.mulan_infer_g4 import (
    create_mulan_model,
    mulan_inference,
)
from pathlib import Path
import json

def parse_args() -> dict:
    """parse args"""
    parser = ArgumentParser(description="report result and metrics")

    parser.add_argument("--results_dir", type=str, help="results_dir")
    parser.add_argument("--mulan_ckpt_path", type=str, help="metric dir", default="/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=014000-median_rank_1=160-kaggle.ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    args, _ = parser.parse_known_args()
    return vars(args)

def infer_results(results_dir, mulan_ckpt_path, device):
    assert Path(results_dir).exists()
    # by default metrics are saved to resutls_dir/metrics.json
    run_wer_metrics(results_dir, device=device)

    mulan_model = create_mulan_model(mulan_ckpt_path, device=device)
    requires = { "mulan_infer_fn": mulan_inference, "mulan": mulan_model }
    run_mcs_metrics(requires, results_dir, device=device)

    metrics_fp = Path(results_dir)/'metrics.json'
    with open(metrics_fp, 'r') as f:
        metrics = json.load(f)
    return metrics, metrics_fp


if __name__ == '__main__':
    op_args = parse_args()
    metrics, metrics_fp = infer_results(**op_args)
    # TODO: upload results

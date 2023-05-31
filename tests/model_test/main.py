import sys
import torch
import numpy as np
import random
from hyperpyyaml import load_hyperpyyaml
import logging

from samantha.utils.hparams import DotDict
from tests.model_test.utils.parser import parse_arguments
from tests.model_test.metrics.commons import setup_tester, run_metrics, generate_reports

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

if __name__ == "__main__":

    hparams_file = parse_arguments(sys.argv[1:])

    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)

    if cfg.benchmark_params.action not in ['generate', 'train']:
        raise NotImplementedError(
            f"action `{cfg.benchmark_params.action}` is not implemented, should be `train` or `generate`.")

    # Generate tester for model
    tester = setup_tester(cfg)

    # Run metrics
    full_results = run_metrics(cfg, tester)

    # Generate reports
    generate_reports(cfg, tester, full_results)

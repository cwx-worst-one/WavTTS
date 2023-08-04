import time

import torch.cuda
from pytorch_lightning.trainer.states import RunningStage

from samantha.utils.model_metric import ModelMetric
from samantha.models.flash_llama import LlamaModel, LlamaConfig


def test_model_metric():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = ModelMetric(16, LlamaModel(LlamaConfig())).to(device)

    global_step = 0
    interval = 3
    batch_size = 1
    seq_len = 1e10
    for _ in range(2):
        for train in range(5):
            global_step += 1
            time.sleep(0.2)
            batch_size, seq_len = 1, 1e10
            metric.update(
                batch_size * seq_len,
                RunningStage.TRAINING,
                1,
                model_kwargs={"batch_size": batch_size, "seq_len": seq_len}
            )
            if global_step % interval == 0:
                print(metric.compute(global_step))
        for val in range(2):
            global_step += 1
            time.sleep(0.1)
            metric.update(
                batch_size * seq_len,
                RunningStage.VALIDATING,
                0.5,
                model_kwargs={"batch_size": batch_size, "seq_len": seq_len}
            )
            if global_step % interval == 0:
                print(metric.compute(global_step))

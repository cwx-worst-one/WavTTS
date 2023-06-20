import time

from pytorch_lightning.trainer.states import RunningStage

from samantha.utils.model_metric import ModelMetric
from samantha.models.flash_llama import LlamaModel, LlamaConfig


def test_model_metric():
    metric = ModelMetric(16, LlamaModel(LlamaConfig()))

    global_step = 0
    interval = 3
    for _ in range(2):
        for train in range(10):
            global_step += 1
            time.sleep(2)
            metric.update(1, 1e10, RunningStage.TRAINING, 1)
            if global_step % interval == 0:
                print(metric.compute(global_step))
        for val in range(5):
            global_step += 1
            time.sleep(1)
            metric.update(1, 1e10, RunningStage.VALIDATING, 0.5)
            if global_step % interval == 0:
                print(metric.compute(global_step))

import logging
import os
import warnings

import torch
from cruise.configuration.cli import CruiseArgumentParser, CruiseCLI

import samantha
from apps.mariana.mariana.data.audio.multitask_datamodule import (
    AudioMultiTaskDataModule,
)
from samantha.dataio.bigmusic.lite import MusicLiteDataModule

logger = logging.getLogger(__name__)
_LOG_LEVEL = (
    logging.DEBUG if int(os.getenv("LITE_TRANSFORM_DEBUG", "0")) != 0 else logging.ERROR
)
logger.setLevel(_LOG_LEVEL)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def test_musiclitedatamodule(
    cfg_path="tests/unittests/dataio/assets/v5_dataloader_mariana.yaml", test_step=100
):
    import sys

    cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    sys.argv = ["", "--config", cfg_path]
    cfg, trainer, model, datamodule = cli.parse_args()
    datamodule: MusicLiteDataModule

    worker_id = int(os.getenv("DMLC_WORKER_ID", "0"))
    worker_num = int(os.getenv("ARNOLD_WORKER_NUM", "1"))
    gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    rank = int(
        os.getenv(
            "RANK", int(os.getenv("OMPI_COMM_WORLD_RANK", "0")) + worker_id * gpu_num
        )
    )
    world_size = int(os.getenv("WORLD_SIZE", worker_num * gpu_num))

    dataloader = datamodule.train_dataloader()
    count = 0
    for idx, batch_data in enumerate(dataloader):
        logger.info(
            f"rank:{rank}, world_size:{world_size} iter:{idx}, {batch_data['uttid']}"
        )
        logger.info(f"batch_data keys:{batch_data.keys()}")
        if "target_token_ids" in batch_data:
            print(f'{batch_data["target_token_ids"].shape=}')
            print(f'{batch_data["target_tokens_length"]=}')

        count += 1
        if test_step > 0 and count >= test_step:
            break

    print(f"number of data:{count}")
    datamodule.teardown()


def test_musiclitedatamodule_eval(
    cfg_path="tests/unittests/dataio/assets/v5_dataloader_mariana.yaml", test_step=1
):
    import sys

    cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    sys.argv = ["", "--config", cfg_path]
    cfg, trainer, model, datamodule = cli.parse_args()
    datamodule: MusicLiteDataModule

    worker_id = int(os.getenv("DMLC_WORKER_ID", "0"))
    worker_num = int(os.getenv("ARNOLD_WORKER_NUM", "1"))
    gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    rank = int(
        os.getenv(
            "RANK", int(os.getenv("OMPI_COMM_WORLD_RANK", "0")) + worker_id * gpu_num
        )
    )
    world_size = int(os.getenv("WORLD_SIZE", worker_num * gpu_num))

    dataloader = datamodule.predict_dataloader()
    count = 0
    for idx, batch_data in enumerate(dataloader):
        logger.info(
            f"rank:{rank}, world_size:{world_size} iter:{idx}, {batch_data['uttid']}"
        )
        logger.info(f"batch_data keys:{batch_data.keys()}")

        print(batch_data.keys())
        print(batch_data["style_text"])

        count += 1
        if test_step > 0 and count >= test_step:
            break

    print(f"number of data:{count}")
    datamodule.teardown()


def test_musiclitedatamodule_mafl(
    # cfg_path="/opt/tiger/samantha/tests/unittests/dataio/assets/musiclish.yaml",
    # cfg_path="/opt/tiger/samantha/apps/bigmusic/mariana_tasks/v5_m8_pt_debug_on_1gpu_mafl.yaml",
    cfg_path="/opt/tiger/samantha/apps/bigmusic/mariana_tasks/conf/v5_m8_680m_bpe_simple_1gpu.yaml",
    test_step=10000,
):
    import sys

    cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    sys.argv = ["", "--config", cfg_path]
    cfg, trainer, model, datamodule = cli.parse_args()
    datamodule: MusicLiteDataModule

    # worker_id = int(os.getenv("DMLC_WORKER_ID", "0"))
    # worker_num = int(os.getenv("ARNOLD_WORKER_NUM", "1"))
    # gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    # rank = int(
    #     os.getenv(
    #         "RANK", int(os.getenv("OMPI_COMM_WORLD_RANK", "0")) + worker_id * gpu_num
    #     )
    # )
    # world_size = int(os.getenv("WORLD_SIZE", worker_num * gpu_num))

    dataloader = datamodule.train_dataloader()
    count = 0
    for idx, batch_data in enumerate(dataloader):

        # logger.info(
        #     f"rank:{rank}, world_size:{world_size} iter:{idx}, {batch_data['uttid']}"
        # )
        logger.info(f"batch_data keys:{batch_data.keys()}")

        count += 1
        if test_step > 0 and count >= test_step:
            break

    print(f"number of data:{count}")
    datamodule.teardown()


def visualize_task(task_distribution_over_step):
    # visualize task_distribution_over_step, plot curve, save to local
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    data = {"step": [], "task": [], "count": []}
    for step, task_distribution in task_distribution_over_step.items():
        for task, count in task_distribution.items():
            data["step"].append(step)
            data["task"].append(task)
            data["count"].append(count)
    df = pd.DataFrame(data)

    plt.cla()
    # plot curve, each curve show a task count over step
    for task in df["task"].unique():
        df_task = df[df["task"] == task]
        df_task = df_task.sort_values(by=["step"])
        plt.plot(df_task["step"], df_task["count"], label=task)
    plt.legend()
    plt.savefig("task_distribution.png")
    plt.close()


def test_multitaskdatamodule(
    cfg_path="apps/bigmusic/mariana_tasks/conf/v5_m8_680m_bpe_multitask_1gpu.yaml",
    test_step=10000,
):
    import sys

    cli = CruiseCLI(datamodule_class=AudioMultiTaskDataModule)
    sys.argv = ["", "--config", cfg_path]
    cfg, trainer, model, datamodule = cli.parse_args()
    datamodule: AudioMultiTaskDataModule
    datamodule.setup()
    logger.info("datamodule setup done")

    worker_id = int(os.getenv("DMLC_WORKER_ID", "0"))
    worker_num = int(os.getenv("ARNOLD_WORKER_NUM", "1"))
    gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    rank = int(
        os.getenv(
            "RANK", int(os.getenv("OMPI_COMM_WORLD_RANK", "0")) + worker_id * gpu_num
        )
    )
    world_size = int(os.getenv("WORLD_SIZE", worker_num * gpu_num))

    dataloader = datamodule.train_dataloader()
    count = 0
    task_distribution = {}
    task_distribution_over_step = {}
    total = 0
    for idx, batch_data in enumerate(dataloader):
        if "task" in batch_data:
            for task in batch_data["task"]:
                total += 1
                if task not in task_distribution:
                    task_distribution[task] = 0
                task_distribution[task] += 1
                if total % 1000 == 0:
                    task_distribution_over_step[total] = task_distribution.copy()
                    visualize_task(task_distribution_over_step)
                    task_distribution = {}

        # for k, v in batch_data.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape}")

        logger.info(f"rank:{rank}, world_size:{world_size} iter:{idx}")
        logger.info(f"batch_data keys:{batch_data.keys()}")

        count += 1
        if test_step > 0 and count >= test_step:
            break

    print(f"number of data:{count}")
    datamodule.teardown()


if __name__ == "__main__":
    # test_musiclitedatamodule()
    # test_musiclitedatamodule_eval()
    # test_musiclitedatamodule_mafl()
    test_multitaskdatamodule()

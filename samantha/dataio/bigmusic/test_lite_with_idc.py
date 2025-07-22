import logging
import os

from cruise.configuration.cli import CruiseArgumentParser, CruiseCLI

import samantha
from samantha.dataio.bigmusic.lite import MusicLiteDataModule

logger = logging.getLogger(__name__)
_LOG_LEVEL = (
    logging.DEBUG if int(os.getenv("LITE_TRANSFORM_DEBUG", "0")) != 0 else logging.INFO
)
logger.setLevel(_LOG_LEVEL)


def test_musiclitedatamodule(
    cfg_path="tests/unittests/dataio/assets/v5_dataloader_mariana_with_idc.yaml",
    test_step=100,
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


if __name__ == "__main__":
    test_musiclitedatamodule()
    # test_musiclitedatamodule_eval()

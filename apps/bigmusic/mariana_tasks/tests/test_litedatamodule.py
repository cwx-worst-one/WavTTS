import logging
import os
import warnings
import yaml

from samantha.dataio.bigmusic.lite import MusicLiteDataModule

logger = logging.getLogger(__name__)
_LOG_LEVEL = (
    logging.DEBUG if int(os.getenv("LITE_TRANSFORM_DEBUG", "0")) != 0 else logging.ERROR
)
logger.setLevel(_LOG_LEVEL)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def test_musiclitedatamodule(test_step=10):
    config = yaml.safe_load(open("apps/bigmusic/mariana_tasks/tests/test_data.yaml", "r"))
    datamodule = MusicLiteDataModule(**config["data"])
    datamodule.rank_zero_prepare()
    datamodule.local_rank_zero_prepare()
    datamodule.setup()

    def loop_dataloader(dataloader, max_step):
        count = 0
        for idx, batch_data in enumerate(dataloader):
            count += 1
            if max_step > 0 and count >= max_step:
                break
        return count

    count = loop_dataloader(datamodule.train_dataloader(), test_step)
    assert count == test_step
    count = loop_dataloader(datamodule.val_dataloader(), test_step)
    assert count == test_step
    count = loop_dataloader(datamodule.predict_dataloader(), test_step)
    assert count == test_step
    datamodule.teardown()

if __name__ == "__main__":
    test_musiclitedatamodule()
import sys
import yaml
import pytest
import tempfile
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.info(f"run training with {_LITE_USE_MULTITASK=}")
# from lite.module.datamodule import LiteDataModule
try:
    from samantha.dataio.bigmusic.lite_multitask import MusicLiteDataModule
except Exception as e:
    print(f"MusicLiteDataModule not support, error={e}")
    MusicLiteDataModule = None
from cruise.configuration.cli import CruiseArgumentParser, CruiseCLI

@pytest.mark.skip(reason="ci env not support")
@pytest.mark.parametrize("cfg_path", [
    # "apps/bigmusic/mariana_tasks/conf/v5_m8_680m_bpe_multitask.yaml",
    "/mnt/bn/data-storage-hl/user/zhangshuo/code/samantha_debug/v5.0.4_moe_680m_multitask_slice_pt.yaml",
])
def test_lite_datamodule(
        cfg_path,
        test_step=100,
        warmup_step=10,
        ):
    with open(cfg_path, "r") as f:
        data_config = yaml.safe_load(f)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.safe_dump({"data": data_config["data"]}, temp_file)
    data_yaml_path = temp_file.name 
    print(f"{data_yaml_path=}")
    datamodule = MusicLiteDataModule(data_config["data"]["config"])

    # cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    # sys.argv = ["", "--config", data_yaml_path]
    # cfg, trainer, model, datamodule = cli.parse_args()
    # datamodule: MusicLiteDataModule

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    count = 1
    # warmup_start_time = None
    test_start_time = None
    test_times = []
    warmup_start_time = time.time()
    for idx, batch_data in enumerate(train_dataloader):
        if count >= warmup_step:
            if test_start_time is None:
                test_start_time = time.time()
            iter_start = time.time()

        if isinstance(batch_data, list):
            print(f"loop out list with {len(batch_data)} items")
            # for data in batch_data:
            #     if isinstance(data, dict):
            #         print(f"\tuttid={data['uttid']} keys={data.keys()}")
        elif isinstance(batch_data, dict):
            print(f"loop out dict with keys={batch_data.keys()}")
            # for key, value in batch_data.items():
            #     print(f"\tkey={key} value={value}")
           
        if count >= warmup_step and test_start_time is not None:
            iter_end = time.time()
            test_times.append(iter_end - iter_start)
            if count == warmup_step:
                warmup_end_time = iter_end
                

        count += 1
        if test_step > 0 and count >= test_step:
            break

    warmup_duration = 0.0
    if warmup_start_time is not None and warmup_step > 0:
        warmup_duration = warmup_end_time - warmup_start_time

    test_total_time = 0.0
    test_avg_time = 0.0
    if test_start_time is not None and test_times:
        test_total_time = sum(test_times)
        test_avg_time = test_total_time / len(test_times)
    
    print(f"Warmup time: {warmup_duration:.4f} seconds")
    print(f"Test total time (excluding warmup): {test_total_time:.4f} seconds")
    print(f"Test average time per batch: {test_avg_time:.4f} seconds")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test item processing with YAML configuration"
    )
    parser.add_argument(
        "--input", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100000,
        help="Number of test items to process (default: 100000)",
    )

    args = parser.parse_args()

    test_lite_datamodule(cfg_path=args.input, test_step=args.num)
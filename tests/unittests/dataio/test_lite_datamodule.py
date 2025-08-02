import sys
import yaml
import pytest
import tempfile
# from lite.module.datamodule import LiteDataModule
try:
    from samantha.dataio.bigmusic.lite_multitask import MusicLiteDataModule
except Exception as e:
    print(f"MusicLiteDataModule not support, error={e}")
    MusicLiteDataModule = None
from cruise.configuration.cli import CruiseArgumentParser, CruiseCLI

@pytest.mark.skip(reason="ci env not support")
@pytest.mark.parametrize("cfg_path", [
    "apps/bigmusic/mariana_tasks/conf/v5_m8_680m_bpe_multitask.yaml",
])
def test_lite_datamodule(
        cfg_path,
        test_step=10,
        ):
    with open(cfg_path, "r") as f:
        data_config = yaml.safe_load(f)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.safe_dump({"data": data_config["data"]}, temp_file)
    data_yaml_path = temp_file.name 
    print(f"{data_yaml_path=}")
    # datamodule = MusicLiteDataModule(data_config["data"])

    cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    sys.argv = ["", "--config", data_yaml_path]
    cfg, trainer, model, datamodule = cli.parse_args()
    # datamodule: MusicLiteDataModule

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    count = 0
    for idx, batch_data in enumerate(train_dataloader):
        if isinstance(batch_data, list):
            print(f"loop out list with {len(batch_data)} items")
            for data in batch_data:
                if isinstance(data, dict):
                    print(f"\tuttid={data['uttid']} keys={data.keys()}")
        elif isinstance(batch_data, dict):
            print(f"loop out dict with keys={batch_data.keys()}")
            for key, value in batch_data.items():
                print(f"\tkey={key} value={value}")


        count += 1
        if test_step > 0 and count >= test_step:
            break



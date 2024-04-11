import os
from hyperpyyaml import load_hyperpyyaml
import pytorch_lightning as pl

from samantha.utils.parser import parse_arguments

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_yaml(yaml_path):
    args = [
        "fit", "-c", yaml_path,
        "--run_opts.num_workers", "0",
        # "--run_opts.lyrics_tokenizer", "zh_wordpiece",
        # "--run_opts.language", "['EN']",
    ]
    hparams_file, run_opts, overrides = parse_arguments(args)

    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    return hparams


def test_yaml_fit():
    yaml_path = os.path.join(
        curr_dir,
        '../conf/symbolic_music/20240308.melody.yaml',
    )
    hparams = load_yaml(yaml_path)

    pl_module = hparams["pl_module"]
    pl_datamodule = hparams["pl_datamodule"]
    trainer = hparams["trainer"]
    
    ## Skip setup since it requires gpu
    pl_module.setup = lambda stage: None
    metrics = trainer.fit(model=pl_module, datamodule=pl_datamodule)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
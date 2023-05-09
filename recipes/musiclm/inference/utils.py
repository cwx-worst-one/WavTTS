from hyperpyyaml import load_hyperpyyaml

from samantha.utils.hparams import DotDict


def load_config(hparams_file: str):
    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)
    return cfg


def load_model(pl_module, ckpt_path: str, device: str):
    pl_module = pl_module.load_from_checkpoint(ckpt_path)
    pl_module = pl_module.eval()
    return pl_module.to(device)

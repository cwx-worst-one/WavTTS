from argparse import ArgumentParser

from hyperpyyaml import load_hyperpyyaml

from recipes.umm.transforms.speech import SpeechTransform
from samantha.utils.hdfs_tools import hdfs_open
from samantha.utils.hparams import DotDict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    args = parser.parse_args()

    hparams_file = args.conf
    if hparams_file.startswith("hdfs"):
        with hdfs_open(hparams_file, "r") as fin:
            hparams = load_hyperpyyaml(fin)
    else:
        with open(hparams_file, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)

    pl_datamodule = cfg.pl_datamodule

    audio_transform: SpeechTransform = cfg.pl_module.model.audio_transform
    audio_transform.calculate_statistics(pl_datamodule, max_datapoints=100000)

    # set feature_cmvn to produced DataModule_Transform.stats.pt file
    feature_cmvn_fp = cfg.extra_params.feature_cmvn
    audio_transform.check_statistics(feature_cmvn_fp, pl_datamodule, max_batches=20)
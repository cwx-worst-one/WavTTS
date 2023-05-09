from collections import defaultdict

import pytest
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from samantha.utils.hparams import DotDict


@pytest.mark.skip("We only run this on Merlin for data verification")
def test_wav2vec_embedding():
    import librosa.feature
    import pandas as pd
    import seaborn as sns

    hparams_file = (
        "./recipes/audio_diffusion/conf/"
        "140223-singsong/singsong_semantic2semantic-coarse.yaml"
    )

    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)
    feats = defaultdict(list)

    datamodule = cfg.pl_datamodule
    train_loader = datamodule.train_dataloader()

    max_batches = 1000
    for batch_idx, batch in enumerate(tqdm(train_loader, total=max_batches)):
        if batch_idx > max_batches:
            break

        acc_audio, voc_audio, acc_semantic, voc_semantic = batch

        for idx, accompaniment in enumerate(acc_audio):
            rms_accompaniment = librosa.feature.rms(y=accompaniment)
            if rms_accompaniment.max() < 0.1:
                torchaudio.save(
                    (
                        f"{batch_idx}-{idx}-accompaniment-"
                        f"rms-{rms_accompaniment.max()}.mp3"
                    ),
                    accompaniment,
                    24000,
                )
            feats["peak_rms_accompaniment"].append(rms_accompaniment.max())

        for idx, vocal in enumerate(voc_audio):
            rms_vocal = librosa.feature.rms(y=vocal)
            feats["peak_rms_vocal"].append(rms_vocal.max())

            if rms_vocal.max() < 0.1:
                torchaudio.save(
                    f"{batch_idx}-{idx}-vocal-rms-{rms_vocal.max()}.mp3", vocal, 24000
                )

    df = pd.DataFrame.from_dict(feats, orient="columns")

    for k in feats:
        plot = sns.displot(data=df, x=k, kde=True)
        plot.fig.savefig(f"{k}.png")

    plot = sns.displot(data=df, x="peak_rms_accompaniment", y="peak_rms_vocal")
    plot.fig.savefig("peak_rms_accompaniment_vocal.png")

import copy
import pickle

from cruise.configuration.cli import CruiseCLI
from hyperpyyaml import load_hyperpyyaml

from apps.bigmusic.mariana_tasks.semantic_emb_module import SemanticEmbModule
from apps.bigmusic.mariana_tasks.semantic_modules import (
    SemanticEmbModule as SemanticEmbModuleLegacy,
)
from samantha.dataio.bigmusic.lite import MusicLiteDataModule

#################################################################
# Save pickle
#################################################################


def test_musiclitedatamodule(
    cfg_path="./samantha/dataio/bigmusic/temp_directories/v5_dataloader_mariana.yaml",
    test_step=100,
):
    import sys

    cli = CruiseCLI(datamodule_class=MusicLiteDataModule)
    sys.argv = ["", "--config", cfg_path]
    _, _, _, datamodule = cli.parse_args()
    datamodule: MusicLiteDataModule

    dataloader = datamodule.train_dataloader()
    count = 0
    batches = []
    for batch_data in dataloader:
        batches.append(batch_data)
        count += 1
        if test_step > 0 and count >= test_step:
            break

    print(f"number of data:{count}")
    datamodule.teardown()

    with open("data.pkl", "wb") as file:
        batches = pickle.dump(batches, file)


################################################################
# Check emb
################################################################


def test_legacy_and_refactored_emb_modules(pkl_path: str = "data.pkl"):
    with open(pkl_path, "rb") as file:
        batches = pickle.load(file)

    with open(
        "./apps/bigmusic/mariana_tasks/conf/v5_semantic_embedder.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        hps = load_hyperpyyaml(f)
    emb = SemanticEmbModule(**hps)
    emb.setup("fit")
    emb.to("cuda:0")
    x = emb(copy.deepcopy(batches[0]))
    print(x)

    with open(
        "./apps/bigmusic/mariana_tasks/conf/v5_emb.yaml", "r", encoding="utf-8"
    ) as f:
        hps = load_hyperpyyaml(f)
    emb_legacy = SemanticEmbModuleLegacy(**hps)
    emb_legacy.setup("fit")
    emb_legacy.to("cuda:0")
    y = emb_legacy(copy.deepcopy(batches[0]))
    print(y)

    print("done")


if __name__ == "__main__":
    test_musiclitedatamodule(test_step=2)
    test_legacy_and_refactored_emb_modules()

from argparse import ArgumentParser

from recipes.research.diff import DiffConfig, DiffUMM

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--commit_hash", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    args = parser.parse_args()

    diff = DiffUMM(DiffConfig(n_layer=4))
    diff = diff.to("cuda")

    model = diff.model.to_torchscript()
    # diff = diff.to_torchscript()

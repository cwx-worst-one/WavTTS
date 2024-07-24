from argparse import ArgumentParser

from recipes.research.ar import ARUMM, ARConfig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--commit_hash", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    args = parser.parse_args()

    model = ARUMM(ARConfig(n_layer=4))
    model = model.to("cuda")

    model = model.to_torchscript()

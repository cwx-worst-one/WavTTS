import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from samantha.utils.hparams import DotDict

if __name__ == "__main__":
    hparams_file = "./recipes/musiclm/conf/data/default.yaml"

    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)

    train_loader = cfg.pl_datamodule.train_dataloader()
    pl_module = cfg.pl_module

    pl_module = pl_module.to("cuda")

    freqs = torch.zeros(6, 1024).to(pl_module.device)
    tokens = []
    num_batches = 20
    for idx, batch in enumerate(tqdm(train_loader)):
        if idx > num_batches:
            break

        audio = batch[0].to(pl_module.device)
        with torch.no_grad():
            tokens.append(pl_module.audio_model(audio))
            # b, q, s = tokens.shape
            # for b_ in range(b):
            #     for q_ in range(q):
            #         counts = tokens[b_, q_].bincount()
            #         for idx in range(len(counts)):
            #             freqs[q_, idx] += counts[idx]

    tokens = torch.cat(tokens)

    import matplotlib.pyplot as plt

    quants = []
    for q in range(6):
        quants.append([i.cpu().item() for i in tokens[:, q].reshape(-1).bincount()])
        plt.bar(range(len(quants[q])), quants[q])
        plt.savefig(f"hist_{q}.png")
        plt.close()

    import pandas as pd

    df = pd.DataFrame(quants)
    breakpoint()

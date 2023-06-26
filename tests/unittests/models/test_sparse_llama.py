import warnings

import torch
from samantha.models.sparse_llama import ModelArgs, LLaMa


def test_sparse_llama():
    params = ModelArgs()
    model = LLaMa(params).eval()
    params.use_cache = True
    input_tokens = torch.randint(low=0, high=1024, size=[2, 100])

    with torch.no_grad():
        out1 = model(input_tokens)["logits"]

        out2 = []
        for i in range(100):
            out = model(input_tokens[:, i: i + 1], start_pos=i)
            out2.append(out["logits"])
        out2 = torch.cat(out2, dim=1)

    err = out1 - out2
    warnings.warn(f"{out1.abs().mean()=}, {err.abs().max()=}, {out2.abs().mean()=}")

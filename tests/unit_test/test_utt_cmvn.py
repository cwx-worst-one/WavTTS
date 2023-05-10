"""utterance CMVN testcase"""
#!/usr/bin/env python3
import numpy as np
import torch
from core.dataset.preprocess.fbank import UtteranceCmvn, DynamicCmvn


def test():
    """test"""
    eps = 1e-9
    key = "fbank"
    bsz = 10
    max_len = 200
    feat_dim = 80
    fbank = torch.randint(low=-100, high=100, size=(bsz, max_len, feat_dim)).float().numpy()
    utt_cmvn = UtteranceCmvn(key=key)
    dynamic_cmvn = DynamicCmvn(
        key=key,
        cmvn_type="sliding_dynamic",
        norm_var=True,
        center=False,
        min_cmn_window=5,
        cmn_window=10000,
    )
    # test single
    for i in range(bsz):
        data = {key: fbank[i]}
        x = utt_cmvn(data)
        x1 = dynamic_cmvn(data)
        assert np.allclose(x[key], x1[key], rtol=eps, atol=eps)


if __name__ == "__main__":
    test()

import pickle

import numpy as np
import torch
from scipy.io.wavfile import write

with torch.no_grad():
    device = "cuda"
    decoder = (
        torch.jit.load("/mnt/bn/jcong-bn-us/models/wavevae_decoder.pt")
        .to(device)
        .eval()
    )

    bns = pickle.load(
        open("tmp/v07e5bg50000c1mr8rvukv02lhr6uni0_000_0008.bns", "rb")
    ).T[None]
    bns = torch.from_numpy(bns).to(device)
    m, logs = torch.split(bns, 32, dim=1)
    z_outputs = m + torch.rand_like(m) * torch.exp(logs)
    de_wav = decoder(z_outputs)[0][0].cpu().numpy()
    de_wav *= (32767) / max(0.01, np.max(np.abs(de_wav)))
    write(
        "xx/v07e5bg50000c1mr8rvukv02lhr6uni0_000_0008_de.wav",
        24000,
        de_wav.astype(np.int16),
    )
    print(bns.shape)

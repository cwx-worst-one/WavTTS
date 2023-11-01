import argparse
import pickle
import uuid

import numpy as np
import torch
from scipy.io.wavfile import write


def decode(args):

    with torch.no_grad():
        device = "cuda"
        decoder = torch.jit.load(args.decoder).to(device).eval()

        bns = pickle.load(open(args.bn, "rb")).T[None]
        bns = torch.from_numpy(bns).to(device)
        m, logs = torch.split(bns, 32, dim=1)
        z_outputs = m + torch.rand_like(m) * torch.exp(logs)
        de_wav = decoder(z_outputs)[0][0].cpu().numpy()
        de_wav *= (32767) / max(0.01, np.max(np.abs(de_wav)))
        write(args.wav, 24000, de_wav.astype(np.int16))
        print(bns.shape, args.wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=str, default="wavevae_decoder_0.pt")
    parser.add_argument("--bn", type=str, required=True)
    parser.add_argument("--wav", type=str, default=f"{uuid.uuid4().hex}.wav")
    args = parser.parse_args()
    decode(args)

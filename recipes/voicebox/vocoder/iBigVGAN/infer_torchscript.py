from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from tqdm import tqdm


device='cuda:0'
MAX_WAV_VALUE = 32768.0
sampling_rate = 24000


def inference(a):
    decoder = torch.jit.load(a.torchscript_pth).to(device).eval()
    filelist = os.listdir(a.input_mels_dir)
    os.makedirs(a.output_dir, exist_ok=True)
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            if filname[-3:] != "npy":
                continue
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device).unsqueeze(0)
            y_g_hat = decoder.forward(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, sampling_rate, audio)



def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--torchscript_pth')
    a = parser.parse_args()
    inference(a)


if __name__ == '__main__':
    main()

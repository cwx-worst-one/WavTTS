#!/usr/bin/env python
# -*- coding:utf-8 -*-
#***********************************************
#      Filename: projector_csv.py
#        Author: Jeff Pan
#         Email: panjunjie.jeff@bytedance.com
#   Description: --
#        Create: 2023-10-01 15:27:56
# Last Modified: 2023-10-01
#***********************************************

import os
import sys
import re
import json
import librosa
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pickle as pk
import csv
import torch

id2label = {
    "0": "neutral",
    "1": "sad",
    "2": "happy",
    "3": "angry",
    "4": "surprise",
    "5": "hate",
    "6": "scare",
    "7": "<UNK>"
}


def main(args):
    model = torch.jit.load(args.model)
    wav, sr = librosa.load(args.inputs, sr=16000)
    wav = torch.as_tensor(wav).to("cuda:0")
    mean, std = torch.mean(wav), torch.std(wav)
    wav = (wav - mean) / std
    wav = wav.unsqueeze(0)
    mean_hidden_states, hidden_states, classifier_hidden, logits = model(wav)
    print("mean_hidden_states: ", mean_hidden_states.shape)
    print("hidden_states: ", hidden_states.shape)
    print("classifier_hidden: ", classifier_hidden.shape)
    print("logits: ", logits.shape)
    predicts = list(map(str, torch.max(logits[:, :-1],1)[1].cpu().numpy()))[0]
    print(args.inputs.split('/')[-1], id2label[predicts])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
        default="/mnt/bn/nas-jeff-02/resource/models/SER/ser_0.pt",
        dest="model", help="")
    parser.add_argument("--inputs",
        default="",
        dest="inputs", help="")
    parser.add_argument("--outputs",
        default="",
        dest="outputs", help="")
    args = parser.parse_args()
    main(args)

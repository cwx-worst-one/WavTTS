# ser indicates speech emotion recognition


import os
import warnings

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

warnings.simplefilter(action="ignore", category=UserWarning)


def preprocess_audio(audio_bin, *_, **__):
    wav, sr = torchaudio.load(audio_bin, channels_first=False)
    if wav.numel() == 0:
        return None, None
    wav = wav.reshape(1, -1)
    audio_dur = wav.shape[-1] / float(sr)
    return wav, audio_dur


def load_model(device, model_path, *_, **__):
    from speechbrain.pretrained import Pretrained
    from speechbrain.utils.parameter_transfer import Pretrainer

    ckpt_flag = "CKPT+2023-08-03+00-15-11+00"
    hparams_file = os.path.join(model_path, "inference.yaml")
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    loadables = {"wav2vec2": hparams["wav2vec2"], "model": hparams["model"]}
    ckptdir = os.path.join(model_path, "save", ckpt_flag)
    paths = {
        "wav2vec2": os.path.join(ckptdir, "wav2vec2.ckpt"),
        "model": os.path.join(ckptdir, "model.ckpt"),
    }
    pretrainer = Pretrainer(collect_in=ckpt_flag, loadables=loadables, paths=paths)
    pretrainer.collect_files()
    pretrainer.load_collected(device="cpu")
    emo_id = Pretrained(hparams["modules"], hparams, run_opts={"device": device})
    return emo_id


@torch.no_grad()
def process_batch(model, batch, device, *_, **__):
    if not batch:
        yield from batch

    for wav in batch:
        wav = wav.to(device)
        lens = torch.ones(wav.shape[0], device=device)

        outputs = model.mods.wav2vec2(wav)
        outputs = model.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        emo_emb = outputs.cpu().numpy()

        out_tag = model.mods.output_mlp(outputs)
        emo_tag = model.hparams.log_softmax(out_tag).exp().cpu().numpy()

        out_deg = model.mods.output_deg(outputs)
        emo_deg = model.hparams.log_softmax(out_deg).exp().cpu().numpy()

        yield emo_tag.squeeze(), emo_deg.squeeze(), emo_emb.squeeze()

from functools import partial
import os
import librosa
import numpy as np
import time
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import Resample
#from samantha.transforms.audio import FastNormalizeAudio
from typing import Optional

import scripts.utils.pyloudnorm as pyln

SUPPORTED_UMM_ARCH=(
    "conformer_umm", 
    "conv1d_umm",
    "dual_umm_v1",
    "dual_umm_v2",
    "dual_umm_v3",
)

class FastNormalizeAudio(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x, norm_tensor: Optional[torch.Tensor] = None, eps=1e-8
    ) -> torch.Tensor:
        if norm_tensor is None:
            denom = x.abs().max().clamp_min_(eps).expand_as(x)
        else:
            denom = norm_tensor.abs().max().clamp_min_(eps).expand_as(x)
        return torch.div(x, denom)

def sample_audio(wav, window_size=30, hop_size=5, sr=24000):
    """
    inputs:
        wav(torch.FloatTensor): wav tensor with [c,t]
        window_size(int): chunk_shape
        hop_size(int): stride
        sr(int): sample rate
    return:
        wav chunks with [n,c,t'] && last_chunk if pad
        (torch.Tensor, bool) 
    """
    wav_len = wav.shape[-1]
    win_len = window_size*sr
    hop_len = hop_size*sr

    if wav_len < win_len:
        n_chunks = 1
    else:
        n_chunks = (wav_len-win_len)//hop_len + 1
    n_pad = n_chunks * hop_len + win_len - wav_len

    if n_pad>0:
        wav = torch.nn.functional.pad(wav,[0,n_pad],mode="constant",value=0.0)
    return (
        torch.stack([wav[:,cidx*hop_len:cidx*hop_len+win_len] for cidx in range(n_chunks)],dim=0), # chunks
        n_pad>0, # last chunk if pad
    )


def preprocess_audio(audio_bin, sample_rate, resampler, device, *args, **kwargs):
    """
    # TODO: this should be handled centrally within the data/model pipeline, not here.
    # this was added by engineering team and hacky fix is approved for now (2024/02/21).

    # "NOTE" are left for functions  that require extra care.
    # """

    wav, sr = librosa.load(audio_bin, sr=None) # NOTE similar audio loading framework is required
    if wav.size == 0:
        return None, 0
    audio_dur = wav.shape[-1] / float(sr)
    if bool(kwargs.get("norm", False)):
        normalize_fn = FastNormalizeAudio() # NOTE similar audio normalization function is required
        wav = normalize_fn(torch.from_numpy(wav))
    
    if bool(kwargs.get("loudness_norm", False)):
        meter = pyln.Meter(sr) # create BS.1770 meter
        wav = wav.T
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -14.0)
        if not isinstance(wav, np.ndarray):
            return None, 0
        wav = wav.T

    if len(wav.shape) == 2 and wav.shape[-1] == 2:
        wav = wav.mean(dim=1, keepdim=True) # NOTE similar mono-fying logic is required

    wav = torch.as_tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)

    if sr != sample_rate:
        if sr not in resampler:
             # NOTE: similar choice of resampler is required
            resampler[sr] = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resampler[sr](wav)
    
    size = wav.shape[-1]

    # NOTE: padding logic needs to be centralised
    if bool(kwargs.get("padding", False)):
        pad_len = 5120 - size % 5120
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), mode="constant", value=0.0)
        # print(f"(ori,pad)={size,pad_len}")
    return wav, audio_dur


@torch.no_grad()
def process_batch(model, batch, device, target_key, *args, **kwargs):
    if not batch:
        yield from batch

    do_sample = bool(kwargs.get("do_sample", False))
    if do_sample:
        window_size=int(kwargs.get("window_size", 0))
        hop_size=int(kwargs.get("hop_size", 0))
        target_sr=int(kwargs.get("target_sr", 24000))
        assert window_size>0 and hop_size>0
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE",64))

    assert "umm_model_arch" in kwargs
    model_arch = kwargs.get("umm_model_arch", "dual_umm_v3")
    if model_arch in ["dual_umm_v1", "dual_umm_v3"]:
        from recipes.umm.utils.mss import MSSPredictor
        predictor = MSSPredictor().to(device)

    assert isinstance(target_key, list)  and len(target_key) == 1
    target_key = target_key[0]

    for wav in batch:
        if do_sample:
            if wav is None or (isinstance(wav,dict) and (len(wav)==0 or any([w is None for w in wav]))):
                yield None
            else:
                assert isinstance(wav, dict) and len(wav)==1 and 'audio' in wav
                wav = wav["audio"]
                wav_chunks, padding_last = sample_audio(wav, window_size, hop_size, target_sr)
                umm_token_chunks = torch.cat([model.wav2token(wav_chunks[i:i+max_batch_size]) for i in range(0, len(wav_chunks), max_batch_size)],dim=0)
                yield {
                    "chunks": umm_token_chunks.cpu().numpy(),
                    "padding_last": padding_last,
                }
        else:
            if wav is None or (isinstance(wav,dict) and (len(wav)==0 or any([w is None for w in wav]))):
                yield None
            else:
                if model_arch in ["conformer_umm", "conv1d_umm"]:
                    assert isinstance(wav, dict) and len(wav)==1 and target_key in wav
                    wav = wav[target_key]
                    if wav.ndim==2:
                        wav = wav.unsqueeze(0)
                    umm_token = model.wav2token(wav)
                elif model_arch in ["dual_umm_v2"]:
                    assert isinstance(wav, dict) and len(wav)==1 and 'audio' in wav
                    wav = wav["audio"]
                    if wav.ndim==2:
                        wav = wav.unsqueeze(0)
                    token_vocal = model.wav2token(wav, 'vocal').squeeze()
                    token_inst = model.wav2token(wav, 'inst').squeeze()
                    umm_token = torch.stack([token_vocal, token_inst], 1)
                    umm_token = umm_token.reshape(-1).unsqueeze(0)
                elif model_arch in ["dual_umm_v1", "dual_umm_v3"]:
                    assert isinstance(wav, dict) and ('audio' in wav or ('vocal' in wav and 'acc' in wav))
                    if "vocal" in wav and "acc" in wav:
                        voc, acc = wav["vocal"], wav["acc"]
                    else:
                        assert "audio" in wav
                        wav = wav["audio"]
                        if wav.ndim==2:
                            wav = wav.unsqueeze(0)
                        voc, acc = predictor(wav)
                    # print(f"{(voc.shape,acc.shape)=}")
                    token_vocal = model.wav2token(voc[:, None], 'vocal')
                    token_acc = model.wav2token(acc[:, None], 'inst')
                    umm_token = torch.stack([token_vocal, token_acc], 2).flatten(1, 2)
                else:
                    raise NotImplementedError

                # print(f"{model_arch} token={umm_token.shape} wav={wav.shape}")
                yield umm_token.cpu().squeeze(0).numpy()



if __name__ == '__main__':
    from hyperpyyaml import load_hyperpyyaml
    device = torch.device("cuda:0")

    yaml_path = "recipes/umm2/conf/inference_stage3_wav2tokens.yaml"
    hparams = load_hyperpyyaml(open(yaml_path, "r", encoding="utf-8"))
    model = hparams["frontend"]
    required_modules = hparams["required_modules"]

    for module_name, loader_config in hparams["required_modules"].items():
        print(f"loading module {module_name}...")
        _args = {k: v for k, v in loader_config.items() if k != "loader"}
        loader = loader_config["loader"](**_args)
        model = loader.nn_load_model(model)
    # model is here: !new:recipes.umm2.models.tasks.wav2tokens.WAV2TOKENS
    # input is a dict having 'audio'

    model.to(device)
    data = {'audio': torch.randn(1, 24000*30, dtype=torch.float).to(device)}
    token = model.forward(data)['vq_ids']
    print(f"data={data['audio'].shape} token={token.shape}")



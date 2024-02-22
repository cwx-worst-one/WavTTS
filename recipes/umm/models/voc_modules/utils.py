import torch
import numpy as np

def mel2wav(mel, f0, model):
    device = next(model.parameters()).device
    c = mel.unsqueeze(0).to(device).transpose(1, 2)
    if f0 is not None:
        f0 = f0[None, None, :]
    y = model(c, f0)
    return y.reshape(-1)


def denorm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100, pitch_padding=None, min=50, max=900):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2 ** f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, a_min=min, a_max=max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def mel2f0(mel, model):
    if len(mel.shape) == 2:
        mel = mel[None]
    device = next(model.parameters()).device
    mel = mel.to(device)
    f0_mel = model(mel)
    f0_pred = f0_mel[:, :, 0]
    uv_pred = f0_mel[:, :, 1]
    return denorm_f0(f0_pred[0], uv_pred[0])

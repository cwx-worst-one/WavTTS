import random

import numpy as np
import torch
from torch.nn import functional as F

noises = None
noise_idx = 0


def init_gumbel_noise(t, max_length=1000, seed=1995):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    assert (len(t.shape) == 3 and t.shape[1] == 1) or len(
        t.shape
    ) == 2, f"input t={t.shape}"
    if len(t.shape) == 3:
        b, _, d = t.shape
    else:
        b, d = t.shape
    noises = torch.zeros(b, max_length, d).uniform_(0, 1).to(t.device)
    print(
        f"=======init_gumbel_noise======= noises={noises.shape} {noises.mean()} {noises.max()}\n"
    )
    for i in range(50, 60, 1):
        print(f"{noises[0,i,:].mean()} {noises[0,i,:].max()}\n")
    return noises


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t, fixed_noise):
    if fixed_noise:
        global noises
        global noise_idx
        if noises is None:
            noises = init_gumbel_noise(t)
        # assert (
        #     len(noises.shape) == 3
        #     and noises.shape[0] == t.shape[0]
        #     and noises.shape[2] == t.shape[2]
        #     and t.shape[1] == 1
        # )
        if noise_idx < noises.shape[1]:
            if len(t.shape) == 3:
                noise = noises[:, noise_idx : noise_idx + 1, :]
            else:
                noise = noises[:, noise_idx, :]
            noise_idx += 1
        if noise_idx >= noises.shape[1]:
            noise_idx = 0
    else:
        noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, fixed_noise=True):
    return ((t / temperature) + gumbel_noise(t, fixed_noise=fixed_noise)).argmax(
        dim=dim
    )


def top_k(logits, thresh=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thresh) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def sample_legacy(predict_logits, temp, mode="naive", device="cuda:0"):
    predict_logits = predict_logits.float()
    if mode == "naive":
        predict_logits = predict_logits / temp
        predict_logits = top_k(predict_logits, thresh=0.9)
        probs = predict_logits.softmax(dim=1)  # [b, d]
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample().unsqueeze(1).to(device)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thresh=0.9)
        samples = gumbel_sample(predict_logits, temp, fixed_noise=False).unsqueeze(
            dim=1
        )
    else:
        raise NotImplementedError()
    return samples


def top_p_logits(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(
        logits, dtype=sorted_indices_to_remove.dtype
    ).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    out = logits.clone()
    out[indices_to_remove] = -float("Inf")
    return out


def sample(
    predict_logits, temp, thresh=0.9, mode="naive", return_probs=False, eos_weight=1.0
):
    sample_probs = None
    if mode == "naive":
        predict_logits = predict_logits / temp
        predict_logits = top_p_logits(predict_logits, thresh)
        probs = predict_logits.softmax(dim=-1)
        if probs.shape[-1] > 16384:
            probs[..., -2:] *= eos_weight
            probs /= probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample()
        if return_probs:
            sample_probs = torch.gather(probs, -1, samples.unsqueeze(1)).squeeze(1)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thresh=thresh)
        samples = gumbel_sample(predict_logits, temp, fixed_noise=True)
        if return_probs:
            probs = (predict_logits / temp).softmax(dim=-1)
            sample_probs = torch.gather(probs, -1, samples.unsqueeze(1)).squeeze(1)
    elif mode == "greedy":
        samples = torch.argmax(predict_logits, dim=-1)
        if return_probs:
            sample_probs = samples / samples
    else:
        raise NotImplementedError

    if return_probs:
        return samples, sample_probs
    else:
        return samples

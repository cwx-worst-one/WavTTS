import torch


def sample(predict_logits, temp, mode="naive", device="cuda:0", thres=0.9):
    predict_logits = predict_logits.float()
    if mode == "naive":
        predict_logits = predict_logits / temp
        predict_logits = top_k(predict_logits, thres=thres)
        probs = predict_logits.softmax(dim=1) # [b, d]
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample().unsqueeze(1).to(device)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thres=thres)
        samples = gumbel_sample(predict_logits, temp).unsqueeze(dim=1)
    else:
        raise NotImplementedError()
    return samples


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

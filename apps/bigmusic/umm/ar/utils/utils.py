import hashlib
import io
import os
import random
import re
import subprocess
import unicodedata
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from pydub import AudioSegment
from scipy.io.wavfile import write
from torch.nn import functional as F

from samantha.utils.hparams import DotDict

noises = None
noise_idx = 0


def init_gumbel_noise(t, max_length=1000, seed=1995):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    assert (len(t.shape) == 3 and t.shape[1] == 1) or len(t.shape) == 2, f"input t={t.shape}"
    if len(t.shape) == 3:
        b,_,d = t.shape
    else:
        b,d = t.shape
    noises = torch.zeros(b, max_length, d).uniform_(0, 1).to(t.device)
    print(f"=======init_gumbel_noise======= noises={noises.shape} {noises.mean()} {noises.max()}\n")
    for i in range(50,60,1):
        print(f"{noises[0,i,:].mean()} {noises[0,i,:].max()}\n")
    return noises



def load_config(hparams_file: str, overrides=None):
    # Load hyperparameter file with command-line overrides
    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    cfg = DotDict(hparams)
    return cfg


def load_model(pl_module, ckpt_path: str, device: str):
    pl_module = pl_module.load_from_checkpoint(ckpt_path)
    pl_module = pl_module.eval()
    return pl_module.to(device)


def sample(
    predict_logits,
    temp,
    thresh=0.9,
    mode="naive",
    return_probs=False,
    exclude_ids=None,
):
    if exclude_ids is not None:
        for i in exclude_ids:
            predict_logits[..., i] = float("-inf")
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample()
    elif mode == "top_p" or mode == "top_k":
        predict_logits = predict_logits / (temp)
        if mode == "top_p":
            predict_logits = top_p_logits(predict_logits, thresh)
        else:
            predict_logits = top_k(predict_logits, thresh=thresh)
        probs = predict_logits.softmax(dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample()
    elif mode == "gumbel" or mode == "gumbel_fixed_noise":
        predict_logits = top_k(predict_logits, thresh=thresh)
        fixed_noise = mode == "gumbel_fixed_noise"
        samples = gumbel_sample(
            predict_logits, temp, fixed_noise=fixed_noise, exclude_ids=exclude_ids
        )
        probs = (predict_logits / temp).softmax(dim=-1)
    else:
        raise NotImplementedError()

    if return_probs:
        sample_probs = torch.gather(probs, -1, samples.unsqueeze(1)).squeeze(1)
        return samples, sample_probs
    else:
        return samples


def set_seed(seed=1996):
    # reproduction setting
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    return


def generate_hash(string):
    """
    (added by Bochen on 2023-06-12)
    encode string (e.g., text prompt) to 64-byte encoded string that captures nuance
    the encoded string is not random
    it can be used to differentiate filename for similar text prompts
    e.g.,
    - input1: "The main soundtrack of an arcade game. It is fast-paced."
        - output1: "cd14cca5dca6bb29f284de44f9c3e446b08bbfafab1019d64f38a9294bc4b47a"
    - input2: "The main soundtrack of an arcade game. It is slow-paced."
        - output2: "10b106b469638031217236f8a83bc0d672a18536bd2a2c8830ce5a0fb628e2a5"
    """
    hash_object = hashlib.sha256(string.encode())
    hash_value = hash_object.hexdigest()
    return hash_value


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def format_name(text_prompt):
    """
    (added by Bochen on 2023-06-12)
    format the text_prompt for a unique filename
    """
    formatted_text = slugify(text_prompt)[:128] + "_" + generate_hash(text_prompt)[:4]
    return formatted_text


def dump_wav(audio, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    handle = io.BytesIO()
    write(handle, sr, audio)
    handle.seek(0)
    return handle.read()


def save_wav(audio, output_file, sr=24000, save_mp3=False):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(output_file, audio, sr)
    if save_mp3:
        output_file_mp3 = output_file.replace(".wav", ".mp3")
        subprocess.run(
            f"ffmpeg -y -i {output_file} -ar {sr} -ac 1 -b:a 320k {output_file_mp3}",
            shell=True,
        )
        os.remove(output_file)


def load_wav(path, sr=24000):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=sr)
    else:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sr)
        wav = np.asarray(audio.get_array_of_samples())
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t, fixed_noise):
    if fixed_noise:
        global noises
        global noise_idx
        if noises == None or t.shape[0] != noises.shape[0]:
            noises = init_gumbel_noise(t)
        # assert len(noises.shape) == 3 and noises.shape[0] == t.shape[0] and noises.shape[2] == t.shape[2] and t.shape[1] == 1 
        if noise_idx < noises.shape[1]:
            if len(t.shape) == 3:
                noise = noises[:,noise_idx:noise_idx+1,:]
            else:
                noise = noises[:,noise_idx,:]
            noise_idx += 1
        if  noise_idx >= noises.shape[1]:
            noise_idx = 0
    else:
        noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    t: torch.Tensor,
    temperature=1.0,
    dim=-1,
    fixed_noise=False,
    exclude_ids=None,
):
    gumbel_dist = (t / temperature) + gumbel_noise(t, fixed_noise=fixed_noise)
    if exclude_ids is not None:
        for i in exclude_ids:
            gumbel_dist[..., i] = float("-inf")
    return gumbel_dist.argmax(dim=dim)


def top_k(logits, thresh=0.95):
    num_logits = logits.shape[-1]
    k = max(int((1 - thresh) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs

def top_p_logits(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = torch.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
    out = logits.clone()
    out[indices_to_remove] = -float('Inf')
    return out

def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size

def length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return torch.arange(final_length, device=lengths.device)[None, :] < lengths[:, None]
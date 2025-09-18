import io
import random
import re
import unicodedata
import hashlib
import os
import subprocess
import librosa
import numpy as np
import torch
import torchaudio
import pyloudnorm as pln

from typing import Union, Optional
from pydub import AudioSegment
from scipy.io.wavfile import write
from hyperpyyaml import load_hyperpyyaml
from samantha.utils.hparams import DotDict
from torch import Tensor
from torch.nn import functional as F
from torchaudio import functional as Fa


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
    p_base=0.1,
):
    if exclude_ids is not None:
        for i in exclude_ids:
            predict_logits[..., i] = float("-inf")
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample()
    elif mode == "top_p" or mode == "top_k" or mode == "min_p":
        # pre-temp-scale
        if isinstance(temp, float):
            predict_logits = predict_logits / (temp)
        elif isinstance(temp, torch.Tensor):
            predict_logits = predict_logits / (temp.reshape(-1, 1, 1))
        else:
            raise NotImplementedError()

        if mode == "top_p":
            predict_logits = top_p_logits(predict_logits, thresh)
        elif mode == "min_p":
            min_p_logits = min_p(predict_logits, p_base=p_base)
            predict_logits = top_p_logits(min_p_logits, thresh)
        else:
            predict_logits = top_k(predict_logits, thresh=thresh)
        # post-temp-scale
        # predict_logits = predict_logits / (temp)
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
    elif mode == "greedy":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=-1)
        samples = probs.argmax(dim=-1)
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


def save_wav(audio, output_file, sr=24000, save_mp3=False, normalize_volume=False):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(output_file, audio, sr)

    if save_mp3 and normalize_volume:
        new_ext = ".normalized.mp3"
    elif normalize_volume:
        new_ext = ".normalized.wav"
    elif save_mp3:
        new_ext = ".mp3"
    else:
        # keep original wav
        return output_file

    output_file_target = output_file.replace(".wav", new_ext)
    if normalize_volume:
        command = "ffmpeg-normalize '%s' -t -16 --keep-loudness-range-target -c:a libmp3lame -b:a 320k -o '%s' -f" % (output_file, output_file_target)
    else:
        command = f"ffmpeg -y -i {output_file} -ar {sr} -ac 1 -b:a 320k {output_file_target}"
    subprocess.run(command, shell=True)
    os.remove(output_file)
    return output_file_target


def tensor_resample(
    wav: Tensor, orig_freq: int = 44100, new_freq: int = 24000
) -> Tensor:
    """
    Resample audio with TorchAudio following HiFi-GAN setting which mimics SoX.
    Compatible with GPU computation.
    """
    assert isinstance(wav, Tensor), f"expected torch tensor, but got {type(wav)}"

    # Check whether the last dimension is the time dimension
    for i in range(len(wav.shape) - 1):
        assert wav.shape[i] <= wav.shape[-1]
    if new_freq == orig_freq:
        return wav

    # The resample setting here comes from HiFi-GAN, which mimics SoX
    orig_type = wav.dtype
    wav = Fa.resample(
        wav.double(),
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    ).to(orig_type)

    return wav


def level_norm_lufs(
    wav: Union[Tensor, np.ndarray],
    sr: int = 24000,
    target_lufs: float = -23.0,
    tensor_out: Optional[bool] = None,
) -> Union[Tensor, np.ndarray]:
    """
    Loudness level normalization to target_lufs (-23dB LUFS by default).
    NOTE: this operation uses numpy operations, and thus does not support gradient and always runs on CPU.
    wav should have shape (T, 1) or (T,)

    tensor_out controls whether the output is a torch tensor or a numpy array.
    If tensor_out is None and input wav is a torch tensor, a torch tensor will be returned.
    If tensor_out is None and input wav is a numpy array, a numpy array will be returned.
    """
    # Convert to numpy array
    tensor_wav = isinstance(wav, Tensor)
    tensor_out = tensor_wav if tensor_out is None else tensor_out
    if tensor_wav:
        orig_device, orig_dtype = wav.device, wav.dtype
        wav = wav.cpu().numpy()

    # wav now must be a numpy array with shape either (T, 1) or (T,)
    assert isinstance(wav, np.ndarray), \
        f"wav must be a numpy array, but got {type(wav)}"
    assert 0 < len(wav.shape) <= 2, \
        f"wav must have shape either (T, 1) or (T,), but got {wav.shape}"
    assert len(wav.shape) == 1 or wav.shape[1] < wav.shape[0], \
        "the time dimension must be the first dimension"
    wav = wav.astype(np.float64)

    # Now, actually do the level norm
    loudness_meter = pln.Meter(sr)
    measured_lufs = loudness_meter.integrated_loudness(wav)
    gain_db = target_lufs - measured_lufs
    gain_linear = 10 ** (gain_db / 20.0)
    wav = wav * gain_linear

    if tensor_out:
        wav = torch.from_numpy(wav).to(orig_device).to(orig_dtype)
    return wav


def load_wav(path, sr=24000, mono=True):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=sr, mono=mono)
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
    indices_to_remove = torch.zeros_like(
        logits, dtype=sorted_indices_to_remove.dtype).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    out = logits.clone()
    out[indices_to_remove] = -float('Inf')
    return out


def min_p(logits, p_base=0.01, filter_value=-float('Inf')):
    assert 0 < p_base < 1.0, f"{p_base} in min_p should be float between 0 and 1.0"
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = p_base * top_probs
    indices_to_remove = probs < scaled_min_p
    sorted_indices = torch.argsort(logits, descending=True, dim=-1)
    sorted_indices_to_remove = torch.gather(indices_to_remove, dim=-1, index=sorted_indices)
    # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # sorted_indices_to_remove[..., 0] = 0
    # indices_to_remove = torch.zeros_like(
    #     logits,
    #     dtype=sorted_indices_to_remove.dtype).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    # out = logits.clone()
    # out[indices_to_remove] = filter_value
    # return out
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


class SamplingScheduler:
    def __init__(self, schedule):
        self.schedule = schedule
        self.previous_temp = None

    def get_schedule(self, idx):
        # reverse schedule. choose index higher than start_idx
        for s in reversed(self.schedule):
            if idx >= s['start_idx'] and self.schedule:
                break

        top_p, target_top_k, target_temp = s['top_p'], s['target_top_k'], s['target_temp']
        return top_p, target_top_k, target_temp

    @classmethod
    def default_schedule(cls, top_p=0.8, target_temp=1, target_top_k=100):
        # creative for first 10 tokens. conservative for the rest of sequence
        s1 = { 'start_idx': 0, 'top_p': 0.98, 'target_top_k': None, 'target_temp': target_temp }
        s2 = { 'start_idx': 50, 'top_p': top_p, 'target_top_k': None, 'target_temp': target_temp }
        s3 = { 'start_idx': 100, 'top_p': top_p, 'target_top_k': target_top_k, 'target_temp': target_temp }
        return SamplingScheduler([s1, s2, s3])

def adaptive_sampling(idx, logits, sampling_schedule: SamplingScheduler, exclude_ids=None):
    def get_cum_probs(logits, temp):
        predict_logits = logits / temp.reshape(-1, 1, 1)

        if exclude_ids is not None:
            for i in exclude_ids:
                predict_logits[..., i] = float("-inf")
        probs = F.softmax(predict_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        return cumulative_probs

    top_p, target_top_k, target_temp = sampling_schedule.get_schedule(idx)
    # convert global temperature to temp per batch
    batch_size = logits.shape[0]
    target_temp = torch.full((batch_size,1), fill_value=target_temp, dtype=logits.dtype, device=logits.device)
    if sampling_schedule.previous_temp is None:
        sampling_schedule.previous_temp = target_temp
    adaptive_temp = sampling_schedule.previous_temp.clone()
    # get probs
    cumulative_probs = get_cum_probs(logits, adaptive_temp)
    token_count = (cumulative_probs <= top_p).sum(-1)

    # increase temperature until # of tokens under top_p > target_top_k
    if target_top_k is not None:
        # decrease temperature if token threshold reached.
        decrease_mask = (token_count > target_top_k) & (adaptive_temp > target_temp)
        if decrease_mask.any():
            print("decrease temp for ", decrease_mask)
        adaptive_temp[decrease_mask] += -0.1
        adaptive_temp = torch.max(adaptive_temp, target_temp)
        # increase temperature if token threshold not reached.
        increase_mask = (token_count < target_top_k) & (adaptive_temp < 8.0)
        while increase_mask.any():
            print("increase temp for ", increase_mask)
            adaptive_temp[increase_mask] += 0.2
            cumulative_probs = get_cum_probs(logits, adaptive_temp)
            token_count = (cumulative_probs <= top_p).sum(-1)
            increase_mask = (token_count < target_top_k) & (adaptive_temp < 4.0)

    adaptive_temp = sampling_schedule.previous_temp * 0.5 + adaptive_temp * 0.5
    sampling_schedule.previous_temp = adaptive_temp

    predict_logits = logits / (adaptive_temp.reshape(-1, 1, 1))
    if exclude_ids is not None:
        for i in exclude_ids:
            predict_logits[..., i] = float("-inf")
    predict_logits = top_p_logits(predict_logits, top_p)
    probs = predict_logits.softmax(dim=-1)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    samples = dist.sample()

    return samples

import os
import random

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from pydub import AudioSegment
from pytorch_lightning.profilers import PassThroughProfiler
from scipy.io.wavfile import write
from tqdm import tqdm

from samantha.utils.hparams import DotDict

from ...requires.mulan.mulan_infer import mulan_inference, mulan_rvq_indexs
from ...utils.hdfs_tools import hdfs_cp, hdfs_mkdir


def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


def load_wav(path):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=24000)
    else:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(24000)
        wav = np.asarray(audio.get_array_of_samples())
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav


def log(t, eps=1e-20):
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


def set_seed(seed=1996):
    # reproduction setting
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    return


def is_audio(fname):
    fname = fname.split(".")[-1]
    if fname in ["wav", "mp3", "mp4", "m4a", "npy"]:
        return True
    return False


class CoarseInfer(pl.LightningModule):
    def __init__(self, output_path, extra_params, required_modules, cache_dir=".tmp"):
        super().__init__()
        self.save_hyperparameters()

        self.requires = {}
        self.extra_params = DotDict(extra_params)

        self.output_path = output_path
        self.cache_path = f"{cache_dir}/{os.path.basename(self.output_path)}"
        hdfs_mkdir(self.output_path)
        hdfs_mkdir(self.cache_path)

        self.num_res = self.extra_params.num_res
        self.num_coarse = self.extra_params.num_coarse
        self.num_fine = self.extra_params.num_res - self.num_coarse
        self.frequency = self.extra_params.sample_rate // self.extra_params.hop_size

    def setup(self, stage: str) -> None:
        self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def predict_step(self, batch, batch_idx):  # batch: list of path
        set_seed(seed=1996)

        device = "cuda:{}".format(self.local_rank)
        mulan_embeds = []
        keys = []
        prefixs = []
        wavs = []

        # #Mulan Infer from audio
        for j, l in enumerate(batch):
            wav = load_wav(l)
            if wav.shape[-1] < 24000 * 10:
                wav = np.pad(wav, (0, 24000 * 10 - wav.shape[-1]))
            wav_len = wav.shape[-1]
            beg = max(0, wav_len // 2 - 24000 * 5)
            wav = wav[beg : beg + 24000 * 10]
            save_wav(
                wav,
                os.path.join(
                    self.cache_path, "{}_prompt.wav".format(os.path.basename(l))
                ),
                sr=self.extra_params.sample_rate,
            )
            hdfs_cp(
                os.path.join(
                    self.cache_path, "{}_prompt.wav".format(os.path.basename(l))
                ),
                os.path.join(
                    self.output_path, "{}_prompt.wav".format(os.path.basename(l))
                ),
                True,
            )
            wav = torch.from_numpy(wav).float().unsqueeze(0).to(device)
            wavs.append(wav)
            keys.append(os.path.basename(l))
            prefixs.append("inner_test_300")
        wavs = torch.cat(wavs, dim=0)
        mulan_embeds = mulan_inference(
            self.requires["mulan"], music=wavs, device=device
        )

        # coarse model inference
        b = mulan_embeds.size(0)
        eos_ids = (
            torch.zeros([b, 1], dtype=torch.long, device=device)
            + 1024
            + self.num_coarse * 1024
        )  # offset: w2v-bert + coarse
        mulan_tokens, ds = mulan_rvq_indexs(
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        mulan_tokens = (
            mulan_tokens + 1024 + self.num_coarse * 1024 + 2
        )  # offset: w2v-bert + coarse + EOS 2
        input_tokens = torch.cat([mulan_tokens, eos_ids + 1], dim=1)

        coarse_samples = []
        past_key_values = None
        for i in tqdm(
            range(10 * self.num_coarse * self.frequency), desc="infer_coarse"
        ):  # 30
            coarse_outputs = self.requires["coarse"].ar(
                input_tokens, past_key_values=past_key_values, use_cache=True
            )
            logits = coarse_outputs["logits"]  # [b, t, d]
            # choose the correct logits
            layer_idx = i % self.num_coarse
            predict_logits = logits[
                :, -1, 1024 + layer_idx * 1024 : 1024 + (layer_idx + 1) * 1024
            ]  # [b,d] # offset: w2v-bert

            # naive sampling
            predict_logits = predict_logits / (1.0)  # * temperature
            probs = predict_logits.softmax(dim=1)  # [b, d]

            dist = torch.distributions.categorical.Categorical(probs=probs)
            samples = dist.sample().unsqueeze(1).to(device)

            # gumbel sampling
            # predict_logits = top_k(predict_logits, thres=0.9)
            # samples = gumbel_sample(predict_logits, 1.0).unsqueeze(dim=1)

            samples = samples + 1024 + layer_idx * 1024  # [b, 1], # offset: w2v-bert
            past_key_values = coarse_outputs["past_key_values"]
            input_tokens = samples
            coarse_samples.append(samples)  # [b, 1]
        coarse_samples = torch.cat(coarse_samples, dim=1)  # [b, k*t]
        coarse_samples = coarse_samples - 1024  # remove offset: w2v-bert

        # # fine model inference
        slice_range = []
        beg = 0
        while True:
            end = (
                beg + 10 * self.frequency * self.num_coarse
            )  # coarse prompt length for fine model
            if end >= 10 * self.frequency * self.num_coarse:  # 30
                end = 10 * self.frequency * self.num_coarse  # 30
                beg = end - 10 * self.frequency * self.num_coarse
                slice_range.append([beg, end])
                break
            else:
                slice_range.append([beg, end])
            beg += (
                10 * self.frequency * self.num_coarse
                - 5 * self.frequency * self.num_coarse
            )
        fine_samples = None
        prev_coarse_end = 0
        for coarse_beg, coarse_end in slice_range:
            # 定义续写长度
            cache_len = (
                (prev_coarse_end - coarse_beg) // self.num_coarse * self.num_fine
            )
            prev_coarse_end = coarse_end
            b = coarse_samples.size(0)
            coarse_slice = coarse_samples[:, coarse_beg:coarse_end]  # [b, coarse*t]
            eos_ids = (
                torch.zeros(size=[b, 1], dtype=coarse_slice.dtype, device=device)
                + self.num_res * 1024
            )  # [b, 1]
            if cache_len == 0:
                input_ids = torch.cat([coarse_slice, eos_ids], dim=1)
            else:
                prefix_fine_samples = fine_samples[
                    :,
                    coarse_beg
                    // self.num_coarse
                    * self.num_fine : coarse_beg
                    // self.num_coarse
                    * self.num_fine
                    + cache_len,
                ]
                input_ids = torch.cat(
                    [coarse_slice, eos_ids, prefix_fine_samples], dim=1
                )
            past_key_values = None
            for i in tqdm(
                range(10 * self.num_fine * self.frequency - cache_len),
                desc="fine_infer_slice_{}_{}".format(coarse_beg, coarse_end),
            ):
                fine_outputs = self.requires["fine"].ar(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
                logits = fine_outputs["logits"]  # [b, t, d]
                # choose the correct logits
                layer_idx = i % self.num_fine + self.num_coarse
                predict_logits = logits[
                    :, -1, layer_idx * 1024 : (layer_idx + 1) * 1024
                ]  # [b, d]
                predict_logits = predict_logits / 0.4  # * temperature
                probs = predict_logits.softmax(dim=1)  # [b, d]
                dist = torch.distributions.categorical.Categorical(probs=probs)
                samples = (
                    dist.sample().unsqueeze(1).to(device) + layer_idx * 1024
                )  # [b, 1], add offset
                # input_ids = torch.cat([input_ids, samples], dim=1) # [b, t+1]
                past_key_values = fine_outputs["past_key_values"]
                input_ids = samples
                if fine_samples is None:
                    fine_samples = samples
                else:
                    fine_samples = torch.cat([fine_samples, samples], dim=1)

        # vqgan inference
        coarse_samples = coarse_samples.view([b, -1, self.num_coarse])
        fine_samples = fine_samples.view([b, -1, self.num_fine])
        vqgan_inputs = (
            torch.cat([coarse_samples, fine_samples], dim=2)
            - torch.arange(self.num_res).to(device) * 1024
        )  # [b, t, n_codebook]
        vqgan_inputs = vqgan_inputs.transpose(
            1, 2
        )  # [b, t, n_codebook] -> [b, n_codebook, t]
        wavs = self.requires["ss_dec"](vqgan_inputs)
        wavs = wavs.cpu().squeeze(1).numpy()  # [b, 1, t] -> [b, t]
        for j, wav in enumerate(wavs):
            save_wav(
                wav,
                os.path.join(self.cache_path, "{}_generate.wav".format(keys[j])),
                sr=self.extra_params.sample_rate,
            )
            hdfs_cp(
                os.path.join(self.cache_path, "{}_generate.wav".format(keys[j])),
                os.path.join(self.output_path, "{}_generate.wav".format(keys[j])),
                True,
            )

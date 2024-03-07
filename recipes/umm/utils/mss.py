import os
import subprocess

import torch
from filelock import FileLock
from torch import nn
from torchaudio.transforms import Resample

from recipes.umm.utils.mss_model import BSTransformer, MssModel
import numpy as np
import torch.nn.functional as F


def noramlize_audio_to_f32(tensor):
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.dtype == torch.float64:
        return tensor.float()
    info = torch.iinfo(tensor.dtype)
    abs_max = 2 ** (info.bits - 1)
    return tensor.float() / abs_max


def load_ema_checkpoint(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # divide param group
    no_decay = ["bn", "bias", "norm.bias", "norm.weight", "rotary"]
    base_params = {}
    no_decay_params = {}
    for name, param in model.named_parameters():
        _found = False
        for k in no_decay:
            if k in name:
                no_decay_params[name] = param
                _found = True
                break
        if not _found:
            base_params[name] = param
    # combine the two dictionaries into one
    new_state_dict = {}
    new_keys = []
    for k, v in base_params.items():
        new_state_dict[k] = v
        new_keys.append(k)
    for k, v in no_decay_params.items():
        new_state_dict[k] = v
        new_keys.append(k)
    for idx, k in enumerate(new_keys):
        # assert new_state_dict[k].shape == ckpt["optimizer_states"][0]["ema"][idx].shape
        if new_state_dict[k].shape != ckpt["optimizer_states"][0]["ema"][idx].shape:
            print(f'{k=}, {new_state_dict[k].shape=}, {ckpt["optimizer_states"][0]["ema"][idx].shape=}')
        new_state_dict[k] = ckpt["optimizer_states"][0]["ema"][idx]
    model.load_state_dict(new_state_dict)
    return model


class MSSPredictor(nn.Module):
    def __init__(self, ckpt_path=None, sr=24000, cache_dir='.module_cache', *args, **kwargs):
        """
        clone git repo `lab_audio/sami_ai_models` and it's git submodule `lab_audio/torch-museval` to ~/sami_ai_models,
        then update ray runtime environment yaml file as bellow:

        ```yaml
        pip:
            - hyperpyyaml
            - torchaudio==2.1.0
            - rotary_embedding_torch
        py_modules:
            - ~/sami_ai_models/recipes
            - ~/sami_ai_models/sami_ai
            - ~/sami_ai_models/recipes/mss/torch-museval/torch_museval
        ```

        """

        super().__init__(*args, **kwargs)
        if not ckpt_path:
            ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/sunyakun.king/sami_models/mss-checkpoint-epoch=235-val_median_sdr_0=12.49.ckpt"

        model_dir = cache_dir
        model_file_name = 'mss.model'
        model_file_path = os.path.join(model_dir, model_file_name)
        os.makedirs(model_dir, exist_ok=True)
        with FileLock(os.path.join(model_dir, '.lock')):
            # download mss model
            os.makedirs(model_dir, exist_ok=True)
            finish_flag = os.path.join(model_dir, '.finish')
            if not os.path.exists(finish_flag):
                if os.path.exists(model_file_path):
                    os.remove(model_file_path)
                subp = subprocess.Popen([
                    'hdfs', 'dfs', '-get', ckpt_path, model_file_path
                ])
                subp.wait()
                if subp.poll() != 0:
                    raise Exception("download model fail, returncode=%d" % subp.returncode)
                with open(finish_flag, 'w'):
                    pass
                print(f"| save model to {model_file_path}")

        model = MssModel(
            input_names=["waveform"],
            output_names=["waveform"],
            stages=[
                BSTransformer(
                    takes=["waveform"],
                    provides=["waveform"],
                    input_channels=2,
                    output_channels=2,
                    target_sources_num=1,
                    use_flash_attn=True,
                    enforce_dropout=0,  # newly added
                    mel_bands=0
                )
            ]
        )
        for k, v in model.stages[0].stft_funcs.items():
            model.stages[0].stft_funcs[k] = v

        model.stages[0] = load_ema_checkpoint(model_file_path, model.stages[0])
        self.model = model
        if sr != 44100:
            self.upsample = Resample(sr, 44100)
            self.downsample = Resample(44100, sr)
        self.segment_samples = 44100 * 8
        self.hop_samples = 44100 * int(round(8 * 0.5))
        self.sr = sr

    def forward(self, wav):
        if len(wav.shape) == 2:
            wav = wav[:, None]
        sr = self.sr
        self.model.eval()
        # repeat to 2 channels
        with torch.no_grad():
            device = next(self.model.parameters()).device
            for k, v in self.model.stages[0].stft_funcs.items():
                self.model.stages[0].stft_funcs[k] = v.to(device)
            wav = wav.to(device)
            wav = torch.repeat_interleave(wav, 2, 1)
            if sr != 44100:
                wav = self.upsample(wav).clamp(-1, 1)
            B = wav.shape[0]
            wav_pad = self.pad_audio(wav)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                (voc_audio,) = self.model(wav_pad)
            voc_audio = self.unpad_audio(voc_audio, B)
            voc_audio = voc_audio[..., :wav.shape[-1]].clamp(-1, 1)
            acc_audio = wav - voc_audio
            acc_audio = acc_audio.clamp(-1, 1)
            if sr != 44100:
                voc_audio = self.downsample(voc_audio).mean(1)
                acc_audio = self.downsample(acc_audio).mean(1)
        return voc_audio, acc_audio

    def unpad_audio(self, audio, B):
        pad_len = (self.segment_samples - self.hop_samples) // 2
        B_nS, c, L_seg = audio.shape
        nS = B_nS // B
        segments = audio.reshape(B, nS, c, L_seg)
        unpadded_audio = torch.zeros(B, c, (nS - 1) * self.hop_samples + self.segment_samples).to(audio.device)
        cnt = torch.zeros_like(unpadded_audio)
        for i in range(nS):
            start = i * self.hop_samples
            end = start + L_seg
            # Accumulate segments with overlap
            unpadded_audio[:, :, start:end] += segments[:, i, :, :]
            cnt[..., start: end] += 1
        unpadded_audio = unpadded_audio / cnt
        unpadded_audio = unpadded_audio[..., pad_len:-pad_len]
        return unpadded_audio

    def pad_audio(self, audio):
        r"""Pad the audio with zero in the end so that the length of audio can
        be evenly divided by segment_samples.

        Args:
            audio: (channels_num, audio_samples)

        Returns:
            padded_audio: (channels_num, audio_samples)
        """
        B, channels_num, audio_samples = audio.shape

        # Number of segments
        segments_num = int(np.ceil(audio_samples / self.hop_samples))

        pad_samples = segments_num * self.hop_samples - audio_samples
        padded_audio = F.pad(audio, [0, pad_samples], "constant")

        # Pad additional zeros as buffer at both side
        pad_len = (self.segment_samples - self.hop_samples) // 2
        if pad_len > 0:
            padded_audio = F.pad(padded_audio, [pad_len, pad_len], "constant")
        padded_audio = padded_audio.unfold(-1, self.segment_samples, self.hop_samples)
        padded_audio = padded_audio.permute(0, 2, 1, 3).flatten(0, 1)

        return padded_audio

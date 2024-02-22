import io
import os
import subprocess
import sys
import threading
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from filelock import FileLock
from torchaudio.transforms import Resample


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


class Separator:
    def __init__(
            self,
            model: nn.Module,
            segment_samples: int,
            hop_samples: int,
            batch_size: int,
            device: str,
    ):
        r"""Separate to separate an audio clip into a target source.

        Args:
            model: nn.Module, trained model
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            hop_samples: int, length of hop size, e.g., 44100*10
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
        """
        self.model = model
        self.segment_samples = segment_samples
        self.hop_samples = hop_samples
        self.batch_size = batch_size
        self.device = device

    def separate(self, input_dict: Dict) -> np.array:
        r"""Separate an audio clip into a target source.

        Args:
            input_dict: dict, e.g., {
                waveform: (channels_num, audio_samples),
                ...,
            }

        Returns:
            sep_audio: (channels_num, audio_samples) | (target_sources_num, channels_num, audio_samples)
        """
        audio = input_dict["waveform"]

        audio_samples = audio.shape[-1]

        # Pad the audio with zero in the end so that the length of audio can be
        # evenly divided by segment_samples.
        padded_audio = self.pad_audio(audio)

        # Enframe long audio into segments.
        segments = self.enframe(padded_audio)
        # (segments_num, channels_num, segment_samples)

        segments_input_dict = {"waveform": segments}

        # Separate in mini-batches.
        sep_segments = self._forward_in_mini_batches(
            self.model, segments_input_dict, self.batch_size
        )["waveform"]
        # (segments_num, channels_num, segment_samples)

        # Deframe segments into long audio.
        sep_audio = self.deframe(sep_segments)
        # (channels_num, padded_audio_samples)

        sep_audio = sep_audio[:, 0:audio_samples]
        # (channels_num, audio_samples)

        return sep_audio

    def pad_audio(self, audio: np.array) -> np.array:
        r"""Pad the audio with zero in the end so that the length of audio can
        be evenly divided by segment_samples.

        Args:
            audio: (channels_num, audio_samples)

        Returns:
            padded_audio: (channels_num, audio_samples)
        """
        channels_num, audio_samples = audio.shape

        # Number of segments
        segments_num = int(np.ceil(audio_samples / self.hop_samples))

        pad_samples = segments_num * self.hop_samples - audio_samples
        padded_audio = np.concatenate(
            (audio, np.zeros((channels_num, pad_samples))), axis=1
        )
        # (channels_num, padded_audio_samples)

        # Pad additional zeros as buffer at both side
        pad_len = (self.segment_samples - self.hop_samples) // 2
        if pad_len > 0:
            padded_audio = np.pad(padded_audio, ((0, 0), (pad_len, pad_len)), "constant")

        return padded_audio

    def enframe(self, audio: np.array) -> np.array:
        r"""Enframe long audio into segments.

        Args:
            audio: (channels_num, audio_samples)
            segment_samples: int
            hop_samples: int

        Returns:
            segments: (segments_num, channels_num, segment_samples)
        """
        audio_samples = audio.shape[1]
        segments = []

        pad_len = (self.segment_samples - self.hop_samples) // 2
        pointer = pad_len

        while pointer - pad_len + self.segment_samples <= audio_samples:
            start = pointer - pad_len
            end = start + self.segment_samples
            segments.append(audio[:, start: end])
            pointer += self.hop_samples

        segments = np.array(segments)
        return segments

    def deframe(self, segments: np.array) -> np.array:
        r"""Deframe segments into long audio.

        Args:
            segments: (segments_num, channels_num, segment_samples)

        Returns:
            output: (channels_num, audio_samples)
        """
        (segments_num, n_channel, segment_samples) = segments.shape

        if segments_num == 1:
            return segments[0]

        # (N, n_channel, segment_samples, dim) = x.shape
        output = np.zeros(
            shape=(
                n_channel,
                (segments_num - 1) * self.hop_samples + self.segment_samples,
            )
        )
        cnt = np.zeros_like(output)

        for i in range(segments_num):
            start = i * self.hop_samples
            end = start + self.segment_samples

            output[:, start: end] += segments[i]
            cnt[:, start: end] += 1

        # # Take the average
        output /= cnt

        # Remove front and end padding
        assert (
                       self.segment_samples - self.hop_samples
               ) % 2 == 0, "hop_size must be even percentage of segment_samples"
        pad_len = int(self.segment_samples - self.hop_samples) // 2
        if pad_len > 0:
            output = output[:, pad_len:-pad_len]

        return output

    def _is_integer(self, x: float) -> bool:
        if x - int(x) < 1e-10:
            return True
        else:
            return False

    def _forward_in_mini_batches(
            self, model: nn.Module, segments_input_dict: Dict, batch_size: int
    ) -> Dict:
        r"""Forward data to model in mini-batch.

        Args:
            model: nn.Module
            segments_input_dict: dict, e.g., {
                'waveform': (segments_num, channels_num, segment_samples),
                ...,
            }
            batch_size: int

        Returns:
            output_dict: dict, e.g. {
                'waveform': (segments_num, channels_num, segment_samples),
            }
        """
        output_dict = {}

        pointer = 0
        segments_num = len(segments_input_dict["waveform"])

        while True:
            if pointer >= segments_num:
                break

            batch_input_dict = {}

            for key in segments_input_dict.keys():
                batch_input_dict[key] = torch.Tensor(
                    segments_input_dict[key][pointer: pointer + batch_size]
                ).to(self.device)

            pointer += batch_size
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    model.eval()
                    (output,) = model(batch_input_dict["waveform"])

            batch_output_dict = {"waveform": output}
            for key in batch_output_dict.keys():
                self._append_to_dict(
                    output_dict, key, batch_output_dict[key].data.cpu().numpy()
                )

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict

    def _append_to_dict(self, dict, key, value):
        if key in dict.keys():
            dict[key].append(value)
        else:
            dict[key] = [value]


class Predictor:

    def __init__(self, device, ckpt_path=None):
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

        if not ckpt_path:
            ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/sunyakun.king/sami_models/mss-checkpoint-epoch=235-val_median_sdr_0=12.49.ckpt"

        model_dir = "/opt/tiger/sami_ai_models/models"
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
                print("| save model to %s", model_file_path)

        print(f"{device=}, {ckpt_path=}, {model_file_path=}")

        sys.path.append('./sami_ai_models')
        from sami_ai_models.recipes.mss.models.BSTransformer import \
            BSTransformer
        from sami_ai_models.recipes.mss.models.model import MssModel

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
            model.stages[0].stft_funcs[k] = v.to(device)

        model.stages[0] = load_ema_checkpoint(model_file_path, model.stages[0])
        model.to(device)
        self.separator = Separator(
            model=model,
            segment_samples=44100 * 8,
            hop_samples=44100 * int(round(8 * 0.5)),
            batch_size=8,
            device=device,
        )
        self.lock = threading.Lock()

    def predict_one(self, wav: bytes):
        t, sample_rate = torchaudio.load(io.BytesIO(wav), normalize=True)
        # repeat to 2 channels
        t = torch.repeat_interleave(t, 2, 0)
        if sample_rate != 44100:
            upsample = Resample(sample_rate, 44100)
            audio = upsample(t).numpy()
        else:
            audio = t.numpy()

        # separate
        len_min = audio.shape[1] / (44100 * 60)
        if len_min > 10:
            len_slice = 44100 * 60 * 5
            sep_audios, acc_audios = [], []
            for _, i in enumerate(range(0, audio.shape[1], len_slice)):
                input_dict = {"waveform": audio[:, i:i + len_slice]}
                with self.lock:
                    sep_audio = self.separator.separate(input_dict)
                acc_audio = np.clip(audio[:, i:i + len_slice] - np.clip(sep_audio, -1.0, 1.0), -1.0, 1.0)
                sep_audios.append(sep_audio)
                acc_audios.append(acc_audio)
            sep_audio = np.concatenate(sep_audios, axis=1)
            acc_audio = np.concatenate(acc_audios, axis=1)
        else:
            input_dict = {"waveform": audio}
            with self.lock:
                sep_audio = self.separator.separate(input_dict)
            acc_audio = np.clip(audio - np.clip(sep_audio, -1.0, 1.0), -1.0, 1.0)
        # avg 2 channels
        sep_audio = sep_audio.mean(axis=0, keepdims=True)
        acc_audio = acc_audio.mean(axis=0, keepdims=True)

        sep_audio = torch.tensor(sep_audio).float()
        acc_audio = torch.tensor(acc_audio).float()

        # downsample
        if sample_rate != 44100:
            downsample = Resample(44100, sample_rate)
            sep_audio = downsample(torch.tensor(sep_audio).float())
            acc_audio = downsample(torch.tensor(acc_audio).float())

        sep_wav = io.BytesIO()
        torchaudio.save(
            sep_wav,
            sep_audio,
            sample_rate,
            format='wav',
            encoding="PCM_S",
            bits_per_sample=16)

        acc_wav = io.BytesIO()
        torchaudio.save(
            acc_wav,
            acc_audio,
            sample_rate,
            format='wav',
            encoding="PCM_S",
            bits_per_sample=16)

        return sep_wav.getvalue(), acc_wav.getvalue()

    def predict(self, batch: List[bytes]):
        result = []
        for item in batch:
            result.append(self.predict_one(item))
        return result


import soundfile as sf


def preprocess_to_bytes(input_audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert a numpy float array to a byte stream in wav format.

    :param input_audio: Numpy float array of audio samples.
    :param sample_rate: Sample rate of the audio.
    :return: Byte stream of audio in wav format.
    """
    buffer = io.BytesIO()
    # Write the numpy array to the buffer as a WAV file.
    sf.write(buffer, input_audio, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)  # Rewind the buffer to the beginning.
    return buffer.getvalue()


def postprocess_to_np(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    """
    Convert audio byte stream back to a numpy float array.

    :param audio_bytes: Byte stream of audio in wav format.
    :param sample_rate: Expected sample rate for the output audio.
    :return: Numpy float array of audio samples.
    """
    buffer = io.BytesIO(audio_bytes)
    data, sr = sf.read(buffer, dtype='float32')
    if sr != sample_rate:
        # Optionally, resample if the output sample rate is different from the expected one.
        # This requires additional implementation not shown here.
        pass
    return data


if __name__ == '__main__':
    predictor = Predictor('cuda:0')
    predictor.predict_one(open('tmp/test.wav', 'rb').read())

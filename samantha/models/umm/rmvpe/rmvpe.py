import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from samantha.models.umm.rmvpe.constants import *
from samantha.models.umm.rmvpe.model import E2E, E2E0
from samantha.models.umm.rmvpe.spec import MelSpectrogram


def to_local_average_cents(salience, center=None, thred=0.03):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, "cents_mapping"):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = 20 * np.arange(N_CLASS) + CONST

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array(
            [
                to_local_average_cents(salience[i, :], None, thred)
                for i in range(salience.shape[0])
            ]
        )

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, "transition"):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    # Convert to probability
    prob = salience.T
    prob = prob / prob.sum(axis=0)

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)

    return np.array(
        [
            to_local_average_cents(salience[i, :], path[i], thred)
            for i in range(len(path))
        ]
    )


def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]  # [B=1, T=1, N]
    idx_cents = idx * 20 + CONST  # [B=1, N]
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - 4, min=0)  # [B, T, 1]
    end = torch.clip(center + 5, max=N_CLASS)  # [B, T, 1]
    idx_mask = (idx >= start) & (idx < end)  # [B, T, N]
    weights = hidden * idx_mask  # [B, T, N]
    product_sum = torch.sum(weights * idx_cents, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    cents = product_sum / (
        weight_sum + (weight_sum == 0)
    )  # avoid dividing by zero, [B, T]
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred  # [B, T]
    f0 = f0 * ~uv
    return f0


def to_viterbi_f0(hidden, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, "transition"):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    # Convert to probability
    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)

    return to_local_average_f0(hidden, center=center, thred=thred)


class MelExtractor(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(self, hop_length=160):
        self.resample_kernel = {}
        model = E2E0(4, 1, (2, 2))
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        )
        self.resample_kernel = {}

    def load_and_eval(self, state_dict):
        try:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Load RMVPE successed!")
        except Exception as e:
            print(e)

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def infer_from_audio(
        self, audio, sample_rate=16000, device=None, thred=0.03, use_viterbi=False
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, 16000, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(device)
            audio_res = self.resample_kernel[key_str](audio)

        mel_extractor = self.mel_extractor.to(device)
        self.model = self.model.to(device)
        mel = mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0

    @torch.no_grad()
    def batch_infer(self, audios, sample_rate, thred=0.03, use_viterbi=False):
        if sample_rate == 16000:
            audio_res = audios
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, 16000, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(
                audios.device
            )
            audio_res = self.resample_kernel[key_str](audios)
        mel_extractor = self.mel_extractor.to(audios.device)
        self.model = self.model.to(audios.device)
        mel = mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0

import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from samantha.dataio.lite.utils.mel import spectral_normalize_torch


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    name = "sr_{}_nfft_{}_win_{}_hop_{}_fmin_{}_fmax_{}_device_{}".format(
        sampling_rate, n_fft, win_size, hop_size, fmin, fmax, dtype_device
    )
    if name not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[name] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if name not in hann_window:
        hann_window[name] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    # torch 1.8 compatable. stft could support half type
    old_type = y.dtype
    y = y.float()
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[name],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = spec.to(old_type)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[name], spec)
    spec = spectral_normalize_torch(spec)
    return spec


class Wave:
    def __init__(
        self, encoder, decoder, version=3.1, hop_size=300, win_size=1200
    ) -> None:
        self.version = version
        self.encoder = encoder
        self.decoder = decoder
        self.hop_size = hop_size
        self.win_size = win_size

    def encode(self, wav):
        if (
            self.version == 3.1 or self.version == 2.1
        ):  # 3.1 和 2.1 的输入是 [wav, spec, mel_spec]
            wav = wav.unsqueeze(1).float()
            wav = F.pad(
                wav,
                (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0),
                value=0.0,
            )
            spec = spectrogram_torch(
                wav.squeeze(1), 2048, 24000, self.hop_size, self.win_size
            )
            mel_spec = mel_spectrogram_torch(
                y=wav.squeeze(1),
                n_fft=2048,
                num_mels=80,
                sampling_rate=24000,
                hop_size=self.hop_size,
                win_size=self.win_size,
                fmin=0.0,
                fmax=None,
            )
            _, m, logs = self.encoder(wav, spec, mel_spec)  # for wvae v3.1
            m = m.transpose(2, 1)
            logs = logs.transpose(2, 1)
            bn = torch.cat([m, logs], -1)
        elif self.version == 2.0:  # 2.0 的输入是 [wav, spec]
            wav = wav.unsqueeze(1).float()
            wav = F.pad(
                wav,
                (0, (wav.size(-1) // 600 + 1) * 600 - wav.size(-1), 0, 0, 0, 0),
                value=0.0,
            )
            spec = spectrogram_torch(
                wav.squeeze(1), 2048, 24000, self.hop_size, self.win_size
            )
            _, m, logs = self.encoder(wav, spec)
            m = m.transpose(2, 1)
            logs = logs.transpose(2, 1)
            bn = torch.cat([m, logs], -1)
        else:
            raise NotImplementedError
        return bn

    def decode(self, bn):
        m, logs = torch.split(bn, bn.shape[-1] // 2, dim=-1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        wav = self.decoder(z.transpose(1, 2))[0, 0]
        return wav

    def decoder_from_z(self, z):
        wav = self.decoder(z.transpose(1, 2))[0, 0]
        return wav

    def reconstruct(self, wav):
        bn = self.encode(wav)
        reconstruct_wav = self.decode(bn)
        return reconstruct_wav

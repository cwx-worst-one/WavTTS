import numpy as np
import scipy
import torch
from torch import nn, view_as_real, view_as_complex


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            if self.padding == "center":
                # Fallback to pytorch native implementation
                return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
            elif self.padding == "same":
                pad_left = (self.win_length - self.hop_length) // 2
                pad_right = (self.win_length - self.hop_length + 1) // 2
            else:
                raise ValueError("Padding must be 'center' or 'same'.")

            assert spec.dim() == 3, "Expected a 3D tensor as input"
            B, N, T = spec.shape

            # Inverse FFT
            ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
            ifft = ifft * self.window[None, :, None]

            # Overlap and Add
            output_size = (T - 1) * self.hop_length + self.win_length
            y = torch.nn.functional.fold(
                ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
            )[:, 0, 0, pad_left:-pad_right]

            # Window envelope
            window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
            window_envelope = torch.nn.functional.fold(
                window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
            ).squeeze()[pad_left:-pad_right]

            # Normalize
            assert (window_envelope > 1e-11).all()
            y = y / window_envelope

        return y

class AMP_PHA_Spectrum(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, audio_channels=1, normalize_spec=False, center=True, atan2_magnitude_threshold_ratio=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.audio_channels = audio_channels
        self.normalize_spec=normalize_spec
        self.center = center
        self.atan2_magnitude_threshold_ratio = atan2_magnitude_threshold_ratio

    def forward(self, y):
        assert len(y.shape) == 3 and y.shape[1] == self.audio_channels, f"Invalid number of channels {y.shape}"
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            B, C, L = y.shape
            y = y.float()
            y = y.reshape(B * C, L) # flatten stereo channels
            specs = amp_pha_spectrum(y, self.n_fft, self.hop_length, self.win_length, normalize_spec=self.normalize_spec, center=self.center, atan2_magnitude_threshold_ratio=self.atan2_magnitude_threshold_ratio)
            if self.center:
                N, T = self.n_fft//2+1, L // self.hop_length + 1 # N frequencies, T frames
            else: # self pad does not have extra frame
                N, T = self.n_fft//2+1, L // self.hop_length # N frequencies, T frames
            assert N == specs[0].shape[-2] and T == specs[0].shape[-1], f"Invalid shape {specs[0].shape}"
            log_amplitude, phase, rea, imag = [s.reshape(B, C, N, T) for s in specs] # restore channels
        return log_amplitude, phase, rea, imag

def amp_pha_spectrum(y, n_fft, hop_length, win_length, normalize_spec=False, center=True, atan2_magnitude_threshold_ratio=0.0):
    hann_window=torch.hann_window(win_length).to(y.device)
    if not center:
        y = torch.nn.functional.pad(
            y,
            (
                (win_length - hop_length) // 2,
                (win_length - hop_length + 1) // 2,
            ),
            mode="reflect",
        )
    stft_spec=torch.stft(y, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window, center=center, return_complex=True) #[batch_size, n_fft//2+1, frames, 2]
    if normalize_spec: # Music2Latent
        stft_spec = normalize_complex(stft_spec)
    rea, imag = torch.real(stft_spec), torch.imag(stft_spec)

    log_amplitude=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]
    if atan2_magnitude_threshold_ratio > 0.0:
        # stabilize the atan2 by setting small values to 0
        magnitude = torch.sqrt(rea**2 + imag**2)
        threshold = torch.max(magnitude) * atan2_magnitude_threshold_ratio
        mask = magnitude < threshold
        rea = torch.where(mask, torch.zeros_like(rea), rea)
        imag = torch.where(mask, torch.zeros_like(imag), imag)
    phase=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, rea, imag

def normalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    return beta_rescale*(x.abs()**alpha_rescale).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def denormalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    x = x/beta_rescale
    return (x.abs()**(1./alpha_rescale)).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

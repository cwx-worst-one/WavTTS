import math

import librosa
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        """Calculate DFT, IDFT, RDFT, IRDFT.
        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        """Calculate DFT of signal.
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        """Calculate IDFT of signal.
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(
            x_imag, self.inv_W_imag
        )
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(
            x_real, self.inv_W_imag
        )

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        """Calculate right DFT of signal.
        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part
        Returns:
          z_real: (n // 2 + 1,), output real part
          z_imag: (n // 2 + 1,), output imag part
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0:n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0:n_rfft])

        if self.norm is None:
            pass
        elif self.norm == "ortho":
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        """Calculate inverse right DFT of signal.
        Args:
          x_real: (n // 2 + 1,), signal real part
          x_imag: (n // 2 + 1,), signal imag part
        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)

        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        x_imag = torch.cat((x_imag, -1.0 * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(
            x_imag, self.inv_W_imag
        )

        if self.norm is None:
            z_real /= self.n
        elif self.norm == "ortho":
            z_real /= math.sqrt(n)

        return z_real


class STFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        """Implementation of STFT with Conv1d. The function has the same output
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = scipy.signal.windows.hann(self.win_length)
        self.window = np.sqrt(self.window)

        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(
            self.window, self.win_length, fftbins=True
        )

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)
        # fft_window = np.sqrt(fft_window)   #sqrthanning window

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0:out_channels] * fft_window[:, None]).T
        )[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)
        Returns:
          real:  (batch_size, time_steps, n_fft // 2 + 1)
          imag:  (batch_size, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, :, :].transpose(1, 2)
        imag = imag[:, :, :].transpose(1, 2)
        # (batch_size, time_steps, n_fft // 2 + 1)

        # real = real[:, None, :, :].transpose(2, 3)
        # imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    mag = (real**2 + imag**2) ** 0.5
    cos = real / torch.clamp(mag, 1e-10, np.inf)
    sin = imag / torch.clamp(mag, 1e-10, np.inf)
    return mag, cos, sin


class OVERLAP_ADD(DFTBase):
    def __init__(
        self, n_fft=2048, hop_length=None, win_length=None, window="hann", center=True
    ):
        """Implementation of overlap and add"""
        super(OVERLAP_ADD, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.overlap_add = nn.ConvTranspose2d(
            in_channels=n_fft,
            out_channels=1,
            kernel_size=(n_fft, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

        self.ifft_window_sum = []
        self.overlap_add.weight.data = torch.Tensor(
            np.eye(self.n_fft)[:, None, :, None]
        )
        """(n_fft, 1, n_fft, 1)"""

    def get_ifft_window(self, n_frames):
        device = next(self.parameters()).device

        ifft_window_sum = librosa.filters.window_sumsquare(
            self.window,
            n_frames,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        ifft_window_sum = np.clip(ifft_window_sum, 1e-8, np.inf)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        return ifft_window_sum

    def forward(self, s_real, length):
        s_real = s_real[..., None]  # (batch_size, n_fft, time_steps, 1)
        y = self.overlap_add(s_real)[:, 0, :, 0]  # (1, samples_num)

        # Divide window
        if len(self.ifft_window_sum) != y.shape[1]:
            frames_num = s_real.shape[2]
            self.ifft_window_sum = self.get_ifft_window(frames_num)

        y = y / self.ifft_window_sum[None, 0 : y.shape[1]]

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat(
                    (y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1
                )
        return y


class IFFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        """Implementation of ISTFT with Conv1d. The function has the same output
        of librosa.core.istft
        """
        super(IFFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.reverse = nn.Conv1d(
            in_channels=n_fft // 2 + 1,
            out_channels=n_fft // 2 - 1,
            kernel_size=1,
            bias=False,
        )

        self.ifft_window_sum = []

        self.init_weights()

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        ifft_window = librosa.filters.get_window(
            self.window, self.win_length, fftbins=True
        )
        """(win_length,)"""

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(ifft_window, self.n_fft)
        # ifft_window = np.sqrt(ifft_window)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1:-1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        """(n_fft // 2 - 1, n_fft // 2 + 1, 1)"""

    def forward(self, real_stft, imag_stft):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          real: (batch_size, data_length)
        """
        # assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        device = next(self.parameters()).device
        batch_size = real_stft.shape[0]

        real_stft = real_stft[:, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft, using flip is not supported by ONNX.
        # full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        # full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, -self.reverse(imag_stft)), dim=1)
        """(batch_size, n_fft, time_steps)"""

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        return s_real


class ISTFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        """Implementation of ISTFT with Conv1d. The function has the same output
        of librosa.core.istft
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = scipy.signal.windows.hann(self.win_length)
        self.window = np.sqrt(self.window)

        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.reverse = nn.Conv1d(
            in_channels=n_fft // 2 + 1,
            out_channels=n_fft // 2 - 1,
            kernel_size=1,
            bias=False,
        )

        self.overlap_add = nn.ConvTranspose2d(
            in_channels=n_fft,
            out_channels=1,
            kernel_size=(n_fft, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

        self.ifft_window_sum = []

        self.init_weights()

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        ifft_window = librosa.filters.get_window(
            self.window, self.win_length, fftbins=True
        )
        """(win_length,)"""

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(ifft_window, self.n_fft)
        # ifft_window = np.sqrt(ifft_window)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1:-1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        """(n_fft // 2 - 1, n_fft // 2 + 1, 1)"""

        self.overlap_add.weight.data = torch.Tensor(
            np.eye(self.n_fft)[:, None, :, None]
        )
        """(n_fft, 1, n_fft, 1)"""

    def get_ifft_window(self, n_frames):
        device = next(self.parameters()).device

        ifft_window_sum = librosa.filters.window_sumsquare(
            self.window,
            n_frames=n_frames,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        ifft_window_sum = np.clip(ifft_window_sum, 1e-8, np.inf)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        return ifft_window_sum

    def forward(self, real_stft, imag_stft, length):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          real: (batch_size, data_length)
        """
        # assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        device = next(self.parameters()).device
        batch_size = real_stft.shape[0]

        real_stft = real_stft.transpose(1, 2)
        imag_stft = imag_stft.transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft, using flip is not supported by ONNX.
        # full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        # full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, -self.reverse(imag_stft)), dim=1)
        """(1, n_fft, time_steps)"""

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        s_real = s_real[..., None]  # (batch_size, n_fft, time_steps, 1)
        y = self.overlap_add(s_real)[:, 0, :, 0]  # (1, samples_num)

        # Divide window
        if len(self.ifft_window_sum) != y.shape[1]:
            frames_num = real_stft.shape[2]
            self.ifft_window_sum = self.get_ifft_window(frames_num)

        y = y / self.ifft_window_sum[None, 0 : y.shape[1]]

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat(
                    (y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1
                )

        return y


class ISTFT2(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        """Implementation of ISTFT with Conv1d. The function has the same output
        of librosa.core.istft
        """
        super(ISTFT2, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = scipy.signal.windows.hann(self.win_length)
        self.window = np.sqrt(self.window)
        self.window = torch.from_numpy(self.window).to(dtype=torch.float32)

        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if self.win_length is None:
            self.win_length = self.n_fft

        # Set the default hop, if it's not already specified
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.conv_imag = nn.Conv1d(
            in_channels=n_fft,
            out_channels=n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.reverse = nn.Conv1d(
            in_channels=n_fft // 2 + 1,
            out_channels=n_fft // 2 - 1,
            kernel_size=1,
            bias=False,
        )

        self.overlap_add = nn.ConvTranspose2d(
            in_channels=n_fft,
            out_channels=1,
            kernel_size=(n_fft, 1),
            stride=(self.hop_length, 1),
            bias=False,
        )

        # self.ifft_window_sum = []
        self.ifft_window_sum = torch.tensor([])
        self.init_weights()

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def get_window(self, window="hann", Nx=0, fftbins=True):
        """
        Return a window of a given length and type.

        Args:
            window (string or tuple): the window type, see below for
                details
            Nx (int): the number of samples in the window. If zero or
                less, an empty array is returned.
            fftbins (bool): if True, create a "periodic" window ready to
                use with ifftshift & co. If False, create a "symmetric"
                window, for use with convolutions, etc.

        Returns:
            window (Tensor): a window of length `Nx` and type `window`
        """

        # Compute FFT window, use symmetric when possible
        if fftbins:
            fft_norm = "backward"
        else:
            fft_norm = "forward"

        # Hamming window
        if window == ("hamming" or "Hamming"):
            w = torch.hamming_window(Nx, periodic=fftbins)

        # Hanning window
        elif window == ("hann" or "Hann"):
            w = torch.hann_window(Nx, periodic=fftbins)

        # Blackman window
        elif window == ("blackman" or "Blackman"):
            w = torch.blackman_window(Nx, periodic=fftbins)

        else:
            raise ValueError(f"Window type {window} is not supported.")

        return w

    def init_weights(self):
        ifft_window = librosa.filters.get_window(
            self.window.numpy(), self.win_length, fftbins=True
        )
        """(win_length,)"""

        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(ifft_window, self.n_fft)
        # ifft_window = np.sqrt(ifft_window)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T
        )[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        tmp = np.zeros((self.n_fft // 2 - 1, self.n_fft // 2 + 1, 1))
        tmp[:, 1:-1, 0] = np.array(np.eye(self.n_fft // 2 - 1)[::-1])
        self.reverse.weight.data = torch.Tensor(tmp)
        """(n_fft // 2 - 1, n_fft // 2 + 1, 1)"""

        self.overlap_add.weight.data = torch.Tensor(
            np.eye(self.n_fft)[:, None, :, None]
        )
        """(n_fft, 1, n_fft, 1)"""

    def get_ifft_window(self, n_frames):
        # device = next(self.parameters()).device
        device = "cpu"
        ifft_window_sum = librosa.filters.window_sumsquare(
            self.window,
            n_frames,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        ifft_window_sum = np.clip(ifft_window_sum, 1e-8, np.inf)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        return ifft_window_sum

    def dev_get_ifft_window(self, n_frames):
        ifft_window_sum = self.window_sumsquare(
            self.window,
            n_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
        )
        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-8, np.inf)
        return ifft_window_sum

    def window_sumsquare(
        self,
        window,
        n_frames,
        hop_length=200,
        win_length=800,
        n_fft=800,
        dtype=torch.float32,
        norm=None,
    ):
        if win_length is None:
            win_length = n_fft

        # Create window tensor and fill with the window function values
        win_tensor = torch.zeros(win_length, dtype=dtype)
        win_vals = torch.from_numpy(window)
        win_tensor[: win_vals.shape[0]] = win_vals

        # Compute squared window and pad to length n_fft
        win_tensor = win_tensor**2
        win_tensor = torch.nn.functional.pad(win_tensor, [0, n_fft - win_length])

        # Calculate overlapping frames and sum of squared values
        n = n_fft + hop_length * (n_frames - 1)
        x = torch.zeros(n, dtype=dtype)
        for i in range(n_frames):
            sample = i * hop_length
            x[sample : (sample + n_fft)] += win_tensor[: min(n_fft, n - sample)]
        return x

    def forward(self, real_stft, imag_stft, length):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          real: (batch_size, data_length)
        """
        # assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        device = real_stft.device
        batch_size = real_stft.shape[0]

        real_stft = real_stft.transpose(1, 2)
        imag_stft = imag_stft.transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft, using flip is not supported by ONNX.
        # full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        # full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_real_stft = torch.cat((real_stft, self.reverse(real_stft)), dim=1)
        full_imag_stft = torch.cat((imag_stft, -self.reverse(imag_stft)), dim=1)
        """(1, n_fft, time_steps)"""

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        s_real = s_real[..., None]  # (batch_size, n_fft, time_steps, 1)
        y = self.overlap_add(s_real)[:, 0, :, 0]  # (1, samples_num)

        # Divide window
        if self.ifft_window_sum.numel() == 0:
            n_frames = real_stft.shape[2]

            if self.win_length is None:
                self.win_length = self.n_fft

            # Create window tensor and fill with the window function values
            win_tensor = torch.zeros(self.win_length, dtype=torch.float32).to(device)
            win_vals = self.window
            win_tensor[: win_vals.shape[0]] = win_vals

            # Compute squared window and pad to length n_fft
            win_tensor = win_tensor**2
            win_tensor = torch.nn.functional.pad(
                win_tensor, [0, self.n_fft - self.win_length]
            )

            # Calculate overlapping frames and sum of squared values
            n = self.n_fft + self.hop_length * (n_frames - 1)
            x = torch.zeros(n, dtype=torch.float32).to(device)
            for i in range(n_frames):
                sample = i * self.hop_length
                x[sample : (sample + self.n_fft)] += win_tensor[
                    : min(self.n_fft, n - sample)
                ]

            self.ifft_window_sum = torch.clamp(x, 1e-8, np.inf)

        y = y / self.ifft_window_sum[None, 0 : y.shape[1]].to(device)

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat(
                    (y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1
                )

        return y


class Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        freeze_parameters=True,
    ):
        """Calculate spectrogram using pytorch. The STFT is implemented with
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def forward(self, input):
        """input: input: (batch_size, data_length)
        Returns:
          spectrogram: (batch_size, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        #  (batch_size, time_steps, n_fft // 2 + 1)

        spectrogram = real**2 + imag**2

        angle = torch.atan2(imag, real)

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram, real, imag, angle


class LogmelFilterBank(nn.Module):
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        n_mels=64,
        fmin=0.0,
        fmax=None,
        is_log=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        freeze_parameters=True,
    ):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax == None:
            fmax = sr // 2

        self.melW = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        ).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.
        Args:
            input: (*, n_fft), spectrogram

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError(
                    "top_db must be non-negative"
                )
            log_spec = torch.clamp(
                log_spec, min=log_spec.max().item() - self.top_db, max=np.inf
            )

        return log_spec


class Enframe(nn.Module):
    def __init__(self, frame_length=2048, hop_length=512):
        """Enframe a time sequence. This function is the pytorch implementation
        of librosa.util.frame
        """
        super(Enframe, self).__init__()

        self.enframe_conv = nn.Conv1d(
            in_channels=1,
            out_channels=frame_length,
            kernel_size=frame_length,
            stride=hop_length,
            padding=0,
            bias=False,
        )

        self.enframe_conv.weight.data = torch.Tensor(
            torch.eye(frame_length)[:, None, :]
        )
        self.enframe_conv.weight.requires_grad = False

    def forward(self, input):
        """input: (batch_size, samples)

        Output: (batch_size, window_length, frames_num)
        """
        output = self.enframe_conv(input[:, None, :])
        return output

    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError(
                    "top_db must be non-negative"
                )
            log_spec = torch.clamp(
                log_spec, min=log_spec.max() - self.top_db, max=np.inf
            )

        return log_spec


class Scalar(nn.Module):
    def __init__(self, scalar, freeze_parameters):
        super(Scalar, self).__init__()

        self.scalar_mean = Parameter(torch.Tensor(scalar["mean"]))
        self.scalar_std = Parameter(torch.Tensor(scalar["std"]))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        return (input - self.scalar_mean) / self.scalar_std

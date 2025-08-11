from typing import Optional

import torch
from torch import nn
from recipes.sacodec.models.components.spectral_ops import ISTFT


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center", atan2_magnitude_threshold_ratio: float = 0.0):
        super().__init__()
        out_dim = int((n_fft + 2) * 1.5) # mag, p_I, p_R
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)
        self.atan2_magnitude_threshold_ratio = atan2_magnitude_threshold_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            x = x.float()
            x = self.out(x.transpose(1, 2)).transpose(1, 2)
            logamp, p_I, p_R = x.chunk(3, dim=1)

            if self.atan2_magnitude_threshold_ratio != 0:
                # stabilize the atan2 by setting small values to 0
                magnitude = torch.sqrt(p_I**2 + p_R**2)
                threshold = torch.max(magnitude) * self.atan2_magnitude_threshold_ratio
                mask = magnitude < threshold
                p_R = torch.where(mask, torch.zeros_like(p_R), p_R)
                p_I = torch.where(mask, torch.zeros_like(p_I), p_I)

            pha = torch.atan2(p_I, p_R)

            rea = torch.exp(logamp)*torch.cos(pha)
            imag = torch.exp(logamp)*torch.sin(pha)

            S = torch.complex(rea, imag)
            audio = self.istft(S)

            outputs = logamp, pha, rea, imag, audio
            outputs = (o.unsqueeze(1) for o in outputs) # keep audio channel dimension
        return outputs


class ISTFTHeadStereo(FourierHead):
    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "center", audio_channels: int = 2, atan2_magnitude_threshold_ratio: float = 0.0):
        super().__init__()
        self.audio_channels = audio_channels
        self.N_feat = 3 # logamp, p_I, p_R
        self.N_dim = n_fft // 2 + 1 # N frequencies
        out_dim = self.N_dim * self.N_feat * self.audio_channels
        if win_length is None:
            win_length = n_fft

        self.out = torch.nn.Linear(dim, out_dim)
        self.atan2_magnitude_threshold_ratio = atan2_magnitude_threshold_ratio
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            x = x.float()
            x = self.out(x.float().transpose(1, 2)).transpose(1, 2)

            # Flatten channels
            B, N, T = x.shape

            x = torch.concat(x.chunk(self.audio_channels, dim=1), dim=0) # B * C, N_dim * N_feat, T
            # x = x.reshape(B * self.audio_channels, self.N_dim * self.N_feat, T)

            # Extract features
            logamp, p_I, p_R = x.chunk(self.N_feat, dim=1)

            if self.atan2_magnitude_threshold_ratio != 0:
                # stabilize the atan2 by setting small values to 0
                magnitude = torch.sqrt(p_I**2 + p_R**2)
                threshold = torch.max(magnitude) * self.atan2_magnitude_threshold_ratio
                mask = magnitude < threshold
                p_R = torch.where(mask, torch.zeros_like(p_R), p_R)
                p_I = torch.where(mask, torch.zeros_like(p_I), p_I)

            pha = torch.atan2(p_I, p_R)

            rea = torch.exp(logamp)*torch.cos(pha)
            imag = torch.exp(logamp)*torch.sin(pha)

            S = torch.complex(rea, imag)
            audio = self.istft(S)

            outputs = logamp, pha, rea, imag, audio
            # Restore audio channel dimensions
            # outputs = tuple(o.reshape(B, self.audio_channels, *o.shape[1:]) for o in outputs) # keep audio channel dimension
            outputs = tuple(torch.stack(o.chunk(self.audio_channels, dim=0), dim=1) for o in outputs) # keep audio channel dimension
        return outputs

class ISTFTHeadStereoVocos(FourierHead):
    "Vocos implementation. Directly predicts phase values instead of complex phase values"
    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "center", audio_channels: int = 2):
        super().__init__()
        self.audio_channels = audio_channels
        self.N_feat = 2 # logamp, phase
        self.N_dim = n_fft // 2 + 1 # N frequencies
        out_dim = self.N_dim * self.N_feat * self.audio_channels
        if win_length is None:
            win_length = n_fft

        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            x = x.float()
            x = self.out(x.transpose(1, 2)).transpose(1, 2)

            # Flatten channels
            B, N, T = x.shape

            # x = torch.concat(x.chunk(self.audio_channels, dim=1), dim=0) # B * C, N_dim * N_feat, T
            x = x.reshape(B * self.audio_channels, self.N_dim * self.N_feat, T)
            # # Extract features
            logamp, pha = x.chunk(self.N_feat, dim=1)

            rea = torch.exp(logamp)*torch.cos(pha)
            imag = torch.exp(logamp)*torch.sin(pha)

            S = torch.complex(rea, imag)
            audio = self.istft(S)

            outputs = logamp, pha, rea, imag, audio
            # Restore audio channel dimensions
            outputs = tuple(o.reshape(B, self.audio_channels, *o.shape[1:]) for o in outputs) # keep audio channel dimension
            # outputs = tuple(torch.stack(o.chunk(self.audio_channels, dim=0), dim=1) for o in outputs) # keep audio channel dimension
        return outputs

class ISTFTHeadStereoMusic2Latent(FourierHead):
    "Vocos implementation. Directly predicts phase values instead of complex phase values"
    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "center", audio_channels: int = 2, normalize_spec: bool = False, atan2_magnitude_threshold_ratio: float = 0.0):
        super().__init__()
        self.audio_channels = audio_channels
        self.N_feat = 2 # logamp, phase
        self.N_dim = n_fft // 2 + 1 # N frequencies
        out_dim = self.N_dim * self.N_feat * self.audio_channels
        if win_length is None:
            win_length = n_fft

        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, padding=padding)
        self.normalize_spec = normalize_spec
        self.atan2_magnitude_threshold_ratio = atan2_magnitude_threshold_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False):
            x = x.float()
            x = self.out(x.transpose(1, 2)).transpose(1, 2)

            # Flatten channels
            B, N, T = x.shape

            # x = torch.concat(x.chunk(self.audio_channels, dim=1), dim=0) # B * C, N_dim * N_feat, T
            x = x.reshape(B * self.audio_channels, self.N_dim * self.N_feat, T)
            # # Extract features
            rea, imag = x.chunk(self.N_feat, dim=1)

            S = torch.complex(rea, imag)
            if self.normalize_spec:
                S = denormalize_complex(S)
            audio = self.istft(S).clamp(-1.,1.)

            rea, imag = torch.real(S), torch.imag(S)
            logamp=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]

            if self.atan2_magnitude_threshold_ratio != 0:
                # stabilize the atan2 by setting small values to 0
                magnitude = torch.sqrt(imag**2 + rea**2)
                threshold = torch.max(magnitude) * self.atan2_magnitude_threshold_ratio
                mask = magnitude < threshold
                rea = torch.where(mask, torch.zeros_like(rea), rea)
                imag = torch.where(mask, torch.zeros_like(imag), imag)

            pha=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

            outputs = logamp, pha, rea, imag, audio
            # Restore audio channel dimensions
            outputs = tuple(o.reshape(B, self.audio_channels, *o.shape[1:]) for o in outputs) # keep audio channel dimension
            # outputs = tuple(torch.stack(o.chunk(self.audio_channels, dim=0), dim=1) for o in outputs) # keep audio channel dimension
        return outputs

def denormalize_realimag(x, alpha_rescale=0.65, beta_rescale=0.34):
    x = x/beta_rescale
    return torch.sign(x)*(x.abs()**(1./alpha_rescale))

def normalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    return beta_rescale*(x.abs()**alpha_rescale).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def denormalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    x = x/beta_rescale
    return (x.abs()**(1./alpha_rescale)).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

class ISTFTDecoderBase(nn.Module):
    def __init__(self, backbone_cls, head_cls):
        super().__init__()
        self.backbone = backbone_cls()
        self.head = head_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
    
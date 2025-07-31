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

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        out_dim = int((n_fft + 2) * 1.5) # mag, p_I, p_R
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        logamp, p_I, p_R = x.chunk(3, dim=1)

        pha = torch.atan2(p_I, p_R)

        rea = torch.exp(logamp)*torch.cos(pha)
        imag = torch.exp(logamp)*torch.sin(pha)

        S = torch.complex(rea, imag)
        audio = self.istft(S)

        outputs = logamp, pha, rea, imag, audio
        outputs = (o.unsqueeze(1) for o in outputs) # keep audio channel dimension
        return outputs


class ISTFTHeadStereo(FourierHead):
    def __init__(self, dim: int, n_fft: int, hop_length: int, win_length: int = None, padding: str = "center", audio_channels: int = 2):
        super().__init__()
        self.audio_channels = audio_channels
        self.N_feat = 3 # logamp, p_I, p_R
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
        x = self.out(x.transpose(1, 2)).transpose(1, 2)

        # Flatten channels
        B, N, T = x.shape

        x = torch.concat(x.chunk(self.audio_channels, dim=1), dim=0) # B * C, N_dim * N_feat, T
        # x = x.reshape(B * self.audio_channels, self.N_dim * self.N_feat, T)

        # Extract features
        logamp, p_I, p_R = x.chunk(self.N_feat, dim=1)

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
    
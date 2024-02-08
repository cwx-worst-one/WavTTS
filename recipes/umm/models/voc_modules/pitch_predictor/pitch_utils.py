import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


### Proprocessing/Postprocessing helper functions. ###
def log1p_to_raw_hz(x):
    return torch.expm1(x)


def raw_hz_to_log1p(x):
    return torch.log1p(x)


### Compute Loss functions. ###
def compute_f0_loss(recon_f0, f0, vuv):
    """Compute loss on continuous f0_hz."""
    recon_f0 = recon_f0.contiguous().float()
    f0 = f0.contiguous().float()
    f0_loss = (torch.abs(recon_f0 - f0) * vuv).sum() / (torch.sum(vuv) + 1)
    return f0_loss


def compute_vuv_loss(recon_vuv, vuv):
    """Compute loss on voiced/unvoiced states."""
    recon_vuv = recon_vuv.contiguous().float()
    vuv = vuv.contiguous().float()
    vuv_loss = F.binary_cross_entropy_with_logits(recon_vuv, vuv)
    return vuv_loss


def compute_perceptual_pitch_loss(recon_pitch_h, pitch_h):
    """Compute MSE between pitch predictor hidden states."""
    # recon_pitch_h : [batch_size, n_frames, hidden_state]
    # pitch_h : [batch_size, n_frames, hidden_state]
    assert recon_pitch_h.shape == pitch_h.shape
    return F.mse_loss(recon_pitch_h, pitch_h)


def compute_min_lengths(x, y, tolerance=2, axis=-1):
    """Find shortest vector length. Sometimes the f0_gt can be 1 sample longer than f0_pred."""
    x_len, y_len = list(x.shape)[axis], list(y.shape)[axis]
    assert abs(x_len - y_len) < tolerance
    return min(x_len, y_len)


### Testing Pitch Model
def gen_test_sin(freq=440, amp=0.9, dur_sec=2, sr=24000, add_batch_dim=True):
    dur_sec = 2
    t = torch.linspace(0, dur_sec, dur_sec * sr)
    angular_freq = 2 * np.pi * 440
    sine_signal = torch.sin(amp * angular_freq * t)
    if add_batch_dim:
        sine_signal = sine_signal[None, :]
    return sine_signal


### Figure drawing functions. ###
def _save_figure(fig, save_path):
    """Save figure for debugging."""
    fig.savefig(save_path)
    print(f"Saved to {save_path}")


def _create_fig_mel_with_f0_pred_and_gt(
    mel,
    f0_pred,
    f0_gt,
    vuv_pred=None,
    vuv_gt=None,
    convert_to_raw_hz=True,
    mel_bins_max=160,
    f0_hz_max=800,
    vuv_height_multiplier=10,
):
    """Plot f0_pred and f0_gt ontop of mel spectrogram. TODO: handle np vs torch. This function expects numpy."""
    # Configure plot
    fig = plt.figure(figsize=(16, 8))
    legend_text = []
    # x axis time markings
    t = np.arange(0, len(f0_pred), 1)
    # Plot Mel
    plt.pcolor(mel.T, vmin=-6, vmax=0.5)
    legend_text.append("Mel")
    # Plot Raw F0_Hz Pred
    f0_hz_pred = np.expm1(f0_pred) if convert_to_raw_hz else f0_pred
    f0_hz_pred = (
        f0_hz_pred / f0_hz_max
    ) * mel_bins_max  # scales the f0 values to fit on the mel spectrogram plot.
    plt.plot(t, f0_hz_pred, "red", linewidth=2)
    legend_text.append("F0_hz Pred")
    # Plot Raw F0_Hz GT
    f0_hz_gt = np.expm1(f0_gt) if convert_to_raw_hz else f0_gt
    f0_hz_gt = (
        f0_hz_gt / f0_hz_max
    ) * mel_bins_max  # scales the f0 values to fit on the mel spectrogram plot.
    plt.plot(t, f0_hz_gt, "blue", linewidth=2)
    legend_text.append("F0_hz GT")
    # Plot Voice/Unvoiced
    if vuv_pred is not None:
        vuv_pred = (
            vuv_pred * vuv_height_multiplier
        )  # scales the vuv values to be more visible on the mel spectrogram plot
        plt.plot(t, vuv_pred, "salmon", linestyle="dashed", linewidth=1)
        legend_text.append("VUV Pred")
    if vuv_gt is not None:
        vuv_gt = (
            vuv_gt * vuv_height_multiplier
        ) * 1.2  # Make it slighly taller than the predicted vuv
        plt.plot(t, vuv_gt, "mediumblue", linestyle="dashed", linewidth=1)
        legend_text.append("VUV GT")
    # Titles and legends
    plt.legend(legend_text)
    plt.title("Combined Mel spectrogram, Pred Raw F0, GT Raw F0 and VUV")
    return fig


def post_process_vuv_logits_to_binary_state(vuv, threshold=0.5):
    """By default, the model outputs logits for vuv state. These can be post processed into binary voiced/unvoiced state."""
    vuv[vuv >= threshold] = 1
    vuv[vuv < threshold] = 0
    return vuv


def mask_f0_by_vuv(f0, vuv):
    """Combine f0 and vuv into a single f0 signal. All unvoiced states will force f0 to 0."""
    assert f0.shape == vuv.shape
    return np.multiply(f0, vuv)

from typing import Any
from torch import nn
import pytorch_lightning as pl
import torch
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.requires.model_initializer import init_rmvpe
from recipes.umm.transforms.speech import SpeechTransform
from recipes.umm.models.umm_mkii import get_vuv, f0_normalize
import torch.nn.functional as F
from recipes.umm.utils.mel_utils import torch_wav2spec
import matplotlib.pyplot as plt
import numpy as np
import os

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

def compute_min_lengths(x, y, tolerance=2):
    """Find shortest vector length. Sometimes the f0_gt can be 1 sample longer than f0_pred."""
    x_len, y_len = list(x.shape)[-1], list(y.shape)[-1]
    assert abs(x_len - y_len) < tolerance
    return min(x_len, y_len)

def create_fig(y_pred, y_gt, title):
    """Plot ground truth and predicted f0 for a single batch sample."""
    t = np.arange(0, len(y_pred), 1) 
    fig = plt.figure(figsize=(16, 8))
    plt.plot(t, y_pred, 'r')
    plt.plot(t, y_gt, 'g')
    plt.legend(['pred', 'gt'])
    plt.title(title)
    return fig


class PitchPredictorTask(pl.LightningModule):
    """
    Lightning module for training a perceptual pitch model.

    Args: 
        audio_key (str) - which of the 3 MSS tracks in the dataset to train from ["audio", "audio_vocal", "audio_inst"]
        n_mels_input (int) - all perceptual losses in V2 UMM Dual Arch training are standardized to 160 mel band input
    """
    def __init__(
        self,
        model_cls,
        audio_key = "audio_vocal",
        n_mels_input = 160,
        sample_rate = 24000,
        cache_dir = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.audio_key = audio_key 
        self.n_mels_input = n_mels_input 
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir # required for saving the RMVPE model

    def setup(self, stage) -> None:
        if stage in ["fit", "validate"]:
            self.rmvpe_model = self._setup_gt_pitch_predictor()
        self.mel_transform = self._setup_mel_audio_transform()
        
    def _setup_gt_pitch_predictor(self):
        """Load RMVPE model as ground truth pitch detector."""
         # TODO(@hanoihantrakul) 1-9-2024 You really should make ckpt_path configurable from a YAML
        ckpt_path = "hdfs://haruna/home/byte_speech_sv/user/chenyuanzhe/share/rmvpe.pt"
        state_dict = init_rmvpe(hpath=ckpt_path, local_rank=0, cache_dir=self.cache_dir)["state_dict"]
        rmvpe = RMVPE()
        rmvpe.load_and_eval(state_dict)
        return rmvpe

    # def _setup_mel_audio_transform(self):
    #     """Create the mel spectrogram audio transformation."""
    #     # TODO(@hanoihantrakul) 1-9-2024 You really should make these settings configurable from a YAML
    #     return SpeechTransform(
    #         sample_rate=self.sample_rate,
    #         n_mels=self.n_mels_input,
    #         n_fft=2048,
    #         win_length=2048,
    #         hop_length=240,
    #         f_min=0,
    #         f_max=self.sample_rate // 2,
    #     )

    def _setup_mel_audio_transform(self):
        # We do not use the SpeechTransform() version of the mel transform.
        return lambda x: torch_wav2spec(x, num_mels=self.n_mels_input, sample_rate=self.sample_rate)

    def prepare_feature(self, batch, audio_key):
        """
        When training the perceptual model, it is possible to configure through `audio_key` what data to train on.
        The model could be trained just on the full mix, just the vocal audio or a combination both (not implemented yet)

        Args:
            audio_key: string, choose whether "audio" (full audio) or "audio_vocal" (vocal audio only). 
        """
        # load and prepare audio
        audio = batch[audio_key] 
        audio = audio.squeeze(dim=1).float()
 
        # mel transform
        mel = self.mel_transform(audio)
        return mel

    def compute_ground_truth_pitch(self, x):
        """Use reference RMVPE model to get ground truth labels."""
        gt_out = self.rmvpe_model.batch_infer(x, self.sample_rate, thred=0.03, use_viterbi=False)
        f0 = gt_out[:, :-1]
        vuv = get_vuv(f0)
        f0 = torch.log1p(f0) # No normalization. Only log1p for stability.
        return f0, vuv

    def compute_pitch_losses(self, recon_f0, f0, recon_vuv, vuv):
        """Logic identical to UMMDual.add_pitch_loss()"""
        f0_loss = compute_f0_loss(recon_f0, f0, vuv)
        vuv_loss = compute_vuv_loss(recon_vuv, vuv)
        return f0_loss, vuv_loss

    def _shared_step(self, batch):
        """
        Main logic:
        - Extract audio (either clean MSS vocals or the vocals with instrumental track.)
        - Generate mel spectrograms
        - Model makes predictions
        - Extract f0 and vuv from predictions
        - Compute ground truth labels using RMVPE on clean MSS vocals of same data

        Returns:
            {"f0_pred": torch.Tensor, 
             "f0_gt":  torch.Tensor, 
             "vuv_pred":  torch.Tensor, 
             "vuv_gt":  torch.Tensor}
        """
        # generate mel spectrogram [batch_size, n_frames, n_mels]
        x_mels = self.prepare_feature(batch, self.audio_key) 

        # predict outputs [batch_size, n_frames, 2]
        y_pred = self.model(x_mels) 
        
        # extract f0 and vuv prediction [batch_size, n_frames]
        f0_pred = y_pred[:, :, 0:1].squeeze(-1)
        vuv_pred = y_pred[:, :, 1:].squeeze(-1) 

        # always use clean vocals when getting ground truth f0 from RMVPE model
        # [batch_size, n_samples] 
        x_audio = batch["audio_vocal"].squeeze(dim=1).float()

        f0_gt, vuv_gt = self.compute_ground_truth_pitch(x_audio) 

        trim_len = compute_min_lengths(f0_pred, f0_gt)
        f0_pred, f0_gt = f0_pred[:, :trim_len], f0_gt[:, :trim_len]
        vuv_pred, vuv_gt = vuv_pred[:, :trim_len], vuv_gt[:, :trim_len]
        assert f0_pred.shape == f0_gt.shape # [batch_size, n_frames]
        assert vuv_pred.shape == vuv_gt.shape # [batch_size, n_frames]
        return {"f0_pred": f0_pred, "f0_gt": f0_gt, "vuv_pred": vuv_pred, "vuv_gt": vuv_gt}
  
    def training_step(self, batch, batch_idx):
        """Main logic: self._shared_step() then calculate losses."""
        output_dict = self._shared_step(batch)

        # compute losses
        f0_loss, vuv_loss = self.compute_pitch_losses(output_dict['f0_pred'], output_dict['f0_gt'], output_dict['vuv_pred'], output_dict['vuv_gt'])

        # compute total loss
        loss = f0_loss + vuv_loss 
        
        # logging losses
        loss_dict = {"loss": loss, "f0_loss": f0_loss, "vuv_loss": vuv_loss}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        '''Similar logic to training step, but log graphs instead.'''
        if batch_idx == 0:
            output_dict = self._shared_step(batch)
            figure_dict = {k: v.cpu().detach().numpy() for k, v in output_dict.items()}
            self._create_and_log_figures(figure_dict)

    def _create_and_log_figures(self, figure_dict, max_batch_samples=6):
        """Iterate through batch and log the figures."""
        num_batch_samples = list(figure_dict["f0_pred"].shape)[-1]
        assert max_batch_samples < num_batch_samples
        for i in range(max_batch_samples):
            f0_pred_i, f0_gt_i = figure_dict["f0_pred"][i,:], figure_dict["f0_gt"][i,:]
            fig_f0 = create_fig(f0_pred_i, f0_gt_i, "Log F0")
            self.logger.experiment.add_figure(
                f"val_f0-{i:02d}", fig_f0, global_step=self.global_step
            )

            vuv_pred_i, vuv_gt_i = figure_dict["vuv_pred"][i,:], figure_dict["vuv_gt"][i,:]
            fig_vuv = create_fig(vuv_pred_i, vuv_gt_i, "Voice/Unvoiced")
            self.logger.experiment.add_figure(
                f"val_vuv-{i:02d}", fig_vuv, global_step=self.global_step
            )
            
    def configure_optimizers(self):
        """Simple task. No need for special optimizer settings."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        return optimizer

def _save_figure(fig, save_path):
    """For debugging saved figures."""
    fig.savefig(save_path)
    print(f"Saved to {save_path}")

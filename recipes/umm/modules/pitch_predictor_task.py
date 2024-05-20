from typing import Any
from torch import nn
import pytorch_lightning as pl
import torch
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.requires.model_initializer import init_rmvpe
from recipes.umm.transforms.speech import SpeechTransform
from recipes.umm.models.umm_mkii import get_vuv, f0_normalize
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.DEBUG)

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

class PitchPredictorTask(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        audio_key = "audio_vocal",
        sample_rate = 24000,
        cache_dir = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.audio_key = audio_key # which of the 3 MSS tracks in the dataset to train from ["audio", "audio_vocal", "audio_inst"]
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

    def _setup_mel_audio_transform(self):
        """Create the mel spectrogram audio transformation."""
        # TODO(@hanoihantrakul) 1-9-2024 You really should make these settings configurable from a YAML
        return SpeechTransform(
            sample_rate=self.sample_rate,
            n_mels=160,
            n_fft=2048,
            win_length=2048,
            hop_length=240,
            f_min=0,
            f_max=self.sample_rate // 2,
        )

    def prepare_feature(self, batch, audio_key):
        # load and prepare audio
        audio = batch[audio_key] # could use "audio_vocal" or "audio" directly (to make system more robust)
        audio = audio.squeeze(dim=1).float()
 
        # mel transform
        normalize = False # for UMM training we normalize using the dataset stats. For this application, this is not necessary.
        mel = self.mel_transform(audio, normalize=normalize)
        return mel

    def compute_ground_truth_pitch(self, x):
        """Use reference RMVPE model to get ground truth labels."""
        gt_out = self.rmvpe_model.batch_infer(x, self.sample_rate, thred=0.03, use_viterbi=False)
        f0 = gt_out[:, :-1]
        vuv = get_vuv(f0)
        f0 = f0_normalize(f0)
        return f0, vuv

    def compute_pitch_losses(self, recon_f0, f0, recon_vuv, vuv):
        """Logic identical to UMMDual.add_pitch_loss()"""
        f0_loss = compute_f0_loss(recon_f0, f0, vuv)
        vuv_loss = compute_vuv_loss(recon_vuv, vuv)
        return f0_loss, vuv_loss
  
    def training_step(self, batch, batch_idx):
        """
        Main training logic:
        - Extract audio (either clean MSS vocals or the vocals with instrumental track.)
        - Generate mel spectrograms
        - Model makes predictions
        - Extract f0 and vuv from predictions
        - Compute ground truth labels using RMVPE on clean MSS vocals of same data
        - Compute losses
        """
        # generate mel spectrogram [batch_size, n_frames, n_mels]
        x_mels = self.prepare_feature(batch, self.audio_key) 

        # predict outputs [batch_size, n_frames, 2]
        y_pred = self.model(x_mels) 
        
        # extract f0 and vuv prediction [batch_size, n_frames]
        f0_pred=y_pred[:, :, 0:1].squeeze(-1)
        vuv_pred=y_pred[:, :, 1:].squeeze(-1) 

        # get ground truth pitch signal from RMVPE model (always use the clean vocal)
        # [batch_size, n_samples] 
        x_audio = batch["audio_vocal"].squeeze(dim=1).float()

        f0_gt, vuv_gt = self.compute_ground_truth_pitch(x_audio) 
        assert f0_pred.shape == f0_gt.shape # [batch_size, n_frames]
        assert vuv_pred.shape == vuv_gt.shape # [batch_size, n_frames]

        # compute losses
        f0_loss, vuv_loss = self.compute_pitch_losses(f0_pred, f0_gt, vuv_pred, vuv_gt)

        # compute total loss
        loss = f0_loss + vuv_loss 
        
        # logging losses
        loss_dict = {"loss": loss, "f0_loss": f0_loss, "vuv_loss": vuv_loss}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     '''Identical logic to training step'''
    #     return self.training_step(batch, batch_idx)

    # def predict_step(self, batch, batch_idx):
    #     '''Identical logic to training step'''
    #     return self.training_step(batch, batch_idx)

    # def forward(self, x):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0e-5, betas=[0.9, 0.900], eps=0.00000001, weight_decay= 0.1)
        return optimizer

from torch import optim
import torch
import os
import numpy as np
from torch.nn.functional import mse_loss, binary_cross_entropy

from recipes.structure.utils.eval_structure import eval_a_song
from recipes.structure.utils.eval_structure import merge_multiple_temporal_probs
from recipes.structure.pl_modules.pl_module import LitStructure
from recipes.structure.models.get_models import get_teacher


class LitPseudoStructure(LitStructure):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        alpha=0.5,
        n_fft=1024,
        sample_len=36,
        sampling_rate=16000,
        label_hop=0.192,
        sample_hop=9,
        n_top_bound=14,
        enable_short=True,
        n_boundary=2,
        n_function=7,
        teacher_name="spectnt",
        teacher_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
            alpha=alpha,
            n_fft=n_fft,
            sample_len=sample_len,
            sampling_rate=sampling_rate,
            label_hop=label_hop,
            sample_hop=sample_hop,
            n_top_bound=n_top_bound,
            enable_short=enable_short,
            n_boundary=n_boundary,
            n_function=n_function,
        )

        self.teacher = get_teacher(teacher_name, teacher_path)
        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        sample = batch[0]

        boundary_tar, function_tar = self.teacher({'audio': sample, 'aug_hop_size': self._hop_length})

        boundary_pre, function_pre = self.model({'audio': sample, 'aug_hop_size': self._hop_length})

        boundary_tar = torch.sigmoid(boundary_tar)
        boundary_tar = boundary_tar / boundary_tar.max()
        #boundary_loss = binary_cross_entropy(torch.sigmoid(boundary_pre), torch.sigmoid(boundary_tar))
        boundary_loss = (mse_loss(torch.sigmoid(boundary_pre), boundary_tar, reduction='none')).mean()
        function_loss = (mse_loss(torch.sigmoid(function_pre), torch.sigmoid(function_tar), reduction='none')).mean()

        loss = self._alpha * boundary_loss + (1 - self._alpha) * function_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

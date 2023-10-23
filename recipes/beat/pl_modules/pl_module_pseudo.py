from collections import defaultdict
from typing import Any

import madmom
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, binary_cross_entropy

import recipes.beat.eval.mireval_beat as mireval_beat
from recipes.beat.utils.augment_utils import time_augmentation
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs
from recipes.beat.models.get_models import get_teacher
from recipes.beat.pl_modules.pl_module import LitBeat


class LitPseudoBeat(LitBeat):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        beat_window_length,
        n_beats,
        n_tempo,
        sample_rate,
        sample_len,
        label_hop,
        do_time_augmentation=False,
        do_cqt_augmentation=False,
        do_tempo_loss=True,
        f_measure_threshold=0.07,
        use_tempo_prior=False,
        teacher_name="tcn",
        teacher_path=None,
        hop_factor=1,
        max_pad_second=0
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
            hop_length=hop_length,
            beat_window_length=beat_window_length,
            n_beats=n_beats,
            n_tempo=n_tempo,
            sample_rate=sample_rate,
            sample_len=sample_len,
            label_hop=label_hop,
            do_time_augmentation=do_time_augmentation,
            do_cqt_augmentation=do_cqt_augmentation,
            do_tempo_loss=do_tempo_loss,
            f_measure_threshold=f_measure_threshold,
            use_tempo_prior=use_tempo_prior,
            hop_factor=hop_factor,
            max_pad_second=max_pad_second
        )

        self.teacher = get_teacher(teacher_name, teacher_path)
        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        if self._do_time_augmentation:
            hop_length = time_augmentation(self._hop_length)
        else:
            hop_length = self._hop_length
        inputs["aug_hop_size"] = hop_length

        beat_tar, tempo_tar = self.teacher({"audio": batch[0], "aug_hop_size": hop_length})

        if self._do_cqt_augmentation:
            self.model.stages[0], _ = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1
            )

        # model prediction
        beat_pred, tempo_pred = self.model(inputs)

        loss = self._train_beat(
            beat_pred[0],
            beat_tar[0],
            tempo_pred[0],
            tempo_tar[0],
            self._hop_length,
            inputs["aug_hop_size"],
            self._beat_window_length,
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def _train_beat(
        self,
        beat_pre,
        beat,
        tempo_pre,
        tempo_target,
        hop_length,
        aug_hop_size,
        beat_window_length,
    ):
        beat_loss = torch.zeros(1, device=self.device)
        # filter batches with no beat annotations
        '''
        beat = torch.nn.functional.interpolate(
            beat.unsqueeze(1),
            (
                int(np.round(beat.shape[1] * hop_length / aug_hop_size)),
                beat.shape[2],
            ),
        ).squeeze(1)
        '''

        # get length
        min_length = min(beat_pre.shape[1], beat.shape[1])
        #beat_loss += (mse_loss(torch.sigmoid(beat_pre[:, :min_length]), torch.sigmoid(beat[:, :min_length]), reduction='none')).mean()

        beat_pre = torch.clip(torch.sigmoid(beat_pre[:, :min_length]), min=0, max=1)
        beat = torch.clip(torch.sigmoid(beat[:, :min_length]), min=0, max=1)
        beat_loss += binary_cross_entropy(beat_pre, beat)

        if self._do_tempo_loss:
            beat_loss += mse_loss(torch.sigmoid(tempo_pre), torch.sigmoid(tempo_target))

        return beat_loss

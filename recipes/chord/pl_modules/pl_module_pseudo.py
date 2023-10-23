from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import mse_loss

from recipes.chord.pl_modules.pl_module import LitChord
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs
from recipes.chord.utils.mireval_chord import get_chord_score
from recipes.chord.models.get_models import get_teacher


class LitPseudoChord(LitChord):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        chord_pool,
        sample_rate,
        sample_len,
        resnet_pools,
        hop_factor=6,
        do_cqt_augmentation=False,
        teacher_name="tcn",
        teacher_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
            hop_length=hop_length,
            chord_pool=chord_pool,
            sample_rate=sample_rate,
            sample_len=sample_len,
            resnet_pools=resnet_pools,
            hop_factor=hop_factor,
            do_cqt_augmentation=do_cqt_augmentation,
        )
       
        self.teacher = get_teacher(teacher_name, teacher_path)
        self.teacher.eval()

    def training_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        inputs["audio"] = batch[0]
        inputs["aug_hop_size"] = self._hop_length

        data["chord_root"], data["chord_triad"] = self.teacher(inputs)

        if self._do_cqt_augmentation:
            self.model.stages[0], note_idx = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1
            )

            data["chord_root"][..., 1:] = torch.roll(
                data["chord_root"][..., 1:], note_idx * -1, -1
            )

        # model prediction
        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)

        loss = self._train_chord(outputs, data)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def _train_chord(self, prediction, target):
        chord_loss = 0

        eval_name = ["chord_root", "chord_triad"]
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            min_length = min(chord_pre.shape[1], chord.shape[1])
            chord_pre = chord_pre[:, :min_length]
            chord = chord[:, :min_length]

            chord_loss += (mse_loss(torch.sigmoid(chord_pre), torch.sigmoid(chord), reduction='none')).mean()

        return chord_loss

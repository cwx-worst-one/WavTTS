from collections import defaultdict
from typing import Any

import madmom
import numpy as np
import torch
import random
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, binary_cross_entropy

import recipes.beat.eval.mireval_beat as mireval_beat
from recipes.beat.utils.augment_utils import time_augmentation
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs, merge_multiple_temporal_probs_inference
from recipes.beat.models.get_models import get_teacher
from recipes.beat.pl_modules.pl_module_vocal import LitVocalBeat


class LitPseudoBeat(LitVocalBeat):
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
        observation_lambda=14,
        transition_lambda=150,
        correction=False,
        max_pad_second=4,
        do_time_augmentation=False,
        do_cqt_augmentation=False,
        do_tempo_loss=True,
        f_measure_threshold=0.07,
        use_tempo_prior=False,
        add_noise=False,
        tempo_dir=None,
        hop_factor=1,
        teacher_name="tcn",
        teacher_path=None,
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
            observation_lambda=observation_lambda,
            transition_lambda=transition_lambda,
            correction=correction,
            tempo_dir=tempo_dir,
            hop_factor=hop_factor,
            max_pad_second=max_pad_second
        )
        self._hop_length = hop_length
        self._beat_window_length = beat_window_length
        self._n_beats = n_beats
        self._n_tempo = n_tempo
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._max_pad_second = max_pad_second
        self._label_hop = label_hop
        self._do_time_augmentation = do_time_augmentation
        self._do_cqt_augmentation = do_cqt_augmentation
        self._do_tempo_loss = do_tempo_loss
        self._f_measure_threshold = f_measure_threshold
        self._use_tempo_prior = use_tempo_prior
        self.reduce_sample = (
            int(self._max_pad_second / self._label_hop) if max_pad_second != 0 else 0
        )
        self.teacher = get_teacher(teacher_name, teacher_path)
        self.teacher.eval()

        self._add_noise = add_noise

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0][:, 0]
        if self._add_noise:
            std = torch.std(inputs["audio"])
            noise_std = random.uniform(0.03 * std, 0.03 * std)
            noise = torch.distributions.normal.Normal(loc=0.0, scale=noise_std).sample(inputs["audio"].shape)
            inputs["audio"] = inputs["audio"] + noise.to(inputs["audio"].device)

        beat_tar, tempo_tar = self.teacher({"audio": batch[0].sum(1), "aug_hop_size": self._hop_length})

        if self._do_time_augmentation:
            inputs["aug_hop_size"] = time_augmentation(self._hop_length)
        else:
            inputs["aug_hop_size"] = self._hop_length

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

    def validation_step(self, batch, batch_idx):
        inputs = {}

        if self._do_cqt_augmentation:
            self.model.stages[0], _ = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1, note="C1"
            )

        sample = batch[0]
        
        pad_sample = int(self._sample_rate * self._max_pad_second)
        sample = torch.nn.functional.pad(sample, (pad_sample, pad_sample), mode='constant', value=0)
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(1, self._train_num_samples + pad_sample * 2, int(self._train_num_samples / self._hop_factor))

        inputs["audio"] = sample.squeeze(0).float()
        inputs["beats"] = batch[1]
        inputs["tempo_label"] = batch[2]
        orig_beats = batch[3][0]

        inputs["aug_hop_size"] = self._hop_length

        if self._add_noise:
            std = torch.std(inputs["audio"])
            noise_std = random.uniform(0.03 * std, 0.03 * std)
            noise = torch.distributions.normal.Normal(loc=0.0, scale=noise_std).sample(inputs["audio"].shape)
            inputs["audio"] = inputs["audio"] + noise.to(inputs["audio"].device)

        beat_pred, tempo_pred = self.model(inputs)

        preds, _ = self._eval_beat(
            beat_pred,
            inputs["beats"],
            tempo_pred,
            inputs["tempo_label"],
            self._n_beats,
            int(self._sample_len / self._label_hop),
            int(self._sample_len / self._hop_factor / self._label_hop),
        )

        metrics, _, _ = self._get_beat_score(
            preds["pred"]["beat_probs"],
            orig_beats,
            preds["truth"]["tempo"],
            self._label_hop,
        )

        for k, v in metrics.items():
            if v is None:
                metrics[k] = 0
        metrics["val_summary"] = np.array(list(metrics.values())).mean() if len(list(metrics.values())) != 0 else 0

        self.log_dict(metrics, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(metrics)

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
        beat = torch.nn.functional.interpolate(
            beat.unsqueeze(1),
            (
                int(np.round(beat.shape[1] * hop_length / aug_hop_size)),
                beat.shape[2],
            ),
        ).squeeze(1)

        # get length
        min_length = min(beat_pre.shape[1], beat.shape[1])
        beat_pre = torch.clip(torch.sigmoid(beat_pre[:, :min_length]), min=0, max=1)
        beat = torch.clip(torch.sigmoid(beat[:, :min_length]), min=0, max=1)
        beat_loss += binary_cross_entropy(beat_pre, beat)

        if self._do_tempo_loss:
            beat_loss += mse_loss(torch.sigmoid(tempo_pre), torch.sigmoid(tempo_target))

        return beat_loss
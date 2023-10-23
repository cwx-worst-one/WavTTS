from collections import defaultdict
from typing import Any

import madmom
import numpy as np
import os
import torch
import random
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss, binary_cross_entropy

import recipes.beat.eval.mireval_beat as mireval_beat
from recipes.beat.utils.augment_utils import time_augmentation
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs, merge_multiple_temporal_probs_inference
from recipes.beat.models.get_models import get_teacher
from recipes.beat.pl_modules.pl_module import LitBeat


class LitVocalBeat(LitBeat):
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
        max_pad_second=5,
        do_time_augmentation=False,
        do_cqt_augmentation=False,
        do_tempo_loss=True,
        f_measure_threshold=0.07,
        use_tempo_prior=False,
        add_noise=False,
        hop_factor=1,
        tempo_dir = None,
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
            add_noise=add_noise,
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
        self._observation_lambda = observation_lambda
        self._correction = correction
        self._transition_lambda = transition_lambda
        self.reduce_sample = (
            int(self._max_pad_second / self._label_hop) if max_pad_second != 0 else 0
        )
        self._tempo_dir = tempo_dir

        self._add_noise = add_noise

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
        sample = sample.unfold(1, self._train_num_samples + pad_sample * 2, self._train_num_samples / self._hop_factor)

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
        metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics["val_summary"] = np.array(list(metrics.values())).mean()

        self.log_dict(metrics, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(metrics)

    def _eval_beat(
        self, beat_pre, beats, tempo_pre, tempo, n_beats, sample_len, sample_hop
    ):
        out = defaultdict(dict)

        beat_pre = (
            beat_pre[0][:, self.reduce_sample : -self.reduce_sample]
            if self.reduce_sample != 0
            else beat_pre[0]
        )
        tempo_pre = tempo_pre[0]

        # non-beat
        non_beat = 1 - beats.sum(-1).unsqueeze(-1)
        non_beat[non_beat < 0] = 0
        beat_labels = torch.cat((beats, non_beat), -1)
        beat_labels = beat_labels.squeeze(0)
        beat_pre = merge_multiple_temporal_probs(
            torch.softmax(beat_pre, -1), beat_labels, sample_len, sample_hop
        )
        min_length = min(beat_pre.shape[0], beat_labels.shape[0])
        beat_labels = beat_labels[:min_length]
        beat_preds = beat_pre[:min_length]

        loss = binary_cross_entropy_with_logits(
            beat_preds, beat_labels[..., :n_beats].float()
        )
        out["pred"]["beat_probs"] = beat_preds[..., :2].cpu().numpy()
        out["pred"]["tempo"] = tempo_pre.cpu().numpy()
        out["truth"]["beat_labels"] = beat_labels.cpu().numpy()
        out["truth"]["tempo"] = tempo.cpu().numpy()

        return out, loss

    def _post_process(self, fps, prediction, min_bpm=50.0, max_bpm=215.0, key=None):

        if key is None or self._tempo_dir is None or not os.path.exists(os.path.join(self._tempo_dir, key+'.txt')):
            _min_bpm, _max_bpm = min_bpm, max_bpm
        else:
            with open(os.path.join(self._tempo_dir, key+'.txt')) as f:
                tempo = float(f.readline().strip())
                _min_bpm = tempo * 0.7
                _max_bpm = tempo * 1.3

        return madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[4],
            observation_lambda=self._observation_lambda,
            transition_lambda=self._transition_lambda,
            correct=self._correction,
            fps=fps,
            min_bpm=_min_bpm,
            max_bpm=_max_bpm,
        )(prediction)


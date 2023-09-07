import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs
from recipes.chord.utils.mireval_chord import get_chord_score
from recipes.mir_benchmark.pl_modules.beat_pl import BaseLightningModule


class LitChord(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        sample_rate,
        sample_len,
        label_hop,
        hop_factor=6,
        max_batch=100,
        do_cqt_augmentation=False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._max_batch = max_batch
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._label_hop = label_hop
        self._hop_factor = hop_factor
        self._do_cqt_augmentation = do_cqt_augmentation

    def training_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        inputs["audio"] = batch[0]
        inputs["aug_hop_size"] = self._hop_length

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]

        if self._do_cqt_augmentation:
            self.model.stages[0], note_idx = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1
            )

            data["chord_root"][..., 1:] = torch.roll(
                data["chord_root"][..., 1:], note_idx * -1, -1
            )
            data["chord_note"][..., :12] = torch.roll(
                data["chord_note"][..., :12], note_idx * -1, -1
            )
            data["chord_note"][..., 12:24] = torch.roll(
                data["chord_note"][..., 12:24], note_idx * -1, -1
            )

        # model prediction
        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)
        loss = self._train_chord(outputs, data)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        if self._do_cqt_augmentation:
            self.model.stages[0], _ = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1, note="C1"
            )

        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(
            1, self._train_num_samples, int(self._train_num_samples / self._hop_factor)
        )

        inputs["audio"] = sample.squeeze(0)
        inputs["aug_hop_size"] = self._hop_length

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]

        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)

        processed_data = self._postprocess_chord(
            data,
            outputs,
            int(self._sample_len / self._label_hop),
            int(self._sample_len / self._hop_factor / self._label_hop),
        )

        metrics = get_chord_score(
            processed_data["pred"], processed_data["truth"], self._label_hop
        )

        # Filter None
        dataset_name = batch[5][0].split("_")[0] + "_"
        partial_metrics = {
            dataset_name + k: v
            for k, v in metrics.items()
            if (v is not None) and (k in ["root", "major_minor"])
        }
        overall_metrics = {
            k: v
            for k, v in metrics.items()
            if (v is not None) and (k in ["root", "major_minor"])
        }

        self.log_dict(partial_metrics, prog_bar=True, batch_size=1, sync_dist=True)
        self.log_dict(overall_metrics, prog_bar=True, batch_size=1, sync_dist=True)

    def on_validation_epoch_end(self):
        print(" ")

    def _train_chord(self, prediction, target, crf=None):
        chord_loss = 0

        eval_name = ["chord_root", "chord_triad"]
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            chord_index = chord.sum(1).sum(1) > 0
            if len(chord_pre[chord_index, :]) > 0:
                min_length = min(chord_pre.shape[1], chord.shape[1])
                chord_pre = chord_pre[:, :min_length]
                chord = chord[:, :min_length]
                ignore = target["chord_ignore"][:, :min_length]
                weight = (1 - ignore).unsqueeze(-1).repeat(1, 1, chord.shape[-1])

                if name == "chord_root":
                    note_weight = torch.ones(chord.shape).to(chord.device)
                    note_weight[..., 1:] += target["chord_note"]
                    weight *= note_weight

                loss = binary_cross_entropy_with_logits(chord_pre, chord, weight)
                chord_loss += loss

        return chord_loss

    def _postprocess_chord(self, target, prediction, sample_len, sample_hop):
        eval_name = ["chord_root", "chord_triad"]
        processed_data = defaultdict(dict)
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            chord = chord.squeeze()
            chord_pre = merge_multiple_temporal_probs(
                chord_pre, chord, sample_len, sample_hop
            )
            chord_pre = torch.softmax(chord_pre, -1)
            min_length = min(chord_pre.shape[0], chord.shape[0])
            chord_labels = chord[:min_length]
            chord_preds = chord_pre[:min_length]
            processed_data["truth"][name] = chord_labels.cpu().numpy()
            processed_data["pred"][name] = chord_preds.cpu().numpy()
        processed_data["truth"]["chord_ignore"] = (
            target["chord_ignore"].squeeze(0).cpu().numpy()
        )

        return processed_data

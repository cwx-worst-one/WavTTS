import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.pl_modules.pl_module import BaseLightningModule
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs
from recipes.chord.utils.mireval_chord import get_chord_score


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

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]

        # pitch shift
        pitch_shift = random.choice(range(-3, 4))
        data["chord_root"][..., 1:] = torch.roll(
            data["chord_root"][..., 1:], -pitch_shift, -1
        )
        data["chord_note"][..., :12] = torch.roll(
            data["chord_note"][..., :12], -pitch_shift, -1
        )
        data["chord_note"][..., 12:24] = torch.roll(
            data["chord_note"][..., 12:24], -pitch_shift, -1
        )

        # model prediction
        inputs["pitch_shift"] = pitch_shift
        outputs["chord_root"], outputs["chord_triad"] = self.model(inputs)
        loss = self._train_chord(outputs, data)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}
        data = {}

        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(
            1, self._train_num_samples, int(self._train_num_samples / self._hop_factor)
        )

        data["chord_root"] = batch[1]
        data["chord_triad"] = batch[2]
        data["chord_note"] = batch[3]
        data["chord_ignore"] = batch[4]
        data["dataset.txt"] = batch[5]

        sample = sample.squeeze(0).float()
        num_iter, rem = divmod(len(sample), self._max_batch)
        if rem > 0:
            num_iter += 1
        root_prd, triad_prd = [], []
        for i in range(num_iter):
            inputs["audio"] = sample[i * self._max_batch : (i + 1) * self._max_batch]
            inputs["pitch_shift"] = 0
            root, triad = self.model(inputs)
            root_prd.append(root)
            triad_prd.append(triad)
        outputs["chord_root"] = torch.cat(root_prd)
        outputs["chord_triad"] = torch.cat(triad_prd)

        processed_data = self._postprocess_chord(
            data,
            outputs,
            int(self._sample_len / self._label_hop),
            int(self._sample_len / self._hop_factor / self._label_hop),
        )

        scores = get_chord_score(
            processed_data["pred"], processed_data["truth"], self._label_hop
        )

        # Filter None
        dataset_name = data["dataset.txt"][0].split("_")[0] + "_"
        partial_scores = {
            dataset_name + k: v for k, v in scores.items() if v is not None
        }
        overall_scores = {k: v for k, v in scores.items() if v is not None}

        self.log_dict(partial_scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.log_dict(overall_scores, prog_bar=True, batch_size=1, sync_dist=True)

    def on_validation_epoch_end(self):
        print(" ")

    def _train_chord(self, prediction, target, crf=None):
        chord_loss = 0

        eval_name = ["chord_root", "chord_triad"]
        for name in eval_name:
            chord_pre, chord = prediction[name], target[name]
            chord_index = chord.sum(1).sum(1) > 0
            if len(chord_pre[chord_index, :]) > 0:
                chord = torch.nn.functional.interpolate(
                    chord.unsqueeze(1),
                    (
                        int(
                            np.round(
                                chord.shape[1] * self._hop_length / self._hop_length
                            )
                        ),
                        chord.shape[2],
                    ),
                ).squeeze(1)
                min_length = min(chord_pre.shape[1], chord.shape[1])
                chord_pre = chord_pre[:, :min_length]
                chord = chord[:, :min_length]
                ignore = target["chord_ignore"][:, :min_length]
                chord_index = chord.sum(1).sum(1) > 0
                if len(chord_pre[chord_index, :]) > 0:
                    weight = (
                        (1 - ignore[chord_index])
                        .unsqueeze(-1)
                        .repeat(1, 1, chord.shape[-1])
                    )
                    if name == "chord_root":
                        note_weight = torch.ones(chord.shape).to(chord.device)
                        note_weight[..., 1:] += target["chord_note"]
                        weight *= note_weight[chord_index, :]

                    loss = binary_cross_entropy_with_logits(
                        chord_pre[chord_index, :], chord[chord_index, :], weight
                    )
                    chord_loss += loss

        if crf:
            tar_root = torch.argmax(target["chord_root"], -1)
            tar_triad = torch.argmax(target["chord_triad"], -1)
            pre_root = torch.argmax(prediction["chord_root"], -1)
            pre_triad = torch.argmax(prediction["chord_triad"], -1)
            ignore[tar_triad > 2] = 1
            tar_triad[tar_triad > 2] = 0
            pre_triad[pre_triad > 2] = 0
            tar_position = torch.nn.functional.relu(tar_root - 1) * 2 + tar_triad
            pre_position = torch.nn.functional.relu(pre_root - 1) * 2 + pre_triad
            tar_position[ignore == 1] = 25
            pre_position[ignore == 1] = 25
            pre_position = torch.nn.functional.one_hot(pre_position, num_classes=26)
            prediction, loss = crf(pre_position, tar_position)
        if chord_loss == 0:
            warnings.warn("There is no chord label.")
            chord_loss = torch.tensor(0.0, requires_grad=True)
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

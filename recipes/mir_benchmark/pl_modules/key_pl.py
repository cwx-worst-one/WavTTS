import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs
from recipes.key_detection.constants import (
    KEY_MAJMIN_NAME,
    KEYMODE_MAJMIN_ID,
    VOCAL_KEYMODE_MAJMIN_ID,
)
from recipes.key_detection.pl_modules.pl_module import get_key_score
from recipes.mir_benchmark.pl_modules.beat_pl import BaseLightningModule


class LitKey(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        label_hop,
        sample_rate,
        sample_len,
        hop_factor=6,
        model_batch_size=16,
        key_map="KEYMODE_MAJMIN_ID",
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._label_key_hop = label_hop
        self._hop_factor = hop_factor
        self._key_map = globals()[key_map]
        self._batch_size = model_batch_size
        assert self._key_map in [KEYMODE_MAJMIN_ID, VOCAL_KEYMODE_MAJMIN_ID]

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]

        # model prediction
        key_pred = self.model(inputs)[0]

        loss = self._train_key(key_pred, batch[1])

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        # prepare data
        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(
            1,
            int(self._sample_len * self._sample_rate),
            int(self._sample_len / self._hop_factor * self._sample_rate),
        ).squeeze(0)

        # model prediction
        key_preds = []
        for b in torch.split(sample, self._batch_size):
            inputs["audio"] = b
            key_pre = self.model(inputs)[0]
            key_preds.append(key_pre.detach().cpu())
        key_preds = torch.cat(key_preds, 0)

        key_pre, key_tar = self._eval_key(
            key_preds,
            batch[1].squeeze(0).detach().cpu(),
            int(self._sample_len / self._label_key_hop),
            int(self._sample_len / self._hop_factor / self._label_key_hop),
        )

        scores, _, _ = get_key_score(
            {"keysig_pred_probs": key_pre, "keysig_truth_labels": key_tar},
            self._key_map,
        )

        # Filter None
        scores = {k: v for k, v in scores.items() if v is not None}
        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(scores)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def _train_key(self, prediction, target):
        keymode_loss = 0
        min_length = min(prediction.shape[1], target.shape[1])
        keymode_loss = binary_cross_entropy_with_logits(
            prediction[:, :min_length], target[:, :min_length]
        )

        return keymode_loss

    def _eval_key(self, key_pred, key_label, sample_len, sample_hop):

        key_pre = torch.nn.functional.softmax(
            merge_multiple_temporal_probs(key_pred, key_label, sample_len, sample_hop),
            -1,
        )
        min_length = min(key_pre.shape[0], key_label.shape[0])
        key_pre = key_pre[:min_length].cpu().numpy()
        key_tar = key_label[:min_length].cpu().numpy()

        return key_pre, key_tar

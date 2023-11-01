import os

import numpy as np
import torch
import subprocess
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from recipes.beat.pl_modules.pl_module import BaseLightningModule
from recipes.structure.utils.eval_structure import (
    eval_a_song,
    merge_multiple_temporal_probs,
    post_process,
)
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs_inference


class LitStructure(BaseLightningModule):
    def __init__(
        self,
        model,
        lr=0.0005,
        scheduler_patience=10,
        scheduler_decay_factor=0.9,
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
        model_batch_size=48,
        pretrain_path=None
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )

        self._hop_length = n_fft // 2
        self._enable_short = enable_short
        self._sampling_rate = sampling_rate
        self._sample_hop = sample_hop
        self._sample_len = sample_len
        self._label_hop = label_hop
        self._n_top_bound = n_top_bound
        self._alpha = alpha
        self._label_window_len = int(sample_len / label_hop)
        self._label_hop_len = self._sample_hop / self._label_hop
        self._audio_window_len = int(self._sample_len * self._sampling_rate)
        self._audio_hop_len = int(self._sample_hop * self._sampling_rate)
        self._n_boundary = n_boundary
        self._n_function = n_function
        self.model_batch_size = model_batch_size

        if os.path.exists("recipes/structure/conf/function_weight.npy"):
            function_weight = np.load("recipes/structure/conf/function_weight.npy")
            function_weight = torch.nan_to_num(
                torch.from_numpy(function_weight), 1, 1, 1
            )
        else:
            function_weight = None
        self.function_weight = function_weight

        if pretrain_path is not None:
            if pretrain_path.startswith("hdfs://"):
                name = pretrain_path.split("/")[-1]
                subprocess.run(f"hdfs dfs -get {pretrain_path}", shell=True)
            else:
                name = pretrain_path
            checkpoint = torch.load(name)
        
            for key in list(checkpoint["state_dict"]):
                if "teacher" in key:
                    checkpoint["state_dict"].pop(key)
                else:
                    checkpoint["state_dict"][key[6:]] = checkpoint["state_dict"].pop(key)

            model_dict = self.model.state_dict()
            model_dict.update(checkpoint["state_dict"]) 
            self.model.load_state_dict(model_dict)

    def training_step(self, batch, batch_idx):
        sample, boundary_tar, function_tar, chorus_only = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

        boundary_pre, function_pre = self.model(
            {"audio": sample, "aug_hop_size": self._hop_length}
        )

        boundary_loss, function_loss = 0, 0
        if chorus_only.sum() > 0:
            chorus_only_weight = chorus_only.sum() / len(boundary_pre)
            boundary_loss += chorus_only_weight * self.wbce_loss(
                boundary_pre[chorus_only, :, 1].unsqueeze(-1), boundary_tar[chorus_only, :, 1].unsqueeze(-1)
            )
            function_loss += chorus_only_weight * self.wbce_loss(
                function_pre[chorus_only, :, 1], function_tar[chorus_only, :, 1]
            )

        if (~chorus_only).sum() > 0:
            non_chorus_only_weight = (~chorus_only).sum() / len(boundary_pre)
            nc_boundary_pre = boundary_pre[~chorus_only, :]
            nc_boundary_tar = boundary_tar[~chorus_only, :]
            boundary_loss += non_chorus_only_weight * self.wbce_loss(
                nc_boundary_pre, nc_boundary_tar
            )
            function_loss += non_chorus_only_weight * self.wbce_loss(
                function_pre[~chorus_only, :],
                function_tar[~chorus_only, :],
                self.function_weight,
                short=self._enable_short,
            )

        loss = self._alpha * boundary_loss + (1 - self._alpha) * function_loss
    
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        (
            sample,
            boundary_tar,
            function_tar,
            _,
            boundary_interval,
            chorus_interval,
            segment_type,
            key,
        ) = (
            batch[0],
            batch[1][0],
            batch[2][0],
            batch[3],
            batch[4][0],
            batch[5][0],
            batch[6],
            batch[7][0],
        )
        n_samples = max(
            int(
                np.ceil(
                    (sample.shape[1] - self._audio_window_len) / self._audio_hop_len
                )
            ),
            0,
        )
        sample = torch.nn.functional.pad(
            sample,
            (
                0,
                int(
                    self._audio_hop_len * n_samples
                    + self._audio_window_len
                    - sample.shape[1]
                ),
            ),
        )
        sample = sample.unfold(1, self._audio_window_len, self._audio_hop_len).squeeze(
            0
        )
        boundary_pre, function_pre = self.model(
            {"audio": sample, "aug_hop_size": self._hop_length}
        )

        function_tar = function_tar.detach().cpu().numpy()
        boundary_tar = boundary_tar.detach().cpu().numpy()

        boundary_pre = torch.sigmoid(boundary_pre).detach().cpu().numpy()
        function_pre = torch.sigmoid(function_pre).detach().cpu().numpy()

        boundary_pre = merge_multiple_temporal_probs(boundary_pre, self._label_hop_len)
        function_pre = merge_multiple_temporal_probs(function_pre, self._label_hop_len)

        min_length = min(boundary_tar.shape[0], boundary_pre.shape[0])
        boundary_pre, function_pre, boundary_tar, function_tar = (
            boundary_pre[:min_length],
            function_pre[:min_length],
            boundary_tar[:min_length],
            function_tar[:min_length],
        )

        scores = eval_a_song(
            boundary_pre,
            function_pre,
            boundary_interval,
            function_tar,
            chorus_interval,
            segment_type,
            self._n_top_bound,
            self._label_hop,
            key,
        )
        scores = {k: v for k, v in scores.items() if v is not None}
        self.log_dict(scores, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(scores)
        return scores

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if type(batch) == dict:
            sample = batch["target_audio"]
        else:
            sample = batch[0]
        bsz = sample.shape[0]
        # Remove channel, assume mono
        if sample.ndim == 3:
            sample = sample.squeeze(1)

        duration = sample.shape[-1] / self._sampling_rate

        n_samples = max(
            int(
                np.ceil(
                    (sample.shape[-1] - self._audio_window_len) / self._audio_hop_len
                )
            ),
            0,
        )
        sample = torch.nn.functional.pad(
            sample,
            (
                0,
                int(
                    self._audio_hop_len * n_samples
                    + self._audio_window_len
                    - sample.shape[1]
                ),
            ),
        )
        # (bsz, num_segments, seg_len)
        sample = sample.unfold(-1, self._audio_window_len, self._audio_hop_len)
        num_segments = sample.shape[1]
        # (bsz * num_segments, seg_len)
        sample = sample.reshape(bsz * num_segments, -1).float()
        batch_boundary, batch_function = [], []
        for s in torch.split(sample, self.model_batch_size):
            b, f = self.model(
                {"audio": s, "aug_hop_size": self._hop_length}
            )

            b = torch.sigmoid(b)
            f = torch.sigmoid(f)
            batch_boundary.append(b)
            batch_function.append(f)
        # (bsz * num_segments, num_frames, output_dim)
        batch_boundary = torch.cat(batch_boundary, dim=0)
        batch_function = torch.cat(batch_function, dim=0)
        # (bsz, num_segments, num_frames, output_dim)
        batch_boundary = batch_boundary.float().reshape(
            bsz, num_segments, batch_boundary.shape[1], -1
        ).detach().cpu().numpy()
        batch_function = batch_function.float().reshape(
            bsz, num_segments, batch_function.shape[1], -1
        ).detach().cpu().numpy()

        batch_raw_segments = []
        for boundary, function in zip(batch_boundary, batch_function):
            boundary = merge_multiple_temporal_probs(boundary, self._label_hop_len)
            # 90M structure model has only 1 output dim for boundary
            if boundary.ndim == 1:
                boundary = boundary[..., np.newaxis]
            function = merge_multiple_temporal_probs(function, self._label_hop_len)
            pred_labels, pred_choruses, segments, choruses, raw_segments = post_process(
                boundary, function, self._n_top_bound, self._label_hop
            )
            batch_raw_segments.append(raw_segments)
        
        return batch_raw_segments

    def wbce_loss(self, prediction, target, weight=None, short=False):
        if weight is not None:
            weight = target * weight.reshape(1, 1, weight.shape[0]) + 1

        if short:
            unmask_idx = target.sum(-1) >= 0
            prediction = prediction[unmask_idx]
            target = target[unmask_idx]
            if weight is not None:
                weight = weight[unmask_idx]

        return binary_cross_entropy_with_logits(prediction, target, weight)

    def configure_optimizers(self):
        # Config optimizer and scheduler
        optimizer = optim.Adam(
            list(self.model.parameters()), lr=self._lr, weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._scheduler_patience,
            factor=self._scheduler_decay_factor,
            verbose=True,
            mode="max",
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "summary"}

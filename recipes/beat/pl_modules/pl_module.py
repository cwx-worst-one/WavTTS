import random
from collections import defaultdict
from typing import Any

try:
    import madmom
except:
    print("[WARNING] Failed to import madmom")
import subprocess
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

import recipes.beat.eval.mireval_beat as mireval_beat
from recipes.beat.models.networks import initialize_filterbank
from recipes.beat.utils.augment_utils import time_augmentation
from recipes.beat.utils.eval_utils import merge_multiple_temporal_probs, merge_multiple_temporal_probs_inference


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._lr = lr
        self._scheduler_patience = scheduler_patience
        self._scheduler_decay_factor = scheduler_decay_factor
        self.validation_step_outputs = []

    def _cqt_augmentation(
        self, model, sample_rate, n_harmonic=6, semitone_scale=2, note=None
    ):
        note_idx = 0
        if note is None:
            note_list = ["Bb0", "B0", "C1", "C#1", "D1"]
            index_list = [-2, -1, 0, 1, 2]
            note_idx = random.choice(index_list)
            note = note_list[note_idx - index_list[0]]
        harmonic_hz, level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale, note
        )
        model.preprocess.hstft.level = level
        model.preprocess.hstft.f0 = torch.tensor(
            harmonic_hz.astype("float32"), device=self.device
        )
        return model, note_idx

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
            mode="min",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def on_validation_epoch_end(self):
        scores = {}
        for metric in self.validation_step_outputs:
            for k in metric.keys():
                if k not in scores:
                    scores[k] = []
                scores[k].append(metric[k])
        for k in scores.keys():
            print(k, round(np.array(scores[k]).mean(), 3))
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

class LitBeat(BaseLightningModule):
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
        beat_window_length=2,
        n_beats=3,
        n_tempo=300,
        do_time_augmentation=False,
        do_cqt_augmentation=False,
        do_tempo_loss=True,
        f_measure_threshold=0.07,
        transition_lambda=110,
        beats_per_bar=[3, 4],
        use_tempo_prior=False,
        hop_factor=3,
        max_pad_second=0,
        add_noise=False,
        pretrain_path=None
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._hop_length = hop_length
        self._beat_window_length = beat_window_length
        self._n_beats = n_beats
        self._n_tempo = n_tempo
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self._label_hop = label_hop
        self._hop_factor = hop_factor
        self._do_time_augmentation = do_time_augmentation
        self._do_cqt_augmentation = do_cqt_augmentation
        self._do_tempo_loss = do_tempo_loss
        self._f_measure_threshold = f_measure_threshold
        self._use_tempo_prior = use_tempo_prior
        self._add_noise = add_noise

        self.transition_lambda = transition_lambda
        self.beats_per_bar = beats_per_bar

        self._max_pad_second = max_pad_second
        self.reduce_sample = (
            int(self._max_pad_second / self._label_hop) if max_pad_second != 0 else 0
        )

        if pretrain_path is not None:
            name = pretrain_path.split("/")[-1]
            subprocess.run(f"hdfs dfs -get {pretrain_path}", shell=True)
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
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["beats"] = batch[1]
        inputs["tempo_label"] = batch[2]

        if self._do_time_augmentation:
            inputs["aug_hop_size"] = time_augmentation(self._hop_length)
        else:
            inputs["aug_hop_size"] = self._hop_length

        if self._do_cqt_augmentation:
            # TODO (jesse.s): Refactor model preprocessing into preprocess stages
            self.model.stages[0], _ = self._cqt_augmentation(
                self.model.stages[0], self._sample_rate, semitone_scale=1
            )

        # model prediction
        beat_pred, tempo_pred = self.model(inputs)

        loss = self._train_beat(
            inputs["beats"],
            self._hop_length,
            inputs["aug_hop_size"],
            beat_pred,
            self._beat_window_length,
            self._n_beats,
            self._n_tempo,
            tempo_pred,
            inputs["tempo_label"],
        )

        # Logging to TensorBoard by default
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        # TODO (jesse.s): Refactor model preprocessing into preprocess stages
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
        # hd5y loader will change to tensor
        if torch.is_tensor(orig_beats):
            orig_beats = orig_beats.cpu().numpy()

        inputs["aug_hop_size"] = self._hop_length

        # model prediction
        # input audio dim should not be 1
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
            batch[-1]
        )
        metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics["val_summary"] = np.array(list(metrics.values())).mean()

        self.log_dict(metrics, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append(metrics)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        audio = batch[0]
        key = batch[-1][0]
        duration = audio.shape[-1] / self._sample_rate

        pad_sample = int(self._sample_rate * self._max_pad_second)
        audio = torch.nn.functional.pad(audio, (pad_sample, pad_sample), mode='constant', value=0)
        audio = torch.nn.functional.pad(
            audio, (0, int(self._sample_rate * self._sample_len))
        )
        audio = audio.unfold(1, self._train_num_samples + pad_sample * 2, int(self._train_num_samples / self._hop_factor))
        inputs = {}
        inputs["audio"] = audio.squeeze(0).float()
        inputs["aug_hop_size"] = self._hop_length

        if self._add_noise:
            std = torch.std(inputs["audio"])
            noise_std = random.uniform(0.03 * std, 0.03 * std)
            noise = torch.distributions.normal.Normal(loc=0.0, scale=noise_std).sample(inputs["audio"].shape)
            inputs["audio"] = inputs["audio"] + noise.to(inputs["audio"].device)

        model_output = self.model(inputs)
        beat_pred, tempo_pred = model_output[0], model_output[1]
        beat_pred = beat_pred = (
            beat_pred[0][:, self.reduce_sample : -self.reduce_sample]
            if self.reduce_sample != 0
            else beat_pred[0]
        )
        
        beat_pre = merge_multiple_temporal_probs_inference(
            torch.softmax(beat_pred, -1), 
            3, 
            int(np.ceil(duration / self._label_hop)), 
            int(self._sample_len / self._label_hop), 
            int(self._sample_len / self._hop_factor / self._label_hop)
        ).cpu().numpy()
       
        beat_probs = np.clip(beat_pre, 1e-16, 0.9999)[..., :2]
        beat_pre = self._post_process(int(1 / self._label_hop), beat_probs, key=key)
        return beat_pre

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

    def _train_beat(
        self,
        beat,
        hop_length,
        aug_hop_size,
        beat_preds,
        beat_window_length,
        n_beats,
        n_tempo,
        tempo_preds=None,
        tempo_target=None,
    ):
        beat_loss = torch.zeros(1, device=self.device)

        # filter batches with no beat annotations
        if len(beat_preds[0]) > 0:
            beat = torch.nn.functional.interpolate(
                beat.unsqueeze(1),
                (
                    int(np.round(beat.shape[1] * hop_length / aug_hop_size)),
                    beat.shape[2],
                ),
            ).squeeze(1)
            weight = beat.clone()
            right_pad, left_pad = torch.zeros(weight.shape).float().to(
                beat.device
            ), torch.zeros(weight.shape).float().to(beat.device)
            for i in range(1, beat_window_length + 1):
                r = torch.roll(weight, i, 1)
                r[:, :i] = 0
                left = torch.roll(weight, -i, 1)
                left[:, -i:] = 0
                right_pad += r
                left_pad += left

            right_pad[right_pad > 1] = 1
            left_pad[left_pad > 1] = 1

            beat += right_pad + left_pad
            beat[beat > 1] = 1

            # non-beat
            non_beat = 1 - beat.sum(-1).unsqueeze(-1)
            non_beat[non_beat < 0] = 0
            beat = torch.cat((beat, non_beat), -1)

            weight = right_pad * 0.5 + left_pad * 0.5 + weight
            weight[weight > 1] = 1
            weight = weight * 5
            weight[..., 1] *= 5

            # non-beat
            weight = torch.cat((weight, beat[:, :, -1:]), -1)
            weight[weight == 0] = 1

            idx = beat[:, :, 1].sum(-1) == 0
            weight[idx, :, 1] = 0

            # get length
            loss_weight = [1]
            for beat_pre, l_weight in zip(beat_preds, loss_weight):
                min_length = min(beat_pre.shape[1], beat.shape[1])
                beat_loss += (
                    binary_cross_entropy_with_logits(
                        beat_pre[:, :min_length],
                        beat[:, :min_length, :n_beats],
                        weight[:, :min_length, :n_beats],
                    )
                    * l_weight
                )

            if self._do_tempo_loss:
                for tempo_pre, l_weight in zip(tempo_preds, loss_weight):
                    beat_loss += (
                        binary_cross_entropy_with_logits(
                            tempo_pre[tempo_target > 0],
                            torch.nn.functional.one_hot(
                                tempo_target[tempo_target > 0].long(),
                                num_classes=n_tempo,
                            ).float(),
                        )
                        * l_weight
                    ) * 2

        return beat_loss

    def _get_beat_score(
        self, beat_probs, beat_labels, tempo_labels, label_hop, key=None
    ):
        result = {}
        beat_probs = np.clip(beat_probs, 1e-16, 0.9999)

        # -inf log probability during Viterbi decoding
        # cannot find a valid path
        # if beat_probs has probability = 1.0,
        # it will cause being divided by log(1.0)=0.0
        beat_time = [frame[0] for frame in beat_labels]

        if all(v[1] == 0 for v in beat_labels):
            downbeat_time = None
        else:
            downbeat_time = [frame[0] for frame in beat_labels if frame[1] == 1]

        beat_res, pred_beat_times = self._beat_score(
            beat_probs, beat_time, downbeat_time, int(1 / label_hop), tempo_labels, key
        )
        if len(pred_beat_times) == 0:
            (
                result["Beat_f1"],
                result["Beat_AMLt"],
                result["Beat_CMLt"],
                result["Downbeat_f1"],
                result["Downbeat_AMLt"],
                result["Downbeat_CMLt"],
            ) = (None, None, None, None, None, None)
            result["Beat_Downbeat_f1"] = None
        else:
            (
                result["Beat_f1"],
                result["Beat_AMLt"],
                result["Beat_CMLt"],
                result["Downbeat_f1"],
                result["Downbeat_AMLt"],
                result["Downbeat_CMLt"],
            ) = beat_res
            if downbeat_time is None:
                result["Beat_Downbeat_f1"] = result["Beat_f1"]
            else:
                result["Beat_Downbeat_f1"] = (
                    result["Beat_f1"] + result["Downbeat_f1"]
                ) / 2
        out = {"beat_probs": beat_probs}
        return result, out, pred_beat_times

    def _beat_score(self, prediction, beat_times, downbeat_time, fps, tempo, key=None):
        if self._use_tempo_prior:
            interval = np.mean(np.diff(np.array(beat_times)))
            tempo = 60 / interval
            pred_beat_times = self._post_process(
                fps, prediction, min_bpm=int(tempo * 0.9), max_bpm=int(tempo * 1.1)
            )
        else:
            pred_beat_times = self._post_process(fps, prediction)
        Downbeat_f1, Downbeat_AMLt, Downbeat_CMLt = None, None, None

        reference_beats = np.array(
            beat_times
        )  # mir_eval.beat.trim_beats(np.array(beat_times))
        estimated_beats = pred_beat_times[
            :, 0
        ]  # mir_eval.beat.trim_beats(pred_beat_times[:, 0])
        Beat_f1 = mireval_beat.f_measure(
            reference_beats,
            estimated_beats,
            f_measure_threshold=self._f_measure_threshold,
        )
        _, Beat_CMLt, _, Beat_AMLt = mireval_beat.continuity(
            reference_beats, estimated_beats
        )

        if downbeat_time:
            pre_downbeat_time = np.array(
                [frame[0] for frame in pred_beat_times if frame[1] == 1]
            )
            reference_downbeats = np.array(downbeat_time)
            estimated_downbeats = np.array(pre_downbeat_time)
            Downbeat_f1 = mireval_beat.f_measure(
                reference_downbeats,
                estimated_downbeats,
                f_measure_threshold=self._f_measure_threshold,
            )
            if len(estimated_downbeats) < 2:
                Downbeat_AMLt, Downbeat_CMLt = 0.0, 0.0
            else:
                _, Downbeat_CMLt, _, Downbeat_AMLt = mireval_beat.continuity(
                    reference_downbeats, estimated_downbeats
                )

        return (
            Beat_f1,
            Beat_AMLt,
            Beat_CMLt,
            Downbeat_f1,
            Downbeat_AMLt,
            Downbeat_CMLt,
        ), pred_beat_times

    def _post_process(self, fps, prediction, min_bpm=55.0, max_bpm=215.0, key=None):
        return madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=self.beats_per_bar,
            observation_lambda=18,
            transition_lambda=self.transition_lambda,
            fps=fps,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )(prediction)

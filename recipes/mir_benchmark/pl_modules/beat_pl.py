import random
from collections import defaultdict
from typing import Any

import madmom
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

import recipes.beat.eval.mireval_beat as mireval_beat
from recipes.beat.models.networks import initialize_filterbank
from recipes.beat.utils.augment_utils import time_augmentation
from recipes.beat.utils.eval_utils import (
    merge_multiple_temporal_probs,
    merge_multiple_temporal_probs_inference_batch,
)


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        *args: Any,
        pretrained_path=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._lr = lr
        self._scheduler_patience = scheduler_patience
        self.validation_step_outputs = []
        self._scheduler_decay_factor = scheduler_decay_factor
        if pretrained_path is not None:
            self.load_from_pretrained(pretrained_path)

    def load_from_pretrained(self, pretrained_path):
        print(f"Loading pre-trained model from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))["state_dict"]
        model_state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
            elif k.endswith(".weight_g") or k.endswith(".weight_v"):
                # Handle older versions with weight_g and weight_v for weight_norm
                # https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/weight_norm.py
                if k.endswith(".weight_g"):
                    new_k = k.replace(".weight_g", f".parametrizations.weight.original0")
                else:
                    new_k = k.replace(".weight_v", f".parametrizations.weight.original1")
                if new_k in model_state_dict:
                    print(f"Changing {k} to {new_k}")
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]
                else:
                    print(f"Can't find {new_k} in model's state dict")
            else:
                print(f"Found extra item in pre-trained state dict: {k}")
        self.load_state_dict(state_dict, strict=True)

    def configure_optimizers(self):
        # Config optimizer and scheduler
        optimizer = optim.Adam([
            {"params": self.model.stages[0].parameters(), "lr": self._lr / 10},
            {"params": self.model.stages[1].parameters(), "lr": self._lr}
        ])

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

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

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


class LitBeat(BaseLightningModule):
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
        hop_factor=2,
        do_tempo_loss=False,
        f_measure_threshold=0.07,
        use_tempo_prior=False,
        model_batch_size=32,
        pretrained_path=None,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
            pretrained_path=pretrained_path,
        )
        self._hop_length = hop_length
        self._beat_window_length = beat_window_length
        self._n_beats = n_beats
        self._n_tempo = n_tempo
        self._sample_rate = sample_rate
        self._beat_sample_len = sample_len
        self._beat_label_hop = label_hop
        self._do_tempo_loss = do_tempo_loss
        self._f_measure_threshold = f_measure_threshold
        self._use_tempo_prior = use_tempo_prior
        self._beat_train_num_samples = int(self._sample_rate * self._beat_sample_len)
        self._beat_hop_factor = hop_factor
        self.model_batch_size = model_batch_size

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["beats"] = batch[1]
        inputs["tempo_label"] = batch[2]
        inputs["aug_hop_size"] = self._hop_length

        # model prediction
        output = self.model(inputs)[0]
        beat_pred, tempo_pred = output['beat']

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
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = self._beat_validation_step(batch)
        dataset_name = batch[4][0].split("_")[0] + "_"
        partial_metrics = {dataset_name + k: v for k, v in metrics.items()}
        overall_metrics = metrics
        overall_metrics["val_summary"] = np.array(list(overall_metrics.values())).mean()

        self.log_dict(partial_metrics, prog_bar=True, batch_size=1, sync_dist=True)
        self.log_dict(overall_metrics, prog_bar=True, batch_size=1, sync_dist=True)

        self.validation_step_outputs.append(overall_metrics)

    def _beat_validation_step(self, batch):
        inputs = {}

        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._beat_sample_len))
        )
        sample = sample.unfold(
                1, 
                int(self._sample_rate * self._beat_sample_len),
                int(self._sample_rate * self._beat_sample_len / self._beat_hop_factor)
                )

        max_batch = 16
        
        inputs["beats"] = batch[1]
        inputs["tempo_label"] = batch[2]
        inputs["aug_hop_size"] = self._hop_length
        

        # model prediction
        # input audio dim should not be 1
        beat_pred, tempo_pred = [], []
        for b in torch.split(sample.squeeze(0), 16):
            inputs["audio"] = b
            output = self.model(inputs)[0]
            _beat, _tempo = output['beat']
            beat_pred.append(_beat)
            tempo_pred.append(_tempo)
        beat_pred = torch.cat(beat_pred, 0)
        tempo_pred = torch.cat(tempo_pred, 0)

        preds, _ = self._eval_beat(
            beat_pred,
            inputs["beats"],
            tempo_pred,
            inputs["tempo_label"],
            self._n_beats,
            int(self._beat_sample_len / self._beat_label_hop),
            int(self._beat_sample_len / self._beat_label_hop / self._beat_hop_factor),
        )

        orig_beats = batch[3][0]
        if torch.is_tensor(orig_beats):
            orig_beats = orig_beats.cpu().numpy()
        metrics, _, _ = self._get_beat_score(
            preds["pred"]["beat_probs"],
            orig_beats,
            preds["truth"]["tempo"],
            self._beat_label_hop,
        )

        metrics = {k: v for k, v in metrics.items() if (k in ["Beat_f1", "Downbeat_f1"])}

        return metrics

    def _beat_test_step(self, batch):
        return self._beat_validation_step(batch)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def _eval_beat(
        self, beat_pre, beats, tempo_pre, tempo, n_beats, sample_len, sample_hop
    ):
        out = defaultdict(dict)

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
            beat_preds, beat_labels[..., :n_beats]
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
        beat_loss = 0
        if tempo_target is not None:
            tempo_target = torch.clamp(tempo_target, min=0, max=299)

        # filter batches with no beat annotations
        if len(beat_preds) > 0:
            beat = torch.nn.functional.interpolate(
                beat.unsqueeze(1),
                (
                    #int(np.round(beat.shape[1] * hop_length / aug_hop_size)),
                    beat_preds.shape[1],
                    beat.shape[2],
                ),
            ).squeeze(1)
            weight = beat.clone()
            right_pad, left_pad = torch.zeros(weight.shape).to(
                beat.device
            ), torch.zeros(weight.shape).to(beat.device)
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
            min_length = min(beat_preds.shape[1], beat.shape[1])
            beat_loss += binary_cross_entropy_with_logits(
                            beat_preds[:, :min_length],
                            beat[:, :min_length, :n_beats],
                            weight[:, :min_length, :n_beats],
                        )
            
                

            if self._do_tempo_loss:
                beat_loss += binary_cross_entropy_with_logits(
                                tempo_preds[tempo_target > 0],
                                torch.nn.functional.one_hot(
                                    tempo_target[tempo_target > 0].float().long(),
                                    num_classes=n_tempo,
                                )
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

        beat_probs = np.nan_to_num(beat_probs)
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
            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            result["Beat_Downbeat_f1"] = 0.0
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

        if type(beat_times[0]) == torch.Tensor:
            beat_times = [b.cpu().item() for b in beat_times]
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
            if type(downbeat_time[0]) == torch.Tensor:
                downbeat_time = [b.cpu().item() for b in downbeat_time]
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

    def _post_process(self, fps, prediction, min_bpm=55.0, max_bpm=215.0):
        try:
            return madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4],
            observation_lambda=16,
            transition_lambda=110,
            fps=fps,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )(prediction)
        except IndexError as e:
            print("no beat")
            return np.empty((0, 2))

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        if type(batch) == dict:
            audio = batch["target_audio"]
        else:
            audio = batch[0]
        # Remove channel, assume mono
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        assert audio.ndim == 2, f"Expect (bsz, seq_len), got {audio.shape}"
        duration = audio.shape[-1] / self._sample_rate

        audio = torch.nn.functional.pad(
            audio, (0, int(self._sample_rate * self._beat_sample_len))
        )
        # (bsz, num_segments, seg_len)
        audio = audio.unfold(1, self._beat_train_num_samples, int(self._beat_train_num_samples / self._beat_hop_factor))
        bsz, num_segments, _ = audio.shape
        # (bsz * num_segments, seg_len)
        audio = audio.reshape(bsz * num_segments, -1).float()

        inputs = {}
        inputs["aug_hop_size"] = self._hop_length

        beat_pred, tempo_pred = [], []
        for b in torch.split(audio, self.model_batch_size):
            inputs["audio"] = b
            output = self.model(inputs)[0]
            _beat, _tempo = output['beat']
            beat_pred.append(_beat)
            tempo_pred.append(_tempo)
        beat_pred = torch.cat(beat_pred, 0)
        # (bsz, num_segments, seg_len, output_dim)
        beat_pred = beat_pred.reshape(bsz, num_segments, -1, beat_pred.shape[-1])
        tempo_pred = torch.cat(tempo_pred, 0)
        
        beat_pre = merge_multiple_temporal_probs_inference_batch(
            torch.softmax(beat_pred, -1), 
            3, 
            int(np.ceil(duration / self._beat_label_hop)), 
            int(self._beat_sample_len / self._beat_label_hop), 
            int(self._beat_sample_len / self._beat_hop_factor / self._beat_label_hop)
        )
       
        beat_probs = torch.clamp(beat_pre, min=1e-16, max=0.9999)[..., :2]
        beat_probs = beat_probs.cpu().numpy()
        beat_pre = [
            self._post_process(int(1 / self._beat_label_hop), beat_probs[i])
            for i in range(bsz)
        ]
        return beat_pre


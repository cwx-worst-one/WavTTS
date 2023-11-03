from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

import lib.mir.mir.beat.mireval_beat as mireval_beat
from mir_tools.beat.utils import merge_multiple_temporal_probs
from sam.data.mir.beat import BeatDataResult
from sam.models.base import LossDict


def loss(
    beat: torch.Tensor,
    hop_length: int,
    aug_hop_size: int,
    beat_preds: List[torch.Tensor],
    beat_window_length,
    n_beats,
    n_tempo,
    do_tempo_loss: bool,
    tempo_preds: Optional[List[torch.Tensor]] = None,
    tempo_target: Optional[torch.Tensor] = None,
) -> LossDict:
    # TODO this requires tests and a refactoring

    beat_loss = torch.zeros(1, device=beat.device)
    if tempo_target is not None:
        tempo_target = torch.clamp(tempo_target, min=0, max=299)

    # filter batches with no beat annotations
    if len(beat_preds[0]) > 0:
        beat = torch.nn.functional.interpolate(
            beat.unsqueeze(1),
            (
                # int(np.round(beat.shape[1] * hop_length / aug_hop_size)), # TODO review logic
                beat_preds[0].shape[1],
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
                F.binary_cross_entropy_with_logits(
                    beat_pre[:, :min_length],
                    beat[:, :min_length, :n_beats],
                    weight[:, :min_length, :n_beats],
                )
                * l_weight
            )

        if do_tempo_loss:
            if tempo_preds is None or tempo_target is None:
                raise Exception(
                    "`tempo_preds` and `tempo_target` need to be defined in order to compute tempo loss"
                )

            for tempo_pre, l_weight in zip(tempo_preds, loss_weight):
                beat_loss += (
                    F.binary_cross_entropy_with_logits(
                        tempo_pre[tempo_target > 0],
                        torch.nn.functional.one_hot(
                            tempo_target[tempo_target > 0].long(),
                            num_classes=n_tempo,
                        ).float(),
                    )
                    * l_weight
                ) * 2

    return {"loss": beat_loss}


def val_test_step(
    beat_logits,
    tempo_logits,
    beat_labels,
    tempo_labels,
    orig_beats,
    n_beats,
    duration,
    label_hop,
    dataset_name: str,
    use_tempo_prior: bool,
):
    # TODO refactor
    beat_preds = [beat_logits]
    tempo_preds = [tempo_logits]

    preds, _ = eval_beat(
        beat_preds[0],  # TODO list type inherited from sami_ai_models
        beat_labels,
        tempo_preds[0],  # TODO list type inherited from sami_ai_models
        tempo_labels,
        n_beats,
        int(duration / label_hop),
        int(duration / label_hop),
    )

    # TODO: only 1 batch size is supported
    orig_beats = orig_beats[0]
    if torch.is_tensor(orig_beats):
        orig_beats = orig_beats.cpu().numpy()

    metrics, _, _ = get_beat_score(
        preds["pred"]["beat_probs"],
        orig_beats,
        preds["truth"]["tempo"],
        label_hop,
        use_tempo_prior,
    )
    dataset_name = dataset_name.split("_")[0] + "_"
    partial_metrics = {
        dataset_name + k: v
        for k, v in metrics.items()
        if (v is not None) and (k in ["Beat_f1", "Downbeat_f1"])
    }
    overall_metrics = {
        k: v
        for k, v in metrics.items()
        if (v is not None) and (k in ["Beat_f1", "Downbeat_f1"])
    }
    overall_metrics["val_summary"] = torch.tensor(list(overall_metrics.values())).mean()
    return partial_metrics, overall_metrics


def eval_beat(beat_pre, beats, tempo_pre, tempo, n_beats, sample_len, sample_hop):
    # TODO this requires tests and a refactoring
    out = defaultdict(dict)

    # beat_pre = beat_pre[0]
    # tempo_pre = tempo_pre[0]

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

    loss = F.binary_cross_entropy_with_logits(
        beat_preds, beat_labels[..., :n_beats].float()
    )
    out["pred"]["beat_probs"] = beat_preds[..., :2].cpu().numpy()
    out["pred"]["tempo"] = tempo_pre.cpu().numpy()
    out["truth"]["beat_labels"] = beat_labels.cpu().numpy()
    out["truth"]["tempo"] = tempo.cpu().numpy()
    return out, loss


def get_beat_score(
    beat_probs, beat_labels, tempo_labels, label_hop, use_tempo_prior: bool, key=None
):
    # TODO this requires tests and a refactoring
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
    beat_res, pred_beat_times = beat_score(
        beat_probs,
        beat_time,
        downbeat_time,
        int(1 / label_hop),
        tempo_labels,
        use_tempo_prior,
        key,
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
            result["Beat_Downbeat_f1"] = (result["Beat_f1"] + result["Downbeat_f1"]) / 2
    out = {"beat_probs": beat_probs}
    return result, out, pred_beat_times


def beat_score(
    prediction,
    beat_times,
    downbeat_time,
    fps,
    tempo,
    f_measure_threshold: float,
    use_tempo_prior: bool,
    key=None,
):
    # TODO this requires tests and a refactoring
    if use_tempo_prior:
        interval = np.mean(np.diff(np.array(beat_times)))
        tempo = 60 / interval
        pred_beat_times = post_process(
            fps, prediction, min_bpm=int(tempo * 0.9), max_bpm=int(tempo * 1.1)
        )
    else:
        pred_beat_times = post_process(fps, prediction)
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
        f_measure_threshold=f_measure_threshold,
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
            f_measure_threshold=f_measure_threshold,
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


def post_process(fps, prediction, min_bpm=55.0, max_bpm=215.0):
    # TODO requires tests
    try:
        return DBNDownBeatTrackingProcessor(
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

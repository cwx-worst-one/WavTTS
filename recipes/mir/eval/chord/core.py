from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from mir_tools.beat.utils import merge_multiple_temporal_probs
from mir_tools.chord.mireval_chord import get_chord_score
from sam.data.mir.chord import ChordDataResult
from sam.models.base import LossDict


def chord_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    chord_note: torch.Tensor,
    chord_ignore: torch.Tensor,
    chord_root: bool,
) -> torch.Tensor:
    # TODO: tests
    loss = torch.tensor(0.0)
    chord_index = targets.sum(1).sum(1) > 0
    if len(preds[chord_index, :]) > 0:
        min_length = min(preds.shape[1], targets.shape[1])
        preds = preds[:, :min_length]
        targets = targets[:, :min_length]
        ignore = chord_ignore[:, :min_length]
        weight = (1 - ignore).unsqueeze(-1).repeat(1, 1, targets.shape[-1])

        if chord_root:
            note_weight = torch.ones(targets.shape).to(targets.device)

            chord_note = chord_note[:, :min_length]  # TODO verify
            note_weight[..., 1:] += chord_note
            weight *= note_weight

        loss = F.binary_cross_entropy_with_logits(preds, targets.float(), weight)
    return loss


def loss(
    root_logits: torch.Tensor,
    root_labels: torch.Tensor,
    triad_logits: torch.Tensor,
    triad_labels: torch.Tensor,
    chord_note: torch.Tensor,
    chord_ignore: torch.Tensor,
) -> LossDict:
    chord_root_loss = chord_loss(
        root_logits,
        root_labels,
        chord_note,
        chord_ignore,
        chord_root=True,
    )
    chord_triad_loss = chord_loss(
        triad_logits,
        triad_labels,
        chord_note,
        chord_ignore,
        chord_root=False,
    )

    loss = chord_root_loss + chord_triad_loss
    return {
        "loss": loss,
        "chord_root_loss": chord_root_loss,
        "chord_triad_loss": chord_triad_loss,
    }


def val_test_step(
    root_logits: torch.Tensor,
    root_labels: torch.Tensor,
    triad_logits: torch.Tensor,
    triad_labels: torch.Tensor,
    chord_ignore: torch.Tensor,
    duration: float,
    label_hop: float,
    hop_factor: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # TODO: test and refactor
    chord_ignore = chord_ignore.squeeze(0).cpu().numpy()
    prediction_root, target_root = postprocess_chord(
        root_logits, root_labels, duration, label_hop, hop_factor
    )
    prediction_triad, target_triad = postprocess_chord(
        triad_logits, triad_labels, duration, label_hop, hop_factor
    )

    preds = {
        "chord_root": prediction_root,
        "chord_triad": prediction_triad,
        "chord_ignore": chord_ignore,
    }
    targets = {
        "chord_root": target_root,
        "chord_triad": target_triad,
        "chord_ignore": chord_ignore,
    }

    metrics = get_chord_score(preds, targets, self.config.label_hop)

    # Filter None
    dataset_name = batch.dataset_name[0].split("_")[0] + "_"
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
    return partial_metrics, overall_metrics


def postprocess_chord(
    prediction: torch.Tensor,
    target: torch.Tensor,
    duration: float,
    label_hop: float,
    hop_factor: float,
):
    duration = int(duration / label_hop)  # TODO verify why int
    sample_hop = int(duration / hop_factor / label_hop)  # TODO verify why int

    target = target.squeeze()
    prediction = merge_multiple_temporal_probs(prediction, target, duration, sample_hop)
    prediction = torch.softmax(prediction, -1)
    min_length = min(prediction.shape[0], target.shape[0])

    prediction = prediction[:min_length].cpu().numpy()
    target = target[:min_length].cpu().numpy()
    return prediction, target

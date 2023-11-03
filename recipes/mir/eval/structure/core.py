import torch
import torch.nn.functional as F

from mir_tools.structure.eval import (  # TODO same as beat?
    eval_a_song,
    merge_multiple_temporal_probs,
)
from sam.data.mir.structure import StructureDataResult
from sam.models.base import LossDict


def wbce_loss(preds, targets, weight=None, short=False):
    # TODO test
    if weight is not None:
        weight = targets * weight.reshape(1, 1, weight.shape[0]) + 1

    if short:
        unmask_idx = targets.sum(-1) >= 0
        preds = preds[unmask_idx]
        targets = targets[unmask_idx]
        if weight is not None:
            weight = weight[unmask_idx]

    return F.binary_cross_entropy_with_logits(preds, targets.float(), weight)


def loss(
    function_logits: torch.Tensor,
    function_labels: torch.Tensor,
    boundary_logits: torch.Tensor,
    boundary_labels: torch.Tensor,
    boundary_loss_scaler: float,
) -> LossDict:
    function_loss = wbce_loss(function_logits, function_labels, short=True)
    boundary_loss = wbce_loss(boundary_logits, boundary_labels)
    loss = (
        boundary_loss_scaler * boundary_loss
        + (1 - boundary_loss_scaler) * function_loss
    )

    return {"loss": loss}


def val_test_step(
    function_logits: torch.Tensor,
    function_labels: torch.Tensor,
    boundary_logits: torch.Tensor,
    boundary_labels: torch.Tensor,
    duration: float,
    hop_factor: float,
    n_top_bound,
    key,
):
    function_preds = torch.sigmoid(function_logits).cpu().numpy()
    boundary_preds = torch.sigmoid(boundary_logits).cpu().numpy()
    boundary_preds = merge_multiple_temporal_probs(
        boundary_preds, hop_factor / duration  # TODO expecting int?
    )
    function_preds = merge_multiple_temporal_probs(
        function_preds, hop_factor / duration  # TODO expecting int?
    )
    function_tar = function_labels.squeeze(0)  # TODO address squeeze
    min_length = min(function_tar.shape[0], function_preds.shape[0])
    boundary_preds, function_preds, function_tar = (
        boundary_preds[:min_length],
        function_preds[:min_length],
        function_tar[:min_length],
    )

    scores = eval_a_song(
        boundary_preds[..., None],
        function_preds,
        boundary_labels[0].cpu().numpy(),
        function_tar.cpu().numpy(),
        n_top_bound,
        duration,
        key[0],
    )
    scores = {k: v for k, v in scores.items() if v is not None}
    return scores

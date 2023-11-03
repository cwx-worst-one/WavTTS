import torch
import torch.nn.functional as F
from mir_tools.beat.utils import merge_multiple_temporal_probs
from mir_tools.key_detection.eval import get_key_score
from sam.models.base import LossDict


def loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> LossDict:
    # TODO requires testing
    min_length = min(preds.shape[1], targets.shape[1])
    keymode_loss = F.binary_cross_entropy_with_logits(
        preds[:, :min_length],
        targets[:, :min_length].float(),
    )
    return {
        "loss": keymode_loss,
    }


def val_test_step(
    key_logits: torch.Tensor,
    key_labels: torch.Tensor,
    duration,
    label_hop,
    hop_factor,
    key_map,
):
    # TODO requires testing
    key_preds = key_logits.cpu()
    key_labels = key_labels.squeeze(0).cpu()  # TODO no squeeze

    key_pre, key_tar = eval_key(key_preds, key_labels, duration, label_hop, hop_factor)

    scores, _, _ = get_key_score(
        {"keysig_pred_probs": key_pre, "keysig_truth_labels": key_tar},
        key_map,
    )

    # Filter None
    scores = {k: v for k, v in scores.items() if v is not None}
    return scores


def eval_key(key_pred, key_label, duration: float, label_hop: float, hop_factor):
    # TODO test, also verify why cast to int
    duration = int(duration / label_hop)
    sample_hop = int(duration / hop_factor / label_hop)

    key_pre = torch.nn.functional.softmax(
        merge_multiple_temporal_probs(key_pred, key_label, duration, sample_hop),
        -1,
    )
    min_length = min(key_pre.shape[0], key_label.shape[0])
    key_pre = key_pre[:min_length].cpu().numpy()
    key_tar = key_label[:min_length].cpu().numpy()

    return key_pre, key_tar

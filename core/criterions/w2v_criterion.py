"""wav2vec criterion"""

import torch
import torch.nn.functional as F


def wav2vec_criterion(
    logits, target, result, batch_data, weights, reduce, infonce, loss_weights=None
):
    """wav2vec criterion"""
    sample_size = target.numel() if infonce else target.long().sum().item()
    bsz = (1 - batch_data['src_mask']).int().bool().size(0)

    if infonce:
        contrastive_loss = F.cross_entropy(
            logits,
            target,
            reduction="mean" if reduce else "none",
        )
    else:
        contrastive_loss = F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            weights,
            reduction="mean" if reduce else "none",
        )

    if loss_weights is not None:
        extra_losses = get_extra_losses(result)
        if torch.is_tensor(extra_losses):
            extra_losses = [extra_losses]
        if len(loss_weights) == 1 and len(extra_losses) != 1:
            loss_weights = [loss_weights[0]] * len(extra_losses)
        assert len(extra_losses) == len(loss_weights), f'{len(extra_losses)}, {len(loss_weights)}'
        diversity_loss = extra_losses[0]
        feature_l2_loss = extra_losses[1]
        backward_loss = (
            contrastive_loss + loss_weights[0] * diversity_loss + loss_weights[1] * feature_l2_loss
        )
    else:
        diversity_loss = contrastive_loss * 0
        feature_l2_loss = contrastive_loss * 0
        backward_loss = contrastive_loss

    if infonce:
        with torch.no_grad():
            if logits.numel() == 0:
                corr = 0
                count = 0
            else:
                assert logits.dim() > 1, logits.shape
                max_size = logits.argmax(-1) == 0
                min_size = logits.argmin(-1) == 0
                both = max_size & min_size
                corr = max_size.long().sum().item() - both.long().sum().item()
                count = max_size.numel()

            accuracy = corr / count
    else:
        accuracy = 0.0

    encoder_out = {}

    encoder_out['loss'] = backward_loss
    encoder_out['backward_loss'] = backward_loss
    encoder_out['contrastive_loss'] = contrastive_loss
    encoder_out['diversity_loss'] = diversity_loss
    encoder_out['feature_l2_loss'] = feature_l2_loss

    encoder_out['acc'] = accuracy
    encoder_out['frame_size'] = batch_data['src_mask'].float().sum()
    encoder_out['tgt_size'] = sample_size
    encoder_out['batch_size'] = bsz

    return encoder_out


def get_extra_losses(net_output):
    """get_extra_losses"""
    pen = []

    if "prob_perplexity" in net_output:
        pen.append(
            (net_output["num_vars"] - net_output["prob_perplexity"]) / net_output["num_vars"]
        )

    if "features_pen" in net_output:
        pen.append(net_output["features_pen"])

    return pen

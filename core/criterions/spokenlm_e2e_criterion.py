"""wav2vec criterion"""

import torch
from torch import nn
from .criterion import Xentropy


class SpokenLmE2eCriterion(nn.Module):
    """
    SpokenLM E2E Criterion
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ce_loss = Xentropy(args=args)
        self.diversity_loss_factor = self.args.diversity_loss_factor
        self.label_smooth_factor = self.args.label_smooth_factor
        self.vocab_size = self.args.vocab_size

    def forward(
        self, logits, logits_mask, targets, targets_mask, spoken_tokens_probs
    ):
        """
        Compute SpokenLM E2E Criterion
        """
        # cross-entropy loss
        forward_out = self.ce_loss(
            logits=logits,
            src_mask=logits_mask,
            target=targets,
            target_mask=targets_mask,
        )

        # diversity loss
        bsz, tsz, vsz = spoken_tokens_probs.shape
        avg_probs = spoken_tokens_probs.view(bsz * tsz, vsz).mean(dim=0)
        prob_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1))
        diversity_loss = (vsz - prob_perplexity) / vsz
        diversity_loss = self.diversity_loss_factor * diversity_loss

        forward_out['diversity_loss'] = diversity_loss
        forward_out['backward_loss'] = forward_out['backward_loss'] + diversity_loss
        forward_out['loss'] = forward_out['loss'] + diversity_loss

        return forward_out

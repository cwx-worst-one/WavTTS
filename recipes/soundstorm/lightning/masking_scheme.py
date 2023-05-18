import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn


def cosine_schedule(ratio: torch.Tensor) -> torch.Tensor:
    """Cosine schedule as proposed in the MaskGIT paper. Given a ratio in [0, 1),
    we generate a masking ratio. During training, the input ratio is uniformly
    sampled; during inference, the input ratio is based on the step number
    divided by the total iteration number: t/T.

    We clip the values between 1e-6 and 1. as per:
    https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py#LL51C37-L51C45  # noqa

    Args:
        batch_size (int): _description_
        seq_len (int): _description_
        device (str): _description_

    Returns:
        torch.Tensor: The mask rate
    """
    return torch.cos(ratio * math.pi / 2.0)


def sample_qs(batch_size: int, n_quantizers: int, device):
    n_randperms = math.ceil(batch_size / n_quantizers)
    qs = []
    for _ in range(n_randperms):
        qs.append(torch.randperm(n_quantizers, device=device))
    return torch.cat(qs)[:batch_size]


class MaskingScheme(nn.Module):
    def __init__(self):
        super().__init__()


class SoundStormMaskingScheme(MaskingScheme):
    """
    This module implements the SoundStorm masking scheme, which is proposed for
    training accordingly:
    1. To enable voice prompting, we randomly sample a timestep t ∈ {1, . . . , T }, where T denotes
    the maximum sequence length, and we do not mask any tokens before this timestep.
    The conditioning tokens are never masked.

    2. Sample the current RVQ level q ~ U {1, Q};

    3. Sample the mask M ∈ {0, 1}T according to a cosine schedule (Chang et al., 2022)
    for level q, i.e., sample the masking ratio p = cos(u) where u ~ U [0, π/2],
    and sample iid Mi ~ Bernoulli(p).

    4. Mask the selected non-prompt tokens at the current RVQ level
    and all non-prompt tokens at finer RVQ levels.
    """

    def __init__(self, sample_q_uniformly: bool, sample_t: bool):
        super().__init__()
        self.sample_q_uniformly = sample_q_uniformly
        self.sample_t = sample_t

    def forward(
        self, audio_tokens: torch.Tensor, mask_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_quantizers, seq_len = audio_tokens.shape
        device = audio_tokens.device

        if self.sample_t:
            rand_ts = torch.randint(0, seq_len - 1, (batch_size,), device=device)

        if self.sample_q_uniformly:
            rand_qs = sample_qs(batch_size, n_quantizers, device)
        else:
            rand_qs = (
                (
                    1
                    - cosine_schedule(
                        torch.empty(batch_size, device=device).uniform_(0, 1)
                    )
                )
                * n_quantizers
            ).long()

        rand_times = torch.empty(batch_size, 1, device=device).uniform_(0, 1)
        rand_probs = cosine_schedule(rand_times)  # [b, ]
        num_tokens_mask = (rand_probs * seq_len).clamp(min=1.0).long()

        mask = torch.full(audio_tokens.shape, False, device=audio_tokens.device)
        for batch_idx, q in enumerate(rand_qs):
            rand_indices_no_replacement = torch.randperm(seq_len, device=device)[
                : num_tokens_mask[batch_idx]
            ]

            t = rand_ts[batch_idx] if self.sample_t else 0
            rand_indices_no_replacement = rand_indices_no_replacement[
                rand_indices_no_replacement >= t
            ]

            mask[batch_idx, q][rand_indices_no_replacement] = True
            mask[batch_idx, q + 1 :, t:] = True

        masked_audio_tokens = torch.where(mask, mask_token_id, audio_tokens.clone())
        return masked_audio_tokens, mask, rand_qs


class DucMaskingScheme(MaskingScheme):
    def __init__(
        self,
        coarse_layers: int = 1,
        mask_all_prob: float = 0.0,
        mask_all_fine: bool = False,
        coarse_prob: Optional[float] = None,
    ):
        super().__init__()
        self.coarse_layers = coarse_layers
        self.mask_all_prob = mask_all_prob
        self.mask_all_fine = mask_all_fine
        self.coarse_prob = coarse_prob

    def forward(
        self, audio_tokens: torch.Tensor, mask_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_quantizers, seq_len = audio_tokens.shape
        device = audio_tokens.device

        rand_qs = []
        rand_times = torch.empty(batch_size, 1, device=device).uniform_(0, 1)
        rand_probs = cosine_schedule(rand_times)  # [b, ]
        num_tokens_mask = (rand_probs * seq_len).clamp(min=1.0).long()

        mask = torch.full(audio_tokens.shape, False, device=audio_tokens.device)

        for batch_idx in range(batch_size):
            prob = 1 - math.cos(random.random() * math.pi / 2.0)
            if self.coarse_prob is None:
                # Select q according to cosine schedule
                q = int(prob * n_quantizers)
            else:
                # Within each bucket, select q according to cosine schedule
                if random.random() <= self.coarse_prob:
                    q = int(prob * self.coarse_layers)
                else:
                    q = self.coarse_layers + int(
                        prob * (n_quantizers - self.coarse_layers)
                    )
            rand_qs.append(q)

            if (
                self.mask_all_fine and q >= self.coarse_layers
            ) or random.random() < self.mask_all_prob:
                mask[batch_idx, q:] = True
            else:
                rand_indices_no_replacement = torch.randperm(seq_len)[
                    : num_tokens_mask[batch_idx]
                ]
                mask[batch_idx, q][rand_indices_no_replacement] = True
                mask[batch_idx, q + 1 :] = True

        masked_audio_tokens = torch.where(mask, mask_token_id, audio_tokens.clone())
        rand_qs = torch.LongTensor(rand_qs).to(device)
        return masked_audio_tokens, mask, rand_qs

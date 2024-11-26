"""
@hanoihantrakul 4NOV2024 Implementation Notes for `umm_mkii_tts_rope_lfr.py`

LFR stands for "Low Frame Rate". 

The problem with the original UMM codebase was the pipeline
hard coded 100hz mel feature rate and 4x downsampling to 25hz token rate. When we wanted to try
20, 15 and 10 token rates, introducing this change into umm_mkii.py was not possible withoout
re writing a few modules.

At the same time, Ju Chiang was working on UMM2 and performing regression tests so we could migrate
from the relatively messy UMM codebase to the cleaner and modularized UMM2. At the time of writing,
we are still regression testing UMM2.

Therefore, I have implemented the LFR version of the ConformerUMM_TTS_ROPE tokenizer using the 
older UMM codebase. Once UMM2 has been properly tested E2E, we can migrate this implementation
into UMM2. I've extensively commented the code to make migration easier. 
"""

from recipes.umm.modules.lit_module import Stage1 
import torch
import random

class Stage1LFR(Stage1):
    """
    @hanoihantrakul 4NOV2024 
    Compared to the original Stage1, I had to rewrite the self.masking() function to avoid
    the hard-coded logic for 4x downsampling assuming 100hz mel features.
    """

    def __init__(
    self,
    model_cls,
    criterion_cls,
    optimizer_cls,
    scheduler_cls,
    required_modules=None,
    checkpointing=False,
    extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
    
    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.model.config.len_masking_raw, device=device)
            < self.model.config.mask_prob
        )
        if torch.all(start_indices == False):
            start_indices[
                random.randint(0, start_indices.size(0) - 1),
                random.randint(0, start_indices.size(1) - 1),
            ] = True
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.model.config.len_masking_raw, dim=1)
        )
        
        """
        The main difference from umm_mkii.Stage1() is this line:
        Originally this line of code was hardcoded with `self.model.config.len_masking_token * 4` 
        which only accounts for the 4x downsampling scenario with 100hz mel features as input.
        """
        mel_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(
                self.model.config.get_mel_mask_len_factor(), dim=1
            )
        )

        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.model.config.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), dtype=x.dtype, device=device)
            * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices, mel_domain_masked_indices
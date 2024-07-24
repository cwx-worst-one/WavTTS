import os
from typing import Optional

import torch

from recipes.research.audio_codec.scripts.compile import get_latest_model_from_commit
from recipes.research.prompt_gpt import PromptGPT
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


def load_promptgpt_model(
    commit_hash: str,
    device: torch.device,
    cache: bool = False,
    ckpt_path: Optional[str] = None,
):
    if ckpt_path is None:
        ckpt_path = get_latest_model_from_commit("prompt_gpt/default", commit_hash)

    model = PromptGPT.load_from_checkpoint(ckpt_path, cache=cache, map_location=device)
    model = model.eval()
    model.freeze()
    model.commit_hash = commit_hash
    model.commit_step = os.path.basename(ckpt_path)
    return model


def sample_prompt_gpt(
    model: PromptGPT,
    text_prompt: str,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.95,
):
    batch_text_prompt = [text_prompt]
    pred_text = model.generate(
        batch_text_prompt,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
    )
    return pred_text[0]

import argparse
from typing import List, Optional

import time
import torch

from recipes.byteformers_example.modules.language_modeling import LanguageModelingModule
from recipes.byteformers_example.tokenizer import LlamaTokenizer
from recipes.musiclm.inference.utils import load_config


def generate(
    prompt: str,
    tokenizer: LlamaTokenizer,
    pl_module: LanguageModelingModule,
    max_new_tokens: int = 50,
    top_k: Optional[int] = None,
    top_p: float = 0.95,
    temperature: float = 0.8,
) -> List[str]:

    tokenizer = LlamaTokenizer("tokenizer.model")
    x = tokenizer.encode(prompt, bos=True, eos=False, device=pl_module.device)[
        None, ...
    ]

    t0 = time.perf_counter()
    ys = pl_module.generate(
        x,
        max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
    )

    for y in ys:
        t = time.perf_counter() - t0
        yield y.cpu()[0]
    print(f"Inference time: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    cfg = load_config("./recipes/byteformers_example/conf/finetune.yaml")

    tokenizer = cfg.tokenizer
    pl_module = cfg.pl_module.to(device)
    pl_module.eval()

    if args.compile:
        pl_module = torch.compile(pl_module)

    stream = generate(
        args.prompt,
        tokenizer,
        pl_module,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    tokens = []
    for token in stream:
        tokens.append(token)

    print(tokenizer.decode(torch.tensor(tokens, dtype=torch.long)))
    print(f"\nMemory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

from typing import Tuple

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
is_cuda_available = torch.cuda.is_available()


def ids_tensor(shape: Tuple[int], vocab_size: int, device: str = "cpu") -> torch.Tensor:
    return torch.randint(0, vocab_size - 1, shape, device=device).contiguous()

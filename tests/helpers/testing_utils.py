import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
is_cuda_available = torch.cuda.is_available()

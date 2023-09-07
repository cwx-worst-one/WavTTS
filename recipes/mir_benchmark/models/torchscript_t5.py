import torch
from torch import nn


class ExampleTorchscript(nn.Module):
    """CNN + T5 Encoder"""

    def __init__(
        self,
        torchscript_path="/mnt/bn/audio-diffusion/torchscript/best_rq/chromatic_80k.pt",
    ):
        super().__init__()

        # preprocessing
        self.t5 = torch.jit.load(torchscript_path)

    def get_latent(self, x, layer_ix):
        _, hidden_states = self.t5(x)
        return hidden_states[layer_ix]

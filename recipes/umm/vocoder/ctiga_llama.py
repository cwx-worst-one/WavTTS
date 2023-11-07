import torch
import torch.nn.functional as F

from torch import nn

from recipes.text2semantic.modules.llama.based_ctiga_llama import LLaMa
from recipes.umm.vocoder.llama_v100 import ModelArgs


class CtigaLLaMa(LLaMa):
    def __init__(
        self,
        params: ModelArgs,
        provider="ctiga",
        state_dict_path=None,
    ):
        super().__init__(params, provider)
        self.params = params
        self.h_output = nn.Linear(params.dim, params.out_dim, bias=False)
        self.init_weight_and_load_state(state_dict_path)

    def forward(
        self,
        inputs: torch.Tensor,
        start_pos: int = 0,
        inference_params=None,
    ):
        bs, seqlen, _ = inputs.shape
        inputs = super().forward(inputs, seqlen, start_pos, inference_params)
        h_output = self.h_output(inputs)
        return h_output.float()

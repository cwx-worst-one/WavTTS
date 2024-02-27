
from dataclasses import dataclass
from torch import nn
from recipes.voicebox.modules.based_ctiga_llama import LLaMa, ModelArgs
from recipes.voicebox.modules.layers import FrontendEmbedding

@dataclass
class VoiceBoxArgs(ModelArgs):
    causal: bool = False # encoder.
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    wordseg_embed_dim: int = 64
    n_wordseg: int = 8
    n_phone: int = 1000
    n_tone: int = 30


class VoiceBox(LLaMa):
    def __init__(self, params: VoiceBoxArgs, provider="ctiga"):
        super().__init__(params, provider)
        self.init_weight_and_load_state(None)

        self.tok_embeddings = FrontendEmbedding(
                params.phone_embed_dim,
                params.tone_embed_dim,
                params.wordseg_embed_dim,
                n_phone = params.n_phone,
                n_tone = params.n_tone,
                n_wordseg = params.n_wordseg,
                out_dim=params.dim
        )
        self.post_net = nn.Linear(1024, 80, bias=False)

    def forward(self, frontend_inputs, mel, mel_len):
        h =  self.tok_embeddings(frontend_inputs)
        h = super().forward(h, mel_len)
        h = self.post_net(h)
        ret_dict = {}
        ret_dict['pred_mel'] = h
        return ret_dict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
from torch import nn
import torch.nn.functional as F
from recipes.text2semantic.modules.llama.based_ctiga_llama import LLaMa, ModelArgs
from recipes.text2semantic.modules.llama.layers import FrontendEmbedding
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1024  # defined later by tokenizer
    out_dim: int = 1024  # maybe not same as vocab_size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 2048
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    sparse: bool = False
    checkpointing: bool = False

    # for codec
    num_res: int = -1
    num_coarse: int = -1
    num_fine: int = -1

    audio_tokens_num: int = 1024
    phone_tokens_num: int = 200
    lang_vocab_size: int = 200
    spk_vocab_size: int = 8192
    
    # for frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    n_phone: int = 1000
    n_tone: int = 30

class VAELLaMaLangSpk(LLaMa):
    def __init__(
        self,
        params: ModelArgs,
        provider="default",
        state_dict_path=None,
        use_lang_id=False,
        use_spk_id=False,
    ):
        super().__init__(params, provider)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.use_lang_id = use_lang_id
        self.use_spk_id = use_spk_id

        self.tok_embeddings = FrontendEmbedding(
                params.phone_embed_dim,
                params.tone_embed_dim,
                n_phone = params.n_phone,
                n_tone = params.n_tone,
                out_dim=params.dim
                )

        self.lang_embeddings = nn.Embedding(params.lang_vocab_size, params.dim)
        self.spk_embeddings = nn.Embedding(params.spk_vocab_size, params.dim)

        self.stop_token_head = nn.Linear(params.dim, 2, bias=False)
        self.h_output = nn.Linear(params.dim, params.out_dim * 2, bias=False)
        self.prenet = nn.Linear(params.out_dim, params.dim, bias=False)

        self.init_weight_and_load_state(state_dict_path)


    def _repara(self, stats):
        m, logs = torch.split(stats, self.params.out_dim, dim=-1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z

    def forward(
        self,
        frontend_inputs,
        bns: torch.Tensor,
        text_lens: torch.Tensor,
        bn_lens: torch.Tensor,
        start_pos: int = 0,
        use_cache=False,
        inference_params=None,
        lang_seqs=None,
        spk_seqs=None,
    ):

        bsz = text_lens.shape[0]
        seqlen = max(text_lens + bn_lens)

        if bns.shape[1] > 0:  # 正常训练和带prompt推理都应该走这个
            bn_in_z = self._repara(bns)
            bn_in_h = self.prenet(bn_in_z)   
        else:
            bn_in_z = torch.zeros([1, 0, self.params.out_dim]).to(bns.device)
            bn_in_h = torch.zeros([1, 0, self.params.dim]).to(
                bns.device
            )  # noprompt 推理用的占位符, T为0所以等于没有任何内容

        if self.use_lang_id:
            lang_embeds = self.lang_embeddings(lang_seqs)
            for i in range(bsz):
                bn_in_h[i, :bn_lens[i], :] += lang_embeds[i, :bn_lens[i], :]

        if self.use_spk_id:
            spk_embeds = self.spk_embeddings(spk_seqs)
            for i in range(bsz):
                bn_in_h[i, :bn_lens[i], :] += spk_embeds[i, :bn_lens[i], :]

        if use_cache:
            assert bsz == 1
            assert bn_in_h is not None
            h = bn_in_h
            seqlen = 1
        else:
            token_in_h = self.tok_embeddings(frontend_inputs)

            h = torch.zeros([bsz, seqlen, bn_in_h.shape[-1]], device=bn_in_h.device)
            for i in range(bsz):
                h[i, :text_lens[i], :] = token_in_h[i, :text_lens[i], :]
                h[i, text_lens[i]:text_lens[i]+bn_lens[i], :] = bn_in_h[i, :bn_lens[i], :]

        h = super().forward(h, seqlen, start_pos, inference_params)

        h_output = self.h_output(h)
        stop_token= self.stop_token_head(h)

        output_dict = {"dense": h_output.float(), "stop_token": stop_token.float(), "bn_in_z": bn_in_z}
        return output_dict



if __name__ == "__main__":
    pass
    # pass
    # when resume model from converted ckpt
    #
    # if only weights:
    # bash recipes/valle/custom_scripts/run_vae_t2s.sh --model_cls.provider ctiga --model_cls.state_dict_path ctiga_only_weights.ckpt
    #
    # else:
    # bash recipes/valle/custom_scripts/run_vae_t2s.sh --model_cls.provider ctiga --ckpt_path ctiga.ckpt

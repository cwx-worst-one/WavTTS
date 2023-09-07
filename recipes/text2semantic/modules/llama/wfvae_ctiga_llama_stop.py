# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
from torch import nn
import torch.nn.functional as F
from recipes.text2semantic.modules.llama.based_ctiga_llama import LLaMa, ModelArgs


class VAELLaMaStop(LLaMa):
    def __init__(
        self,
        params: ModelArgs,
        use_speaker_id=False,
        provider="default",
        state_dict_path=None,
        use_lang_id=False,
        use_phoneme_loss=True
    ):
        super().__init__(params, provider)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.use_speaker_id = use_speaker_id
        self.use_lang_id = use_lang_id

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.use_phoneme_loss = use_phoneme_loss
        if self.use_phoneme_loss:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.h_output = nn.Linear(params.dim, params.out_dim * 2, bias=False)
        self.prenet = nn.Linear(params.out_dim, params.dim, bias=False)
        self.stop_token_head = nn.Linear(params.dim, 2, bias=False)

        self.init_weight_and_load_state(state_dict_path)


    def _repara(self, stats):
        m, logs = torch.split(stats, self.params.out_dim, dim=-1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z

    def forward(
        self,
        text_ids: torch.Tensor,
        text_id_lens: torch.Tensor,
        bns: torch.Tensor,
        bn_lens: torch.Tensor,
        inputs: torch.Tensor,
        start_pos: int = 0,
        use_cache=False,
        inference_params=None,
    ):

        extra_shift_num = 2
        if self.use_speaker_id:
            extra_shift_num += 1

        if self.use_lang_id:
            extra_shift_num += 1

        bsz, seqlen = inputs.shape
        token_in_h = self.tok_embeddings(inputs)

        if bns.shape[1] > 0:  # 正常训练和带prompt推理都应该走这个
            bn_in_z = self._repara(bns)
            bn_in_h = self.prenet(bn_in_z)
        else:
            bn_in_z = torch.zeros([1, 0, self.params.out_dim]).to(bns.device)
            bn_in_h = torch.zeros([1, 0, self.params.dim]).to(
                bns.device
            )  # noprompt 推理用的占位符, T为0所以等于没有任何内容

        if use_cache:
            assert bsz == 1
            h = bn_in_h
        else:
            h = []
            for i in range(bsz):
                h.append(
                    torch.cat(
                        # bos_text_sep  + bn + eos_pad0
                        (
                            token_in_h[i, : text_id_lens[i] + extra_shift_num, :],
                            bn_in_h[i, : bn_lens[i], :],
                            token_in_h[i, bn_lens[i] + text_id_lens[i] + extra_shift_num :, :],
                        ),
                        dim=-2,
                    )
                )
            h = torch.stack(h)

        h = super().forward(h, seqlen, start_pos, inference_params)

        if self.use_phoneme_loss:
            output = self.output(h)
        h_output = self.h_output(h)
        stop_token= self.stop_token_head(h)

        if self.use_phoneme_loss:
            output_dict = {"logits": output.float(), "dense": h_output.float(), "stop_token": stop_token.float()}
        else:
            output_dict = {"dense": h_output.float(), "stop_token": stop_token.float()}
        return output_dict, bn_in_z



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

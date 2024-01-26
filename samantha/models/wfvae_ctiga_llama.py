# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import logging
import math
import os
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import T5EncoderModel

import samantha.utils.hdfs_helper as hh
from samantha.models.based_ctiga_llama import LLaMa
from samantha.models.ECAPA_TDNN_bias import ECAPA_TDNN
from samantha.models.grl import GradientReversal

logger = logging.getLogger(__name__)


def load_torch_script_module(module_path, device):
    module = torch.jit.load(module_path, map_location=device).to(device).eval()
    return module


def init_sami_ser(hpath, local_rank, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")

    sami_ser = {}

    # wav2vec2
    h_wav2vec2 = f"{hpath}/wav2vec2_{local_rank}.pt"
    local_wav2vec2_path = f"{cache_dir}/wav2vec2_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_wav2vec2_path):
            if not hh.get(h_wav2vec2, local_wav2vec2_path):
                raise ConnectionError(f"Cannot retrieve file from {h_wav2vec2}.")
        sami_ser["wav2vec2"] = load_torch_script_module(local_wav2vec2_path, device)
    else:
        sami_ser["wav2vec2"] = load_torch_script_module(h_wav2vec2, device)

    # output_deg
    h_output_deg = f"{hpath}/output_deg_{local_rank}.pt"
    local_output_deg_path = f"{cache_dir}/output_deg_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_output_deg_path):
            if not hh.get(h_output_deg, local_output_deg_path):
                raise ConnectionError(f"Cannot retrieve file from {h_output_deg}.")
        sami_ser["output_deg"] = load_torch_script_module(local_output_deg_path, device)
    else:
        sami_ser["output_deg"] = load_torch_script_module(h_output_deg, device)

    # output_mlp
    h_output_mlp = f"{hpath}/output_mlp_{local_rank}.pt"
    local_output_mlp_path = f"{cache_dir}/output_mlp_{local_rank}.pt"
    if hpath.startswith("hdfs://"):
        if not os.path.exists(local_output_mlp_path):
            if not hh.get(h_output_mlp, local_output_mlp_path):
                raise ConnectionError(f"Cannot retrieve file from {h_output_mlp}.")
        sami_ser["output_mlp"] = load_torch_script_module(local_output_mlp_path, device)
    else:
        sami_ser["output_mlp"] = load_torch_script_module(h_output_mlp, device)

    return sami_ser


class FrontendEmbedding(nn.Module):
    def __init__(
        self,
        phone_embed_dim,
        tone_embed_dim,
        out_dim,
        padding_idx=0,
        n_phone=200,
        n_tone=20,
        # n_phone=148,
        # n_tone=13,
        # n_wordcateg=6,
        # n_prosody=7
    ):
        super().__init__()
        self.phone_embedding = nn.Embedding(
            n_phone, phone_embed_dim, padding_idx=padding_idx
        )
        self.tone_embeddig = nn.Embedding(
            n_tone, tone_embed_dim, padding_idx=padding_idx
        )
        input_dim = phone_embed_dim + tone_embed_dim

        self.out_linear = nn.Linear(input_dim, out_dim, bias=False)

    def forward(self, inputs):
        phone_emb = self.phone_embedding(inputs["phone"])
        tone_emb = self.tone_embeddig(inputs["tone"])
        emb = torch.cat([phone_emb, tone_emb], dim=-1)

        return self.out_linear(emb)


def sequence_mask_binary(seq_lens, max_len=None, device="cpu"):
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask >= (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        logger.info(f"Created pos_embedding with size:{pos_embedding.size()}")

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        token_embedding = token_embedding.transpose(0, 1)
        res = self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )
        res = res.transpose(0, 1)
        return res


@dataclass
class VAELLaMaModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    cross_attention_n_heads: int = 16
    vocab_size: int = 1024  # defined later by tokenizer
    out_dim: int = 1024  # maybe not same as vocab_size
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-6

    max_batch_size: int = 32
    max_seq_len: int = 2048
    pos_pdrop: float = 0.1
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
    tag_vocab_size: int = 4

    # for frontend
    phone_embed_dim: int = 512
    tone_embed_dim: int = 64
    n_phone: int = 1000
    n_tone: int = 30
    n_phonetone: int = 30000

    # prenet
    n_layers_prenet: int = 1
    prenet_dim: int = 256
    n_layers_gr: int = 1
    gr_dim: int = 256

    # ser_tag
    ser_tag_vocab_size: int = 100

    # ref_enc
    use_ref_enc: bool = False
    ref_enc_dim: int = 512
    ref_enc_bn_dim: int = 1024

    # text_encoder d_model
    text_encoder_d_model: int = 1536


class VAELLaMaLangSpkSer(LLaMa):
    def __init__(
        self,
        params: VAELLaMaModelArgs,
        provider="default",
        state_dict_path=None,
        use_lang_id=False,
        use_spk_id=False,
        use_phoneme_loss=False,
        text_encoder_type=None,
        text_encoder_path=None,
        use_extra_tag=False,
        spk_type="add",
        attn_type="mha",
        use_lang_grloss=False,
        input_type="2dim",
        use_ser_tag=False,
        use_ser_tag_loss=False,
    ):
        super().__init__(params, provider)
        self.params = params

        # for engine deploy
        dim_per_head = int(self.params.dim / self.params.n_heads)
        assert dim_per_head & (dim_per_head - 1) == 0 and dim_per_head <= 128, (
            self.params.dim,
            self.params.n_heads,
            dim_per_head,
        )

        self.n_layers = params.n_layers
        self.use_lang_id = use_lang_id
        logger.info(f"use_lang_id: {self.use_lang_id}")
        self.use_spk_id = use_spk_id
        logger.info(f"use_spk_id: {self.use_spk_id}")
        self.text_encoder_type = text_encoder_type
        logger.info(f"text_encoder_type: {self.text_encoder_type}")
        self.text_encoder_path = text_encoder_path
        logger.info(f"text_encoder_path: {self.text_encoder_path}")
        self.attn_type = attn_type
        logger.info(f"attn_type: {self.attn_type}")
        self.use_extra_tag = use_extra_tag
        logger.info(f"use_extra_tag: {self.use_extra_tag}")
        self.use_phoneme_loss = use_phoneme_loss
        logger.info(f"use_phoneme_loss: {self.use_phoneme_loss}")
        self.spk_type = spk_type
        logger.info(f"spk_type: {self.spk_type}")
        self.use_lang_grloss = use_lang_grloss
        logger.info(f"use_lang_grloss: {self.use_lang_grloss}")
        self.input_type = input_type
        logger.info(f"input_type: {self.input_type}")
        self.use_ser_tag = use_ser_tag
        logger.info(f"use_ser_tag: {self.use_ser_tag}")
        self.use_ser_tag_loss = use_ser_tag_loss
        logger.info(f"use_ser_tag_loss: {self.use_ser_tag_loss}")
        self.use_ref_enc = params.use_ref_enc
        logger.info(f"use_ref_enc: {self.use_ref_enc}")

        if self.input_type == "2dim":
            self.tok_embeddings = FrontendEmbedding(
                params.phone_embed_dim,
                params.tone_embed_dim,
                n_phone=params.n_phone,
                n_tone=params.n_tone,
                out_dim=params.dim,
            )
        elif self.input_type == "1dim":
            self.tok_embeddings = nn.Embedding(params.n_phonetone, params.dim)
        else:
            raise NotImplementedError

        if self.use_lang_id:
            self.lang_embeddings = nn.Embedding(params.lang_vocab_size, params.dim)

        if self.use_spk_id:
            if self.spk_type == "cln":
                self.spk_embeddings = nn.Embedding(
                    params.spk_vocab_size, params.dim * 2
                )
            else:
                self.spk_embeddings = nn.Embedding(params.spk_vocab_size, params.dim)
        if self.use_extra_tag:
            self.tag_embeddings = nn.Embedding(params.tag_vocab_size, params.dim * 2)

        self.stop_token_head = nn.Linear(params.dim, 2, bias=False)
        if self.use_phoneme_loss:
            if self.input_type == "2dim":
                self.output = nn.Linear(params.dim, params.n_phone, bias=False)
            elif self.input_type == "1dim":
                self.output = nn.Linear(params.dim, params.n_phonetone, bias=False)
            else:
                raise NotImplementedError

        self.h_output = nn.Linear(params.dim, params.out_dim * 2, bias=False)
        if params.n_layers_prenet == 1:
            self.prenet = nn.Linear(params.out_dim, params.dim, bias=False)
        else:
            self.prenet = []
            for i in range(params.n_layers_prenet - 1):
                self.prenet.append(
                    nn.Linear(params.out_dim, params.prenet_dim, bias=False)
                )
                self.prenet.append(nn.ReLU())
            self.prenet.append(nn.Linear(params.prenet_dim, params.dim, bias=False))
            self.prenet = nn.Sequential(*self.prenet)

        if self.use_lang_grloss:
            self.gradient_rev = GradientReversal(1.0)
            if params.n_layers_gr == 1:
                self.lang_output = nn.Linear(
                    params.dim, params.lang_vocab_size, bias=False
                )
            else:
                self.lang_output = []
                for i in range(params.n_layers_gr - 1):
                    self.lang_output.append(
                        nn.Linear(params.dim, params.gr_dim, bias=False)
                    )
                    self.lang_output.append(nn.ReLU())
                self.lang_output.append(
                    nn.Linear(params.gr_dim, params.lang_vocab_size, bias=False)
                )
                self.lang_output = nn.Sequential(*self.lang_output)

        if self.use_ser_tag:
            self.ser_tag_embeddings = nn.Embedding(
                params.ser_tag_vocab_size, params.dim
            )

        if self.use_ser_tag_loss:
            self.ser_tag_output = nn.Linear(
                params.dim, params.ser_tag_vocab_size, bias=False
            )

        if params.use_ref_enc:
            self.ref_enc = ECAPA_TDNN(
                params.out_dim, params.ref_enc_dim, params.dim, params.ref_enc_bn_dim
            )
            if self.spk_type == "concat" or self.spk_type == "add":
                self.ref_enc_proj = nn.Linear(params.dim, params.dim, bias=False)
            elif self.spk_type == "cln":
                self.ref_enc_proj = nn.Linear(params.dim, params.dim * 2, bias=False)
            else:
                raise NotImplementedError

        self.init_weight_and_load_state(state_dict_path)

        # load pretrained_models after [ self.init_weight_and_load_state ],
        # especially when you want to freeze these parameters
        if self.text_encoder_type in ["flan-T5-large", "byte-T5-base"]:
            logger.info(
                f"Using text encoder: {self.text_encoder_type}, {self.text_encoder_path}"
            )
            assert self.text_encoder_path != ""
            self.use_text_encoder = True
            # self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_path)
            self.text_linear = nn.Linear(
                self.params.text_encoder_d_model, self.params.dim, bias=False
            )
            if self.attn_type == "mha":
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=params.dim,
                    num_heads=params.cross_attention_n_heads,
                    batch_first=True,
                    bias=False,
                )
                self.positional_encoding = PositionalEncoding(
                    params.dim, dropout=params.pos_pdrop, maxlen=params.max_seq_len
                )
                logger.info(
                    f"Created positional_encoding with maxlen: {params.max_seq_len}"
                )
            else:
                raise NotImplementedError
        else:
            self.use_text_encoder = False

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
        byt5_embeds=None,
        byt5_lens=None,
        tag_ids=None,
        ser_tags=None,
        crop_bn=None,
        spk_embd_masks=None,
    ):
        if self.use_ref_enc and crop_bn is not None:
            if not use_cache:
                self.prompt_cond = self.get_prompt_cond(crop_bn).unsqueeze(1)
            else:
                pass
        else:
            self.prompt_cond = None

        bsz = text_lens.shape[0]
        seqlen = max(text_lens + bn_lens)

        if bns.shape[1] > 0:  # 正常训练和带prompt推理都应该走这个
            bn_in_z = self._repara(bns)
            bn_in_h = self.prenet(bn_in_z)
            if self.use_lang_grloss:
                bn_in_h_rev = self.gradient_rev(bn_in_h)
                lang_output = self.lang_output(bn_in_h_rev)
            else:
                lang_output = None
        else:
            bn_in_z = torch.zeros([1, 0, self.params.out_dim]).to(bns.device)
            bn_in_h = torch.zeros([1, 0, self.params.dim]).to(
                bns.device
            )  # noprompt 推理用的占位符, T为0所以等于没有任何内容
            lang_output = None

        if self.use_lang_id:
            lang_embeds = self.lang_embeddings(lang_seqs)
            for i in range(bsz):
                bn_in_h[i, : bn_lens[i], :] += lang_embeds[i, : bn_lens[i], :]

        if self.use_spk_id and spk_seqs is not None:
            if (self.spk_type == "concat" and not use_cache) or self.spk_type == "cln":
                spk_embeds = self.spk_embeddings(spk_seqs[:, 0:1])
            elif self.spk_type == "add":
                spk_embeds = self.spk_embeddings(spk_seqs)
            elif self.spk_type == "cln" and not use_cache:
                spk_embeds = self.spk_embeddings(spk_seqs[:, 0:1])

        if self.use_ref_enc and crop_bn is not None:
            assert self.spk_type == "cln" or self.spk_type == "concat", self.spk_type
            spk_embeds_ref = self.ref_enc_proj(self.prompt_cond)
            spk_embeds = (
                spk_embd_masks.unsqueeze(-1).unsqueeze(-1) * spk_embeds
                + (1 - spk_embd_masks.unsqueeze(-1).unsqueeze(-1)) * spk_embeds_ref
            )

        if self.use_spk_id and self.spk_type == "add" and spk_seqs is not None:
            for i in range(bsz):
                bn_in_h[i, : bn_lens[i], :] += spk_embeds[i, : bn_lens[i], :]

        cond = None
        if self.use_extra_tag:
            assert tag_ids is not None
            cond = self.tag_embeddings(tag_ids).unsqueeze(1)

        if self.use_spk_id and self.spk_type == "cln" and spk_seqs is not None:
            assert spk_embeds is not None
            cond = spk_embeds if cond is None else (cond + spk_embeds)

        if self.use_ser_tag:
            ser_tag_embeds = self.ser_tag_embeddings(ser_tags)

        attn_weights = None
        if use_cache:
            assert bsz == 1
            assert bn_in_h is not None
            h = bn_in_h
            seqlen = 1
        else:
            if self.input_type == "2dim":
                token_in_h = self.tok_embeddings(frontend_inputs)
            elif self.input_type == "1dim":
                token_in_h = self.tok_embeddings(frontend_inputs["phonetone"])
            if self.use_spk_id and self.spk_type == "concat":
                seqlen += 1
            if self.use_ser_tag:
                seqlen += 1
            h = torch.zeros([bsz, seqlen, bn_in_h.shape[-1]], device=bn_in_h.device)

            if self.use_text_encoder:
                # bpe_in_h = self.text_encoder(bpe_seqs)[0]
                bpe_in_h = self.text_linear(byt5_embeds)
                # 1. prepare key & value: bpe sequence
                max_byt5_len = max(byt5_lens)
                bpe_mask = sequence_mask_binary(
                    byt5_lens, max_len=max_byt5_len, device=bns.device
                )
                # 3. prepare query: phone sequence
                max_text_len = max(text_lens)
                text_mask = sequence_mask_binary(
                    text_lens, max_len=max_text_len, device=bns.device
                )

                # 4. prepare attention_mask [B, phone_len, bpe_len]
                attention_mask = text_mask.unsqueeze(-1) & bpe_mask.unsqueeze(1)
                # 4.1 expand to [B*heads, phone_len, bpe_len]
                attention_mask = (
                    attention_mask.unsqueeze(0)
                    .expand(self.params.cross_attention_n_heads, -1, -1, -1)
                    .reshape(
                        bsz * self.params.cross_attention_n_heads,
                        max_text_len,
                        max_byt5_len,
                    )
                )

                # 5. calculate cross-attention
                # text_in_h [B, bpe_len, dim] -> [B, phone_len, dim]
                attn_bpe_in_h, attn_weights = self.cross_attention(
                    query=self.positional_encoding(token_in_h),
                    key=self.positional_encoding(bpe_in_h),
                    value=self.positional_encoding(bpe_in_h),
                    key_padding_mask=bpe_mask,
                    attn_mask=attention_mask,
                    average_attn_weights=False,
                )

                # 6. add attened text_in_h into the phone part of token_in_h
                for i in range(bsz):
                    # add phone embeds and attended-bpe embeds
                    h[i, : text_lens[i], :] = (
                        token_in_h[i, : text_lens[i], :]
                        + attn_bpe_in_h[i, : text_lens[i], :]
                    )
                    if self.use_spk_id:
                        if self.spk_type == "concat":
                            # add spk embds
                            h[i, text_lens[i] : text_lens[i] + 1, :] = spk_embeds[i]
                            if self.use_ser_tag:
                                # insert bn embeds
                                h[
                                    i, text_lens[i] + 1 : text_lens[i] + 2, :
                                ] = ser_tag_embeds[i]
                                h[
                                    i,
                                    text_lens[i] + 2 : text_lens[i] + 2 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                            else:
                                # insert bn embeds
                                h[
                                    i,
                                    text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                        elif self.spk_type == "add" or self.spk_type == "cln":
                            if self.use_ser_tag:
                                h[
                                    i, text_lens[i] : text_lens[i] + 1, :
                                ] = ser_tag_embeds[i]
                                h[
                                    i,
                                    text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                            else:
                                h[
                                    i, text_lens[i] : text_lens[i] + bn_lens[i], :
                                ] = bn_in_h[i, : bn_lens[i], :]

                    else:
                        if self.use_ser_tag:
                            h[i, text_lens[i] : text_lens[i] + 1, :] = ser_tag_embeds[i]
                            h[
                                i, text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i], :
                            ] = bn_in_h[i, : bn_lens[i], :]
                        else:
                            h[i, text_lens[i] : text_lens[i] + bn_lens[i], :] = bn_in_h[
                                i, : bn_lens[i], :
                            ]

            else:
                for i in range(bsz):
                    h[i, : text_lens[i], :] = token_in_h[i, : text_lens[i], :]
                    if self.use_spk_id:
                        if self.spk_type == "concat":
                            # add spk embds
                            h[i, text_lens[i] : text_lens[i] + 1, :] = spk_embeds[i]
                            if self.use_ser_tag:
                                # insert bn embeds
                                h[
                                    i, text_lens[i] + 1 : text_lens[i] + 2, :
                                ] = ser_tag_embeds[i]
                                h[
                                    i,
                                    text_lens[i] + 2 : text_lens[i] + 2 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                            else:
                                # insert bn embeds
                                h[
                                    i,
                                    text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                        elif self.spk_type == "add" or self.spk_type == "cln":
                            if self.use_ser_tag:
                                h[
                                    i, text_lens[i] : text_lens[i] + 1, :
                                ] = ser_tag_embeds[i]
                                h[
                                    i,
                                    text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i],
                                    :,
                                ] = bn_in_h[i, : bn_lens[i], :]
                            else:
                                h[
                                    i, text_lens[i] : text_lens[i] + bn_lens[i], :
                                ] = bn_in_h[i, : bn_lens[i], :]
                    else:
                        if self.use_ser_tag:
                            h[i, text_lens[i] : text_lens[i] + 1, :] = ser_tag_embeds[i]
                            h[
                                i, text_lens[i] + 1 : text_lens[i] + 1 + bn_lens[i], :
                            ] = bn_in_h[i, : bn_lens[i], :]
                        else:
                            h[i, text_lens[i] : text_lens[i] + bn_lens[i], :] = bn_in_h[
                                i, : bn_lens[i], :
                            ]

        h = super().forward(h, seqlen, start_pos, inference_params, cond=cond)

        if self.use_phoneme_loss:
            output = self.output(h)
        else:
            output = None

        h_output = self.h_output(h)
        stop_token = self.stop_token_head(h)

        if self.use_ser_tag_loss:
            ser_tag_output = self.ser_tag_output(h)
        else:
            ser_tag_output = None

        output_dict = {
            "logits": output.float() if output is not None else None,
            "dense": h_output.float(),
            "stop_token": stop_token.float(),
            "bn_in_z": bn_in_z,
            "attn_weights": attn_weights,
            "lang_output": lang_output.float() if lang_output is not None else None,
            "ser_tag_logits": ser_tag_output.float()
            if ser_tag_output is not None
            else None,
        }

        return output_dict

    def get_prompt_cond(self, bns, bn_mask=None):
        bn_in_z = self._repara(bns)
        if bn_mask is not None:
            bn_in_z = bn_in_z * bn_mask.unsqueeze(-1)
        prompt_cond = self.ref_enc(bn_in_z.transpose(1, 2))
        return prompt_cond

import collections
from typing import Callable, OrderedDict

import torch
from s3a.providers.ctiga.models.gpt import GPTLMHeadModel
from s3a.providers.ctiga.models.llama import remap_state_dict_huggingface_llama
from transformers import GPT2Config

from .sparse_llama import LLaMa as SparseLLama


def create_ctiga_llama(model_cls: Callable):
    model_obj = model_cls()
    if isinstance(model_obj, SparseLLama):
        return create_ctiga_from_sparse_llama(model_obj)
    return None


def get_n_inner_dim(n_embd: int, multiple_of=256):
    n_inner = 4 * n_embd
    n_inner = int(2 * n_inner / 3)
    N = multiple_of
    return ((n_inner - 1) // N) * N + N


def create_ctiga_from_sparse_llama(orig_model_obj: SparseLLama) -> GPTLMHeadModel:
    args = orig_model_obj.params
    ctiga_state_dict = remap_state_dict_huggingface_llama(
        remap_hf_llama(orig_model_obj, args.n_layers, args.dim, args.n_heads),
        args.n_layers,
        args.n_heads,
        args.dim,
    )
    ctiga_config = GPT2Config(
        n_positions=0,
        vocab_size=args.vocab_size,
        n_embd=args.dim,
        n_head=args.n_heads,
        n_layer=args.n_layers,
        n_inner=get_n_inner_dim(args.dim, args.multiple_of),
        layer_norm_epsilon=args.norm_eps,
        attn_pdrop=args.attn_pdrop,
        resid_pdrop=args.resid_pdrop,
        embd_pdrop=0.0,
        activation_function="swiglu",
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        rotary_emb_compat="default",
        tie_word_embeddings=False,
        initializer_range=0.02,
        rms_norm=True,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        use_flash_attn=True,
        fused_bias_fc=True,
        fused_mlp=False,
        fused_dropout_add_ln=True,
        residual_in_fp32=True,
        checkpointing=args.checkpointing,
    )
    ctiga_model_obj = GPTLMHeadModel(ctiga_config)
    ctiga_model_obj.load_state_dict(ctiga_state_dict, strict=False)
    return ctiga_model_obj


def remap_hf_llama_state_dict(
    orig_state_dict: OrderedDict[str, torch.Tensor],
    n_layer: int,
    hidden_size: int,
    n_head: int,
):
    remap_state_dict: OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    remap_state_dict["model.embed_tokens.weight"] = (
        orig_state_dict["tok_embeddings.weight"].clone().detach()
    )
    remap_state_dict["model.norm.weight"] = (
        orig_state_dict["norm.weight"].clone().detach()
    )
    remap_state_dict["lm_head.weight"] = (
        orig_state_dict["output.weight"].clone().detach()
    )
    head_dim = hidden_size // n_head
    for i in range(n_layer):
        attn_layer = f"layers.{i}.attention"
        qw = orig_state_dict[f"{attn_layer}.wq.weight"].clone().detach()
        qw = (
            qw.reshape(n_head, head_dim // 2, 2, hidden_size)
            .transpose(1, 2)
            .reshape(hidden_size, hidden_size)
        )
        remap_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = qw

        kw = orig_state_dict[f"{attn_layer}.wk.weight"].clone().detach()
        kw = (
            kw.reshape(n_head, head_dim // 2, 2, hidden_size)
            .transpose(1, 2)
            .reshape(hidden_size, hidden_size)
        )
        remap_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = kw

        vw = orig_state_dict[f"{attn_layer}.wv.weight"].clone().detach()
        remap_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = vw
        ow = orig_state_dict[f"{attn_layer}.wo.weight"].clone().detach()
        remap_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = ow

        mlp = f"layers.{i}.feed_forward"
        gate_w = orig_state_dict[f"{mlp}.w1.weight"].clone().detach()
        remap_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = gate_w
        down_w = orig_state_dict[f"{mlp}.w2.weight"].clone().detach()
        remap_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = down_w
        up_w = orig_state_dict[f"{mlp}.w3.weight"].clone().detach()
        remap_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = up_w

        input_norm = f"layers.{i}.attention_norm"
        remap_state_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            orig_state_dict[f"{input_norm}.weight"].clone().detach()
        )
        ffn_norm = f"layers.{i}.ffn_norm"
        remap_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            orig_state_dict[f"{ffn_norm}.weight"].clone().detach()
        )
    return remap_state_dict


def remap_hf_llama(ref_model: SparseLLama, n_layer: int, hidden_size: int, n_head: int):
    return remap_hf_llama_state_dict(
        ref_model.state_dict(), n_layer, hidden_size, n_head
    )


if __name__ == "__main__":
    # [
    #     'epoch',
    #     'global_step',
    #     'pytorch-lightning_version',
    #     'state_dict',
    #     'loops',
    #     'callbacks',
    #     'optimizer_states',
    #     'lr_schedulers',
    #     'MixedPrecisionPlugin',
    #     'hparams_name',
    #     'hyper_parameters',
    # ]
    model_ckpt = torch.load("/workspace/merlin_data/assets/last_ar.ckpt")
    print(type(model_ckpt), model_ckpt.keys())
    for k, v in model_ckpt.items():
        print("*" * 100 + "\n" + f"** {k} **")
        # if k == "optimizer_states":
        #     print(type(v))
        #     continue
        if isinstance(v, dict):
            for _k, _v in v.items():
                if isinstance(_v, torch.Tensor):
                    print(f"[{k}:{_k}] {_v.shape}")
                elif isinstance(_v, dict):
                    print(f"[{k}:{_k}] {_v}")
                else:
                    print(f"[{k}:{_k}] {_v}")
        elif isinstance(v, list):
            for i, _v in enumerate(v):
                print(f"[{k}:{i}] {_v['state'][0],_v['param_groups']}")
        else:
            print(f"[{k}] {v}")
        print("*" * 100 + "\n")

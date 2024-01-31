# Copyright (c) 2023, Tri Dao.

import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers import GPT2Config

from samantha.components.activations import sqrelu_fwd
from samantha.components.ctiga.block import Block, ParallelBlock
from samantha.components.ctiga.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from samantha.components.ctiga.mha import MHA, ParallelMHA
from samantha.components.ctiga.mlp import FusedMLP, GatedMlp, Mlp, ParallelFusedMLP
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.ctiga.localmask import ELEMWISE_WINDOW_MASK, WINDOW_MASK_TYPES
from samantha.utils.ctiga.padding import pad_input, unpad_input
from samantha.utils.ctiga.pretrained import state_dict_from_pretrained
from samantha.utils.distributed import all_gather_raw, sync_shared_params

try:
    from samantha.components.ctiga.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from samantha.components.ctiga.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from samantha.components.ctiga.ops.layer_norm import (
        dropout_add_layer_norm_parallel_residual,
    )
except ImportError:
    dropout_add_layer_norm_parallel_residual = None

try:
    from samantha.components.ctiga.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from samantha.components.ctiga.ops.rms_norm import (
        dropout_add_rms_norm_parallel_residual,
    )
except ImportError:
    dropout_add_rms_norm_parallel_residual = None

try:
    from samantha.components.ctiga.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None


logger = logging.getLogger(__name__)


def create_mixer_cls(
    config, layer_idx=None, process_group=None, device=None, dtype=None
):
    factory_kwargs = {"device": device, "dtype": dtype}
    flashattn_version = getattr(config, "flashattn_version", 2)
    assert flashattn_version in [1, 2, 2.3]
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    if dwconv:
        assert process_group is None, "TensorParallel MHA does not support dwconv yet"
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    rotary_emb_compat = getattr(config, "rotary_emb_compat", "default")
    causal = getattr(config, "causal", True)
    blocksparse = getattr(config, "blocksparse", False)
    blockmask = getattr(config, "blockmask", None)
    grad_checkpointing = getattr(config, "grad_checkpointing", False)

    use_window_mask = getattr(config, "use_window_mask", False)
    window_size = getattr(config, "window_size", [-1, -1])
    window_type = getattr(config, "window_type", ELEMWISE_WINDOW_MASK)
    if flashattn_version == 2.3:
        assert (
            use_flash_attn and process_group is None
        ), "flashattn_2.3 only support use_flash_attn=True and process_group is None"
        assert not (
            causal and use_window_mask
        ), "don't support causal=True and use_window_mask=True meanwhile"
        if use_window_mask and not causal:
            assert window_type in WINDOW_MASK_TYPES and len(window_size) == 2
        elif not use_window_mask and causal:
            window_size = [-1, 0]
            window_type = ELEMWISE_WINDOW_MASK
        else:
            window_size = [-1, -1]
            window_type = ELEMWISE_WINDOW_MASK
        window_type = WINDOW_MASK_TYPES[window_type]
    else:
        assert (
            not use_window_mask
        ), f"only support use_window_mask=True in flashattn_version=2.3 now, but got {flashattn_verison}"

    if blocksparse:
        assert (
            isinstance(blockmask, List) and len(blockmask) == config.num_hidden_layers
        )
    if not fused_bias_fc:
        assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    mha_cls = MHA if process_group is None else ParallelMHA
    serial_kwargs = (
        {"fused_bias_fc": fused_bias_fc, "dwconv": dwconv}
        if process_group is None
        else {}
    )
    parallel_kwargs = (
        {
            "process_group": process_group,
            "sequence_parallel": getattr(config, "sequence_parallel", True),
        }
        if process_group is not None
        else {}
    )

    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=causal,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        rotary_emb_compat=rotary_emb_compat,
        use_flash_attn=use_flash_attn,
        checkpointing=grad_checkpointing,
        blocksparse=blocksparse,
        blockmask=blockmask[layer_idx] if blocksparse else None,
        window_size=window_size,
        window_type=window_type,
        version=flashattn_version,
        **serial_kwargs,
        **parallel_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
        ]
    fused_dense_sqrelu_dense = getattr(config, "fused_dense_sqrelu_dense", False)
    if fused_dense_sqrelu_dense:
        assert config.activation_function == "sqrelu", (
            "fused_dense_sqrelu_dense only "
            "supports approximate activation_function sqrelu"
        )
    assert not (fused_dense_sqrelu_dense and fused_mlp)
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"
    if not fused_mlp and not fused_dense_sqrelu_dense:
        assert config.activation_function in [
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            activation = (
                F.sigmoid
                if config.activation_function == "glu"
                else (F.silu if config.activation_function == "swiglu" else F.gelu)
            )
            mlp_cls = partial(
                GatedMlp,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                multiple_of=getattr(config, "multiple_of", 256),
                **factory_kwargs,
            )
        else:
            if config.activation_function == "relu":
                activation = partial(F.relu, inplace=True)
            elif config.activation_function == "sqrelu":
                activation = sqrelu_fwd
            else:
                approximate = (
                    "tanh"
                    if config.activation_function
                    in ["gelu_new", "gelu_fast", "gelu_approx"]
                    else "none"
                )
                activation = partial(F.gelu, approximate=approximate)
            mlp_cls = partial(
                Mlp,
                hidden_features=config.n_inner,
                activation=activation,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **factory_kwargs,
            )
    else:
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        if fused_mlp:
            if FusedMLP is None:
                raise ImportError("fused_dense is not installed")
            activation = (
                "gelu_approx"
                if config.activation_function
                in ["gelu_new", "gelu_fast", "gelu_approx"]
                else config.activation_function
            )
            mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
            parallel_kwargs = (
                {
                    "process_group": process_group,
                    "sequence_parallel": getattr(config, "sequence_parallel", True),
                }
                if process_group is not None
                else {}
            )
            mlp_cls = partial(
                mlp_cls,
                hidden_features=config.n_inner,
                activation=activation,
                checkpoint_lvl=mlp_checkpoint_lvl,
                bias1=mlp_fc1_bias,
                bias2=mlp_fc2_bias,
                **parallel_kwargs,
                **factory_kwargs,
            )
        elif fused_dense_sqrelu_dense:
            assert FusedDenseSqreluDense is not None
            mlp_cls = partial(
                FusedDenseSqreluDense,
                hidden_features=config.n_inner,
                checkpoint_lvl=mlp_checkpoint_lvl,
                **factory_kwargs,
            )
        else:
            raise RuntimeError("MLP type not supported")
    return mlp_cls


def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)
    flashattn_verison = getattr(config, "flashattn_version", 2)
    mixer_cls = create_mixer_cls(
        config, layer_idx, process_group=process_group, **factory_kwargs
    )
    mlp_cls = create_mlp_cls(
        config, layer_idx, process_group=process_group, **factory_kwargs
    )
    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=config.layer_norm_epsilon,
        **factory_kwargs,
    )
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = (
        config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    )
    prenorm = getattr(config, "prenorm", True)
    parallel_block = getattr(config, "parallel_block", False)
    if not parallel_block:
        block = Block(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            prenorm=prenorm,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
            version=flashattn_verison,
            device=device,
            dtype=dtype,
        )
    else:
        assert prenorm
        block = ParallelBlock(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            tied_norm=getattr(config, "parallel_block_tied_norm", False),
            fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
            version=flashattn_verison,
        )
    block.layer_idx = layer_idx
    return block


class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config
        self.gradient_checkpointing = False

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        config,
        *args,
        strict=True,
        device=None,
        dtype=None,
        world_size=1,
        rank=0,
        **kwargs,
    ):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        self.gradient_checkpointing = True


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # print(f"rescale_prenorm_residual {name}")
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                )


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.process_group = process_group
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        # For GPT-J, GPT-NeoX
        self.parallel_block = getattr(config, "parallel_block", False)

        if process_group is None:
            self.embeddings = GPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                word_embed_proj_dim=word_embed_proj_dim,
                **factory_kwargs,
            )
        else:
            self.embeddings = ParallelGPT2Embeddings(
                config.hidden_size,
                vocab_size,
                config.max_position_embeddings,
                process_group=process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(
                    config, layer_idx=i, process_group=process_group, **factory_kwargs
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.use_unet_style_skip_connect = getattr(
            config, "use_unet_style_skip_connect", False
        )

        if self.use_unet_style_skip_connect:
            self.skip_combiner = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size * 2, config.hidden_size)
                    for _ in range(config.num_hidden_layers // 2 - 1)
                ]
            )
        self.num_hidden_layers = config.num_hidden_layers

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if (not self.parallel_block and dropout_add_layer_norm is None) or (
                self.parallel_block and dropout_add_layer_norm_parallel_residual is None
            ):
                raise ImportError("dropout_layer_norm is not installed")
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(
                config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs
            )
        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True

        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=getattr(
                    config, "rescale_prenorm_residual", True
                ),
            )
        )
        self.tie_weights()

    # adapt for huggingface
    def wte(self, input_ids):
        return self.embeddings.word_embeddings(input_ids)

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        inference_params=None,
        return_attn_probs=False,
        cond=None,
    ):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention layers need to know the seqlen.
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds.")

        embedding_kwargs = (
            {"combine_batch_seqlen_dim": True}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )

        if inputs_embeds is None:
            seqlen = input_ids.size(1)
            hidden_states = self.embeddings(
                input_ids, position_ids=position_ids, **embedding_kwargs
            )
        else:
            seqlen = inputs_embeds.size(1)
            hidden_states = inputs_embeds
            if self.embeddings.project_in is not None:
                hidden_states = self.embeddings.project_in(hidden_states)
            if self.embeddings.max_position_embeddings > 0:
                if position_ids is None:
                    position_ids = torch.arange(
                        seqlen, dtype=torch.long, device=inputs_embeds.device
                    )
                position_embeddings = self.embeddings.position_embeddings(position_ids)
                hidden_states += position_embeddings

        if self.parallel_block:
            hidden_states2 = None
        residual = None
        mixer_kwargs = (
            {"seqlen": input_ids.shape[1]}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params

        if attention_mask is not None:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, attention_mask
            )
            mixer_kwargs["cu_seqlens"] = cu_seqlens
            mixer_kwargs["max_seqlen"] = max_seqlen_in_batch

            if getattr(self.config, "rotary_emb_fraction", 0.0) > 0.0:
                mixer_kwargs["indices"] = indices
                mixer_kwargs["key_padding_mask"] = attention_mask

        if return_attn_probs:
            assert (
                inference_params is None and self.training
            ), "only support return_attn_probs=True in training"
            all_attn_probs = []

        if self.use_unet_style_skip_connect:
            skip_connects = []

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            mixer_kwargs=mixer_kwargs,
                            return_attn_probs=return_attn_probs,
                        )

                    return custom_forward

                if self.prenorm:
                    if not self.parallel_block:
                        layer_outs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer), hidden_states, residual
                        )
                        if return_attn_probs:
                            assert len(layer_outs) == 3
                            hidden_states, residual, attn_probs = layer_outs
                        else:
                            assert len(layer_outs) == 2
                            hidden_states, residual = layer_outs
                    else:
                        # NOTE: no support return_attn_probs now
                        (
                            hidden_states,
                            hidden_states2,
                        ) = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            hidden_states,
                            hidden_states2,
                            residual,
                        )
                else:
                    if not self.parallel_block:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer), hidden_states
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer), hidden_states
                        )

            else:
                if self.prenorm:
                    if not self.parallel_block:
                        layer_outs = layer(
                            hidden_states,
                            residual,
                            mixer_kwargs=mixer_kwargs,
                            return_attn_probs=return_attn_probs,
                        )
                        if return_attn_probs:
                            assert len(layer_outs) == 3
                            hidden_states, residual, attn_probs = layer_outs
                        else:
                            assert len(layer_outs) == 2
                            hidden_states, residual = layer_outs
                    else:
                        # NOTE: no support return_attn_probs now
                        hidden_states, hidden_states2, residual = layer(
                            hidden_states,
                            hidden_states2,
                            residual,
                            mixer_kwargs=mixer_kwargs,
                        )
                else:
                    if not self.parallel_block:
                        layer_outs = layer(
                            hidden_states,
                            mixer_kwargs=mixer_kwargs,
                            return_attn_probs=return_attn_probs,
                        )
                        if return_attn_probs:
                            assert len(layer_outs) == 2
                            hidden_states, attn_probs = layer_outs
                        else:
                            assert len(layer_outs) == 1
                            hidden_states = layer_outs[0]
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if return_attn_probs:
                all_attn_probs.append(attn_probs)

            if cond is not None:
                scln_scale, scln_bias = cond.chunk(2, dim=-1)
                hidden_states = scln_scale * hidden_states + scln_bias

            if self.use_unet_style_skip_connect:
                if i < self.num_hidden_layers // 2 - 1:  # [0,1,2]
                    skip_connects.append(hidden_states)
                elif (
                    i >= self.num_hidden_layers // 2 and i < self.num_hidden_layers - 1
                ):  # [4,5,6,7]
                    skip_connect = skip_connects.pop()
                    hidden_states = torch.cat([hidden_states, skip_connect], dim=-1)
                    hidden_states = self.skip_combiner[i - self.num_hidden_layers // 2](
                        hidden_states
                    )

        if attention_mask is not None:
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            residual = pad_input(residual, indices, batch, seqlen)

        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                else:
                    dropped2 = self.drop_f(hidden_states2)
                    residual = (
                        (residual + dropped + dropped2)
                        if residual is not None
                        else dropped + dropped2
                    )
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if not self.parallel_block:
                    fused_add_norm_fn = (
                        dropout_add_rms_norm
                        if isinstance(self.ln_f, RMSNorm)
                        else dropout_add_layer_norm
                    )
                    hidden_states = fused_add_norm_fn(
                        hidden_states,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                else:
                    fused_add_norm_fn = (
                        dropout_add_rms_norm_parallel_residual
                        if isinstance(self.ln_f, RMSNorm)
                        else dropout_add_layer_norm_parallel_residual
                    )
                    hidden_states, _ = fused_add_norm_fn(
                        hidden_states,
                        hidden_states2,
                        residual,
                        self.ln_f.weight,
                        self.ln_f.bias,
                        None,
                        None,
                        self.drop_f.p if self.training else 0.0,
                        self.ln_f.eps,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
        if return_attn_probs:
            return (hidden_states, all_attn_probs)
        else:
            return hidden_states

    def fwd_flop_per_token(self, seq_len):
        # an estimate of the total non-embedding forward compute
        # QKVO projection
        N = 4 * self.config.n_layer * self.config.hidden_size * self.config.hidden_size

        # Mlp(gpt) or GatedMlp(llama)
        for l in self.layers:
            N += l.mlp.fc1.weight.numel() + l.mlp.fc2.weight.numel()

        # attention QKT and VScore
        flops_per_token = (
            2 * N + 4 * self.config.n_layer * seq_len * self.config.hidden_size
        )

        # rotary emb
        if isinstance(self.ln_f, RMSNorm):
            flops_per_token += 6 * self.config.n_layer * self.config.hidden_size

        return flops_per_token


class GPTLMHeadModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.process_group = process_group
        self.transformer = GPTModel(
            config, process_group=process_group, **factory_kwargs
        )
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = (
            config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        )
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(
                config.n_embd, embed_dim, bias=False, **factory_kwargs
            )
        else:
            self.project_out = None

        num_logits = getattr(config, "num_logits", None) or vocab_size
        if process_group is None:
            self.lm_head = nn.Linear(
                embed_dim, num_logits, bias=lm_head_bias, **factory_kwargs
            )
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                embed_dim,
                num_logits,
                process_group,
                bias=lm_head_bias,
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                rescale_prenorm_residual=getattr(
                    config, "rescale_prenorm_residual", True
                ),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        self.transformer.gradient_checkpointing_enable()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        inference_params=None,
        last_token_only=False,
        return_attn_probs=False,
        output_hidden_states=False,
    ):
        """
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        last_token_only: whether to return the logit for the last token only,
            of shape (batch_size, vocab_size)
        """
        assert (inputs_embeds is None) ^ (input_ids is None)
        if return_attn_probs:
            use_flash_attn = getattr(self.config, "use_flash_attn", False)
            fa_version = getattr(self.config, "flashattn_version", 2)
            assert use_flash_attn and fa_version in [
                2
            ], "return_attn_probs=True only support in this ctiga version 2"
            assert (
                inference_params is None and self.training
            ), "only support return_attn_probs=True in training"

        outs = self.transformer(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            inference_params=inference_params,
            attention_mask=attention_mask,
            return_attn_probs=return_attn_probs,
        )

        if return_attn_probs:
            hidden_states, all_attn_probs = outs
        else:
            hidden_states = outs
        if last_token_only:
            hidden_states = hidden_states[:, -1]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if (
            isinstance(self.lm_head, ColumnParallelLinear)
            and inference_params is not None
        ):
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(
                lm_logits, "(n b) ... d -> b ... (n d)", b=hidden_states.shape[0]
            )
        return_dict = {"logits": lm_logits}
        if return_attn_probs:
            return_dict["attn_probs"] = all_attn_probs
        if output_hidden_states:
            return_dict["hidden_states"] = hidden_states
        CausalLMOutput = namedtuple("CausalLMOutput", list(return_dict.keys()))
        return CausalLMOutput(**return_dict)

    def load_state_dict(self, state_dict: Dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(
                f"transformer.layers.{n_layers - 1}.norm2.weight"
            )
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(
                        f"transformer.layers.{l - 1}.norm2.weight"
                    )
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict["transformer.layers.0.norm1.weight"] = ln_weight
            state_dict["transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)

    def generate_forward(self, input_token_ids):
        batch_size = input_token_ids.size(0)
        with torch.inference_mode():
            if self._infer_params.sequence_len_offset == 0:
                position_ids = None
            else:
                position_ids = torch.full(
                    (batch_size, 1),
                    self._infer_params,
                    dtype=torch.long,
                    device=input_token_ids.device,
                )
            logits = self.forward(
                input_token_ids,
                position_ids=position_ids,
                inference_params=self._infer_params,
                last_token_only=True,
            ).logits
        self._infer_params.sequence_len_offset += 1
        return logits

    def init_cache(self, seq_len, batch_size):
        # TODO: support ft_kernel
        setattr(
            self,
            "_infer_params",
            InferenceParams(
                max_sequence_len=seq_len,
                max_batch_size=batch_size,
                fused_ft_kernel=False,
            ),
        )

    def deinit_cache(self):
        if hasattr(self, "_infer_params"):
            delattr(self, "_infer_params")

    def fwd_flop_per_token(self, seq_len):
        N = self.lm_head.weight.numel()
        if self.project_out:
            N += self.project_out.weight.numel()
        flops_per_token = 2 * N + self.transformer.fwd_flop_per_token(seq_len)
        return flops_per_token


def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[0] // world_size
        state_dict[key] = x[rank * dim : (rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        x = state_dict[key]
        dim = x.shape[-1] // world_size
        state_dict[key] = x[..., rank * dim : (rank + 1) * dim]

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(
            x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ..."
        )

    shard_first_dim(state_dict, "transformer.embeddings.word_embeddings.weight")
    if "lm_head.weight" in state_dict:
        shard_first_dim(state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        shard_last_dim(state_dict, "transformer.embeddings.position_embeddings.weight")
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mixer.out_proj.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mixer.out_proj.bias")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.weight")
        shard_first_dim(state_dict, f"transformer.layers.{i}.mlp.fc1.bias")
        shard_last_dim(state_dict, f"transformer.layers.{i}.mlp.fc2.weight")
        if rank != 0:
            state_dict.pop(f"transformer.layers.{i}.mlp.fc2.bias")
    return state_dict


def combine_state_dicts_tp(state_dicts, config):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    world_size = len(state_dicts)
    # keys = state_dicts[0].keys()
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    # Sometimes the word embeddings are sharded on the 0th dim, sometimes on the 1st dim.
    # vocab_size // world_size coordinates are nonzero.
    def combine_word_embeddings(state_dicts, state_dict, key):
        dim = 0 if state_dicts[0][key].shape[0] == vocab_size // world_size else 1
        state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_dim(state_dicts, state_dict, key, dim=-1):
        if key in state_dict:
            state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_qkv_headdim(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [
                rearrange(s[key], "(three d) ... -> three d ...", three=3)
                for s in state_dicts
            ]
            state_dict[key] = rearrange(
                torch.cat(xs, dim=1), "three d ... -> (three d) ..."
            )

    def combine_gated_mlp(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [
                rearrange(s[key], "(two d) ... -> two d ...", two=2)
                for s in state_dicts
            ]
            state_dict[key] = rearrange(
                torch.cat(xs, dim=1), "two d ... -> (two d) ..."
            )

    state_dict = state_dicts[0].copy()  # don't modify state_dict[0] inplace
    combine_word_embeddings(
        state_dicts, state_dict, "transformer.embeddings.word_embeddings.weight"
    )
    if "lm_head.weight" in state_dict:
        combine_word_embeddings(state_dicts, state_dict, "lm_head.weight")
    if "transformer.embeddings.position_embeddings.weight" in state_dict:
        combine_dim(
            state_dicts,
            state_dict,
            "transformer.embeddings.position_embeddings.weight",
            -1,
        )
    mlp_combine_fn = (
        combine_gated_mlp
        if config.activation_function in ["glu", "swiglu", "geglu"]
        else partial(combine_dim, dim=0)
    )
    for i in range(config.num_hidden_layers):
        combine_qkv_headdim(
            state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.weight"
        )
        combine_qkv_headdim(
            state_dicts, state_dict, f"transformer.layers.{i}.mixer.Wqkv.bias"
        )
        combine_dim(
            state_dicts, state_dict, f"transformer.layers.{i}.mixer.out_proj.weight", -1
        )
        mlp_combine_fn(
            state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.weight"
        )
        combine_dim(state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc1.bias", 0)
        combine_dim(
            state_dicts, state_dict, f"transformer.layers.{i}.mlp.fc2.weight", -1
        )
    return state_dict


def remap_state_dict_hf_gpt2(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("wte.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict[
        "transformer.embeddings.word_embeddings.weight"
    ]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(
            r"^h.(\d+).ln_(1|2).(weight|bias)", r"transformer.layers.\1.norm\2.\3", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f"h.{d}.mlp.c_fc.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc1.weight"] = W1.t()
        W2 = state_dict.pop(f"h.{d}.mlp.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc2.weight"] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub(
            r"^h.(\d+).mlp.c_fc.bias", r"transformer.layers.\1.mlp.fc1.bias", key
        )
        key = re.sub(
            r"^h.(\d+).mlp.c_proj.bias", r"transformer.layers.\1.mlp.fc2.bias", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        state_dict.pop(f"h.{d}.attn.bias")  # We don't store this bias
        Wqkv = state_dict.pop(f"h.{d}.attn.c_attn.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = Wqkv.t()
        Wout = state_dict.pop(f"h.{d}.attn.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mixer.out_proj.weight"] = Wout.t()

    def key_mapping_attn(key):
        key = re.sub(
            r"^h.(\d+).attn.c_attn.bias", r"transformer.layers.\1.mixer.Wqkv.bias", key
        )
        key = re.sub(
            r"^h.(\d+).attn.c_proj.bias",
            r"transformer.layers.\1.mixer.out_proj.bias",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def remap_state_dict_megatron(state_dict, config):
    def key_mapping_transformer(key):
        key = re.sub(r"^language_model.encoder.", "transformer.", key)
        key = re.sub(r"^language_model.", "transformer.", key)
        return key

    state_dict = OrderedDict(
        (key_mapping_transformer(k), v) for k, v in state_dict.items()
    )

    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^wpe.", "transformer.embeddings.position_embeddings.", key)

    state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embedding.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple)
        * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict[
        "transformer.embeddings.word_embeddings.weight"
    ]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(
            r"^transformer.final_layernorm.(weight|bias)", r"transformer.ln_f.\1", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).input_layernorm.(weight|bias)",
            r"transformer.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).post_attention_layernorm.(weight|bias)",
            r"transformer.layers.\1.norm2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_h_to_4h.(weight|bias)",
            r"transformer.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_4h_to_h.(weight|bias)",
            r"transformer.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.rotary_emb.inv_freq",
            r"transformer.layers.\1.mixer.rotary_emb.inv_freq",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.query_key_value.(weight|bias)",
            r"transformer.layers.\1.mixer.Wqkv.\2",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.dense.(weight|bias)",
            r"transformer.layers.\1.mixer.out_proj.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    # Megatron stores Wqkv as ((nheads 3 headdim), hidden_dim)
    # while we store Wqkv as ((3 nheads headdim), hidden_dim)
    headdim = config.hidden_size // config.num_attention_heads
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = rearrange(
            Wqkv,
            "(nheads three headdim) ... -> (three nheads headdim) ...",
            three=3,
            headdim=headdim,
        )
        bqkv = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.bias")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.bias"] = rearrange(
            bqkv,
            "(nheads three headdim) -> (three nheads headdim)",
            three=3,
            headdim=headdim,
        )

    return state_dict

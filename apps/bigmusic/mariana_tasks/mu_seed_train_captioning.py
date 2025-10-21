# This file describe training and infer tasks of semantic seed model.

from collections import defaultdict
import datetime
import gc
import itertools
import os
import sys
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import init_empty_weights, init_on_device
from accelerate.utils.modeling import set_module_tensor_to_device
from cruise import CruiseCLI, CruiseConfig, CruiseModule, last_cli
from cruise.configuration.cli import namespace_to_cruise_config
from cruise.module.model_io import _partial_load_from_checkpoint
from cruise.trainer.callback import ModelCheckpoint
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hput
from panther.custom_ops.torch.flash_attn import unpad_input
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from transformers import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig

from mariana.models.audio.batch_dispatcher.llm_batch_dispatcher import LLMBatchDispatcher
from mariana.models.audio.speech_checkpoint import SpeechModelCheckpoint
import samantha # noqa: F401, resolve mariana python path
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.models.audio.weight_init import ModuleInitializer
from samantha.dataio.bigmusic.lite import MusicLiteDataModule
from samantha.criterion.masked_loss import sequence_mask
from apps.bigmusic.mariana_tasks.semantic_emb_module import SemanticEmbModule
from apps.bigmusic.mariana_tasks.utils.speech_data_collect_callback import SpeechDataCollectCallback
from mariana.models.audio.gpt2_audio import (
    GPT2LMHeadModel,
    inplace_update_megatron_state_dict,
)
from mariana.models.audio.moe import MoETopkCapGate
from mariana.models.audio.utils._fsdp_utils import (
    _uppdate_fsdp_post_backward_reshard_hook,
)
from mariana.utils.exp_helper import ExpHelper
from tasks.audio.audio_trainer import AudioTrainer
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.utils.comm_utils import get_formated_model_summary_table
from tasks.audio.utils import (
    get_optimizer_grouped_parameters,
    upload_single_compressed_trace,
    upload_trace,
)
from tasks.audio.ndtimeline import get_ndtimeline_profile
from torch.nn.utils.rnn import pad_sequence

import random
from tqdm.auto import tqdm
from recipes.bigmusic.lightning.base_modules import TokenBuffer
from recipes.musiclm.inference.utils import sample, adaptive_sampling, SamplingScheduler
from samantha.utils.hparams import DotDict

from recipes.umm2.models.config import UMMConfig
from recipes.umm2.models.umm_fm import UMMModified as UMM

from mariana.models.audio.acoustic_head import (
    TimePatchHead,
)
from types import SimpleNamespace
from pathlib import Path
import json


logger = AudioLogger()

# default config for M8 MoE-680M LLM
_m8_network_config = {
    "llm_empty_init": True,
    "tokenizer_path": "hdfs://haruna/home/byte_data_aml_research/user/anzhecheng/tokenizer/bbpe155k-v6.4.3-ml.pret",
    "ignored_llm_missing_keys": ["inv_freq"],
    # Arch
    "arch": "m8",
    "hidden_size": 1152,
    "n_embed": 1152,  # vocab embedding
    "n_inner": 576,
    "n_layer": 28,
    "vocab_size": 155136,
    "tie_weight": True,
    "pad_idx": 1,
    "activation_function": "swiglu",
    "embd_pdrop": 0.0,
    "resid_pdrop": 0.1,
    "initializer_range": 0.0161,
    # ROPE
    "max_position_embeddings": 32768,
    "position_embeddings_type": "rope",
    "rope_mode": "ntk",
    "rope_base": 10000,
    "rope_scale": 50,
    "rope_dim": 32,

    # Attn
    "kv_mirror_imitated_layers": [0,1,2,3,4,5,6,7],
    "kv_mirror_layers": [20,21,22,23,24,25,26,27],
    "query_head_scale_factor": 2,
    "use_attention_bias": False,
    "attn_pdrop": 0.1,
    "n_head": 12,
    "n_shared_qhead": 3,

    # Pre-post norm
    "pre_post_layernorm_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    "layer_norm_type": "rmsnorm",
    "layer_norm_epsilon": 1.0e-5,

    # MoE
    "transformer_kernel": "default",
    "mlp_type": "moe",
    "moe_expert_type": 'swiglu-melego',
    "moe_gate_type": 'cap-lego',
    "moe_expert_op_version": 'V6',
    "moe_expert_eq_dim_factor": 1.0,
    "moe_topk": 6,
    "moe_num_expert": 128,
    "moe_share_expert_num": 2,
    "moe_ema_calc_type": 'local',
    "moe_aux_loss_weight": 0.005,
    #
    "mlp_bias": False,
    "ignore_ln2": False,  # applicable to native implementation only.

    #
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "noop_transformer_layers": [],
    "use_non_sequential_block": False,
    "init_weight": True,  # original is False
    "eos_loss_scalar": 1.0,

    # HPC configs
    "use_flash_attn": True,
    "use_fused_layernorm": True,
    "use_bumi": True,
    "use_bumi_gemm": False,
    "gpt_use_fused_block": False,
    "gpt_selective_recompute": 0,
    "gpt_fused_block_full_recompute_start_layer_index": 100,
    "use_flash_attn_kvcache": False,   # infer with kvcache
    # precision configs
    "use_llm_fp16": False,
    "use_llm_bf16": False,
    "freeze_llm": False,
    "freeze_umm": False,
    "use_flash_ce": False,
    # checkpointing configs
    "gradient_checkpointing_start_layers": 100,
    "gradient_checkpointing": False,
    # longcontext configs
    "distributed_sequence_parallel_size": -1,
    "context_parallel_size": -1,
    #
    "gpt_activation_offload": False,
    "gpt_activation_offload_memory_pool_size_in_gb": 30,
    "expert_parallel_size": 1,
    "use_bumi_ep": False,
    "expert_group_capacity": 1.0,
    "share_expert_num": 0,
    # balance configs
    "balance_llm": False,
    "llm_balance_policy": "llm_tokens",
    "balance_group_size": -1,
    # lora configs
    "use_lora": False,
    "lora_config": {
        "rank": 64,  # rank of lora adapter's weights
        "target_modules": ["attn", "mlp"],  # modules to use lora
        "lora_alpha": 32,  # lora_alpha / rank is the scale factor for lora adapter's output
        "lora_dropout": 0.01,  # lora adapter' dropout rate
        "bias": "all",  # 'all' means unfreeze bias, 'none' means freeze bias
        "exclude_embedding": False,  # whether to unfreeze wte embedding
        "exclude_all": False,  # whether to unfreeze all the parameters except the weights in Transformer
        "merge_lora": False,  # whether to merge lora adapter under inference / export mode
        "group_size": -1,  # to use qa-lora, set it to the group size used for quantization. -1 means using conventional lora  # noqa
    },
    # others
    "strict_fsdp_memory": False,
    "profile_memory": False,
    "memory_profile_end_step": 10,
    "trace_ndtimeline": False,
    "ndtimeline_range": "(20, 40)",
    "return_moe_metric": False,

    # semantic emb config,
    'emb_config_path': 'apps/bigmusic/mariana_tasks/conf/v5_emb.yaml',
    # Refactored emb config
    # 'emb_config_path': 'apps/bigmusic/mariana_tasks/conf/v5_semantic_embedder.yaml',
    'emb_config':{
        'extra_params': {
            'bpe_tokenizer_path': 'bbpe155k-v6.4.3-ml.pret'
        }
    }
}

_umm_network_config = {
    "feature_cmvn": "./recipes/datasets/mcc/data_ids=794.stats.pt",
    "sample_rate": 24000,
    "n_mels": 128,
    "n_mels_tgt": 160,
    "add_ctc": True,
    "add_pitch": True,
    "add_chroma": True,
    "n_chroma": 12,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 240,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "max_source_positions": 2500,
    "w_loss_mel": 1.0,
    "w_loss_chroma": 1.0,
    "w_loss_pitch": 1.0,
    "w_loss_ctc": 1.0,
    "tokenizer": "bert-base-multilingual-uncased",
    "frame_rate": 25,
    "vocab_size": 105880,
    "ctc_blank_id": 105879,
    "ctc_zero_infinity": True,
    "use_bn": False,
    "first_conformer": True,
    "act_fn": "gelu",
    "mask_mel": False,
    "rope_enhance_pos": 15000,
    "add_ctc": False,
    "add_chroma": False,
    "add_mel": False,
    "add_pitch": False,
    "downsample_size": -1,
    "use_fused_kernel": True
}


_inference_config = {
    "num_beams": 1,
    "temperature": 0.8,
    "max_new_tokens": 30000,  
    "top_p": 0.9,
    "top_k": 1,
    "do_sample": True,
    "early_stopping": False,
    "length_penalty": 1,
    "return_dict_in_generate": True,
    "num_return_sequences": 1,
    "bos_token_id": 0,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "use_cache": True,
    "result_file": "./results_tagging.json",
    "result_dir": "./results"
}

def _get_local_path(hdfs_path):
    
    if hdfs_path:
        local_path = os.path.join("./", os.path.basename(hdfs_path))
    else:
        local_path = None
    return local_path

class MaskedAvgPool1d(nn.Module):
    """ A module that performs masked average pooling. """

    def __init__(self):
        super().__init__()


    def forward(self, x: torch.Tensor, mask: torch.Tensor):

        x_masked = x.masked_fill(~mask.unsqueeze(-1), 0.0)  # (B, T, D)

        sum_pooled = x_masked.sum(dim=1, keepdim=True)  # (B, 1, D)

        num_valid = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

        pooled = sum_pooled / (num_valid.float() + 1e-8)  # (B, 1, D)

        new_mask = torch.ones((x.size(0), 1), dtype=torch.bool, device=x.device)  # (B, 1)

        return pooled, new_mask


class AudioAdapter(nn.Module):
    def __init__(self, downsample_size, input_size, output_size):
        super().__init__()
        downsample_size = int(os.environ.get('DOWNSAMPLE'))
        # print('downsample_size:', downsample_size)
        logger.info(f"[!!] downsample_size: {downsample_size}")

        if downsample_size > 1:
            logger.info(f"[!!] downsample_size > 1 Pass")
            args = SimpleNamespace(patch_size=downsample_size)
            self.downsampler = TimePatchHead(args)
            self.projection = nn.Linear(input_size * downsample_size, output_size)
        elif downsample_size == -1:
            logger.info(f"[!!] downsample_size == -1 Pass")
            self.downsampler = MaskedAvgPool1d()
            self.projection = nn.Linear(input_size, output_size)
        else:
            self.downsampler = nn.Identity()
            self.projection = nn.Linear(input_size, output_size)

    def forward(self, x, x_mask):
        
        if isinstance(self.downsampler, (TimePatchHead, MaskedAvgPool1d)):

            x, mask = self.downsampler(x, x_mask)
        else:
            x = self.downsampler(x)
            mask = x_mask

        x = self.projection(x)

        return x, mask

class SemanticLlmModel(CruiseModule):
    def __init__(
            self,
            network: CruiseConfig = CruiseConfig(dict(_m8_network_config)),
            adapter_path='',
            llm_path='',
            umm_path='',
            hybrid_shard_group_size=-1,
            ddp_gate=False,
            inference_config: CruiseConfig = CruiseConfig(dict(_inference_config)),
    ):
        super().__init__()
        self.save_hparams()
        self.hybrid_shard_group_size = hybrid_shard_group_size
        self.ddp_gate = ddp_gate
        self.skip_num: Dict[int, Dict[str, int]] = {}  # record skip status from dataloader
        self.local_llm_path = _get_local_path(self.hparams.llm_path)
        self.local_umm_path = _get_local_path(self.hparams.umm_path)

        self.inference_config = inference_config
        
        self.local_tokenizer_path = _get_local_path(self.hparams.network.get("tokenizer_path", None))

        # NOTE: add partial pretrain loading if we need

        with init_on_device(torch.device('cpu')): # if DIST_ENV.local_rank == 0 else init_empty_weights():
            umm_config = UMMConfig(**_umm_network_config)
            self.audio_encoder = UMM(umm_config)
            self.gpt2 = GPT2LMHeadModel(self.hparams)
            
            self.audio_adapter = AudioAdapter(
                                 downsample_size=umm_config.downsample_size, ##### downsampling
                                 input_size=umm_config.hidden_size, 
                                 output_size=self.hparams.network.hidden_size)

            self.trace_upload_finished = False
            if self.hparams.network.get("trace_ndtimeline", False):
                self.ndtimeline = get_ndtimeline_profile(self.hparams.network.ndtimeline_range)

        self.BOS_IDX = self.hparams.network.get('bos_idx', 0)
        self.PAD_IDX = self.hparams.network.get('pad_idx', 1)  
        self.EOS_IDX = self.hparams.network.get('eos_idx', 2)
        self.label_smoothing = self.hparams.network.get('label_smoothing', 0.0)
        self.eos_loss_scalar = self.hparams.network.get('eos_loss_scalar', 1.0)
        # generate with cuda graph
        self.past_key_values = None
        self.graph = None
        self.graph_uncond = None

        self.generation_config = GenerationConfig(**self.inference_config)



    def local_rank_zero_prepare(self):
        """Download all necessary files like model weights, tokenizer, and UMM files.
        Only executed on rank zero in distributed training, with progress logging."""

        # Download LLM weights if needed
        if self.local_llm_path and (not os.path.isfile(self.local_llm_path)):
            logger.info(f"Downloading {self.hparams.llm_path} to {self.local_llm_path}")
            hcopy(self.hparams.llm_path, self.local_llm_path, chunk_thread_num=64)
        logger.info(f"LLM weights available at: {self.local_llm_path}")
        
        # Download tokenizer if needed
        if self.local_tokenizer_path and (not os.path.isfile(self.local_tokenizer_path)):
            logger.info(f"Downloading {self.hparams.network.tokenizer_path} to {self.local_tokenizer_path}")
            hcopy(self.hparams.network.tokenizer_path, self.local_tokenizer_path, chunk_thread_num=64)
        logger.info(f"Tokenizer available at: {self.local_tokenizer_path}")
        
        # Download UMM files if needed
        if hasattr(self, 'local_umm_path') and self.local_umm_path:
            os.makedirs(os.path.dirname(self.local_umm_path), exist_ok=True)
            if not os.path.isfile(self.local_umm_path):
                logger.info(f"Downloading {self.hparams.umm_path} to {self.local_umm_path}")
                hcopy(self.hparams.umm_path, self.local_umm_path, chunk_thread_num=64)
            logger.info(f"UMM files available at: {self.local_umm_path}")

    def update_process_group(self):
        if not self.check_strategy('fsdp'):
            return
        if (
            self.trainer._strategy.fsdp_params['sharding_strategy'] != ShardingStrategy.HYBRID_SHARD
            or self.hybrid_shard_group_size < 0
        ):
            return
        if self.hybrid_shard_group_size >= DIST_ENV.world_size:
            self.trainer._strategy.fsdp_params['sharding_strategy'] = ShardingStrategy.FULL_SHARD
        else:
            assert DIST_ENV.world_size % self.hybrid_shard_group_size == 0
            # setup dsp/cp group after setup model and traning strategy
            if DIST_ENV.world_size > 1 and self.hybrid_shard_group_size > 1:
                # create fsdp shard group and replicate group,
                # if world size = 8, rank_id = [0,1,2,3,4,5,6,7],
                # for hybrid_shard_group_size=4,
                # the shard group would be[[0,1,2,3],[4,5,6,7]]
                # and the replicate group would be [[0,4],[1,5],[2,6],[3,7]]
                shard_group = None
                replicate_group = None
                timeout_seconds = int(os.environ.get("CRS_NCCL_TIMEOUT_SECOND", 1800))
                # create shard_group
                num_shard_group = DIST_ENV.world_size // self.hybrid_shard_group_size
                for i in range(num_shard_group):
                    rank_ids = [i * self.hybrid_shard_group_size + j for j in range(self.hybrid_shard_group_size)]
                    tmp_group = dist.new_group(
                        rank_ids,
                        backend='nccl',
                        timeout=datetime.timedelta(seconds=timeout_seconds),
                    )
                    if DIST_ENV.rank in rank_ids:
                        shard_group = tmp_group
                # create replicate group
                for i in range(self.hybrid_shard_group_size):
                    rank_ids = [i + j * self.hybrid_shard_group_size for j in range(num_shard_group)]
                    tmp_group = dist.new_group(
                        rank_ids,
                        backend='nccl',
                        timeout=datetime.timedelta(seconds=timeout_seconds),
                    )
                    if DIST_ENV.rank in rank_ids:
                        replicate_group = tmp_group
                self.trainer._strategy.fsdp_params['process_group'] = tuple([shard_group, replicate_group])

        self.rank_zero_info(
            f'fsdp shard strategy: {self.trainer._strategy.fsdp_params["sharding_strategy"]}, '
            f'process group: {self.trainer._strategy.fsdp_params["process_group"]}'
        )
    def init_ignored_params(self):
        if not self.check_strategy('fsdp'):
            return
        # update ignored state from config
        ignored_module_filters = self.trainer._strategy.fsdp_params.get('ignored_module_filters', None)
        ignored_parameter_filters = self.trainer._strategy.fsdp_params.get('ignored_parameter_filters', None)
        ignored_modules = self.trainer._strategy.fsdp_params.get('ignored_modules', None)
        self.trainer._strategy._update_ignored_modules()
        self.trainer._strategy._update_ignored_module_by_filters()
        self.trainer._strategy._update_ignored_parameters_by_filters()
        ignored_params = set()
        if 'ignored_states' in self.trainer._strategy.fsdp_params:
            ignored_params = set(self.trainer._strategy.fsdp_params["ignored_states"])
        elif 'ignored_parameters' in self.trainer._strategy.fsdp_params:
            ignored_params = set(self.trainer._strategy.fsdp_params["ignored_parameters"])

        device = torch.device("cuda", torch.cuda.current_device())
        # ignored modules
        for name, module in self.trainer.model.named_modules():
            if module not in self.trainer._strategy.fsdp_params['ignored_modules']:
                continue
            self.rank_zero_print(f"module:{name} has been init")
            if DIST_ENV.rank != 0:
                module.to_empty(device=device)
            module = module.to(device)
            if torch.distributed.is_initialized():
                for _name, param in module.named_parameters():
                    torch.distributed.broadcast(param.data, src=0)
        # ignored parameters
        ignored_name = set()
        for name, param in self.trainer.model.named_parameters():
            if param not in ignored_params:
                continue
            ignored_name.add(name)
            if DIST_ENV.rank != 0:
                set_module_tensor_to_device(
                    module=self.trainer.model,
                    tensor_name=name,
                    device=device,
                    value=torch.empty_like(param.data, device=device),
                )
        for name, param in self.trainer.model.named_parameters():
            if name in ignored_name:
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(param.data.to(device), src=0)
                self.rank_zero_print(f"param:{name} has been init")
        # reset ignored config for fsdp init
        if ignored_module_filters is not None:
            self.trainer._strategy.fsdp_params['ignored_module_filters'] = ignored_module_filters
        if ignored_parameter_filters is not None:
            self.trainer._strategy.fsdp_params['ignored_parameter_filters'] = ignored_parameter_filters
        if ignored_modules is not None:
            self.trainer._strategy.fsdp_params['ignored_modules'] = ignored_modules

    def load_emb_weights(self):
        state_dict = None
        if self.local_emb_path and DIST_ENV.local_rank == 0:
            state_dict_all = torch.load(self.local_emb_path, map_location='cpu', weights_only=True)
            if 'model' in state_dict_all:
                # dolphin model
                state_dict = state_dict_all['model']
            elif "state_dict" in state_dict_all:
                # mariana model and samantha model
                state_dict = state_dict_all['state_dict']
            else:
                state_dict = state_dict_all
        # NOTE: redefine names in state dict if need
        if state_dict is not None:
            incompatible_keys = self.emb.load_state_dict(state_dict, strict=False)
            self.rank_zero_info(f"Semantic emb model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"Semantic emb model unexpected keys are {incompatible_keys.unexpected_keys}")
            assert len(incompatible_keys.missing_keys) == 0
            state_dict.clear()
            del state_dict
            gc.collect()

    def load_umm_and_adapter_weights(self):
        state_dict = None
        logger.info(f"load_umm_weights {self.local_umm_path=}")
        if self.local_umm_path and DIST_ENV.local_rank == 0:

            #Current UMM has the following useless keys
            #will remove for next version
            rename_params = {'model.stages.0.': ''}
            span_keys = {f'model.stages.1.spans.{i}.': '' for i in range(4)}
            rename_params.update(span_keys)

            state_dict= _partial_load_from_checkpoint(
                self.local_umm_path,
                rename_params=rename_params,
                map_location='cpu',
                mmap=True)
            
            #Load umm
            incompatible_keys = self.audio_encoder.load_state_dict(state_dict, strict=False)
            self.rank_zero_info(f"UMM model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"UMM model unexpected keys are {incompatible_keys.unexpected_keys}")  

            #Load adapter
            incompatible_keys = self.audio_adapter.load_state_dict(state_dict, strict=False)
            self.rank_zero_info(f"adapter model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"adapter model unexpected keys are {incompatible_keys.unexpected_keys}")  

            state_dict.clear()
            del state_dict
            gc.collect()
            self.rank_zero_info("Successfully loaded umm weights")



    def check_strategy(self, target_strategy):
        if hasattr(self.trainer, '_strategy'):
            strategy = getattr(self.trainer, '_strategy')
            return strategy.name() == target_strategy
        else:
            return False

    def load_llm_weights(self):
        logger.info(f"load_llm_weights {self.local_llm_path=}")
        if self.local_llm_path and DIST_ENV.local_rank == 0:
            state_dict = _partial_load_from_checkpoint(
                self.local_llm_path,
                rename_params={'gpt.': '', 'gpt2': ''},
                map_location='cpu',
                mmap=True,
            )
            state_dict = inplace_update_megatron_state_dict(self.hparams.network, state_dict)
            incompatible_keys = self.gpt2.load_state_dict(state_dict, strict=False)
            self.rank_zero_info(f"LLM model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"LLM model unexpected keys are {incompatible_keys.unexpected_keys}")
            ignored_llm_missing_keys = self.hparams.network.get("ignored_llm_missing_keys", [])

            filtered_missing_keys = []
            for k in incompatible_keys.missing_keys:
                if not any(key in k for key in ignored_llm_missing_keys):
                    filtered_missing_keys.append(k)
            assert len(filtered_missing_keys) == 0, [k for k in filtered_missing_keys]
            state_dict.clear()
            del state_dict
            gc.collect()
            self.rank_zero_info("Successfully loaded llm weights")

    def update_fsdp_mixed_precision(self):
        if not self.check_strategy('fsdp'):
            return
        mixed_precision_policy = self.trainer._strategy.fsdp_config.get('mixed_precision', None)
        if mixed_precision_policy is None:
            return
        if not isinstance(mixed_precision_policy, MixedPrecision):
            dtype_map = {
                'bf16': torch.bfloat16,
                'fp16': torch.half,
                'fp32': torch.float32,
                'None': None,
            }
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype_map[mixed_precision_policy.get('param_dtype', 'None')],
                reduce_dtype=dtype_map[mixed_precision_policy.get('reduce_dtype', 'None')],
            )
        # wrap "MoETopkCapGate" and ignored in mixed precision
        if (
            mixed_precision_policy.param_dtype is not None
            and mixed_precision_policy.param_dtype is not torch.float32
            and not self.ddp_gate
        ):
            mixed_precision_policy._module_classes_to_ignore += (MoETopkCapGate,)

        self.rank_zero_info(f'fsdp mixed precision policy: {mixed_precision_policy}')
        self.trainer._strategy.fsdp_params['mixed_precision'] = mixed_precision_policy

    def setup(self, stage='fit', log_model_info=True, update_wrap=True):
        # load embedder's encoder weight

        self.balance_group_size = self.hparams.network.get('balance_group_size', -1)
        if self.balance_group_size > 0:
            self.balance_group_size = min(self.balance_group_size, DIST_ENV.world_size)
            assert DIST_ENV.world_size % self.balance_group_size == 0
            timeout_seconds = int(os.environ.get("CRS_NCCL_TIMEOUT_SECOND", 1800))
            for i in range(DIST_ENV.world_size // self.balance_group_size):
                rank_ids = [i * self.balance_group_size + j for j in range(self.balance_group_size)]
                cur_group = dist.new_group(
                    rank_ids,
                    backend='nccl',
                    timeout=datetime.timedelta(seconds=timeout_seconds),
                )
                if DIST_ENV.rank in rank_ids:
                    self.balance_group = cur_group
        else:
            self.balance_group = None
        self.llm_batch_dispatcher = LLMBatchDispatcher(balance_policy=self.hparams.network.llm_balance_policy)

        # check sync_module_states flag in FSDP
        if self.check_strategy('fsdp'):
            assert self.trainer._strategy.fsdp_params.get(
                'sync_module_states', False
            ), ('sync_module_states is required to train acllm_model with FSDP.'
                'Please set this flag to True in your fsdp_config or use SemanticLlmCLI.')
            if self.hparams.network.get('moe_type', '') == 'moe':
                mixed_precision_strategy = self.trainer._strategy.fsdp_params.get('mixed_precision', False)
                if mixed_precision_strategy:
                    assert (
                        mixed_precision_strategy.param_dtype == torch.bfloat16
                    ), 'Native moe can only process bf16 params.'
        # build tokenizer
        if self.local_tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_tokenizer_path)

        # build model weight dtype
        if self.hparams.network.use_llm_bf16:
            self.rank_zero_info("convert llm weight to bf16")
            if self.hparams.network.position_embeddings_type != 'absolute':
                for name, param in self.gpt2.named_parameters():
                    if (
                        'mlp.gate' in name
                        or 'mlp_extra.gate' in name
                        or 'wte.text_embedding' in name
                        or 'wte.extra_embedding' in name
                        or not param.is_floating_point()
                    ):
                        continue
                    param.data = param.data.bfloat16()
                for name, buf in self.gpt2.named_buffers():
                    if (
                        'mlp.gate' in name
                        or 'mlp_extra.gate' in name
                        or 'wte.text_embedding' in name
                        or 'wte.extra_embedding' in name
                        or not buf.is_floating_point()
                    ):
                        continue
                    buf.data = buf.data.bfloat16()
            else:
                # keep layernorm and all bias in fp32 for 13b model
                self.gpt2._apply(lambda t: t.bfloat16() if t.is_floating_point() and len(t.shape) > 1 else t)
            assert self._trainer.precision in [
                'bf16',
                'fp32',
                32,  # HACK: for pl trainer
            ], f'use same dtype for llm and `trainer.precision`={self._trainer.precision} for better performance'
        elif self.hparams.network.use_llm_fp16:
            self.rank_zero_info("convert llm weight to fp16")
            if self.hparams.network.position_embeddings_type != 'absolute':
                self.gpt2._apply(lambda t: t.half() if t.is_floating_point() else t)
            else:
                # keep layernorm and all bias in fp32 for 13b model
                self.gpt2._apply(lambda t: t.half() if t.is_floating_point() and len(t.shape) > 1 else t)
            assert self._trainer.precision in [
                'fp16',
                'fp32',
                32,
            ], 'use same dtype for llm and `trainer.precision` for better performance'

        # load llm weights
        self.load_llm_weights()
        self.load_umm_and_adapter_weights()

        # freeze semantic model
        if stage != 'fit' or self.hparams.network.freeze_umm :
            self.audio_encoder.requires_grad_(False)
            logger.info(f"freeze umm model")
        # freeze llm model
        if stage != "fit" or self.hparams.network.freeze_llm:
            self.gpt2.requires_grad_(False)
            logger.info(f"freeze LLM model")

        if self.hparams.network.get('unfreeze_moe_gate', False) and stage == 'fit' and hasattr(self.gpt2, 'transformer'):
            for h in self.gpt2.transformer.h:
                h.mlp.gate.requires_grad_(True)
            logger.info('Unfrozen MoE gates.')

        # update fsdp process_group
        self.update_process_group()
        # init ignored parameters
        self.init_ignored_params()
        # update fsdp mixed precision policy
        self.update_fsdp_mixed_precision()

        # broadcast weight for ddp
        if self.check_strategy('ddp'):
            device = torch.device(f'cuda:{DIST_ENV.local_rank}')
            if DIST_ENV.rank != 0:
                self.trainer.model.to_empty(device=device, recurse=True)
            self.trainer._model = self.trainer._model.to(device)
            if torch.distributed.is_initialized():
                for _name, param in self.trainer.model.named_parameters():
                    torch.distributed.broadcast(param.data, src=0)
                for _name, buffer in self.trainer.model.named_buffers():
                    torch.distributed.broadcast(buffer.data, src=0)
            self.rank_zero_info("Successfully broadcast weight for ddp")

        if DIST_ENV.rank == 0 and log_model_info:
            summary = get_formated_model_summary_table(self)
            logger.info(summary)

    def on_train_start(self):
        torch.cuda.synchronize()
        if self.hparams.network.get("trace_ndtimeline", False):
            self.ndtimeline.start()
        if self.hparams.network.get('profile_memory', False) and DIST_ENV.rank == 0:
            torch.cuda.memory._record_memory_history()
        # hack for fsdp prefetch
        if self.hparams.network.get('strict_fsdp_memory', False) and self.trainer._strategy.name() == 'fsdp':
            self.rank_zero_info(
                "The strict mode of fsdp VRAM has been enabled, will strictly limit the use of fsdp VRAM."
            )
            _uppdate_fsdp_post_backward_reshard_hook()

    def on_validation_end(self):
        if isinstance(self.trainer.model, FSDP):
            _post_backward_final_callback(self.trainer.model, self.trainer.model)

    def on_fit_start(self):
        if DIST_ENV.rank == 0:
            print(f"Model:\n{self.trainer.model}")

        if isinstance(self.trainer.model, FSDP):
            # check all ignored parameters are freezed
            success = True
            for name, p in self.trainer.model.named_parameters():
                if p in self.trainer.model._ignored_params:
                    if p.requires_grad:
                        success = False
                        print(f"Param:{name} is unfreezed but ignored by FSDP.")
            assert success

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.network.get("trace_ndtimeline", False):
            self.ndtimeline.step()
        if DIST_ENV.local_rank == 0 and self.trainer._enable_profiler and not self.trace_upload_finished:
            self.trace_upload_finished = upload_trace(self.trainer.default_root_dir)

        if self.hparams.network.get('strict_fsdp_memory', False) and self.trainer._strategy.name() == 'fsdp':
            # clear all caching memory
            self.trainer.model._free_event_queue._max_num_inflight_all_gathers = 2
            while True:
                event = self.trainer.model._free_event_queue._dequeue()
                if event is None:
                    break
                event.synchronize()
        if self.hparams.network.get('profile_memory', False) and DIST_ENV.rank == 0:
            memory_profile_end_step = int(self.hparams.network.get('memory_profile_end_step', 10))
            if batch_idx == memory_profile_end_step:
                try:
                    file_name = "acllm_memory_profile.pickle"
                    torch.cuda.memory._dump_snapshot(file_name)
                    torch.cuda.synchronize()
                    upload_single_compressed_trace(file_name)
                    hput(file_name, self.trainer.default_hdfs_dir)
                except Exception as e:
                    self.rank_zero_warn(f"Failed to generate memory profile, reason for failure: {e}")
        return 0

    def on_before_backward(self, loss, name=None) -> None:
        if self.hparams.network.get('strict_fsdp_memory', False) and self.trainer._strategy.name() == 'fsdp':
            self.trainer.model._free_event_queue._max_num_inflight_all_gathers = 1


    def compute_eos_accuracy(
        self,
        preds: torch.Tensor,
        shift_labels: torch.Tensor,
        eos_idx: int,
        window_size: int = 1
    ) -> torch.Tensor:
        """

        Args:
            preds: [num_tokens]
            shift_labels: [num_tokens] ground_truth
            eos_idx: eos_idx
            window_size: window size for eos
        Returns:
            EOS 准确率（%），若 window_size>1 则允许前 N 个 token 出现 EOS
        """

        assert preds.shape == shift_labels.shape, "Preds and labels must have same shape"

        device = preds.device
        eos_mask = (shift_labels == eos_idx)
        eos_positions = eos_mask.nonzero().squeeze(-1)


        if len(eos_positions) == 0:
            return torch.tensor(0.0, device=device)

        if window_size == 1:
            correct = (preds[eos_positions] == eos_idx)
        else:
            max_pos = preds.size(0) - 1
            window_starts = torch.clamp(eos_positions - window_size + 1, min=0)

            window_idx = torch.arange(window_size, device=device).expand(len(eos_positions), -1)
            window_idx = window_idx + window_starts.unsqueeze(1)
            window_idx = torch.clamp(window_idx, max=max_pos)

            window_preds = preds[window_idx]  # [num_eos, window_size]
            correct = (window_preds == eos_idx).any(dim=1)

        return correct.float().mean() * 100


    def calc_loss_acc(self, shift_hidden_states, shift_labels, loss_mask,require_eos_acc=False, eos_index_window=1):
        shift_hidden_states = shift_hidden_states.contiguous()
        shift_labels = shift_labels.contiguous()
        preds = None

        if self.hparams.network.get('tie_weight', False):
            loss, acc = self.gpt2.transformer.wte(
                    shift_hidden_states.view(-1, shift_hidden_states.size(-1)).to(torch.bfloat16),
                    treat_as_embedding=False,
                    return_cross_entropy_loss=True,
                    labels=shift_labels.view(-1),
                    use_flash_ce=self.hparams.network.use_flash_ce,
                )
        else:
            #[Deprecated] P6 version
            loss, acc = self.gpt2.lm_head(
                shift_hidden_states.view(-1, shift_hidden_states.size(-1)).to(torch.bfloat16),
                return_cross_entropy_loss=True,
                labels=shift_labels.view(-1),
                use_flash_ce=self.hparams.network.use_flash_ce,
            )

        if self.eos_loss_scalar != 1.0:
            loss += (self.eos_loss_scalar - 1.0) * (shift_labels.view(-1) == self.emb.target_embedder.eos_id).float() * loss

        num_valid_tokens = loss_mask.sum()
        assert num_valid_tokens.item() > 0
        loss = (loss * loss_mask).sum() / num_valid_tokens
        acc = (acc * loss_mask).sum() / num_valid_tokens


        if require_eos_acc:
            assert preds is not None
            eos_acc=self.compute_eos_accuracy(preds, shift_labels, self.emb.target_embedder.eos_id, window_size=eos_index_window)
        else:
            eos_acc=-1.0

        output = dict()
        output["loss"] = loss
        output["acc"] = acc
        output["loss_tokens"] = num_valid_tokens
        output["eos_acc"] = eos_acc
        return output



    def prepare_training_inputs_backup(self, batch):
        

        audio_data = self.audio_encoder.forward_with_audio(batch['audio'], normalize=True)
        audio_projection = self.audio_projection(audio_data['hidden_states'])
        if audio_data.get("mask") is None:
            batch_size, seq_len = audio_projection.size()[:2]
            audio_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=audio_projection.device)
        else:
            audio_mask = audio_data['mask'].bool()
                
        prefix_inputs = batch['prefix']['input_ids']
        prefix_mask = batch['prefix']['attention_mask']
    
        suffix_inputs = batch['suffix']['input_ids']
        suffix_mask = batch['suffix']['attention_mask']

        target_inputs = batch['output']['input_ids']
        target_mask = batch['output']['attention_mask']

        prefix_embedding = self.gpt2.transformer.wte(prefix_inputs)
        suffix_embedding = self.gpt2.transformer.wte(suffix_inputs)
        target_embedding = self.gpt2.transformer.wte(target_inputs)

        # unpad
        prefix_unpad, _, prefix_cu_seqlens, _ = unpad_input(prefix_embedding, prefix_mask)
        suffix_unpad, _, suffix_cu_seqlens, _ = unpad_input(suffix_embedding, suffix_mask)
        target_unpad, _, target_cu_seqlens, _ = unpad_input(target_embedding, target_mask)
        audio_unpad, _, audio_cu_seqlens, _ = unpad_input(audio_projection, audio_mask)

        context_length = prefix_cu_seqlens.diff() + audio_cu_seqlens.diff() + suffix_cu_seqlens.diff()

        # step 2: concate
        token_embed = torch.cat([prefix_unpad, audio_unpad, suffix_unpad, target_unpad], dim=0)
        target_loss_mask = torch.cat([
            torch.zeros(prefix_unpad.size(0) + audio_unpad.size(0) + suffix_unpad.size(0), dtype=torch.bool, device=token_embed.device),
            torch.ones(target_unpad.size(0), dtype=torch.bool, device=token_embed.device)
        ])

        return {
            'token_embeds': token_embed,
            'target_loss_mask': target_loss_mask,
            'prefix_length': context_length,
            'target_length': target_cu_seqlens.diff(),
            'token_length': context_length + target_cu_seqlens.diff(),
        }


    def prepare_training_inputs(self, batch):

        # Audio
        encoder_output = self.audio_encoder({"audio": batch['audio'],
                                             "audio_length": batch['audio_length']})    

        hidden_states = encoder_output['latent']
        attn_mask = encoder_output['attn_mask']
        audio_proj, audio_mask = self.audio_adapter(hidden_states, attn_mask)

        B, L2, D = hidden_states.size()

        # Determine optional components
        has_prefix = 'prefix' in batch
        has_suffix = 'suffix' in batch

        if has_prefix:
            prefix_input_ids = batch['prefix']['input_ids']              # (B, L1)
            prefix_mask = batch['prefix']['attention_mask'].bool()       # (B, L1)
            prefix_emb = self.gpt2.transformer.wte(prefix_input_ids)     # (B, L1, D)

        if has_suffix:
            suffix_input_ids = batch['suffix']['input_ids']
            suffix_mask = batch['suffix']['attention_mask'].bool()
            suffix_emb = self.gpt2.transformer.wte(suffix_input_ids)

        target_input_ids = batch['output']['input_ids']
        target_mask = batch['output']['attention_mask'].bool()
        target_emb = self.gpt2.transformer.wte(target_input_ids)
            
        #build token_embeds / token_ids / loss_mask / attention_mask
        token_embeds_list = []
        token_ids_list = []
        loss_mask_list = []
        attention_mask_list = []
        total_length_list = []


        for b in range(B):
            tokens_emb = []
            tokens_ids = []

            if has_prefix:
                tokens_emb.append(prefix_emb[b][prefix_mask[b]])
                tokens_ids.append(prefix_input_ids[b][prefix_mask[b]])

            tokens_emb.append(audio_proj[b][audio_mask[b]])
            tokens_ids.append(torch.full((audio_mask[b].sum().item(),), self.PAD_IDX, dtype=torch.long, device=audio_proj.device))

            if has_suffix:
                tokens_emb.append(suffix_emb[b][suffix_mask[b]])
                tokens_ids.append(suffix_input_ids[b][suffix_mask[b]])

            tokens_emb.append(target_emb[b][target_mask[b]])
            tokens_ids.append(target_input_ids[b][target_mask[b]])

            token_embed = torch.cat(tokens_emb, dim=0)  # (L_total_b, D)
            token_embeds_list.append(token_embed)
            token_ids_list.append(torch.cat(tokens_ids, dim=0))  # (L_total_b,)

            # loss mask
            num_non_target = sum(t.size(0) for t in tokens_emb[:-1])
            loss_mask = torch.cat([
                torch.zeros(num_non_target, dtype=torch.bool, device=token_embed.device),
                torch.ones(tokens_emb[-1].size(0), dtype=torch.bool, device=token_embed.device),
            ])
            loss_mask_list.append(loss_mask)

            attention_mask_list.append(torch.ones_like(loss_mask))
            total_length_list.append(torch.tensor(token_embed.size(0), device=token_embed.device))



        token_embeds = pad_sequence(token_embeds_list, batch_first=True)        # (B, L_max, D)
        token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=self.PAD_IDX)  # (B, L_max)
        target_loss_mask = pad_sequence(loss_mask_list, batch_first=True)       # (B, L_max)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True)    # (B, L_max)
        total_length = torch.stack(total_length_list)                           # (B,)



        B, T = target_loss_mask.shape
        mask_int = target_loss_mask.int()
        indices = torch.arange(T, device=target_loss_mask.device).expand(B, T)
        masked = torch.where(mask_int == 1, indices, T)

        first_true_indices = masked.min(dim=1).values

        logger.debug(
            f"audio_attention: {audio_mask.sum(dim=1)}\n"
            f"prefix_mask: {prefix_mask.sum(dim=1)}\n"
            f"suffix_mask: {suffix_mask.sum(dim=1)}\n"
            f"target_mask: {target_mask.sum(dim=1)}\n"
            f"Length: {first_true_indices}\n"
            f"total_length: {total_length}"
        )


        return {
            "token_embeds": token_embeds,           # (B, L_max, D)
            "token_ids": token_ids,                 # (B, L_max)
            "target_loss_mask": target_loss_mask,   # (B, L_max)
            "attention_mask": attention_mask,       # (B, L_max)
            "total_length": total_length,           # (B,)
        }


    def forward(self, batch, **kwargs):

        training_inputs = self.prepare_training_inputs(batch)

        token_embeds = training_inputs['token_embeds']
        token_length = training_inputs['total_length']

        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        target_ids = training_inputs['token_ids'][:, 1:] # using padding for audio_placeholder_token
        
        input_token_embeds = training_inputs['token_embeds'][:, :-1]
        seq_len = token_embeds.shape[1]
        shifted_input_mask = sequence_mask(token_length, seq_len, device="cuda")[:, 1:]
        input_embeds_rmpad, _, cu_seqlens_q, max_seqlen_q = unpad_input(input_token_embeds.contiguous(), shifted_input_mask)

        # TODO: Activations offload, Context parallel.
        if self.hparams.network.balance_llm:
            total_seqlens = cu_seqlens_q.diff()
            (input_embeds_rmpad,), cu_seqlens_q, max_seqlen_q = self.llm_batch_dispatcher.balance_rmpad_embed(
                [input_embeds_rmpad],
                total_seqlens=total_seqlens,
                balance_policy=self.hparams.network.llm_balance_policy,
                group=self.balance_group,
            )

        labels_shift_rmpad, _, _, _ = unpad_input(target_ids.unsqueeze(2).contiguous(), attention_mask=shifted_input_mask)
        target_loss_mask, _, _, _ = unpad_input(target_loss_mask.unsqueeze(2).contiguous(), shifted_input_mask)
        hidden_states = self.gpt2(
            inputs_embeds=input_embeds_rmpad.unsqueeze(1).contiguous(),
            return_final_hidden_states=True,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )
        if self.hparams.network.get('return_moe_metric', False):
            hidden_states, expert_cnt = hidden_states
        outputs = self.calc_loss_acc(
            hidden_states,
            labels_shift_rmpad.squeeze(1),
            target_loss_mask.squeeze(1),
            require_eos_acc=kwargs.get("require_eos_acc", False),
            eos_index_window=kwargs.get("eos_index_window", 1),
        )
        outputs['seqlens_q'] = cu_seqlens_q.diff()
        outputs['gpt2_lengths'] = outputs['seqlens_q']
        outputs['tokens'] = input_embeds_rmpad.shape[0]

        outputs['consume_tokens(B)'] = outputs['tokens'] * 1e-9
        outputs['loss_tokens(B)'] = outputs['loss_tokens'] * 1e-9
        if self.hparams.network.get('return_moe_metric', False):
            n_layer = expert_cnt.shape[0]
            for i in range(n_layer):
                outputs[f'expert_cnt_layer{i}'] = expert_cnt[i]
                outputs[f'expert_active_cnt_layer{i}'] = expert_cnt[i].nonzero().numel()
        return outputs

    def _count_dataloader_skip_info(self, batch, need_output=False):
        if "worker_info" not in batch:
            return {}
        worker_info = batch["worker_info"]
        skip_num = worker_info["skip_num"]
        for pid in skip_num.keys():
            if pid not in self.skip_num:
                self.skip_num[pid] = skip_num[pid]
            elif sum(skip_num[pid].values()) > sum(self.skip_num[pid].values()):
                # update
                self.skip_num[pid] = skip_num[pid]
        if need_output:
            all_keys = set(sum([list(k.keys()) for k in self.skip_num.values()], []))
            all_skip = defaultdict(int)
            for k in all_keys:
                all_skip[_get_skip_meter_name(k)] += sum(s.get(k, 0) for s in self.skip_num.values())
            return all_skip
        else:
            return {}


    def training_step(self, batch, batch_idx):

        outputs = self.forward(batch)
        fwd_flops = self.gpt2.calc_flops_rmpad(outputs['gpt2_lengths'])
        bwd_flops = fwd_flops * 2
        if not self.training:
            bwd_flops = 0
        outputs['flops'] = fwd_flops + bwd_flops
        if self.training:
            outputs['lr * 1e3'] = self.trainer.optimizers[0].param_groups[0]['lr'] * 1000
        del outputs['gpt2_lengths']

        # record skip data info
        if self.training:
            should_log = self.trainer.global_step % self.trainer._log_every_n_steps == 0
            should_log = True
            skip_info = self._count_dataloader_skip_info(batch, should_log)
            if should_log:
                outputs.update(skip_info)
                batch_prefix_suffix_output_string=f"STEPS:{self.trainer.global_step}:"

                if self.trainer.global_step % 1000 == 0:
                    if 'prefix_string' in batch:
                        batch_prefix_suffix_output_string += f"prefix_string:{batch['prefix_string'][0]} || "
                    if "suffix_string" in batch:
                        batch_prefix_suffix_output_string += f"suffix_string:{batch['suffix_string'][0]} || "
                    if "output_string" in batch:
                        batch_prefix_suffix_output_string += f"output_string:{batch['output_string'][0]} || "
                    
                    logger.info(batch_prefix_suffix_output_string)
                

        return outputs

    def validation_step(self, batch, batch_idx):

        #By Default Consider previous 1s window
        outputs = self.forward(batch,
                              require_eos_acc=False,
                              eos_index_window=1)
        outputs['flops'] = self.gpt2.calc_flops_rmpad(outputs['gpt2_lengths'])
        del outputs['gpt2_lengths']
        return outputs

    def configure_optimizers(self, model, optimizer_kwargs):
        if self.hparams.network.get('use_layerwise_lr', False):
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                model,
                layerwise_modules=['audio', None],
                layerwise_lr=[self.hparams.network.usm_model_lr, self.hparams.network.llm_model_lr],
                layerwise_weight_decay=[
                    optimizer_kwargs["optimizer"]["params"]["weight_decay"],
                    optimizer_kwargs["optimizer"]["params"]["weight_decay"],
                ],
            )
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                model,
                layerwise_weight_decay=optimizer_kwargs["optimizer"]["params"]["weight_decay"],
            )

        optimizers = super()._configure_optimizers(optimizer_grouped_parameters, optimizer_kwargs)
        lr_schedulers = super()._configure_schedulers(optimizers, optimizer_kwargs)

        return optimizers, lr_schedulers

    @staticmethod
    def get_max_memory():
        '''get max gpu memory.'''
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated()
        try:
            retry = int(torch.cuda.memory_summary().split('|\n|')[3].split(':')[-1].strip())
        except Exception:
            retry = -1
        return mem / (1024 * 1024), retry

    def calc_gpt_flops(self, lengths, **kwargs):
        if not hasattr(self.gpt2, 'calc_flops_rmpad'):
            return 0, 0
        sum_of_T = torch.sum(lengths).item()
        flops = self.gpt2.calc_flops_rmpad(lengths, **kwargs)
        if isinstance(flops, tuple) and self.hparams.network.get("use_extra_moe", False):
            fwd_flops, extra_fwd_flops = flops
            fwd_flops += extra_fwd_flops
            bwd_flops = 2 * extra_fwd_flops
        else:
            fwd_flops = flops
            bwd_flops = 0
        if self.hparams.network.freeze_llm or self.hparams.network.use_lora or not self.gpt2.training:
            bwd_flops += fwd_flops
            if self.hparams.network.use_lora:
                lora_rank = self.hparams.network.lora_config.rank
                embed = self.hparams.network.n_embed
                ffn_hidden = self.hparams.network.n_inner
                shared_heads = self.hparams.network.n_shared_qhead
                lora_flops = 0
                if "attn" in self.hparams.network.lora_config.target_modules:
                    lora_flops += (
                        2
                        * sum_of_T
                        * self.hparams.network.n_layer
                        * (
                            embed * lora_rank
                            + lora_rank * embed
                            + 2 * (embed * lora_rank + lora_rank * embed / shared_heads)  # q_proj
                            + embed * lora_rank  # k/v_proj
                            + lora_rank * embed  # attn_out
                        )
                        * 3
                    )
                if "mlp" in self.hparams.network.lora_config.target_modules:
                    lora_flops += (
                        2
                        * sum_of_T
                        * self.hparams.network.n_layer
                        * (
                            embed * lora_rank
                            + lora_rank * ffn_hidden
                            + ffn_hidden * lora_rank  # ffn_in
                            + lora_rank * embed  # ffn_out
                        )
                        * 3
                    )
                fwd_flops += lora_flops
                bwd_flops += lora_flops * 2
        else:
            bwd_flops += fwd_flops * 2
        return fwd_flops, bwd_flops

    def calc_flops(self, inputs_shape, gpt2_lengths):
        # TODO: update semantic flops calc
        T_text = inputs_shape[1][1]
        audio_input_shape = list(inputs_shape[0])
        # if self.hparams.network.get("prompt_type", "fixed_one") == "fixed_one" and self.hparams.network.get(
        #     "prompt", ""
        # ):
        #     T_text += self.fixed_one_prompt.shape[-1]

        fwd_flops, bwd_flops, out_shape = 0, 0, audio_input_shape
        # # flops of audio encoder
        # [
        #     audio_encoder_fwd_flops,
        #     audio_encoder_bwd_flops,
        #     out_shape,
        # ] = self.audio_encoder.calc_flops(out_shape)
        # fwd_flops += audio_encoder_fwd_flops
        # if not self.hparams.network.freeze_audio_encoder:
        #     bwd_flops += audio_encoder_bwd_flops
        # # flops of audio converter
        # [
        #     audio_converter_fwd_flops,
        #     audio_converter_bwd_flops,
        #     out_shape,
        # ] = self.audio_downsample.calc_flops(out_shape)
        # audio_proj_fwd_flops = (
        #     2 * out_shape[0] * out_shape[1] * self.audio_proj.in_features * self.audio_proj.out_features
        # )
        # fwd_flops += audio_converter_fwd_flops
        # fwd_flops += audio_proj_fwd_flops
        # if self.hparams.network.freeze_converter:
        #     if not self.hparams.network.freeze_audio_encoder:
        #         bwd_flops += audio_converter_fwd_flops
        #         bwd_flops += audio_proj_fwd_flops
        # else:
        #     bwd_flops += audio_converter_bwd_flops
        #     bwd_flops += audio_proj_fwd_flops * 2
        # flops of gpt2
        gpt2_fwd, gpt2_bwd = self.calc_gpt_flops(gpt2_lengths.cpu())
        fwd_flops += gpt2_fwd
        bwd_flops += gpt2_bwd
        # shape not changed.

        return fwd_flops, bwd_flops, out_shape
 
    def left_pad_sequence(self, sequences, batch_first=True, padding_value=0.0):
        reversed_sequences = [torch.flip(s, dims=[0]) for s in sequences]

        padded_reversed = pad_sequence(
            reversed_sequences,
            batch_first=batch_first,
            padding_value=padding_value
        )
        if batch_first:
            padded = torch.flip(padded_reversed, dims=[1])
        else:
            padded = torch.flip(padded_reversed, dims=[0])
            
        return padded


    def prepare_inference_inputs(self, batch):

        device = self.gpt2.device
        # Audio
        encoder_output = self.audio_encoder({"audio": batch['audio'], "audio_length": batch['audio_length']})    
        hidden_states = encoder_output['latent']
        attn_mask = encoder_output['attn_mask']

        print(hidden_states.shape)
        
        audio_proj, audio_mask = self.audio_adapter(hidden_states, attn_mask)

        B, L2, D = hidden_states.size()
        # Determine optional components

        has_prefix = 'prefix' in batch
        has_suffix = 'suffix' in batch

        
        if has_prefix:
            prefix_input_ids = batch['prefix']['input_ids'].to(device)              # (B, L1)
            prefix_mask = batch['prefix']['attention_mask'].to(device).bool()       # (B, L1)
            prefix_emb = self.gpt2.transformer.wte(prefix_input_ids)     # (B, L1, D)

        if has_suffix:
            suffix_input_ids = batch['suffix']['input_ids'].to(device)           
            suffix_mask = batch['suffix']['attention_mask'].to(device).bool()
            suffix_emb = self.gpt2.transformer.wte(suffix_input_ids)

        token_embeds_list = []
        attention_mask_list = []
        total_length_list = []

        for b in range(B):
            tokens_emb = []

            tokens_emb.append(prefix_emb[b][prefix_mask[b]])
            tokens_emb.append(audio_proj[b][audio_mask[b]])
            tokens_emb.append(suffix_emb[b][suffix_mask[b]])
            token_embed = torch.cat(tokens_emb, dim=0)
            token_embeds_list.append(token_embed)

            attention_mask_list.append(torch.ones(token_embed.size(0), dtype=torch.long, device=token_embed.device))
            total_length_list.append(torch.tensor(token_embed.size(0), device=token_embed.device))

        token_embeds = pad_sequence(token_embeds_list, batch_first=True)        # (B, L_max, D)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True)    # (B, L_max)
        total_length = torch.stack(total_length_list)                           # (B,)
        
        token_embeds = self.left_pad_sequence(
            token_embeds_list,
            batch_first=True,
            padding_value=0.0
        )
        attention_mask = self.left_pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )

        return {
            "token_embeds": token_embeds,           # (B, L_max, D)
            "attention_mask": attention_mask,       # (B, L_max)
            "total_length": total_length,           # (B,)
        }
 

    def on_predict_start(self):
        
        # Get GPU rank for multi-GPU setup
        if torch.distributed.is_initialized():
            rank_idx = torch.distributed.get_rank()
        else:
            rank_idx = 0
        
        # Create result directory if it doesn't exist
        result_dir = Path(self.inference_config.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rank-specific result file
        base_name = Path(self.inference_config.result_file).stem
        extension = Path(self.inference_config.result_file).suffix
        self.rank_result_file = result_dir / f"{base_name}_rank_{rank_idx}{extension}"
        
        # Initialize empty list for storing batch results
        self.batch_results = []
        
        logger.info(f"Rank {rank_idx}: Results will be saved to {self.rank_result_file}")


    @torch.no_grad()
    def predict_step(self, batch, batch_idx=0):

        data_inputs = self.prepare_inference_inputs(batch)

        input_embed=data_inputs['token_embeds']
        attention_mask=data_inputs['attention_mask']

        outputs = self.gpt2(
                input_ids=None,
                is_inference=True, 
                inputs_embeds=input_embed,
                inputs_embeds_mask=attention_mask, 
                generation_config=self.generation_config)

        import time, os, threading, math

        # def get_uttid():
        #     ts = int(time.time() * 1e6)  # microseconds
        #     pid = os.getpid()             # process id
        #     tid = threading.get_ident()   # thread id
        #     return f"{ts}_{pid}_{tid}"

        # uttid = get_uttid()
        
        # torch.save(outputs,      f"{uttid}_gpt2_outputs.pt")
        output_ids = outputs.sequences
        # torch.save(output_ids,   f"{uttid}_gpt2_output_ids.pt")
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        # torch.save(output_texts, f"{uttid}_gpt2_output_texts.pt")
        
        # output_scores = [s for s in outputs.scores[0]] 
        output_scores  = [[
            s[184].item() if math.isfinite(s[184].item()) else -1, 
            s[173].item() if math.isfinite(s[173].item()) else -1
            ] for s in outputs.scores[0]]



        ### scores[0] is the logits of the first token [B, Vocab]
        output_texts = [str(text) for text in output_texts]
        
        batch['predict_results'] = output_texts
        batch['scores'] = output_scores
        logger.info("-" * 50)
        logger.info(f"      UTTID: {batch['uttid'][0]}")
        logger.info(f" prediction: {output_texts[0]}")
        logger.info(f"        IDS: {output_ids}")
        logger.info(f"     SCORES: {output_scores}")
        logger.info("-" * 50)
        
        # import pdb; pdb.set_trace()
        if "audio" in batch:
            del batch["audio"]
        
        ##### [TODO] #######
        '''
        {
            'uttid': {...}
        }
        '''
        return batch


    def on_predict_epoch_end(self, outputs, batch_idx=0):
        """
        Save all batch results to a JSON file, separated by GPU rank
        """

        if not hasattr(self, 'rank_result_file'):
            logger.warning("rank_result_file not initialized. Skipping save.")
            return
        
        # Collect all relevant data from batch_results
        all_data = []
        
        for batch in outputs:
            # Handle case where batch might be a list or single item
            if isinstance(batch, list):
                batch = batch[0]  # Take first item if it's a list
            
            batch_size = len(batch['uttid'])
            
            # Create individual records for each item in the batch
            for i in range(batch_size):
                # record = {
                #     'uttid': batch['uttid'][i] if i < len(batch['uttid']) else None,
                #     'audio_tags': batch['audio_tags'][i] if i < len(batch['audio_tags']) else None,
                #     # 'url': batch['url'][i] if i < len(batch['url']) else None,
                #     'scores': batch['scores'][i] if i < len(batch['scores']) else None,
                #     'predict_results': batch['predict_results'][i] if i < len(batch['predict_results']) else None
                # }
                record = {
                    'uttid': batch['uttid'][i] if i < len(batch['uttid']) else None,
                    'audio_tags': batch['audio_tags'][i] if i < len(batch['audio_tags']) else None,
                    'raw': batch['raw'][i] if i < len(batch['raw']) else None,
                    'url': batch['url'][i] if i < len(batch['url']) else None,
                    'scores': batch['scores'][i] if i < len(batch['scores']) else None,
                    'predict_results': batch['predict_results'][i] if i < len(batch['predict_results']) else None
                }
                all_data.append(record)
        
        # Save to JSON file
        try:
            with open(self.rank_result_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            rank_idx = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            logger.info(f"Rank {rank_idx}: Saved {len(all_data)} records to {self.rank_result_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to {self.rank_result_file}: {str(e)}")


    def __del__(self):
        if self.graph is not None:
            self.release_cuda_graph()

    def execution_order(self) -> list[str]:
        """LSDP need this"""
        return ["audio_encoder","audio_adapter", "gpt2"]

def _get_skip_meter_name(transform_name: str):
    return "transform_skip/"+transform_name



class SemanticLlmTrainer(AudioTrainer):
    train_meters = [
        ('loss', {'type': 'Weighted', 'args': ['loss', 'tokens']}),
        ('acc', {'type': 'Weighted', 'args': ['acc', 'tokens']}),
        ('eos_acc',{'type': 'Weighted', 'args': ['eos_acc', 'tokens']} ),
        ('lr * 1e3', {'type': 'Simple', 'args': ['lr * 1e3']}),
        ('loss_tokens(B)', {'type': 'Sum', 'args': ['loss_tokens(B)']}),
        ('consume_tokens(B)', {'type': 'Sum', 'args': ['consume_tokens(B)']}),
        ('flops', {'type': 'Flops', 'args': ['flops']}),
    ]

    valid_meters = [
        ('loss', {'type': 'Weighted', 'args': ['loss', 'loss_tokens']}),
        ('acc', {'type': 'Weighted', 'args': ['acc', 'loss_tokens']}),
        ('eos_acc',{'type': 'Weighted', 'args': ['eos_acc', 'tokens']} ),
        ('loss_tokens(B)', {'type': 'Sum', 'args': ['loss_tokens(B)']}),
        ('consume_tokens(B)', {'type': 'Sum', 'args': ['consume_tokens(B)']}),
        ('flops', {'type': 'Flops', 'args': ['flops']}),
    ]
    flush_rule = [
        'loss',
        'acc',
        'lr * 1e3',
        'flops',
    ]

    def _setup_meters(self):
        self._config
        global_config = last_cli().hparams
        train_transform_names = [t.type for t in global_config.data.train_item_transform]
        # TODO: 或许可以手动将需要观察的transform名字加在这里。其实有很多是不会skip的。
        skip_meters = [
            (_get_skip_meter_name(tn), {"type": "Sum", "args": [_get_skip_meter_name(tn)]})
            for tn in train_transform_names
        ]
        self.train_meters.extend(skip_meters)

        if global_config.model.network.get('return_moe_metric', False):
            moe_meters = []
            for i in range(global_config.model.network.n_layer):
                moe_meters.append(
                    (f'moe/expert_cnt_layer{i}', {'type': 'Histogram', 'args': [f'expert_cnt_layer{i}']})
                )
                moe_meters.append(
                    (f'moe/expert_active_cnt_layer{i}', {'type': 'Simple', 'args': [f'expert_active_cnt_layer{i}']})
                )
            
            self.train_meters.extend(moe_meters)


#Not a good practice
class MuInference:
    def __init__(self, args, trainer, model, datamodule) -> None:
        self._args = args
        self._trainer = trainer
        self._model = model
        self._datamodule = datamodule
        self._extra_args = {}

        # Avoid DDP initialization to save GPU memory.
        # if self._trainer._strategy.name() == 'ddp':
        #     self._trainer._strategy.enable_ddp = False

        rank, world_size, local_rank = self._setup_from_env()
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        gc.enable()

    def _setup_from_env(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer warning
        rank = DIST_ENV.rank
        world_size = DIST_ENV.world_size
        local_rank = DIST_ENV.local_rank
        # assert world_size <= 8, f'Inference not support multi-machine multi-gpu, use single machine instead'

        return rank, world_size, local_rank


    def batch_predict(self):

        if self._rank == 0:
            self._datamodule.rank_zero_prepare()
        DIST_ENV.barrier()
        if self._local_rank == 0:
            self._datamodule.local_rank_zero_prepare()
        DIST_ENV.barrier()
        self._datamodule.setup()

        self._model.local_rank_zero_prepare()
        local_tokenizer_path = self._model.local_tokenizer_path
    
        assert local_tokenizer_path is not None, "Invalid local_tokenizer_path from self._datamodule"
        self._tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        self._tokenizer.padding_side = "left"
        self._trainer._setup_data_model(self._model, stage='predict', test_dataloader=None, datamodule=None)
        if self._trainer.resume_ckpt_path is not None:
            self._trainer._resume_from_ckpt(self._trainer.resume_ckpt_path, load_for_training=False)
        if getattr(DIST_ENV, 'expert_parallel_size', 1) > 1:
            self._model.load_ep_expert_state_dict()
        torch.set_grad_enabled(False)
        self._model.eval()

        predict_dataloader = self._datamodule.predict_dataloader()

            
        for batch in predict_dataloader:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                self._model.predict_step(batch)

        # trainer context fsdp

    


class SemanticLlmCLI(CruiseCLI):
    def _parse_arguments(self, parser) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        self.config = parser.parse_args()
        if DIST_ENV.local_rank == 0:
            sys.stdout.write(parser.dump(self.config, skip_check=True))

        # Force static_sync_limit_val to False to avoid getting
        # the dataloader length incorrectly during validation.
        # self.config.trainer.static_sync_limit_val = False
        def _init_with_reset_params(module: torch.nn.Module):
            """
            to_empty + reset_parameters() init function example for modules
            initailized with device="meta"
            """
            has_meta_states = any(
                t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
            )
            if has_meta_states:
                device = torch.device("cuda", torch.cuda.current_device())
                module.to_empty(device=device, recurse=False)
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        if self.config.trainer.get('strategy', 'ddp') == 'fsdp':
            self.config.trainer.accelerator_kwargs.fsdp_config.param_init_fn = _init_with_reset_params
            self.config.trainer.accelerator_kwargs.fsdp_config.sync_module_states = True
            if self.config.model.network.get('strict_fsdp_memory', False):
                self.config.trainer.accelerator_kwargs.fsdp_config.forward_prefetch = False
        try:
            self._hparams = namespace_to_cruise_config(self.config)
        except Exception:
            self._hparams = CruiseConfig(dict(self.config))


def setup_cli(CLI_Clazz=SemanticLlmCLI):
    helper = ExpHelper(__file__)
    ckpt_save_interval_from_env = int(os.getenv('MARIANA_CUSTOM_SAVE_INTERVAL', 2000))
    ckpter = SpeechModelCheckpoint(
        monitor="step",
        save_last=False,
        save_top_k=-1 if ckpt_save_interval_from_env > 0 else 0,
        every_n_train_steps=ckpt_save_interval_from_env,
        every_n_epochs=0,
        verbose=True,
        save_on_train_epoch_end=False,
        enable_trace=False,
        save_best=False,
    )
    callbacks = [ckpter]
    data_save_interval_from_env = int(os.getenv('MARIANA_SPEECH_DATA_COLLECT_INTERVAL', 0))
    data_save_max_items_from_env = int(os.getenv('MARIANA_SPEECH_DATA_COLLECT_MAX', 32))
    if data_save_interval_from_env > 0:
        collector = SpeechDataCollectCallback(
            keys_to_collect = ["style_text", "freeform_text", "conditions", "duration", "raw_lyrics"],
            audio_key="target_audio",
            audio_duration_key="duration",
            max_items_to_save=data_save_max_items_from_env,
            every_n_train_steps=data_save_interval_from_env,
        )
        callbacks.append(collector)
    cli = CLI_Clazz(
        SemanticLlmModel,
        datamodule_class=MusicLiteDataModule,
        trainer_class=SemanticLlmTrainer,
        trainer_defaults={
            'precision': 16,
            # "logger": "console",
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'find_unused_parameters': False,
            "save_before_val": False,
            "callbacks": callbacks,
        },
    )
    # add inference config here
    cli.add_argument('--inference', default=False, action='store_true', dest='inference')
    helper.report_trial_info()  # report current trial info
    return cli

def parse_args():
    cli = setup_cli(SemanticLlmCLI)
    args, trainer, model, datamodule = cli.parse_args()

    return args, trainer, model, datamodule


if __name__ == '__main__':
    args, trainer, model, datamodule = parse_args()

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint) and callback._every_n_train_steps == 0:
            callback._every_n_train_steps = trainer._checkpoint_kwargs.get('every_n_train_steps', 0)
    try:
        from bytedance.ndtimeline import EmergencyServer

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        EmergencyServer.init(local_rank=local_rank)
    except Exception as e:
        logger.warning(f"Fail to init EmergencyServer, {e}")

    if args.inference:
        trainer.predict(model, datamodule=datamodule)
    else:
        # Training
        trainer.fit(model, datamodule=datamodule)

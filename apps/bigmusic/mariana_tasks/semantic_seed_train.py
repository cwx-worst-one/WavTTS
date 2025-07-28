# This file describe training and infer tasks of semantic seed model.

from collections import defaultdict
import datetime
import gc
import itertools
import os
import sys
import operator
from functools import partial, reduce
from typing import Dict, List
import math
import torch
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

from mariana.models.audio.batch_dispatcher.llm_batch_dispatcher import LLMBatchDispatcher
from mariana.models.audio.speech_checkpoint import SpeechModelCheckpoint
import samantha # noqa: F401, resolve mariana python path
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.models.audio.weight_init import ModuleInitializer
from samantha.dataio.bigmusic.lite import MusicLiteDataModule
from samantha.criterion.masked_loss import sequence_mask
from apps.bigmusic.mariana_tasks.semantic_modules import SemanticEmbModule as SemanticEmbModuleLegacy
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
from tasks.audio.utils import (
    get_optimizer_grouped_parameters,
    upload_single_compressed_trace,
    upload_trace,
)
from mariana.utils.comm_utils import get_formated_model_summary_table
from tasks.audio.ndtimeline import get_ndtimeline_profile

import hyperpyyaml
from tqdm.auto import tqdm
from recipes.musiclm.inference.utils import sample, adaptive_sampling, SamplingScheduler
from samantha.utils.hparams import DotDict


logger = AudioLogger()

# default config for M8 MoE-680M LLM
_m8_network_config = {
    "llm_empty_init": True,
    "tokenizer_path": "hdfs://haruna/home/byte_data_aml_research/user/anzhecheng/tokenizer/bbpe155k-v6.4.3-ml.pret",
    "ignored_llm_missing_keys": ["inv_freq"],
    "fp8_amp": False,
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

_inference_config = {
    "use_cache": True,
    "use_cuda_graph": True,
    "cuda_graph_max_bsz": 2,
    "cuda_graph_max_seqlen": 10240,
}


def _get_local_path(hdfs_path, cache_dir="."):
    if hdfs_path:
        local_path = os.path.join(cache_dir, os.path.basename(hdfs_path))
    else:
        local_path = None
    return local_path


class GenerationConfig:
    def __init__(self, hp, **kwargs):
        # basic
        self.duration = hp.duration
        self.skip_sos = hp.get('skip_sos', False)
        self.frame_rate = hp.get('semantic_frame_rate', 25)

        # forward
        self.use_cache = kwargs.get("use_cache", False)
        self.use_cuda_graph = kwargs.get("use_cuda_graph", False)
        self.cuda_graph_max_bsz = kwargs.get("cuda_graph_max_bsz", 1)
        self.cuda_graph_max_seqlen = kwargs.get("cuda_graph_max_seqlen", 1024)

        # cfg
        self.use_controller_cfg = hp.get('use_controller_cfg', False)
        self.controller_cfg_gamma = hp.get('controller_cfg_gamma', 3)

        # sampling
        self.temperature = hp.semantic_temperature
        self.sample_mode = hp.sample_mode
        self.sample_thresh = hp.get("sample_thresh", 0.9)
        self.repetition_penalty = hp.get('repetition_penalty', 1.0)

        # sob
        self.use_step_out_blank = hp.get('use_step_out_blank', False)
        self.step_out_blank_logic = hp.get('step_out_blank_logic', 'v4')
        self.step_out_blank_max_len = hp.get('step_out_blank_max_len', 8000)

        # eos
        self.exclude_eos_first_secs = hp.get('exclude_eos_first_secs', 0)
        self.exclude_eos_thresh_secs = hp.get('exclude_eos_thresh_secs', 0)
        self.emit_eos_thresh_secs = hp.get('emit_eos_thresh_secs', 0)
        self.stop_eos = hp.get('stop_eos', False)


class GenerationState:
    """Manage generation state like updating output tokens and checking stop criteria."""

    def __init__(self, beam: int, original_batch_size: int, device: torch.device):
        self.beam = beam
        self.original_batch_size = original_batch_size
        self.device = device
        self.is_eos_stop = torch.zeros((beam, original_batch_size), dtype=torch.long, device=device)
        self.output_tokens = None

    def update_output_tokens(self, predict_token: torch.Tensor) -> None:
        self.output_tokens = torch.cat([self.output_tokens, predict_token], dim=1) if self.output_tokens is not None else predict_token

    def check_stopping_criteria(self, predict_token: torch.Tensor, step: int, num_tokens: torch.Tensor,
                                eos_id: int, stop_eos: bool) -> bool:
        if not stop_eos:
            return False

        num_tokens_mask = (step >= num_tokens).broadcast_to(self.beam, self.original_batch_size) # (beam, bsz)
        eos_mask = predict_token.view(self.beam, -1) == eos_id  # (beam, bsz)
        eos_mask = torch.logical_or(eos_mask, num_tokens_mask)
        self.is_eos_stop += eos_mask
        self.output_tokens[eos_mask.view(-1), -1] = eos_id  # force
        return torch.all(self.is_eos_stop > 0)


class TokenBuffer:
    def __init__(self, beam: int, original_batch_size: int, step_out_blank_max_len: int, batch: Dict):
        self.beam = beam
        self.original_batch_size = original_batch_size
        self.step_out_blank_max_len = step_out_blank_max_len
        self.buffers = self._initialize_buffers(batch)

    def _initialize_buffers(self, batch: Dict) -> List[List[int]]:
        buffers = [[] for _ in range(self.beam * self.original_batch_size)] # init previous_tokens

        if "audio_prompt_token_ids" in batch:
            token_ids = batch["audio_prompt_token_ids"].tolist()
            previous_tokens = reduce(operator.add, [[ti[:l]]for l, ti in zip(batch["target_tokens_length"].tolist(), token_ids)] * self.beam)  # (beam * bs, audio_prompt_token_len)
            buffer_len = min(max(len(pt) for pt in previous_tokens), self.step_out_blank_max_len)
            buffers = [[-1] * (buffer_len - len(pt)) + pt[-buffer_len:] for pt in previous_tokens]

        return buffers

    def update(self, predict_token: torch.Tensor) -> None:
        predict_token_cpu = predict_token.cpu().numpy()
        # predict_token_cpu: (beam, b)
        # previous_tokens: (beam_size * b, queue_len)
        for j in range(self.beam * self.original_batch_size):
            self.buffers[j].append(predict_token_cpu[j,0])
            if len(self.buffers[j]) > self.step_out_blank_max_len:
                self.buffers[j] = self.buffers[j][-self.step_out_blank_max_len:]


class LogitsProcessor:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def _apply_cfg(self, logits, batch_size, n_cfg_path):
        if isinstance(self.config.controller_cfg_gamma, list):
            beam_bs = batch_size // (1+n_cfg_path)
            uncond_logits = logits[-beam_bs:]
            cond_cfg_logits = torch.zeros_like(uncond_logits)
            for gg in range(len(self.config.controller_cfg_gamma)):
                this_cond_logits = logits[beam_bs*gg: beam_bs*(gg+1)]
                cond_cfg_logits += (self.config.controller_cfg_gamma[gg] * this_cond_logits)
            logits = cond_cfg_logits
        else:
            uncond_logits = logits[batch_size//2:]     # unconditioned path
            cond_logits = logits[0:batch_size//2]
            logits = self.config.controller_cfg_gamma * cond_logits + (1 - self.config.controller_cfg_gamma) * uncond_logits
        return logits

    def apply_cfg(self, logits: torch.Tensor, batch_size: int, n_cfg_path: int) -> torch.Tensor:
        if self.config.use_controller_cfg:
            return self._apply_cfg(logits, batch_size, n_cfg_path)
        return logits

    def apply_step_out_blank(self, logits: torch.Tensor, previous_tokens: List[List[int]],
                                beam: int, original_batch_size: int) -> torch.Tensor:
        if not self.config.use_step_out_blank or self.config.step_out_blank_logic != 'v4':
            logger.warning("The step_out_blank for v1, v2, and v3 are now deprecated. Only v4 is supported.")
            return logits

        # process previous token, bin counts, apply penalty
        previous_output_tokens = torch.tensor(
            previous_tokens, dtype=torch.long, device=logits.device
        ).reshape(beam * original_batch_size, -1)
        bin_counts = torch.zeros(
            [beam * original_batch_size, logits.size(-1) + 1],
            dtype=torch.long, device=logits.device
        )
        bin_counts.scatter_add_(1, previous_output_tokens, torch.ones_like(previous_output_tokens))
        bin_counts = bin_counts[:, :logits.size(-1)]

        mask = (bin_counts > 0).view(logits.shape)
        negative_mask = logits < 0

        penalty_mask_neg = mask & negative_mask
        penalty_mask_pos = mask & (~negative_mask)

        logits = torch.where(penalty_mask_neg, logits * self.config.repetition_penalty, logits)
        logits = torch.where(penalty_mask_pos, logits / self.config.repetition_penalty, logits)

        return logits

    def apply_eos_control(self, logits: torch.Tensor, step: int, exclude_eos_first_secs: torch.Tensor,
                         frame_rate: int, original_batch_size: int, eos_id: int) -> torch.Tensor:
        for i in range(original_batch_size):
            if step < exclude_eos_first_secs[i] * frame_rate:
                logits[i, 0, eos_id] = -float('Inf')
        return logits


class SemanticLlmModel(CruiseModule):
    def __init__(
            self,
            network: CruiseConfig = CruiseConfig(dict(_m8_network_config)),
            emb_path='',
            llm_path='',
            partial_pretrain='',
            hybrid_shard_group_size=-1,
            ddp_gate=False,
            weighted_loss=False,
            inference: CruiseConfig = CruiseConfig(dict(_inference_config)),
    ):
        super().__init__()
        self.save_hparams()
        self.hybrid_shard_group_size = hybrid_shard_group_size
        self.ddp_gate = ddp_gate
        self.skip_num: Dict[int, Dict[str, int]] = {}  # record skip status from dataloader
        self.local_emb_path = _get_local_path(self.hparams.emb_path)
        self.local_llm_path = _get_local_path(self.hparams.llm_path)
        if self.hparams.partial_pretrain:
            partial_cache_dir = "./partial"
            self.hdfs_partial_pretrain = []
            self.local_partial_pretrain = []
            if '|' in self.hparams.partial_pretrain:
                checkpoint_dir = self.hparams.partial_pretrain.split('|')[0]
                for fl in self.hparams.partial_pretrain.split('|')[1].split(','):
                    self.local_partial_pretrain.append(os.path.join(partial_cache_dir, fl))
                    self.hdfs_partial_pretrain.append(os.path.join(checkpoint_dir, fl))
            else:
                self.local_partial_pretrain.append(
                    os.path.join(partial_cache_dir, os.path.basename(self.hparams.partial_pretrain))
                )
                self.hdfs_partial_pretrain.append(self.hparams.partial_pretrain)
        else:
            self.local_partial_pretrain = None
        self.local_tokenizer_path = _get_local_path(self.hparams.get("tokenizer_path", None))

        # NOTE: add partial pretrain loading if we need

        with init_on_device(torch.device('cpu')): # if DIST_ENV.local_rank == 0 else init_empty_weights():
            network = self.hparams.get('network', '')
            config_path = network.get('emb_config_path', '')
            overrides = network.get('emb_config', {})
            if config_path:
                with open(config_path, 'r') as f:
                    hps = hyperpyyaml.load_hyperpyyaml(f, overrides=overrides)
            else:
                raise ValueError("emb_config_path is not provided")
            # TODO (Yilin): Remove `SemanticEmbModuleLegacy` after it's no longer used.
            emb_cls = SemanticEmbModuleLegacy if 'input_embedders' in hps.get('extra_params', {}) else SemanticEmbModule
            self.emb = emb_cls(**hps)
            self.gpt2 = GPT2LMHeadModel(self.hparams)
            if DIST_ENV.local_rank == 0 and self.hparams.network.get("init_weight", False):
                module_initializer = ModuleInitializer(self.hparams.network)
                module_initializer.apply(self)
            self.extra_params = hps['extra_params']

            self.trace_upload_finished = False
            if self.hparams.network.get("trace_ndtimeline", False):
                self.ndtimeline = get_ndtimeline_profile(self.hparams.network.ndtimeline_range)
        self.BOS_IDX = self.hparams.network.get('bos_idx', 0)
        self.PAD_IDX = self.hparams.network.get('pad_idx', 1)  # 用来ignore的
        self.EOS_IDX = self.hparams.network.get('eos_idx', 2)
        self.label_smoothing = self.hparams.network.get('label_smoothing', 0.0)
        self.eos_loss_scalar = self.hparams.network.get('eos_loss_scalar', 1.0)
        self.extra_params = DotDict(hps['extra_params'])
        # generate with cuda graph
        self.past_key_values = None
        self.graph = None

    def local_rank_zero_prepare(self):
        # Download all files like model weights and tokenizer
        # Sometimes downloading files takes a long time, so it's better to mark progress with logging.

        if self.local_emb_path and (not os.path.isfile(self.local_emb_path)):
            logger.info(f"Downloading {self.hparams.emb_path} to {self.local_emb_path}")
            hcopy(self.hparams.emb_path, self.local_emb_path, chunk_thread_num=64)
        logger.info(f"local_emb_path is: {self.local_emb_path}")
        if self.local_llm_path and (not os.path.isfile(self.local_llm_path)):
            logger.info(f"Downloading {self.hparams.llm_path} to {self.local_llm_path}")
            hcopy(self.hparams.llm_path, self.local_llm_path, chunk_thread_num=64)
        logger.info(f"local_llm_path is: {self.local_llm_path}")
        if self.local_partial_pretrain:
            partial_dir = os.path.dirname(self.local_partial_pretrain[0])
            if not os.path.isdir(partial_dir):
                os.mkdir(partial_dir)
            for hdfs_path, local_path in zip(self.hdfs_partial_pretrain, self.local_partial_pretrain):
                if not os.path.isfile(local_path):
                    logger.info(f"Downloading {self.hparams.partial_pretrain} to {self.local_partial_pretrain}")
                    hcopy(hdfs_path, local_path, chunk_thread_num=64)
        logger.info(f"local_partial_pretrain is: {self.local_partial_pretrain}")
        if self.local_tokenizer_path and (not os.path.isfile(self.local_tokenizer_path)):
            logger.info(f"Downloading {self.hparams.tokenizer_path} to {self.local_tokenizer_path}")
            hcopy(self.hparams.tokenizer_path, self.local_tokenizer_path, chunk_thread_num=64)
        logger.info(f"local_tokenizer_path is: {self.local_tokenizer_path}")

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

    def check_strategy(self, target_strategy):
        if hasattr(self.trainer, '_strategy'):
            strategy = getattr(self.trainer, '_strategy')
            return strategy.name() == target_strategy
        else:
            return False

    def load_llm_weights(self):
        print(f"load_llm_weights {self.local_llm_path=}")
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

    def load_partial_pretrain(self):
        self.rank_zero_info(f"load partial pretrain model {self.local_partial_pretrain=}")
        if self.local_partial_pretrain and DIST_ENV.local_rank == 0:
            state_dict = dict()
            count = 0
            for checkpoint_file in self.local_partial_pretrain:
                st = _partial_load_from_checkpoint(
                    checkpoint_file,
                    rename_params={"module.": ""},
                    map_location="cpu",
                    mmap=True,
                )
                for k, v in st.items():
                    if k not in state_dict:
                        state_dict[k] = v
                    elif torch.is_floating_point(v):
                        total = state_dict[k]
                        avg = total.mul(count / (count + 1)).add(v.div(count + 1))
                        state_dict[k] = avg
                    else:
                        state_dict[k] = v
                count += 1
            IncompatibleKeys = self.load_state_dict(state_dict, strict=False)
            self.rank_zero_warn(f"Partial model missing keys are:{IncompatibleKeys.missing_keys}")
            self.rank_zero_warn(f"Partial model unexpected keys are:{IncompatibleKeys.unexpected_keys}")
            ignored_llm_missing_keys = self.hparams.network.get("ignored_llm_missing_keys", [])

            filtered_missing_keys = []
            for k in IncompatibleKeys.missing_keys:
                if not any(key in k for key in ignored_llm_missing_keys):
                    filtered_missing_keys.append(k)
            assert len(filtered_missing_keys) == 0, [k for k in filtered_missing_keys]
            state_dict.clear()
            del state_dict
            gc.collect()
            self.rank_zero_info("Successfully loaded partial pretrain weights")

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
        if self.emb is not None:
            self.emb.setup(stage)

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
        # load semantic weights
        if self.emb is not None:
            self.load_emb_weights()
        # load llm weights
        self.load_llm_weights()

        self.load_partial_pretrain()

        # freeze semantic model
        if stage != 'fit' or self.hparams.network.freeze_emb :
            self.emb.requires_grad_(False)
        # freeze llm model
        if stage != "fit" or self.hparams.network.freeze_llm:
            self.gpt2.requires_grad_(False)

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

            if require_eos_acc:
                # Kernal version doesn't support return logits
                # require_logits set as True is mandatory for now
                require_logits=True
                loss, acc, logits = self.gpt2.transformer.wte(
                    shift_hidden_states.view(-1, shift_hidden_states.size(-1)).to(torch.bfloat16),
                    treat_as_embedding=False,
                    return_cross_entropy_loss=True,
                    labels=shift_labels.view(-1),
                    use_flash_ce=False,
                    return_logits=require_logits
                )
                preds = logits.argmax(dim=-1)
            else:
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


    def forward(self, batch, **kwargs):

        training_inputs = self.emb(batch)
        prefix_length = training_inputs['prefix_length']
        target_length = training_inputs['target_length']
        token_length = prefix_length + target_length

        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        target_ids = training_inputs['token_ids'][:, 1:] * target_loss_mask  # set non-target ids to 0 to avoid OOB
        if self.emb.is_token_input():
            token_ids = training_inputs['token_ids'][:, :-1]
            seq_len = training_inputs['token_ids'].shape[1]
            shifted_input_mask = sequence_mask(token_length, seq_len, device="cuda")[:, 1:]
            input_ids_rmpad, _, cu_seqlens_q, max_seqlen_q = unpad_input(token_ids.contiguous(), shifted_input_mask)

            input_embeds_rmpad = self.gpt2.transformer.wte(input_ids_rmpad.squeeze(1))
        else:
            input_token_embeds = training_inputs['token_embeds'][:, :-1]
            seq_len = training_inputs['token_embeds'].shape[1]
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

        return outputs

    def validation_step(self, batch, batch_idx):

        #By Default Consider previous 1s window
        outputs = self.forward(batch,
                              require_eos_acc=True,
                              eos_index_window=self.extra_params.get("semantic_frame_rate", 25))
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

    def capture_cuda_graph(self, max_batch_size, max_seqlen):
        """
        Create kvcache and capture CUDA graph for inference. LLM part only.
        """
        gpt2: GPT2LMHeadModel = self.gpt2
        device = torch.device("cuda")
        self.past_key_values = torch.empty(
            gpt2.config.n_layer,
            2,  # k,v
            max_batch_size,
            max_seqlen,
            gpt2.config.n_head // gpt2.config.n_shared_qhead,
            gpt2.config.hidden_size // gpt2.config.n_head,
            dtype=torch.bfloat16,  # only support dtype=torch.bfloat16
            device=device,
        )

        # forward for decode phase with inputs_embeds
        @torch.inference_mode()
        def fwd_func(inputs_embeds, inputs_embeds_mask, cache_seqlens, past_key_values):
            hidden_states = gpt2.transformer(
                inputs_embeds=inputs_embeds,
                inputs_embeds_mask=inputs_embeds_mask,
                cache_seqlens=cache_seqlens,
                past_key_values=past_key_values,
                use_cache=True,
                phase="decode",
            )["last_hidden_state"]
            if gpt2.lm_head is not None:
                logits = gpt2.lm_head(hidden_states)
            else:
                logits = gpt2.transformer.wte(hidden_states, treat_as_embedding=False)
            return logits

        torch.cuda.synchronize()
        assert self.graph is None, "cuda graph has been captured"
        self.graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        # warmup
        inputs_embeds = torch.zeros(max_batch_size, 1, gpt2.config.hidden_size, dtype=torch.bfloat16, device=device)
        inputs_embeds_mask = torch.zeros(max_batch_size, 1, dtype=torch.bool, device=device)
        cache_seqlens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        with torch.cuda.stream(capture_stream):
            fwd_func(inputs_embeds, inputs_embeds_mask, cache_seqlens, self.past_key_values)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        DIST_ENV.barrier()
        # Capture the forward pass.
        with torch.cuda.graph(self.graph):
            logits = fwd_func(inputs_embeds, inputs_embeds_mask, cache_seqlens, self.past_key_values)
        torch.cuda.synchronize()
        DIST_ENV.barrier()
        # Save the input and output buffers.
        self.input_buffers = {
            "inputs_embeds": inputs_embeds,
            "inputs_embeds_mask": inputs_embeds_mask,
            "cache_seqlens": cache_seqlens,
        }
        self.output_buffers = {"logits": logits}


    def release_cuda_graph(self):
        """
        Release CUDA graph.
        """
        del self.past_key_values
        del self.graph
        self.past_key_values = None
        self.graph = None
        self.input_buffers = None
        self.output_buffers = None


    def predict_by_cuda_graph(self, inputs_embeds, inputs_embeds_mask, cache_seqlens, past_key_values):
        """
        Run inference with CUDA graph.
        """
        real_batch_size = inputs_embeds.shape[0]
        max_batch_size = past_key_values.shape[2]
        assert real_batch_size <= max_batch_size, f"real_batch_size: {real_batch_size}, max_batch_size: {max_batch_size}"
        assert self.graph is not None, "cuda graph has not been captured"
        self.input_buffers["inputs_embeds"][:real_batch_size].copy_(inputs_embeds)
        self.input_buffers["inputs_embeds_mask"][:real_batch_size].copy_(inputs_embeds_mask)
        self.input_buffers["cache_seqlens"][:real_batch_size].copy_(cache_seqlens)
        self.graph.replay()
        return {
            "logits": self.output_buffers["logits"][:real_batch_size],
        }

    def predict_step(self, batch, batch_idx):
        output = self.emb.predict(batch)
        # TODO: token to wave
        return super().predict_step(batch, batch_idx)

    # Prediction code
    def sample_logits(self, i, logits, temp, mode, thresh=0.9, exclude_ids=None):
        if mode == 'adaptive_sampling':
            if i == 0: # restart scheduler
                target_top_k = self.extra_params.get('target_top_k', 100)
                self.extra_params['sampling_scheduler'] = SamplingScheduler.default_schedule(
                    thresh, target_temp=temp, target_top_k=target_top_k)
            sampling_schedule = self.extra_params['sampling_scheduler']
            samples = adaptive_sampling(i, logits, sampling_schedule=sampling_schedule, exclude_ids=exclude_ids)
            return samples
        return sample(logits, temp=temp, mode=mode, thresh=thresh, exclude_ids=exclude_ids)

    def _prepare_cuda_graph(self, use_cache: bool, use_cuda_graph: bool, cuda_graph_max_bsz: int = 1, cuda_graph_max_seqlen: int = 1024) -> None:
        # init cuda graph
        if use_cache and use_cuda_graph and self.graph is None:
            logger.info(f"Initializing CUDA graph for {self.gpt2.__class__.__name__} "
                       f"with max_bsz={cuda_graph_max_bsz} and max_seqlen={cuda_graph_max_seqlen}")

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                # max_batch_size must equal to beam_size * real_batch_size
                self.capture_cuda_graph(cuda_graph_max_bsz, cuda_graph_max_seqlen)

    def _forward_step(self, model_input: Dict, gpt2_kwargs: Dict, step: int,
                     use_cache: bool, use_cuda_graph: bool) -> Dict:
        # to enable inference without a trainer, we simply cast the inputs to the expected model type
        # which is either torch.float16 or torch.bfloat16
        model_input["inputs_embeds"] = model_input["inputs_embeds"].to(self.gpt2.transformer.ln_f.weight.dtype)
        B, T, _ = model_input['inputs_embeds'].shape

        if step == 0:
            # prefill
            gpt2_kwargs = self.gpt2.prepare_inputs_for_generation(
                None,
                inputs_embeds=model_input['inputs_embeds'],
                inputs_embeds_mask=model_input['inputs_embeds_mask'],
                **gpt2_kwargs,
            )
        else:
            # decode
            gpt2_kwargs.update(
                inputs_embeds=model_input['inputs_embeds'],
                inputs_embeds_mask=torch.ones(B, T, dtype=torch.bool, device=model_input["inputs_embeds"].device),
            )

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            if use_cache and use_cuda_graph:
                if step == 0:
                    # prefill
                    gpt2_kwargs['past_key_values'] = self.past_key_values
                    output = self.gpt2(**gpt2_kwargs, return_dict=True)
                else:
                    # decode
                    output = self.predict_by_cuda_graph(
                        gpt2_kwargs['inputs_embeds'],
                        gpt2_kwargs['inputs_embeds_mask'],
                        gpt2_kwargs['cache_seqlens'],
                        gpt2_kwargs['past_key_values'],
                    )
            else:
                output = self.gpt2(**gpt2_kwargs, return_dict=True)

        return output, gpt2_kwargs

    def _update_cache(self, gpt2_kwargs: Dict, output: Dict) -> Dict:
        # update generation input for next step
        cache_seqlens = gpt2_kwargs.get('cache_seqlens', None)
        past_key_values = gpt2_kwargs.get('past_key_values', None)
        if past_key_values is None:
            past_key_values = output.get('past_key_values', None)
        this_peer_finished = gpt2_kwargs.get('this_peer_finished', False)

        assert past_key_values is not None, "past_key_values should not be None when using cache"

        if cache_seqlens is None:
            inputs_embeds_mask = gpt2_kwargs.get('inputs_embeds_mask', None)
            assert inputs_embeds_mask is not None, "inputs_embeds_mask should not be None"
            cache_seqlens = inputs_embeds_mask.view(inputs_embeds_mask.shape[0], -1).sum(-1, dtype=torch.int32) - 1

        cache_seqlens = cache_seqlens + 1

        gpt2_kwargs.update({
            'cache_seqlens': cache_seqlens,
            'this_peer_finished': this_peer_finished,
            'past_key_values': past_key_values
        })

        return gpt2_kwargs

    @torch.no_grad()
    def predict(self, batch, hp, beam=1):
        if self.emb.is_token_input():
            raise NotImplementedError("token input has derapcated, please use bpe module")

        # prepare config
        config = GenerationConfig(
            hp=hp,
            **self.hparams.inference,
        )

        # prepare emb results and init cfg batch
        model_input = self.emb.predict_emb(
            batch=batch,
            hp=hp,
            beam=beam)

        original_batch_size = model_input['original_batch_size']
        exclude_ids = model_input['exclude_ids']
        inputs_embeds = model_input['inputs_embeds']

        frame_rate = config.frame_rate
        slice_dur = batch['slice_duration'].ceil().int()
        # Variable slice_dur will be used to calculate the number of tokens to generate.
        # For audio continuation, the prompt duration should be subtracted from the slice duration.
        if "audio_prompt" in batch:
            slice_dur = slice_dur - batch["target_tokens_length"] / frame_rate

        exclude_eos_first_secs = (
            slice_dur - config.exclude_eos_thresh_secs
            if config.exclude_eos_thresh_secs > 0
            else torch.empty(original_batch_size, dtype=torch.int32).fill_(config.exclude_eos_first_secs)
        ).int().cpu()

        num_tokens = (slice_dur + config.emit_eos_thresh_secs) * frame_rate if config.emit_eos_thresh_secs > 0 else config.duration * frame_rate
        if isinstance(num_tokens, torch.Tensor):
            num_tokens_max = num_tokens.int().amax().item()
        else:
            num_tokens_max = num_tokens
            num_tokens = torch.empty(original_batch_size, dtype=torch.int32).fill_(num_tokens_max)

        n_cfg_path = 0
        if config.use_controller_cfg:
            cfg_batch_size = model_input['cfg_batch_size']
            n_cfg_path = cfg_batch_size // original_batch_size

        batch_size = inputs_embeds.size(0)
        logits_processor = LogitsProcessor(config)
        generation_state = GenerationState(beam, original_batch_size, self.gpt2.device)

        token_buffer = None
        if config.use_step_out_blank:
            token_buffer = TokenBuffer(beam, original_batch_size, config.step_out_blank_max_len, batch)

        # prepare cuda graph
        self._prepare_cuda_graph(
            config.use_cache, config.use_cuda_graph,
            config.cuda_graph_max_bsz, config.cuda_graph_max_seqlen,
        )

        # generation loop
        gpt2_kwargs = dict(use_cache=config.use_cache)
        pbar = tqdm(range(num_tokens_max))
        tqdm_name = f"{self.__class__.__name__}.rank{DIST_ENV.local_rank}"

        for step in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens_max}]")
            # forward
            output, gpt2_kwargs = self._forward_step(model_input, gpt2_kwargs, step, config.use_cache, config.use_cuda_graph)

            # update cache
            if config.use_cache:
                gpt2_kwargs = self._update_cache(gpt2_kwargs, output)

            logits = output['logits'].float()[:, -1:, :]

            # cfg
            logits = logits_processor.apply_cfg(logits, batch_size, n_cfg_path)

            # sob
            if token_buffer:
                logits = logits_processor.apply_step_out_blank(logits, token_buffer.buffers, beam, original_batch_size)

            logits = logits_processor.apply_eos_control(
                logits, step, exclude_eos_first_secs, frame_rate,
                original_batch_size, self.emb.target_embedder.eos_id
            )

            # sampling
            predict_token = self.sample_logits(
                step, logits, config.temperature, config.sample_mode, config.sample_thresh, exclude_ids
            )

            if config.use_step_out_blank:
                token_buffer.update(predict_token)

            # get next input embedding
            predict_token_emb = self.emb.target_embedder.embedder(predict_token)
            if config.use_controller_cfg:
                predict_token_emb = predict_token_emb.repeat(n_cfg_path + 1, 1, 1)

            # update cache
            if config.use_cache:
                model_input['inputs_embeds'] = predict_token_emb
            else:
                model_input['inputs_embeds'] = torch.cat((model_input['inputs_embeds'], predict_token_emb), 1)

            generation_state.update_output_tokens(predict_token)

            # stopping criteria
            if generation_state.check_stopping_criteria(predict_token, step, num_tokens, self.emb.target_embedder.eos_id, config.stop_eos):
                break

        return generation_state.output_tokens

    def __del__(self):
        if self.graph is not None:
            self.release_cuda_graph()

    def execution_order(self) -> list[str]:
        """LSDP need this"""
        return ["gpt2"]

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
            "enable_omnistore": True,
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
        results = trainer.predict(model, datamodule=datamodule)
    else:
        # Training
        trainer.fit(model, datamodule=datamodule)

# This file describe training and infer tasks of semantic seed model.

import os
import gc
import torch
from typing import List, Dict, Optional, Union, Tuple
from cruise import CruiseConfig
from cruise.trainer.callback import ModelCheckpoint
from cruise import CruiseCLI, CruiseConfig, CruiseModule, last_cli
import torch.distributed
from functools import partial, reduce
import operator

import samantha # noqa: F401, resolve mariana python path
from accelerate import init_empty_weights, init_on_device
from mariana.models.audio.speech_checkpoint import SpeechModelCheckpoint
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.utils.exp_helper import ExpHelper
from samantha.criterion.masked_loss import sequence_mask
from apps.bigmusic.mariana_tasks.utils.speech_data_collect_callback import SpeechDataCollectCallback, TokenNumPerCategoryParser
from apps.bigmusic.mariana_tasks.semantic_seed_train import (
    SemanticLlmCLI,
    SemanticLlmModel,
    SemanticLlmTrainer,
    _m8_network_config,
    _get_skip_meter_name,
    GenerationConfig,
    GenerationState,
    TokenBuffer,
    LogitsProcessor
)
from apps.bigmusic.mariana_tasks.mtp_module import (
    MultiTokenPredictionModule,
    HierarchicalTokenPredictionModule,
)

import hyperpyyaml
from tqdm.auto import tqdm
from recipes.bigmusic.lightning.base_modules import TokenBuffer
from recipes.musiclm.inference.utils import sample, adaptive_sampling, SamplingScheduler
from samantha.utils.hparams import DotDict
import math
from cruise.utilities.distributed import DIST_ENV
import copy
from panther.custom_ops.torch.flash_attn import unpad_input

logger = AudioLogger()

_LITE_USE_MULTITASK = int(os.getenv("_LITE_USE_MULTITASK", "0")) != 0
if _LITE_USE_MULTITASK:
    from samantha.dataio.bigmusic.lite_multitask import MusicLiteDataModule
else:
    from samantha.dataio.bigmusic.lite import MusicLiteDataModule
logger.info(f"run training with {_LITE_USE_MULTITASK=}")


# import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

class MTPLogitsProcessor(LogitsProcessor):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)

    
    def apply_step_out_blank(self, logits: torch.Tensor, previous_tokens: Union[List[List[int]], List[List[List[int]]]],
                                beam: int, original_batch_size: int, r_idx=None) -> torch.Tensor:
        if not self.config.use_step_out_blank or self.config.step_out_blank_logic != 'v4':
            logger.warning("The step_out_blank for v1, v2, and v3 are now deprecated. Only v4 is supported.")
            return logits

        this_rp = self.config.repetition_penalty
        if isinstance(this_rp, list) and r_idx is not None:
            this_rp = this_rp[r_idx]

        if r_idx is not None:
            previous_output_tokens = torch.tensor(previous_tokens[r_idx], dtype=torch.long, device=logits.device).reshape(beam * original_batch_size, -1)
        else:
            previous_output_tokens = torch.tensor(previous_tokens, dtype=torch.long, device=logits.device).reshape(beam * original_batch_size, -1)

        bin_counts = torch.zeros(
            [beam * original_batch_size, logits.size(-1) + 1],
            dtype=torch.long, device=logits.device
        )
        
        bin_counts.scatter_add_(1, previous_output_tokens, torch.ones_like(previous_output_tokens))
        bin_counts = bin_counts[:, :logits.size(-1)]

        new_logits = logits.clone()
        penalty_logits = torch.gather(logits, dim=-1, index=previous_output_tokens.unsqueeze(1))
        penalty_logits = (penalty_logits / (this_rp ** torch.sign(penalty_logits))).to(new_logits)
        new_logits.scatter_(dim=-1, index=previous_output_tokens.unsqueeze(1), src=penalty_logits)

        mask = (bin_counts > 0).view(logits.shape)
        negative_mask = logits < 0
        penalty_mask_neg = mask & negative_mask
        penalty_mask_pos = mask & (~negative_mask)
        logits = torch.where(penalty_mask_neg, logits * this_rp, logits)
        logits = torch.where(penalty_mask_pos, logits / this_rp, logits)

        flag = torch.allclose(logits, new_logits.to(logits))
        if not flag:
            import pdb; pdb.set_trace()
        return logits
    
    def apply_eos_control(self, logits: torch.Tensor, step: int, exclude_eos_first_n_tokens: torch.Tensor,
                         original_batch_size: int, eos_id: int) -> torch.Tensor:
        for i in range(original_batch_size):
            if step < exclude_eos_first_n_tokens[i]:
                logits[i, 0, eos_id] = -float('Inf')
        return logits
    
class MTPTokenBuffer:
    def __init__(self, beam: int, original_batch_size: int, step_out_blank_max_len: int, batch: Dict, R: int):
        self.beam = beam
        self.original_batch_size = original_batch_size
        self.step_out_blank_max_len = step_out_blank_max_len
        self.R = R
        self.buffers = self._initialize_buffers(batch)

    def _initialize_buffers(self, batch: Dict) -> Union[List[List[int]], List[List[List[int]]]]:
        if self.R == 1:
            buffers = [[] for _ in range(self.beam * self.original_batch_size)]
        else:
            buffers = [[[] for _ in range(self.beam * self.original_batch_size)] for _ in range(self.R)]
        
        if "audio_prompt_token_ids" in batch:
            token_ids = batch["audio_prompt_token_ids"].tolist()

            if self.R == 1:
                previous_tokens = reduce(
                    operator.add,
                    [[ti[:l]] for l, ti in zip(batch["target_tokens_length"].tolist(), token_ids)]
                    * self.beam
                )
                buffer_len = min(
                    max(len(pt) for pt in previous_tokens),
                    self.step_out_blank_max_len
                )
                buffers = [
                    [-1] * (buffer_len - len(pt)) + pt[-buffer_len:] 
                    for pt in previous_tokens
                ]

            else:
                target_tokens_length = batch["target_tokens_length"].tolist()
                for r in range(self.R):
                    previous_tokens_r = reduce(
                        operator.add,
                        [[ti[r][:l[r]]] for l, ti in zip(target_tokens_length, token_ids)]
                        * self.beam
                    )
                    buffer_len = min(
                        max(len(pt) for pt in previous_tokens_r),
                        self.step_out_blank_max_len
                    )
                    buffers[r] = [
                        [-1] * (buffer_len - len(pt)) + pt[-buffer_len:] 
                        for pt in previous_tokens_r
                    ]

        return buffers

    def update(self, predict_token: torch.Tensor, r_idx=None) -> None:
        predict_token_cpu = predict_token.cpu().numpy()
        if self.R > 1:
            assert r_idx is not None, "r_idx is required when R > 1"
        if r_idx is not None:
            for j in range(self.beam * self.original_batch_size):
                self.buffers[r_idx][j].append(predict_token_cpu[j, 0])
                if len(self.buffers[r_idx][j]) >= self.step_out_blank_max_len:
                    self.buffers[r_idx][j] = self.buffers[r_idx][j][-self.step_out_blank_max_len:]
        else:
            for j in range(self.beam * self.original_batch_size):
                self.buffers[j].append(predict_token_cpu[j, 0])
                if len(self.buffers[j]) >= self.step_out_blank_max_len:
                    self.buffers[j] = self.buffers[j][-self.step_out_blank_max_len:]

class MTPGenerationState(GenerationState):
    def check_stopping_criteria(self, predict_token: torch.Tensor, step: int, num_tokens: torch.Tensor,
                                eos_id: int, stop_eos: bool) -> bool:
        """
        predict_token: 
            RVQ-MTP: [beam*BS, G, T=1, R]
            VQ-MTP:  [beam*BS, G, T=1]

        self.output_tokens
            RVQ-MTP: [beam*BS, G, T, R]
            VQ-MTP:  [beam*BS, G*T, 1]
        """
        if not stop_eos:
            return False

        if predict_token.ndim == 4: # RVQ-MTP tokens [B, G, 1, R]
            predict_token = predict_token.view(self.beam, self.original_batch_size, -1, predict_token.shape[-1]) # [B, BS, G*1, R]
            eos_mask = (predict_token == eos_id).any(dim=-2)    # [beam, BS, R]
            eos_mask = eos_mask[..., 0] # [B, BS]
        else:
            predict_token = predict_token.view(self.beam, self.original_batch_size, -1)   # [B, BS, G*1]
            eos_mask = (predict_token == eos_id).any(dim=-1) # [beam, BS]

        if num_tokens.ndim == 1:
            num_tokens = num_tokens.unsqueeze(0).expand(self.beam, -1) 
        num_tokens_mask = (step >= num_tokens)

        eos_mask = torch.logical_or(eos_mask, num_tokens_mask)
        self.is_eos_stop += eos_mask

        if predict_token.ndim == 4: 
            self.output_tokens[eos_mask.view(-1), :, -1] = eos_id
        else:
            self.output_tokens[eos_mask.view(-1), -1] = eos_id  # [beam*BS, G*T, 0]
        return torch.all(self.is_eos_stop > 0)


    def update_output_tokens(self, predict_token: torch.Tensor) -> None:
        """
        predict_token: 
            VQ-MTP:  [original_batch_size, G, T=1]
            RVQ:     [original_batch_size, T=1, R] (deprecated, can be merged to RVQ-MTP when G=1)
            RVQ-MTP: [original_batch_size, G, T=1, R]
        """
        # @qinxin: all types of tokens should be concatenated at T-axis (axis=2), due to historical reasons I left it as G-axis (axis=1)
        concat_dim = 1  
        if predict_token.ndim == 4:     # 2d tokens
            concat_dim = 2  # concat at T-axis
        self.output_tokens = torch.cat([self.output_tokens, predict_token], dim=concat_dim) if self.output_tokens is not None else predict_token

class SemanticLlmMtpTrainer(SemanticLlmTrainer):
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
        ('eos_acc', {'type': 'Weighted', 'args': ['eos_acc', 'tokens']}),
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
        if "rvq" in global_config.model.network.mtp_config_path:
            rvq_meters = []
            for r in range(2): # here fix the output R=2
                rvq_meters.append(
                    (f'rvq/acc_r{r}', {'type': 'Weighted', 'args': [f'tgt_accu_r{r}', 'tokens']})
                )
                rvq_meters.append(
                    (f'rvq/loss_r{r}', {'type': 'Weighted', 'args': [f'tgt_loss_r{r}', 'tokens']})
                )
            self.train_meters.extend(rvq_meters)
        TokenNumPerCategoryParser.initialize(global_config)
        self.train_meters.extend(TokenNumPerCategoryParser.get_train_meters())


class SemanticLlmModelMtp(SemanticLlmModel):

    def __init__(self,
            network: CruiseConfig = CruiseConfig(dict(_m8_network_config)),
            emb_path='',
            llm_path='',
            partial_pretrain='',
            hybrid_shard_group_size=-1,
            ddp_gate=False,
            weighted_loss=False,
            inference: CruiseConfig = CruiseConfig(dict({})),
            emb_cls_name = "SemanticEmbModuleMtp"
            ):
        super().__init__(
            network=network,
            emb_path=emb_path,
            llm_path=llm_path,
            partial_pretrain=partial_pretrain,
            hybrid_shard_group_size=hybrid_shard_group_size,
            ddp_gate=ddp_gate,
            weighted_loss=weighted_loss,
            inference=inference,
            emb_cls_name=emb_cls_name,
        )
        with init_on_device(torch.device('cpu')): # if DIST_ENV.local_rank == 0 else init_empty_weights():
            mtp_config_path = network.get('mtp_config_path')
            with open(mtp_config_path, 'r') as f:
                mtp_hps = hyperpyyaml.load_hyperpyyaml(f)
            if 'codebook_depth' in mtp_hps:
                self.mtp = HierarchicalTokenPredictionModule(**mtp_hps)
            else:
                self.mtp = MultiTokenPredictionModule(**mtp_hps)
    
    def load_emb_weights(self):
        state_dict = None
        mtp_state_dict = dict()
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
        
            mtp_keys = [key for key in state_dict if key.startswith('mtp_module.')]
            for key in mtp_keys:
                new_key = key.replace('mtp_module.', '')
                mtp_state_dict[new_key] = state_dict[key]
                state_dict.pop(key)

        if len(mtp_state_dict) > 0:
            incompatible_keys = self.mtp.load_state_dict(mtp_state_dict, strict=False)
            self.rank_zero_info(f"Semantic mtp model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"Semantic mtp model unexpected keys are {incompatible_keys.unexpected_keys}")
            assert len(incompatible_keys.missing_keys) == 0
            mtp_state_dict.clear()
            del mtp_state_dict
            gc.collect()

        # NOTE: redefine names in state dict if need
        if state_dict is not None:
            incompatible_keys = self.emb.load_state_dict(state_dict, strict=False)
            self.rank_zero_info(f"Semantic emb model missing keys are {incompatible_keys.missing_keys}")
            self.rank_zero_info(f"Semantic emb model unexpected keys are {incompatible_keys.unexpected_keys}")
            assert len(incompatible_keys.missing_keys) == 0
            state_dict.clear()
            del state_dict
            gc.collect()

            
    def setup(self, stage='fit', log_model_info=True, update_wrap=True):
        super().setup(stage, log_model_info, update_wrap)
        # workaround here
        self.emb.set_mtp_module(self.mtp)

    def forward(self, batch, **kwargs):

        training_inputs = self.emb(batch)
        prefix_length = training_inputs['prefix_length']
        target_length = training_inputs['target_length']
        token_length = prefix_length + target_length.to(prefix_length.device)

        input_token_embeds = training_inputs['token_embeds'][:, :-1]
        seq_len = training_inputs['token_embeds'].shape[1]
        shifted_input_mask = sequence_mask(token_length, seq_len, device="cuda")[:, 1:]
        input_embeds_rmpad, indices, cu_seqlens_q, max_seqlen_q = unpad_input(input_token_embeds.contiguous(), shifted_input_mask)

        hidden_states = self.gpt2(
            inputs_embeds=input_embeds_rmpad.unsqueeze(1).contiguous(),
            return_final_hidden_states=True,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )
        if self.mtp.use_lm_head_output and self.gpt2.lm_head is not None:
            hidden_states_norm = self.mtp.lm_head_norm(hidden_states)
            hidden_states = self.gpt2.lm_head(hidden_states_norm)

        token_length = training_inputs["token_length"].long().cpu().tolist()
        # split torch.Tensor to List[torch.Tensor]
        # unpad [T1+T2+...+Tb] to List of [T1], [T2], ... according to token_length
        hidden_states_list, input_embeds_list = [], []
        start_pos = 0
        for seg_len in token_length:
            # exclude sos length (-1)
            end_pos = start_pos + seg_len - 1
            hidden_states_list.append(hidden_states[start_pos:end_pos].squeeze(1))    # [T, D]
            start_pos = end_pos

        target_logits = self.mtp.decode_hidden_state(
                        hidden_state = hidden_states_list,
                        prefix_length = training_inputs["prefix_length"], 
                        # target_length = training_inputs["ori_target_length"],
                        target_length = training_inputs["target_length"],   # @qinxin: change ori_target_length (flattened length) to target_length (grouped length)
                        group_token_embeds = training_inputs["target_group_token_embeds"],
                        target_embedder = self.emb.target_embedder,
                        group_token_ids = training_inputs["target_group_token_ids"],
                        )
    
        outputs = self.calc_loss_acc(
                        target_logits = target_logits, 
                        target_ids = training_inputs["target_ids"],
                        prefix_length = training_inputs["prefix_length"],
                        eos_id = self.emb.target_embedder.eos_id,
                        )

        outputs['seqlens_q'] = cu_seqlens_q.diff()
        outputs['gpt2_lengths'] = outputs['seqlens_q']
        outputs['tokens'] = input_embeds_rmpad.shape[0]

        outputs['consume_tokens(B)'] = outputs['tokens'] * 1e-9
        outputs['loss_tokens(B)'] = outputs['loss_tokens'] * 1e-9
        return outputs


    def calc_loss_acc(
            self, 
            target_ids: List[torch.Tensor], # 
            prefix_length: torch.Tensor,
            target_logits: List[torch.Tensor], # [(T//G+1(eos))*G, n_logits] * B
            target_loss_mask: Optional[torch.Tensor] = None,
            eos_id: int = None,
            ):
        """
        target_ids: list [B] of [1(sos)+(T//G+1(eos)) * G]
        target_logits: list [B] of [(T//G+1(eos))*G, n_logits]
        """
        # remove sos
        assert target_loss_mask is None
        target_ids = torch.cat([target_id[1:] for target_id in target_ids], dim=0)
        target_logits = torch.cat(target_logits, dim=0)
        assert target_ids.shape[0] == target_logits.shape[0], f"batch size mismatch {target_ids.shape=} {target_logits.shape=}"
        result_dict = {}

        # @qinxin: cancel the loss for sos token (because sos token is not given in the form of multi-token), 
        # in this case we need to manully add sos token during inference
        sos_token_pos = prefix_length.long().unsqueeze(1) - 1
        if target_ids.ndim == 2:  # RVQ/HVQ
            target_loss = []
            target_accu = []
            for r in range(target_logits.shape[-2]):
                target_loss_r = self.mtp.criterion(target_logits[..., r,:], target_ids[..., r])
                target_accu_r = (target_logits[..., r,:].argmax(dim=-1) == target_ids[..., r]).float().mean()
                target_loss.append(target_loss_r)
                target_accu.append(target_accu_r)
                result_dict.update({
                    f'tgt_accu_r{r}': target_accu_r.item(),
                    f'tgt_loss_r{r}': target_loss_r.item(),
                })
            target_loss = sum(target_loss) / len(target_loss)
            accu = sum(target_accu) / len(target_accu)
        else:   # VQ
            target_loss = self.mtp.criterion(target_logits, target_ids)
            
        accu = target_logits.argmax(dim=-1) == target_ids
        if eos_id is not None:
            eos_mask = target_ids == eos_id
            accu_eos = accu[eos_mask].float().mean()
        accu = accu.float().mean()
        num_valid_tokens = target_ids.numel()
        
        result_dict.update({
            'loss': target_loss,    # with grad
            'acc': accu.item(),
            'loss_tokens': num_valid_tokens,
            'tgt_loss': target_loss.item(), # without grad
            'eos_acc': accu_eos.item(),
        })
        return result_dict
    
    def _prepare_cuda_graph(self, use_cache: bool, use_cuda_graph: bool, cuda_graph_max_bsz: int = 1, cuda_graph_max_seqlen: int = 1024) -> None:
        # init cuda graph
        if use_cache and use_cuda_graph and self.graph is None:
            logger.info(f"Initializing CUDA graph for {self.gpt2.__class__.__name__} "
                       f"with max_bsz={cuda_graph_max_bsz} and max_seqlen={cuda_graph_max_seqlen}")

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                # max_batch_size must equal to beam_size * real_batch_size
                self.capture_cuda_graph(cuda_graph_max_bsz, cuda_graph_max_seqlen, return_hidden_states=True)

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
                    output = self.gpt2(**gpt2_kwargs, return_final_hidden_states=True, return_dict=True)
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



    @torch.no_grad()
    def predict(self, batch, hp, beam=1):
        model_input = self.emb.predict_emb(
            batch=batch,
            hp=hp,
            beam=beam)

        original_batch_size = model_input['original_batch_size']
        exclude_ids = model_input['exclude_ids']
        inputs_embeds = model_input['inputs_embeds']

        config = GenerationConfig(
            hp=hp,
            original_batch_size=original_batch_size,
            cfg_batch_size = model_input.get('cfg_batch_size', 0),
            **self.hparams.inference,
        )

        frame_rate = config.frame_rate
        slice_dur = batch['slice_duration'].ceil().int()
        if "audio_prompt" in batch:
            slice_dur = slice_dur - batch["target_tokens_length"] / frame_rate
        
        
        def compute_num_tokens(slice_dur, config):
            exclude_eos_first_secs = (
                slice_dur - config.exclude_eos_thresh_secs
                if config.exclude_eos_thresh_secs > 0
                else torch.empty(original_batch_size, dtype=torch.int32).fill_(config.exclude_eos_first_secs)
            ).int().cpu()
            num_tokens = (slice_dur + config.emit_eos_thresh_secs) * frame_rate if config.emit_eos_thresh_secs > 0 else config.duration * frame_rate
            if isinstance(self.mtp, HierarchicalTokenPredictionModule): # 2d pattern (RVQ)
                num_tokens = num_tokens // self.mtp.group + self.mtp.group
                exclude_eos_first_n_tokens = frame_rate * exclude_eos_first_secs // self.mtp.group
                # exclude_1in_first_n_tokens = self.extra_params.semantic_frame_rate * 120 // self.target_embedder.group
            else:   # 1d pattern (VQ)
                num_tokens = num_tokens // self.mtp.group * self.mtp.R + self.mtp.group 
                exclude_eos_first_n_tokens = frame_rate * exclude_eos_first_secs // self.mtp.group * self.mtp.R
                # exclude_1in_first_n_tokens = self.extra_params.semantic_frame_rate * 120 // self.target_embedder.group * self.target_embedder.R

            if isinstance(num_tokens, torch.Tensor):
                num_tokens_max = num_tokens.int().amax().item()
            else:
                num_tokens_max = num_tokens
                num_tokens = torch.empty(original_batch_size, dtype=torch.int32).fill_(num_tokens_max)
            return num_tokens, num_tokens_max, exclude_eos_first_n_tokens

        num_tokens, num_tokens_max, exclude_eos_first_n_tokens = compute_num_tokens(slice_dur, config)
        batch_size = inputs_embeds.size(0)
        mtp_kwargs = {}
        logits_processor = MTPLogitsProcessor(config)
        generation_state = MTPGenerationState(beam, original_batch_size, self.gpt2.device)

        token_buffer = None
        if config.use_step_out_blank:
            token_buffer = MTPTokenBuffer(beam, original_batch_size, config.step_out_blank_max_len, batch, self.mtp.R)
        
        # prepare cuda graph
        self._prepare_cuda_graph(
            config.use_cache, config.use_cuda_graph,
            config.cuda_graph_max_bsz, config.cuda_graph_max_seqlen,
        )

        # generation loop
        gpt2_kwargs = dict(use_cache=config.use_cache)
        pbar = tqdm(range(num_tokens_max))
        tqdm_name = f"{self.__class__.__name__}.rank{DIST_ENV.local_rank}"


        mtp_pattern = hp.get("multi_token_pattern", None)
        print(f"{num_tokens=}, {self.mtp.group=}, {self.mtp.R=}")
        if mtp_pattern in ["parallel-rq", "parallel-arq", "delay-2d-rq", "delay-2d-arq", "delay-2d-context-rq"]:
            self.mtp.enable_kv_cache(batch_size=batch_size, target_embedder=self.emb.target_embedder)

        def sample_single_token(logits, r_idx=None):
            logits = logits_processor.apply_cfg(logits, batch_size, config.n_cfg_path)
            if token_buffer:
                logits = logits_processor.apply_step_out_blank(logits, token_buffer.buffers, beam, original_batch_size, r_idx)
            
            logits = logits_processor.apply_eos_control(
                logits, step, exclude_eos_first_n_tokens, original_batch_size, self.emb.target_embedder.eos_id
            )

            temp = config.temperature
            if r_idx is not None and isinstance(config.temperature, list):
                temp = config.temperature[r_idx]
            thresh = config.sample_thresh
            if r_idx is not None and isinstance(config.sample_thresh, list):
                thresh = config.sample_thresh[r_idx]

            predict_token = self.sample_logits(step, logits, temp, config.sample_mode, thresh, exclude_ids)
            if config.use_step_out_blank:
                token_buffer.update(predict_token, r_idx)
            
            predict_token = predict_token.reshape(original_batch_size, 1)
            return predict_token

        for step in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens_max}]")
            output, gpt2_kwargs = self._forward_step(model_input, gpt2_kwargs, step, config.use_cache, config.use_cuda_graph)

            # update cache
            if config.use_cache:
                gpt2_kwargs = self._update_cache(gpt2_kwargs, output)

            if step == 0:
                hidden_states = output['last_hidden_states'][:, -1:, :]
            else:
                hidden_states = output['logits'][:, -1:, :]

            # add for extra lm head
            if self.mtp.use_lm_head_output and self.gpt2.lm_head is not None:
                hidden_states_norm = self.mtp.lm_head_norm(hidden_states)
                hidden_states = self.gpt2.lm_head(hidden_states_norm)

            predict_token = self.mtp.predict(
                hidden_states = hidden_states, 
                mtp_pattern = mtp_pattern, 
                target_embedder = self.emb.target_embedder,
                use_controller_cfg = config.use_controller_cfg,
                n_cfg_path = config.n_cfg_path,
                sample_single_token_func = sample_single_token,
                **mtp_kwargs,
            )

            if mtp_pattern in ["parallel-rq", "parallel-arq"]:  # [B, G]
                predict_token = predict_token.reshape(original_batch_size, self.mtp.group, 1) # [B, G, T=1]
                predict_token_emb = self.emb.embed_token_id(token_ids=predict_token, frame_idx=step)    # [B, G, T=1, D]
                predict_token_emb = self.mtp.encode_token_emb(predict_token_emb)    # [B, T=1, D]
            elif mtp_pattern in ["delay-2d-rq", "delay-2d-arq", "delay-2d-context-rq"]: # [B, T=1, G, R]
                mtp_kwargs = {'last_predict_token': predict_token}  # [B, T=1, G, R]
                predict_token = predict_token.transpose(1, 2).reshape(original_batch_size, self.mtp.group, 1, self.mtp.R) # [B, G, T=1, R]
                this_input_R = self.mtp.R
                if hp.get("input_R", None) is not None:
                    this_input_R = int(hp.get("input_R", None))
                predict_token_emb, _ = self.emb.embed_token_id(predict_token[...,:this_input_R], frame_idx=step) # [B, G, T=1, D]
                predict_token_emb = self.mtp.encode_token_emb(predict_token_emb)    # [B, T=1, D]
            else:
                predict_token_emb = self.emb.target_embedder.embedder(predict_token)

            if config.use_controller_cfg:
                predict_token_emb = predict_token_emb.repeat(config.n_cfg_path + 1, 1, 1)

            if config.use_cache:
                model_input['inputs_embeds'] = predict_token_emb
            else:
                model_input['inputs_embeds'] = torch.cat((model_input['inputs_embeds'], predict_token_emb), 1)

            generation_state.update_output_tokens(predict_token) 

            if generation_state.check_stopping_criteria(predict_token, step, num_tokens, self.emb.target_embedder.eos_id, config.stop_eos):
                break

        # @qinxin: here no need to ungroup_token for parallel-rq/arq, 
        # as output_tokens have been flattened and concatenated across time axis.
        output_tokens = torch.stack([self.mtp.ungroup_token(
                                group_ids=generation_state.output_tokens[b], 
                                sos_id = self.emb.target_embedder.sos_id,
                                eos_id = self.emb.target_embedder.eos_id,
                                delete_null=False, 
                                add_sos=False, 
                                delay_back=False) for b in range(generation_state.output_tokens.shape[0])], 0)
        print(output_tokens.shape)
        return output_tokens
    

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
        SemanticLlmModelMtp,
        datamodule_class=MusicLiteDataModule,
        trainer_class=SemanticLlmMtpTrainer,
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

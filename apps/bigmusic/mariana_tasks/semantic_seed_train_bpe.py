"""A Semantic Module that supports the simple BPE-based implementation"""

import os
import torch
from cruise import CruiseConfig
from cruise.trainer.callback import ModelCheckpoint
import torch.distributed

import samantha # noqa: F401, resolve mariana python path
from mariana.models.audio.speech_checkpoint import SpeechModelCheckpoint
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.utils.exp_helper import ExpHelper
from samantha.dataio.bigmusic.lite import MusicLiteDataModule
from apps.bigmusic.mariana_tasks.utils.speech_data_collect_callback import SpeechDataCollectCallback
from apps.bigmusic.mariana_tasks.semantic_seed_train import (
    SemanticLlmCLI,
    SemanticLlmModel,
    SemanticLlmTrainer,
    _m8_network_config,
    _inference_config,
)
from mariana.models.audio.gpt2_audio import (
    GPT2LMHeadModel,
)

from tqdm.auto import tqdm
import math
from cruise.utilities.distributed import DIST_ENV
import copy
from panther.custom_ops.torch.flash_attn import unpad_input


logger = AudioLogger()

class SemanticLlmModelBpe(SemanticLlmModel):

    def __init__(self,
            network: CruiseConfig = CruiseConfig(dict(_m8_network_config)),
            emb_path='',
            llm_path='',
            partial_pretrain='',
            hybrid_shard_group_size=-1,
            ddp_gate=False,
            inference: CruiseConfig = CruiseConfig(dict(_inference_config))):
        super().__init__(
            network=network,
            emb_path=emb_path,
            llm_path=llm_path,
            partial_pretrain=partial_pretrain,
            hybrid_shard_group_size=hybrid_shard_group_size,
            ddp_gate=ddp_gate,
            inference=inference,
        )
        # NOTE: This is a hack to remove the embedding module while keeping other methods reusable
        del self.emb
        self.emb = None


    def forward(self, batch, *_args, **_kwargs):

        # gpt forward in naive & elegant way
        # outputs = self.gpt2(
        #     input_ids = batch["input_ids"],
        #     attention_mask = batch["attention_mask"],
        #     labels = batch["input_ids"],
        #     label_mask = batch["token_type_ids"][:,1:].contiguous(),  # the loss mask is not shifted inside the model
        # )

        # gpt forward in messy & inelegant way, to save memory in CE calculation 
        hidden_states = self.gpt2(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            return_final_hidden_states=True,
        )
        attention_mask = batch["attention_mask"][:,1:].contiguous()
        labels_shift_rmpad, _, _, _ = unpad_input(batch["input_ids"][..., 1:].unsqueeze(2).contiguous(), attention_mask)
        hidden_states_rmpad, _, _, _  = unpad_input(hidden_states[:, :-1, :].contiguous(), attention_mask)
        loss_mask_rmpad, _, _, _ = unpad_input(batch["token_type_ids"][:,1:].unsqueeze(2).contiguous(), attention_mask)
        loss_mask_rmpad = loss_mask_rmpad.view(-1)

        loss, acc = self.gpt2.transformer.wte(
            hidden_states_rmpad.view(-1, hidden_states.size(-1)).to(torch.bfloat16),
            treat_as_embedding=False,
            return_cross_entropy_loss=True,
            labels=labels_shift_rmpad.view(-1),
            use_flash_ce=self.hparams.network.use_flash_ce,
        )

        num_valid_tokens = loss_mask_rmpad.sum()
        outputs = dict()
        loss = (loss * loss_mask_rmpad).sum() / loss_mask_rmpad.sum()
        acc = (acc * loss_mask_rmpad).sum() / loss_mask_rmpad.sum()
        outputs["loss"] = loss
        outputs["acc"] = acc
        outputs["eos_acc"] = -1.0


        outputs['tokens'] = batch["attention_mask"].sum()
        outputs["loss_tokens"] = batch['token_type_ids'].sum()
        outputs['consume_tokens(B)'] = outputs['tokens'] * 1e-9
        outputs['loss_tokens(B)'] = outputs['loss_tokens'] * 1e-9
        outputs['gpt2_lengths'] = torch.sum(batch["attention_mask"][:, 1:], dim=1) # TODO: check
        if self.hparams.network.get('return_moe_metric', False):
            expert_cnt = outputs['moe_metric']
            n_layer = expert_cnt.shape[0]
            for i in range(n_layer):
                outputs[f'expert_cnt_layer{i}'] = expert_cnt[i]
                outputs[f'expert_active_cnt_layer{i}'] = expert_cnt[i].nonzero().numel()

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        # TODO add it back
        fwd_flops = self.gpt2.calc_flops_rmpad(outputs['gpt2_lengths'])
        bwd_flops = fwd_flops * 2
        if not self.training:
            bwd_flops = 0
        outputs['flops'] = fwd_flops + bwd_flops
        if self.training:
            outputs['lr * 1e3'] = self.trainer.optimizers[0].param_groups[0]['lr'] * 1000
        del outputs['gpt2_lengths']
        return outputs

    def validation_step(self, batch, batch_idx, *_args, **_kwargs):
        return self.training_step(batch, batch_idx)
    
    
    def capture_cuda_graph(self, max_batch_size, max_seqlen):
        """
        Create kvcache and capture CUDA graph for inference. LLM part only.
        """
        gpt2: GPT2LMHeadModel = self.gpt2
        device = torch.device("cuda")
        past_key_values = torch.empty(
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
            fwd_func(inputs_embeds, inputs_embeds_mask, cache_seqlens, past_key_values)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        DIST_ENV.barrier()
        # Capture the forward pass.
        with torch.cuda.graph(self.graph):
            logits = fwd_func(inputs_embeds, inputs_embeds_mask, cache_seqlens, past_key_values)
        torch.cuda.synchronize()
        DIST_ENV.barrier()
        # Save the input and output buffers.
        self.input_buffers = {
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "inputs_embeds_mask": inputs_embeds_mask,
            "cache_seqlens": cache_seqlens,
        }
        self.output_buffers = {"logits": logits}
    
    def release_cuda_graph(self):
        """
        Release CUDA graph.
        """
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
        self.input_buffers["past_key_values"].copy_(past_key_values)
        self.input_buffers["inputs_embeds"][:real_batch_size].copy_(inputs_embeds)
        self.input_buffers["inputs_embeds_mask"][:real_batch_size].copy_(inputs_embeds_mask)
        self.input_buffers["cache_seqlens"][:real_batch_size].copy_(cache_seqlens)
        self.graph.replay()
        return {
            "logits": self.output_buffers["logits"][:real_batch_size],
            "past_key_values": self.input_buffers["past_key_values"],
        }

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, rl_training=False):
        frame_rate = self.extra_params.get("semantic_frame_rate")
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode
        sample_thresh = hp.get('sample_thresh', 0.9)
        use_controller_cfg = hp.get('use_controller_cfg', False)
        controller_cfg_gamma = hp.get('controller_cfg_gamma', 1)
        use_step_out_blank = hp.get('use_step_out_blank', False)
        step_out_blank_logic = hp.get('step_out_blank_logic', 'v2')
        step_out_blank_max_len = hp.get('step_out_blank_max_len', 10)
        repetition_penalty = hp.get('repetition_penalty', 1.0)
        exclude_eos_first_secs = hp.get('exclude_eos_first_secs', 0)
        exclude_eos_thresh_secs = hp.get('exclude_eos_thresh_secs', 0)
        exclude_text_tokens = hp.get('exclude_text_tokens', True)
        emit_eos_thresh_secs = hp.get('emit_eos_thresh_secs', 0)
        skip_sos = hp.get('skip_sos', False)
        stop_eos = hp.get('stop_eos', False)
    
        model_inputs = [] # model_inputs is a list of batch, e.g. [batch, uncond_batch], compatible with cfg
        eos_id = batch['eos_id'] # TODO get all required fields
        text_codebook_size = batch['text_codebook_size'] # TODO get all required fields

        model_inputs.append({'token_ids': batch['input_ids']})
        
        if use_controller_cfg:
            controller_cfg_gamma = [controller_cfg_gamma] if isinstance(controller_cfg_gamma, (float, int)) else controller_cfg_gamma # to list
            for idx in range(len(controller_cfg_gamma)):
                if idx == 0: # hack, will be removed in future version
                    model_inputs.append({'token_ids': batch[f'input_ids_uncond']})
                    continue
                model_inputs.append({'token_ids': batch[f'input_ids_uncond{idx}']})

        batch_size, seq_len = model_inputs[0]['token_ids'].size()

        if 'slice_duration' in batch:
            slice_dur = math.ceil(batch['slice_duration'].item())
            exclude_eos_first_secs = slice_dur - exclude_eos_thresh_secs if exclude_eos_thresh_secs > 0 else exclude_eos_first_secs
            num_tokens = (slice_dur + emit_eos_thresh_secs) * 25 if emit_eos_thresh_secs > 0 else num_tokens

        pbar = tqdm(range(num_tokens))
        # is_eos_stop = torch.zeros((batch_size * beam), dtype=torch.long, device=self.device)
        tqdm_name = f"{self.__class__.__name__}.rank{DIST_ENV.local_rank}" 

        def apply_cfg(logits_list, controller_cfg_gamma, batch_size, n_cfg_path):
            if len(controller_cfg_gamma) > 2: # group cfg TODO
                beam_bs = batch_size // (1+n_cfg_path)
                uncond_logits = logits[-beam_bs:]
                cond_cfg_logits = torch.zeros_like(uncond_logits)
                for gg in range(len(controller_cfg_gamma)):
                    this_cond_logits = logits[beam_bs*gg: beam_bs*(gg+1)]
                    cond_cfg_logits += (controller_cfg_gamma[gg] * this_cond_logits)
                logits = cond_cfg_logits
            elif len(logits_list) == 2: # vanilla cfg
                cond_logits, uncond_logits = logits_list
                controller_cfg_gamma = controller_cfg_gamma[0]
                logits = controller_cfg_gamma * cond_logits + (1 - controller_cfg_gamma) * uncond_logits
            else: # no cfg
                logits = logits_list[0]
            return logits


        if use_controller_cfg:
            cfg_batch_size = batch_size
        else:
            cfg_batch_size = None

        assert beam == 1, "predict_token not support beam > 1"
        assert step_out_blank_logic == "v4", "predict_token support step_out_blank_logic v4 only"

        original_batch_size = batch_size
        exclude_ids = []

        output_tokens = None
        if use_step_out_blank:
            # previous_tokens is a series of token buffers.
            # Suppose beam == 2, original_batch_size == 2, the buffer order is:
            #  |------ batch_size ------|  |------ batch_size ------|
            #  |-------- beam#0 --------|  |-------- beam#1 --------|
            # [beam#0_seq#0, beam#0_seq#1, beam#1_seq#0, beam#1_seq#1]
            previous_tokens = [[] for _ in range(beam * original_batch_size)]
            if "audio_prompt_token_ids" in batch:
                previous_tokens = torch.concat([batch["audio_prompt_token_ids"]] * beam).tolist()   # (beam * bs, audio_prompt_token_len)
                buffer_len = min(min(len(pt) for pt in previous_tokens), step_out_blank_max_len)
                previous_tokens = [pt[-buffer_len:] for pt in previous_tokens]

        # TODO: adapt to batch infer
        # slice_dur = math.ceil(batch['slice_duration'].item())
        # exclude_eos_first_secs = slice_dur - exclude_eos_thresh_secs if exclude_eos_thresh_secs > 0 else exclude_eos_first_secs
        # num_tokens = (slice_dur + emit_eos_thresh_secs) * 25 if emit_eos_thresh_secs > 0 else num_tokens

        pbar = tqdm(range(num_tokens))
        is_eos_stop = torch.zeros((beam, original_batch_size), dtype=torch.long, device=self.gpt2.device)

        for model_input in model_inputs:
            model_input["inputs_embeds"] = self.gpt2.transformer.wte(model_input["token_ids"].contiguous())

        use_cache = self.hparams.inference.get('use_cache', False)
        use_cuda_graph = self.hparams.inference.get('use_cuda_graph', False)
        if use_cache:
            # init cuda graph
            if use_cuda_graph and self.graph is None:
                cuda_graph_max_bsz = self.hparams.inference.get('cuda_graph_max_bsz', 1)
                cuda_graph_max_seqlen = self.hparams.inference.get('cuda_graph_max_seqlen', 1024)
                logger.info(f"init cuda graph for {self.gpt2.__class__.__name__} with max_bsz={cuda_graph_max_bsz} and max_seqlen={cuda_graph_max_seqlen}")
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    # max_batch_size must equal to beam_size * real_batch_size
                    self.capture_cuda_graph(cuda_graph_max_bsz, cuda_graph_max_seqlen)
                
        gpt2_kwargs = []
        for _ in model_inputs:
            gpt2_kwargs.append(dict(use_cache=use_cache))

        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")

            # to enable inference without a trainer, we simply cast the inputs to the expected model type
            # which is either torch.float16 or torch.bfloat16
            for model_input in model_inputs:
                model_input["inputs_embeds"] = model_input["inputs_embeds"].to(self.gpt2.transformer.ln_f.weight.dtype)
            
            for idx, gpt2_kwarg in enumerate(gpt2_kwargs):
                inputs_embeds = model_inputs[idx]["inputs_embeds"]
                                                # B, T
                inputs_embeds_mask = torch.ones(*model_inputs[idx]['inputs_embeds'].shape[:2], dtype=torch.bool, device=model_inputs[idx]["inputs_embeds"].device)
                if i == 0:
                    gpt2_kwargs[idx] = self.gpt2.prepare_inputs_for_generation(
                        None,
                        inputs_embeds=inputs_embeds,
                        inputs_embeds_mask=inputs_embeds_mask,
                        **gpt2_kwarg,
                    )
                else:
                    gpt2_kwarg.update(
                        inputs_embeds=inputs_embeds,
                        inputs_embeds_mask=inputs_embeds_mask
                    )

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = []
                for gpt2_kwarg in gpt2_kwargs:
                    if use_cache and use_cuda_graph:
                        if i == 0:
                            # prefill
                            gpt2_kwarg['past_key_values'] = self.input_buffers["past_key_values"].clone() # past_key_values generated by gpt2.prepare_inputs_for_generation as a length of 32768, overwrite it here
                            output = self.gpt2(**gpt2_kwarg, return_dict=True)  
                        else:
                            # decode
                            output = self.predict_by_cuda_graph(
                                gpt2_kwarg['inputs_embeds'],
                                gpt2_kwarg['inputs_embeds_mask'],
                                gpt2_kwarg['cache_seqlens'],
                                gpt2_kwarg['past_key_values'],
                            )
                        cache_seqlens = gpt2_kwarg.get('cache_seqlens', None)
                        this_peer_finished = gpt2_kwarg.get('this_peer_finished', False)
                        if cache_seqlens is None:
                            inputs_embeds_mask = gpt2_kwarg.get('inputs_embeds_mask', None)
                            assert inputs_embeds_mask is not None
                            cache_seqlens = inputs_embeds_mask.view(inputs_embeds_mask.shape[0], -1).sum(-1, dtype=torch.int32) - 1
                        cache_seqlens = cache_seqlens + 1

                        gpt2_kwarg['cache_seqlens'] = cache_seqlens
                        gpt2_kwarg['this_peer_finished'] = this_peer_finished
                        gpt2_kwarg['past_key_values'].copy_(output['past_key_values'])
                    else:
                        output = self.gpt2(**gpt2_kwarg, return_dict=True)
                    outputs.append(copy.deepcopy(output)) # prevent the second call to cuda graph from overwriting the first output

            logits_list = []
            for output in outputs:
                logits = output['logits']
                logits = logits.float()
                logits = logits[:, -1:, :] # only predicting on last logit.
                logits_list.append(logits)
            logits = apply_cfg(logits_list, controller_cfg_gamma, batch_size, cfg_batch_size)

            if use_step_out_blank:
                previous_output_tokens = torch.tensor(previous_tokens, dtype=torch.long, device='cuda').reshape(beam * original_batch_size, -1)
                bin_counts = torch.zeros([beam * original_batch_size, logits.size(-1)+1], dtype=torch.long, device='cuda')
                bin_counts.scatter_add_(1, previous_output_tokens, torch.ones_like(previous_output_tokens))

                bin_counts = bin_counts[:, 0:logits.size(-1)]

                mask = (bin_counts > 0).view(logits.shape)
                negative_mask = logits < 0

                logits = torch.where(mask.logical_and(negative_mask), logits * repetition_penalty, logits)
                logits = torch.where(mask.logical_and(negative_mask.logical_not()), logits / repetition_penalty, logits)

            if i < 25 * exclude_eos_first_secs:
                # Ensure that it generates at least 30s music.
                for j in range(original_batch_size):
                    logits[j, 0, eos_id] = -float('Inf')

            if exclude_text_tokens:
                logits[..., :text_codebook_size] = -float('Inf') # not using exclude_ids to prevent performance issues
                logits[..., eos_id+1:] = -float('Inf')
            predict_token = self.sample_logits(
                i, logits, temperature, sample_mode, sample_thresh, exclude_ids
            )

            if use_step_out_blank:
                predict_token_cpu = predict_token.cpu().numpy()
                # predict_token_cpu: (beam, b)
                # previous_tokens: (beam_size * b, queue_len)
                for beam_idx in range(beam):
                    for j in range(original_batch_size):
                        batch_slice = slice(beam_idx*original_batch_size, (beam_idx+1)*original_batch_size)
                        if len(previous_tokens[batch_slice][j]) < step_out_blank_max_len:
                            previous_tokens[batch_slice][j].append(predict_token_cpu[beam_idx][j])
                        else:
                            previous_tokens[batch_slice][j] = previous_tokens[batch_slice][j][-step_out_blank_max_len:]
                            previous_tokens[batch_slice][j].append(predict_token_cpu[beam_idx][j])

            # predict_token_emb = self.emb.target_embedder.embedder(predict_token)
            predict_token_emb = self.gpt2.transformer.wte(predict_token)

            for model_input in model_inputs:
                if use_cache:
                    model_input['inputs_embeds'] = predict_token_emb
                else:
                    model_input['inputs_embeds'] = torch.cat((model_input['inputs_embeds'],predict_token_emb), 1) 

            output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token

            if stop_eos:
                eos_mask = predict_token == (eos_id)  # (beam, b)
                is_eos_stop += eos_mask
                if torch.all(is_eos_stop > 0):
                    break 

        output_tokens -= text_codebook_size
        return output_tokens
    
class SemanticLlmBpeTrainer(SemanticLlmTrainer):
    train_meters = [
        ('loss', {'type': 'Weighted', 'args': ['loss', 'tokens']}),
        ('acc', {'type': 'Weighted', 'args': ['acc', 'tokens']}),
        ('lr * 1e3', {'type': 'Simple', 'args': ['lr * 1e3']}),
        ('loss_tokens(B)', {'type': 'Sum', 'args': ['loss_tokens(B)']}),
        ('consume_tokens(B)', {'type': 'Sum', 'args': ['consume_tokens(B)']}),
        ('flops', {'type': 'Flops', 'args': ['flops']}),
    ]

    valid_meters = [
        ('loss', {'type': 'Weighted', 'args': ['loss', 'loss_tokens']}),
        ('acc', {'type': 'Weighted', 'args': ['acc', 'loss_tokens']}),
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
            keys_to_collect = ["prompt","lyrics"],
            audio_key=None,
            audio_duration_key=None,
            max_items_to_save=data_save_max_items_from_env,
            every_n_train_steps=data_save_interval_from_env,
        )
        callbacks.append(collector)

    cli = CLI_Clazz(
        SemanticLlmModelBpe,  # <- modified
        datamodule_class=MusicLiteDataModule,
        trainer_class=SemanticLlmBpeTrainer,  # <- modified
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

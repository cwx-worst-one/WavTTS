from typing import List, Optional, Dict, Union
import readline # required, support delete in terminal
import math
import sys
import os
import traceback
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
import wandb

from cruise import CruiseModule, CruiseCLI, CruiseConfig, last_cli
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hexists
from mariana.utils.exp_helper import ExpHelper
from mariana.models.gpt2 import GPT2LMHeadModel, get_subsequent_mask, Conv1D
from mariana.utils.generate import sample_generate, play_file
from mariana.optim import mariana_optimizer_kwargs_defaults
from mariana.utils.checkpoint_utils import load_zero3_state_dict, is_zero3
from mariana.utils.eta_utils import ETAMeter

try:
    from megatron.core import mpu
except ImportError:
    print("Unable to import megatron, skipping...")

from rich.console import Console
from rich.markdown import Markdown

from samantha.dataio.dataloader.asr_dataloader import AsrDataModule
from samantha.utils.mariana_utils.merge_zero3_ckpt import merge_zero_checkpoints

# Config adapter
network_config = {
    "hidden_size": 2048,
    "n_embed": 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 1025,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "gradient_checkpointing_mlp": False,
    "gradient_checkpointing_ln": False,
    "tie_weight": True,
    "pad_idx": 2,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad": False,
  }


class GPT2Model(CruiseModule):
    """Deberta pretrain"""
    def __init__(self,
                 network: CruiseConfig = network_config,
                 freeze_prefix: Optional[List[str]] = None,
                 partial_pretrain: Optional[str] = None,
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams
        
        self.hparams.micro_batch_size = network.get('megatron_micro_batch_size', 1)

        # 文本
        self.gpt = GPT2LMHeadModel(self.hparams)
        # self.init_weights()  # will always load pretrained weights
        self.freeze_params(self.hparams.freeze_prefix or [])
        self.consume_tokens = 0
        self.last_consume_tokens = -1
        self.last_time = time.time()
        self.mfu = 0
        self.eta_meter = ETAMeter(window=100)

    def setup(self):
        _is_zero3 = is_zero3(last_cli().hparams)
        # In DDP rank 0 load pretrain weights is enough
        if (_is_zero3 or self.trainer.global_rank == 0) and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            from cruise.utilities.cloud_io import load as crs_load
            if 'mp_rank' in self.hparams.partial_pretrain:
                # zero2 checkpoints has key 'module'
                state_dict = crs_load(self.hparams.partial_pretrain, map_location='cpu')
                state_dict = state_dict['module']
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if _is_zero3:
                self.print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} start to load checkpoint to a zero3 partitioned model')
                if self.trainer.global_rank == 0:
                    state_dict = crs_load(self.hparams.partial_pretrain, map_location='cpu')
                else:
                    state_dict = None
                DIST_ENV.barrier()  # add barrier in case rank 0 download ckpt takes too long
                self.print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} loading checkpoint to a zero3 partitioned model')
                metadata = getattr(state_dict, "_metadata", None)
                error_msgs = []
                load_zero3_state_dict(state_dict, self, metadata, error_msgs, prefix="")
            else:
                state_dict = crs_load(self.hparams.partial_pretrain, map_location='cpu')
                self.partial_load_from_checkpoints(
                    state_dict,
                    rename_params=rename_params, verbose=True)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.rank_zero_print('freeze_params:', name)
                    param.requires_grad = False

    def count_tokens(self, token_masks):
        # group = mpu.get_data_parallel_group() if self.transformer_kernel == 'megatron' else None
        realbsz = token_masks.shape[0]
        batch_tokens = token_masks.sum().item()
        batch_elem = token_masks.numel()
        group = None
        group_size = torch.distributed.get_world_size(group=group)
        all_batch_info = [torch.LongTensor(3).to(f'cuda:{self.trainer.local_rank}') for i in range(group_size)]
        batch_info = torch.LongTensor([batch_tokens, realbsz, batch_elem]).to(f'cuda:{self.trainer.local_rank}')
        torch.distributed.all_gather(all_batch_info, batch_info, group=group)
        batch_tokens = sum([x[0] for x in all_batch_info]).item()
        total_realbsz = sum([x[1] for x in all_batch_info]).item()
        total_batch_elem = sum([x[2] for x in all_batch_info]).item()
        avg_seq_len = batch_tokens / total_realbsz
        self.consume_tokens += batch_tokens
        train_batch_size = self.trainer._datamodule.hparams.train_batch_size
        train_steps = self.trainer.total_steps
        bsz = train_batch_size * DIST_ENV.world_size
        seq_len = self.trainer._datamodule.hparams.max_seq_len
        self.log('avg_sample_seq_len', avg_seq_len, console=True)
        self.log('avg_effetive_len', batch_tokens/bsz, console=True)
        self.log('total_real_bsz', total_realbsz, console=True)
        self.log('total_valid_token_ratio', batch_tokens/total_batch_elem, console=True)
        self.average_token_rate = batch_tokens / bsz / seq_len
        if self.trainer.global_step <= 0:
            return
        rank_print = 0
        if self.trainer.global_step % self.trainer._log_every_n_steps == 0 and self.trainer.global_rank == rank_print:
            cur_time = time.time()
            delta_time = cur_time - self.last_time
            delta_tokens = self.consume_tokens-self.last_consume_tokens
            tokens_per_second = delta_tokens / delta_time
            self.mfu = self.estimate_mfu(delta_tokens, delta_time)
            self.eta_meter.step()
            eta_timestamp = (self.trainer.total_steps - self.trainer.global_step) / \
                self.trainer._log_every_n_steps * self.eta_meter.sec_per_step() + cur_time
            self.last_time = cur_time
            self.last_consume_tokens = self.consume_tokens

            self.trainer.logger.log_metrics({'consume_tokens(M)': self.consume_tokens / 1e6}, step=self.trainer.global_step,
                                            step_size=None, reduce_fx=lambda x: x[0] if isinstance(x, list) else x, pause_flush=True)
            self.trainer.logger.log_metrics({'consume_tokens(B)': self.consume_tokens / 1e9}, step=self.trainer.global_step,
                                            step_size=None, reduce_fx=lambda x: x[0] if isinstance(x, list) else x, pause_flush=True)
            self.trainer.logger.log_metrics({'average_token_rate': self.average_token_rate * 100}, step=self.trainer.global_step,
                                            step_size=None, reduce_fx=lambda x: x[0] if isinstance(x, list) else x, pause_flush=True)
            self.trainer.logger.log_metrics({'tokens_per_second(M)': tokens_per_second / 1e6}, step=self.trainer.global_step,
                                            step_size=None, reduce_fx=lambda x: x[0] if isinstance(x, list) else x, pause_flush=True)
            self.trainer.logger.log_metrics({'mfu': self.mfu}, step=self.trainer.global_step,
                                            step_size=None, reduce_fx=lambda x: x[0] if isinstance(x, list) else x, pause_flush=True)
            # update directly to sql, easier to parse in merlin
            self.logger.wandb.summary['training/eta'] = int(eta_timestamp)
            self.logger.wandb.summary['training/mfu'] = self.mfu

    def estimate_mfu(self, tokens, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        cfg = self.hparams.network
        L, H, T, V = cfg["n_layer"], cfg["n_embed"], cfg["max_position_embeddings"], cfg["vocab_size"]
        N = 12*H*H*L + V*H
        flops_per_token = 6*N + 12*L*H*T
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_token * tokens * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        world_size = self.trainer.world_size  # GPU count
        mfu = flops_achieved / flops_promised / world_size
        return mfu

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        pad_output=False,
    ):
        attention_mask = get_subsequent_mask(attention_mask)
        model_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pad_output=pad_output)
        return model_out

    def training_step(self, batch, batch_idx):
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch['labels'] = batch['input_ids']

        forward_batch = {}
        forward_batch["input_ids"] = batch['input_ids']
        forward_batch["attention_mask"] = batch['attention_mask']
        forward_batch["labels"] = batch['labels']
        forward_batch["pad_output"] = True

        self.count_tokens(batch['attention_mask'])
        model_out = self.forward(**forward_batch)

        logits = model_out["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        pred = torch.softmax(shift_logits, dim=-1).argmax(-1)
        bsz, seqlen, vsz = shift_logits.shape
        nlogprobs = F.cross_entropy(shift_logits.view(bsz*seqlen, vsz), shift_labels.view(-1), reduction='none')
        nlogprobs = nlogprobs.view(bsz, seqlen)

        mask = batch['attention_mask'][..., 1:]
        prefix_mask = batch['prefix_masks'][..., 1:]
        affix_mask = batch['affix_masks'][..., 1:]

        acc = ((pred==shift_labels).float() * mask).sum() / mask.sum()
        nll = (nlogprobs * mask).sum() / mask.sum()
        prefix_acc = ((pred==shift_labels).float() * prefix_mask).sum() / prefix_mask.sum()
        affix_acc = ((pred==shift_labels).float() * affix_mask).sum() / affix_mask.sum()
        prefix_nll = (nlogprobs * prefix_mask).sum() / prefix_mask.sum()
        affix_nll = (nlogprobs * affix_mask).sum() / affix_mask.sum()
        
        
        self.log('acc', acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('nll', nll.item(), console=True, tb=True, reduce_fx="mean")
        self.log('prefix_acc', prefix_acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('affix_acc', affix_acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('prefix_nll', prefix_nll.item(), console=True, tb=True, reduce_fx="mean")
        self.log('affix_nll', affix_nll.item(), console=True, tb=True, reduce_fx="mean")

        loss = model_out['loss']
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch['labels'] = batch['input_ids']

        forward_batch = {}
        forward_batch["input_ids"] = batch['input_ids']
        forward_batch["attention_mask"] = batch['attention_mask']
        forward_batch["labels"] = batch['labels']
        model_out = self.forward(**forward_batch)

        logits = model_out["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        pred = torch.softmax(shift_logits, dim=-1).argmax(-1)
        nlogprobs = F.cross_entropy(shift_logits, shift_labels, reduction='none')


        mask = batch['attention_mask'][..., 1:]
        prefix_mask = batch['prefix_masks'][..., 1:]
        affix_mask = batch['affix_masks'][..., 1:]

        acc = ((pred==shift_labels).float() * mask).sum() / mask.sum()
        nll = (nlogprobs * mask).sum() / mask.sum()
        prefix_acc = ((pred==shift_labels).float() * prefix_mask).sum() / prefix_mask.sum()
        affix_acc = ((pred==shift_labels).float() * affix_mask).sum() / affix_mask.sum()
        prefix_nll = (nlogprobs * prefix_mask).sum() / prefix_mask.sum()
        affix_nll = (nlogprobs * affix_mask).sum() / affix_mask.sum()
        
        
        self.log('acc', acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('nll', nll.item(), console=True, tb=True, reduce_fx="mean")
        self.log('prefix_acc', prefix_acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('affix_acc', affix_acc.item(), console=True, tb=True, reduce_fx="mean")
        self.log('prefix_nll', prefix_nll.item(), console=True, tb=True, reduce_fx="mean")
        self.log('affix_nll', affix_nll.item(), console=True, tb=True, reduce_fx="mean")

        loss = model_out['loss']
        return {'val_loss': loss}

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, input_mask: torch.Tensor, *args, **kwargs):
        """For generation task"""
        model_out = self.gpt(input_ids=input_ids, attention_mask=input_mask)
        return model_out

    def configure_optimizers(self, optimizer_kwargs):
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm', 'ln']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': optimizer_kwargs["optimizer"]["params"]["weight_decay"]}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict]

        optimizers = super()._configure_optimizers(optimizer_grouped_parameters, optimizer_kwargs)
        lr_schedulers = super()._configure_schedulers(optimizers, optimizer_kwargs)
        return optimizers, lr_schedulers

    def lr_scheduler_step(
        self,
        schedulers,
        **kwargs,
    ) -> None:
        r"""
        默认是per epoch的lr schedule, 改成per step的
        """
        # if self.trainer.global_step == 0:
        #     # skip first step
        #     return
        for scheduler in schedulers:
            scheduler.step()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj.weight" in name or 'attn_ow' in name: # deepspeed transformer kernel 是 attn_ow
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer)))


def play_console(tokenizer, model, trial_num=1,
                 steps=2048, temperature=0.6, do_sample=True,
                 top_k=5, top_p=None,
                 dynamic_top_p=None, 
                 omega = 0.3, 
                 decay_lambda=0.9, until_n_eos=1):
    console = Console()
    while True:
        try:
            text = console.input(">> ")
            if text == "/exit":
                console.print("See you!")
                break
            token_ids = tokenizer(text)["input_ids"]
            # token_ids.append(tokenizer.sep_token_id)
            print(
                f'Prompt: {text}, Tokens: {tokenizer.decode(token_ids)}, Token Ids: {token_ids}')
            input_ids = torch.tensor(token_ids).unsqueeze(0).long()
            for i in range(trial_num):
                y = sample_generate(model,
                                    input_ids=input_ids.to(model.device),
                                    steps=steps, 
                                    temperature=temperature, 
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p, 
                                    omega=omega, 
                                    decay_lambda=decay_lambda, 
                                    eos=tokenizer.eos_token_id,
                                    until_n_eos=until_n_eos,
                                    full_stop_input_ids=[])

                response = ''.join(tokenizer.decode(y))
                # response.replace('##', '')

                # md = Markdown(response)
                # console.print(md)
                print(response)
                # import pdb; pdb.set_trace()
        except Exception as e:
            print(e)
            print(traceback.format_exc())

def play_file(tokenizer, model, trial_num=1,
                 steps=2048, temperature=0.6, do_sample=True,
                 top_k=5, top_p=None,
                 dynamic_top_p=None, 
                 omega = 0.3, 
                 decay_lambda=0.9, until_n_eos=1):
    console = Console()
    while True:
        try:
            text = console.input(">> ")
            if text == "/exit":
                console.print("See you!")
                break
            token_ids = tokenizer(text)["input_ids"]
            # token_ids.append(tokenizer.sep_token_id)
            print(
                f'Prompt: {text}, Tokens: {tokenizer.decode(token_ids)}, Token Ids: {token_ids}')
            input_ids = torch.tensor(token_ids).unsqueeze(0).long()
            for i in range(trial_num):
                y = sample_generate(model,
                                    input_ids=input_ids.to(model.device),
                                    steps=steps, temperature=temperature, do_sample=do_sample,
                                    top_k=top_k,
                                    top_p=top_p, 
                                    omega=omega, 
                                    decay_lambda=decay_lambda, 
                                    eos=tokenizer.eos_token_id,
                                    until_n_eos=until_n_eos,
                                    full_stop_input_ids=[])
                response = ''.join(tokenizer.decode(y))
                # response.replace('##', '')

                # md = Markdown(response)
                # console.print(md)
                print(response)
                # import pdb; pdb.set_trace()
        except Exception as e:
            print(e)
            print(traceback.format_exc())

if __name__ == '__main__':
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint
    ckpter = ModelCheckpoint(monitor='step',
                             save_last=False,
                             save_top_k=-1,
                             every_n_train_steps=2000,
                             every_n_epochs=1,
                             save_on_train_epoch_end=True,
                             save_weights_only=False,
                             enable_trace=False)
    cli = CruiseCLI(
        GPT2Model,
        datamodule_class=AsrDataModule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'log_every_n_steps': 20,
            'find_unused_parameters': False,
            'max_epochs': 10,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': -1,
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'callbacks': [ckpter],
            'optimizer_kwargs': mariana_optimizer_kwargs_defaults,
        })
    cli.add_argument('--val-only', default=False, action='store_true', dest='val_only')
    cli.add_argument('--play', default=False, action='store_true', dest='play')
    cli.add_argument('--play-file', default='', type=str, help='generate by samples loaded from file')
    cli.add_argument('--play-file-limit', default=-1, type=int, help="If >0, limit how many lines to generate.")
    cli.add_argument('--generate-trial-num', default=1, type=int, help="generation trial num, default is 5")
    cli.add_argument('--generate-steps', default=2048, type=int, help='decode sequence length/steps')
    cli.add_argument('--generate-temp', default=0.7, type=float, help='Smaller tempreature logits become more steep')
    cli.add_argument('--generate-do-sample', default=False, type=bool, help='multinomial sample if True')
    cli.add_argument('--generate-topk', default=None, type=int, help='sample top-k')
    cli.add_argument('--generate-topp', default=0.9, type=float, help='sample at least top-p probability')
    cli.add_argument('--generate-n-eos', default=1, type=int, help='Stop until n-eos tokens')
    cli.add_argument('--merge_zero3_states', default=False, type=bool, help='Whether to merge zero3 state dict')
    cli.add_argument('--merge_ckpt_dtype', default="fp16", type=str, help='The data type of merged zero3 state dict')
    cli.add_argument('--merge_cache_dir', default='./', type=str, help='Directory to merge zero3 state dict')


    cfg, trainer, model, datamodule = cli.parse_args()
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    elif cfg.play_file or cfg.play:
        assert DIST_ENV.world_size == 1, "Play mode only support single card"
        datamodule.rank_zero_prepare()
        datamodule.local_rank_zero_prepare()
        datamodule.setup()
        model.rank_zero_prepare()
        model.local_rank_zero_prepare()
        model.setup()
        if cfg.play_file:
            print("\nFile play mode.")
            play_file(cfg.play_file, datamodule.tokenizer, model.cuda(), cfg.generate_trial_num,
                      steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                      top_k=cfg.generate_topk, top_p=cfg.generate_topp, until_n_eos=cfg.generate_n_eos,
                      limit_samples=cfg.play_file_limit)
        else:
            print("\nConsole play mode.")
            play_console(datamodule.tokenizer, model.cuda(), cfg.generate_trial_num,
                         steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                         top_k=cfg.generate_topk, top_p=cfg.generate_topp, until_n_eos=cfg.generate_n_eos)
            
    else:
        trainer.fit(model, datamodule)
        if cfg.merge_zero3_states and trainer.global_rank == 0:
            ckpt_path = os.path.join(trainer.default_hdfs_dir, "checkpoints", "global_step_" + str(trainer.global_step))
            merge_zero_checkpoints(ckpt_path, cfg.merge_cache_dir, cfg.merge_ckpt_dtype)

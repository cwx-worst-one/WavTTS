import os
import math
import contextlib
from typing import Any, Dict, Tuple, Union, Optional

import hyperpyyaml
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.distributed
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel, PretrainedConfig
from copy import deepcopy
from accelerate import init_empty_weights, init_on_device
from transformers.modeling_utils import no_init_weights

import samantha
from apps.bigmusic.pl_tasks.p6d import P6DenseModel, P6DenseForCausalLM, P6DenseConfig
from apps.bigmusic.mariana_tasks.semantic_emb_module import SemanticEmbModule
from apps.bigmusic.pl_tasks.optim import get_cosine_schedule_with_warmup_lrdecay
from samantha.callbacks import Timer, HDFSModelCheckpoint

import logging
from hdfs_io import hcopy

logger = logging.getLogger(__file__)

os.environ["LITE_USE_PL_MODULE"] = "1"
from samantha.dataio.bigmusic.lite import MusicLiteDataModule


def _init_emb_module(config: AttributeDict):
    config = deepcopy(config)
    config_path = config.pop("config_path", "")
    if config_path:
        with open(config_path, "r") as f:
            hps = hyperpyyaml.load_hyperpyyaml(f, overrides=config)
    else:
        raise ValueError("emb config_path is not provided")
    return SemanticEmbModule(**hps)


def _get_local_path(hdfs_path):
    if hdfs_path is not None and hdfs_path.startswith("hdfs://"):
        return os.path.join("./", os.path.basename(hdfs_path))
    return hdfs_path

def _hf_model_init(
    hf_model_pth: str,
    rank_zero_init_context=contextlib.nullcontext,
    rank_nonzero_init_context=contextlib.nullcontext,
    **kwargs,
) -> Tuple[PreTrainedModel, PretrainedConfig]:
    """
    Initialize model parameters on the meta device but not for global_rank0.
    """
    config = AutoConfig.from_pretrained(hf_model_pth, **kwargs)
    if torch.distributed.get_rank() == 0:
        with rank_zero_init_context():
            model = AutoModelForCausalLM.from_pretrained(hf_model_pth, **kwargs)
        model.to(device="cuda")
    else:
        with rank_nonzero_init_context():
            model = AutoModelForCausalLM.from_pretrained(hf_model_pth, **kwargs)
            # FIXME: xieshuangyi: lm_head has different behavior between from_pretrained and from_config
            # model = AutoModelForCausalLM.from_config(
            #     AutoConfig.from_pretrained(hf_model_pth, **kwargs),
            # )
        model.to_empty(device="cuda")
    return model, config


class MusicSeedModule(LightningModule):
    emb: SemanticEmbModule
    llm_model: P6DenseForCausalLM
    llm_config: P6DenseConfig
    tokenizer: transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast

    def __init__(
        self,
        emb_config: Optional[Dict] = None,
        emb_path: Optional[str] = None,
        hf_model_path: Optional[str] = None,
        network_config: Optional[Dict] = None,
        optimizer_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # download emb weight, hf llm model weight could load from hdfs directly.
        self.local_emb_path = _get_local_path(self.hparams.emb_path)
        self.local_hf_model_path = _get_local_path(self.hparams.hf_model_path)

    def setup(self, stage):
        if self.local_rank == 0:
            # Download all files like model weights and tokenizer
            # Sometimes downloading files takes a long time, so it's better to mark progress with logging.
            def is_downloaded(p):
                return os.path.isdir(p) or os.path.isfile(p)

            if self.local_emb_path and (not is_downloaded(self.local_emb_path)):
                logger.info(f"Downloading {self.hparams.emb_path} to {self.local_emb_path}")
                hcopy(self.hparams.emb_path, self.local_emb_path, chunk_thread_num=64)
            logger.info(f"local_emb_path is: {self.local_emb_path}")

            if self.local_hf_model_path and (not is_downloaded(self.local_hf_model_path)):
                logger.info(f"Downloading {self.hparams.hf_model_path} to {self.local_hf_model_path}")
                hcopy(self.hparams.hf_model_path, self.local_hf_model_path, chunk_thread_num=64)
            logger.info(f"local_hf_model_path is: {self.local_hf_model_path}")

        self.trainer.strategy.barrier()

        self.emb = _init_emb_module(self.hparams.emb_config)
        self.emb.setup(stage)
        if self.local_emb_path:
            self.emb.load_state_dict(self.local_emb_path)

        network_config = self.hparams.network_config or {}
        self.llm_model, self.llm_config = _hf_model_init(self.local_hf_model_path, **network_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.get("tokenizer_path", self.local_hf_model_path))


    def forward(self, batch, inference=False, batch_idx=None):
        training_inputs = self.emb(batch)

        prefix_length = training_inputs["prefix_length"]
        target_length = training_inputs["target_length"]
        token_length = prefix_length + target_length
        input_token_embeds = training_inputs["token_embeds"]
        target_loss_mask = training_inputs["target_loss_mask"]
        num_valid_tokens = target_loss_mask[..., 1: ].sum()

        output = dict()
        output["token_length"] = token_length  # for flops

        batch_size, seq_len, _ = input_token_embeds.shape
        attention_mask = torch.arange(0, seq_len, device=prefix_length.device)[None, :] < token_length[:, None]
        target_ids = training_inputs["token_ids"]
        target_ids[target_loss_mask.logical_not()] = -100

        model_outputs = self.llm_model(
            inputs_embeds=input_token_embeds,
            attention_mask=attention_mask,
            labels=target_ids,
        )

        loss = (model_outputs.loss * target_loss_mask[..., 1:].reshape(-1) ).sum() / num_valid_tokens
        output["loss"] = loss
        output["loss_tokens"] = num_valid_tokens
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch, inference=False, batch_idx=batch_idx)
        loss = outputs["loss"]

        # TODO: metric:
        # - flops, mfu, tokens_per_second(TPS), tokens_per_day
        # - seqlen, valid tokens, consumed tokens
        # maybe we can calculate these in a callback
        llm_flops = self.calc_llm_flops_rmpad(outputs["token_length"])
        if self.training:
            outputs['lr * 1e3'] = self.trainer.optimizers[0].param_groups[0]['lr'] * 1000
            llm_flops += 2 * llm_flops
        
        self.log_dict({"training/loss": loss}, on_step=True, on_epoch=False, prog_bar=True)
        return outputs

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self(*args, **kwargs)
        super().validation_step
        return outputs

    def configure_optimizers(self):
        optimizer_config = self.hparams.optimizer_kwargs.get("optimizer", {})
        optimizer = torch.optim.AdamW(self.parameters(), **optimizer_config)

        max_steps = self.trainer.max_steps
        lr_scheduler_config = self.hparams.optimizer_kwargs.get("scheduler", {})
        warmup_step_rate = lr_scheduler_config.pop("warmup_step_rate", 0)
        num_warmup_steps = int(max_steps * warmup_step_rate)
        lr_scheduler = get_cosine_schedule_with_warmup_lrdecay(optimizer, num_warmup_steps, max_steps, **lr_scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        }

    def calc_llm_flops_rmpad(self, lengths):  # lengths: shape of [B], one item for each sequence
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        sum_of_T = lengths.sum().item()

        flops_transformer = 0
        # attn inproj flops
        query_head_scale_factor = self.llm_config.query_head_scale_factor
        flops_transformer += (
            2
            * sum_of_T
            * self.llm_config.hidden_size
            * (
                self.llm_config.hidden_size * query_head_scale_factor
                + 2 * self.llm_config.hidden_size // (self.llm_config.num_attention_heads//self.llm_config.num_key_value_heads)
            )
        )
        # attn flops
        attn_scale = 0.5 if getattr(self.llm_config, "attn_causal", True) else 1
        flops_transformer += (
            2 * 2 * (lengths**2).sum().item() * self.llm_config.hidden_size * query_head_scale_factor * attn_scale
        )
        # attn outproj flops
        flops_transformer += 2 * sum_of_T * self.llm_config.hidden_size * self.llm_config.hidden_size * query_head_scale_factor
        # mlp flops
        mlp_flops = 2 * sum_of_T * self.llm_config.hidden_size * self.llm_config.intermediate_size
        mlp_flops = mlp_flops * 3 if self.llm_config.hidden_act == 'silu' else mlp_flops * 2
        flops_transformer += mlp_flops
        # mul number of layers
        flops_transformer *= self.llm_config.num_hidden_layers
        return flops_transformer

class MusicSeedModelsCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(HDFSModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "model_checkpoint.monitor": "step",
                "model_checkpoint.save_top_k": -1,
                "model_checkpoint.every_n_train_steps": 2000,
            }
        )
        parser.add_class_arguments(WandbLogger, "wandb")
        return super().add_arguments_to_parser(parser)

def cli_main():
    # NOTE: @xieshuangyi
    # pytorch_lightning CLI could read command line args from sys.argv,
    # but it looks like not flexible enough. It not support such as `run_ops.xxx` anymore.
    # Maybe we could parse args from cli by ourself and convert to pytorch_lightning args.
    cli = MusicSeedModelsCLI(
        model_class=MusicSeedModule,
        datamodule_class=MusicLiteDataModule,
        parser_kwargs={"parser_mode": "yaml"},
        trainer_class=pl.Trainer,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "precision": "bf16-mixed",
            "devices": int(os.getenv("ARNOLD_WORKER_GPU", 0)) or "auto",  # for debug cases
            "num_nodes": int(os.getenv("ARNOLD_WORKER_NUM", 1)),
        },
    )

if __name__ == "__main__":
    cli_main()

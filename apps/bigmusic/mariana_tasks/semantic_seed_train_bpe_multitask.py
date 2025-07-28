"""A Semantic Module that supports the simple BPE-based implementation"""

import os
import torch
from cruise import CruiseConfig
from cruise.trainer.callback import ModelCheckpoint
from panther.custom_ops.torch.flash_attn import unpad_input
import torch.distributed

import samantha # noqa: F401, resolve mariana python path
from mariana.models.audio.speech_checkpoint import SpeechModelCheckpoint
from mariana.utils.audio.audio_logger import AudioLogger
from mariana.utils.exp_helper import ExpHelper
from apps.bigmusic.mariana_tasks.utils.speech_data_collect_callback import SpeechDataCollectCallback
from apps.mariana.mariana.data.audio.multitask_datamodule import AudioMultiTaskDataModule

from apps.bigmusic.mariana_tasks.semantic_seed_train import (
    SemanticLlmCLI,
    _m8_network_config,
    _inference_config,
)
from apps.bigmusic.mariana_tasks.semantic_seed_train_bpe import (
    SemanticLlmModelBpe,
    SemanticLlmBpeTrainer,
)

from tqdm.auto import tqdm
from cruise.utilities.distributed import DIST_ENV


logger = AudioLogger()

class SemanticLlmModelMultitaskBpe(SemanticLlmModelBpe):

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
        """
        unpad input ids and attention mask, save GPU memory
        """
        input_ids = batch["input_ids"][:, :-1]
        shifted_input_mask = batch["attention_mask"][:, 1:]
        input_ids_rmpad, _, cu_seqlens_q, max_seqlen_q = unpad_input(
            input_ids.contiguous().unsqueeze(-1), shifted_input_mask
        )
        input_embeds_rmpad = self.gpt2.transformer.wte(input_ids_rmpad.squeeze(1))

        target_loss_mask = batch["token_type_ids"][:, 1:]
        target_ids = (
            batch["input_ids"][:, 1:] * target_loss_mask
        )  # set non-target ids to 0 to avoid OOB
        labels_shift_rmpad, _, _, _ = unpad_input(
            target_ids.contiguous().unsqueeze(-1), attention_mask=shifted_input_mask
        )
        target_loss_mask, _, _, _ = unpad_input(
            target_loss_mask.contiguous().unsqueeze(-1), shifted_input_mask
        )
        hidden_states = self.gpt2(
            inputs_embeds=input_embeds_rmpad.unsqueeze(1).contiguous(),
            return_final_hidden_states=True,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
        )
        outputs = self.calc_loss_acc(
            hidden_states,
            labels_shift_rmpad.squeeze(1),
            target_loss_mask.squeeze(1),
            require_eos_acc=_kwargs.get("require_eos_acc", False),
            eos_index_window=_kwargs.get("eos_index_window", 1),
        )
        outputs["tokens"] = batch["attention_mask"].sum()
        outputs["loss_tokens"] = batch["attention_mask"].sum()
        outputs["consume_tokens(B)"] = outputs["tokens"] * 1e-9
        outputs["loss_tokens(B)"] = outputs["loss_tokens"] * 1e-9
        outputs["gpt2_lengths"] = torch.sum(
            batch["attention_mask"][:, 1:], dim=1
        )  # TODO: check

        return outputs

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, rl_training=False):
        output_tokens = super().predict(batch, hp, beam, rl_training)
        text_codebook_size = batch['text_codebook_size']
        output_tokens += text_codebook_size

        # Seperate audio tokens and text tokens
        output_tokens_list = output_tokens.flatten().tolist()

        audio_vocab = batch['audio_vocab']
        audio_output_tokens = [i for i in output_tokens_list if i in audio_vocab]
        text_output_tokens = [i for i in output_tokens_list if i not in audio_vocab]

        bpe_tokenizer = batch['tokenizer']
        output_string = bpe_tokenizer.decode(text_output_tokens)
        batch['output_string'] = output_string

        if len(audio_output_tokens) == 0:
            # fake audio output
            output_audio = torch.tensor([0] * 100, dtype=torch.long, device=output_tokens.device).view(1, -1)
        else:
            output_audio = torch.tensor(audio_output_tokens, dtype=torch.long, device=output_tokens.device).view(1, -1)
            output_audio -= text_codebook_size
        return output_audio

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
        SemanticLlmModelMultitaskBpe,
        datamodule_class=AudioMultiTaskDataModule,  # <- modified
        trainer_class=SemanticLlmBpeTrainer, 
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
        trainer.predict(model, datamodule=datamodule)
    else:
        # Training
        trainer.fit(model, datamodule=datamodule)

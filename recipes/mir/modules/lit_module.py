import math
import random
from typing import Optional, Union

import pytorch_lightning as pl
import time
import torch
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from recipes.umm.models.utils import clip_grad_value_, mel_spectrogram_torch
from recipes.umm.modules.criterion_vocoder import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from samantha.dataio.webdataset import ShardWriter
from samantha.utils.hparams import DotDict
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams


def sample(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove = indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    probs = logits.softmax(-1)
    dist = torch.distributions.categorical.Categorical(probs=probs)
    samples = dist.sample()
    return samples


class UnifiedDecoder(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = {}

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        for name, item in self.hparams.required_modules.items():
            hpath = item["ckpt_path"]
            init_fn = item["init_fn"]
            cache_dir = item["cache_dir"]
            print(f"Loading {name} from {hpath}")
            self.requires.update(
                init_fn(hpath, local_rank=self.local_rank, cache_dir=cache_dir)
            )

    def _shared_step(self, batch):
        token_seq, token_length, target_length = self.prepare_feature(batch)
        input_seq = token_seq[:, :-1]
        target_seq = token_seq[:, 1:]
        if isinstance(self.model, gpt.GPTLMHeadModel):
            logits = self.model(input_ids=input_seq)[0]
        else:
            logits = self.model(input_ids=input_seq)["logits"]
        loss_mask = torch.zeros_like(target_seq)
        for i, (tok_len, tar_len) in enumerate(zip(token_length, target_length)):
            loss_mask[i, tok_len - tar_len - 1 : tok_len - 1] = 1
        loss_dict = self.criterion(logits, target_seq, mask=loss_mask)
        accu = ((logits.argmax(dim=-1) == target_seq).float() * loss_mask).sum() / sum(
            target_length
        )
        loss_dict.update(accu=accu)
        loss_dict["training/loss"] = loss_dict["loss"]
        loss_dict["aux/num_target_tokens"] = sum(target_length)
        loss_dict["aux/num_input_tokens"] = sum(token_length)
        loss_dict["aux/num_tokens"] = token_seq.size(0) * token_seq.size(1)
        return loss_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def assemble_tokens(self, umm_tokens, text_tokens, text_length, token_map):
        text_sos_id = torch.LongTensor([self.extra_params.text_sos_id]).to(umm_tokens)
        audio_sos_id = torch.LongTensor([self.extra_params.audio_sos_id]).to(umm_tokens)
        break_id = torch.LongTensor([self.extra_params.break_id]).to(umm_tokens)
        eos_id = torch.LongTensor([self.extra_params.eos_id]).to(umm_tokens)
        token_seq = []
        token_length = []
        target_length = []
        for task_type, task_id, it_i, ia_i, tt_i, ta_i in token_map:
            tokens = []
            t_len = 0
            if it_i is not None:
                tokens.append(text_sos_id)
                tokens.append(text_tokens[it_i][: text_length[it_i]])
            if ia_i is not None:
                tokens.append(audio_sos_id)
                tokens.append(umm_tokens[ia_i])
            tokens.append(break_id)
            if tt_i is not None:
                tokens.append(text_sos_id)
                t_len += 1
                tokens.append(text_tokens[tt_i][: text_length[tt_i]])
                t_len += text_length[tt_i]
            if ta_i is not None:
                tokens.append(audio_sos_id)
                t_len += 1
                tokens.append(umm_tokens[ta_i])
                t_len += umm_tokens[ta_i].size(-1)
            tokens.append(eos_id)
            t_len += 1
            tokens = torch.cat(tokens, dim=-1)
            token_length.append(tokens.size(-1))
            token_seq.append(tokens)
            target_length.append(t_len)
        return token_seq, token_length, target_length

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        mono_map = batch["mono_map"]
        homo_map = batch["homo_map"]
        mono_audio = batch["mono_audio"]
        homo_audio = batch["homo_audio"]
        mono_text = batch["mono_text"]
        homo_text = batch["homo_text"]
        mono_text_length = batch["mono_text_length"]
        homo_text_length = batch["homo_text_length"]
        token_seq = []
        token_length = []
        target_length = []

        if len(mono_map) > 0:
            mono_umm_tokens = (
                self.get_umm_tokens(mono_audio) + self.extra_params.vocab_size_text
            )
            mono_assembled = self.assemble_tokens(
                mono_umm_tokens, mono_text, mono_text_length, mono_map
            )
            token_seq = token_seq + mono_assembled[0]
            token_length = token_length + mono_assembled[1]
            target_length = target_length + mono_assembled[2]
        if len(homo_map) > 0:
            homo_umm_tokens = (
                self.get_umm_tokens(homo_audio) + self.extra_params.vocab_size_text
            )
            homo_assembled = self.assemble_tokens(
                homo_umm_tokens, homo_text, homo_text_length, homo_map
            )
            token_seq = token_seq + homo_assembled[0]
            token_length = token_length + homo_assembled[1]
            target_length = target_length + homo_assembled[2]
        token_seq = pad_sequence(token_seq, batch_first=True, padding_value=0)
        if not self.extra_params.dynamic_batch:
            token_seq = F.pad(
                token_seq,
                [0, self.extra_params.max_length - token_seq.size(1)],
                mode="constant",
                value=0,
            )
        return token_seq, token_length, target_length

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_umm_tokens(self, x):
        tokens = self.requires["Stage3"].wav2token(x.float())
        return tokens

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     # Dummy function for triggering callbacks
    #     return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(loss_dict)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            val_loss_dict = {}
            for loss in outputs:
                for k, v in loss.items():
                    k = f"val_{dataloader_idx}/{k}"
                    if k not in val_loss_dict:
                        val_loss_dict[k] = v
                    else:
                        val_loss_dict[k] = val_loss_dict[k] + v
            for k, v in val_loss_dict.items():
                val_loss_dict[k] = v / len(outputs)
            self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=True)
    def predict(self, input_ids, top_k, top_p, max_length=None):
        b, t = input_ids.size()
        if max_length is None:
            max_length = t * 2
        output_samples = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            inference_params = InferenceParams(
                max_sequence_len=max_length, max_batch_size=b
            )
        else:
            past_key_values = None
        pbar = tqdm(range(max_length - t))
        stop = [False for _ in range(b)]
        for _ in pbar:
            pbar.set_description(
                f"[Sampling] {b=} - {t=} - {max_length=}"
            )
            if isinstance(self.model, gpt.GPTLMHeadModel):
                logits = self.model(
                    input_ids,
                    inference_params=inference_params,
                    position_ids=None,
                    last_token_only=False,
                ).logits
                inference_params.sequence_len_offset += input_ids.size(1)
            else:
                model_output = self.model(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
            predict_logits = logits[:, -1:, :]
            samples = sample(
                predict_logits,
                top_k=top_k,
                top_p=top_p,
            )
            input_ids = samples
            for i in range(b):
                if input_ids[i].item() == self.extra_params.eos_id:
                    stop[i] = True
            if output_samples is None:
                output_samples = samples
            else:
                output_samples = torch.cat([output_samples, samples], dim=1)
            if all(stop):
                break
        return output_samples

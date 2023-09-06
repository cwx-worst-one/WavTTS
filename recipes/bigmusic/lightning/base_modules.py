
from typing import Optional, Tuple, Union, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.profilers import PassThroughProfiler
from s3a.providers.ctiga.models import gpt
from tqdm.auto import tqdm

from samantha.utils.hparams import DotDict
from recipes.musiclm.inference.utils import sample
from recipes.bigmusic.lightning.embedding_modules import TokenEmbedder, BaseEmbedder

class BaseModule(pl.LightningModule):
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
        self.save_hyperparameters(ignore=["input_embedders", "target_embedder"])
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    # Disable gradient logging for faster performance
    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

        pretrained_path = self.extra_params.get('pretrained_path')
        if stage == "fit" and pretrained_path is not None:
            self.load_from_pretrained(pretrained_path)

    def load_from_pretrained(self, pretrained_path=None):
        print('Loading pre-trained model from checkpoint', pretrained_path)
        state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
            else:
                print(f"Dropping parameter {k}")

        self.load_state_dict(state_dict, strict=False)

    def load_required_modules(self, ignore=()):
        for name, item in self.hparams.required_modules.items():
            if name in ignore: continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_training_inputs(batch)
        logits = self.model(**input_ids)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -target_ids.size(1):, :]
        loss = self.criterion(x, target_ids)        
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100        
        return loss, accu

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch)
        self.log_dict({"tr_loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            for l, a in outputs:
                loss += l
                accu += a
            loss /= len(outputs)
            accu /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        if isinstance(self.model, gpt.GPTLMHeadModel):
            optimizer = self.hparams.optimizer_cls(self.parameters())
        else:
            params = []
            for name, p in self.named_parameters():
                if "bias" in name or "layernorm" in name or "ln_" in name:
                    print(f"Skip weight decay: {name}")
                    params.append({"params": [p], "weight_decay": 0.0})
                else:
                    params.append({"params": [p]})
            optimizer = self.hparams.optimizer_cls(params)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

class BaseContinuousEmbedModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        input_embedders: Mapping[str, BaseEmbedder],
        target_embedder: TokenEmbedder,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(model_cls, criterion_cls, optimizer_cls, scheduler_cls, required_modules, checkpointing, extra_params)
        delete_embedding_module(self.model)
        self.input_embedders = input_embedders
        self.target_embedder = target_embedder
        self.input_embedders.apply(self.model._init_weights)
        self.target_embedder.apply(self.model._init_weights)
        self.use_cross_attn = self.extra_params.get("use_cross_attn", False)

    def infer_batch_size(self, batch):
        batch_size = [len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)][0]
        return batch_size

    def prepare_inputs_embeddings(self, batch):
        raise NotImplementedError()

    def prepare_training_inputs(self, batch):        
        target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False)
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(batch)
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)
        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        if self.target_embedder.eos_id is not None:
            eos_ids = self.target_embedder.get_eos_token(batch_size)
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(target_ids.device)
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
        if self.use_cross_attn:
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds
            }, torch.cat([target_ids, eos_ids], dim=1)
        input_embeds = torch.cat([inputs_embeds, sos_embeds, target_embeds], dim=1)
        target_ids = torch.cat([target_ids, eos_ids], dim=1)        
        return {"inputs_embeds":  input_embeds}, target_ids

    # Prediction code
    def sample_logits(self, i, logits, temp, mode):
        return sample(logits, temp=temp, mode=mode)

    @torch.no_grad()
    def predict(self, inputs_embeds, num_tokens, temperature=1, sample_mode="gumbel", tqdm_name=None):
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        batch_size = inputs_embeds.size(0)
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)

        if self.use_cross_attn:
            model_input = { 
                "inputs_embeds": sos_embeds,
                "encoder_hidden_states": inputs_embeds
            }
        else:
            model_input = { "inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1) }
        
        output_tokens = None
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")
            model_output = self.model(
                **model_input, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            past_key_values = model_output["past_key_values"]
            logits = model_output["logits"]
            logits = logits[:, -1:, :] # only predicting on last logit.

            predict_token = self.sample_logits(i, logits, temperature, sample_mode)
            model_input['inputs_embeds'] = self.target_embedder.embedder(predict_token)
            output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token
        return output_tokens

    @torch.no_grad()
    def predict_slice(self, inputs_embeds, input_framerate, output_framerate, target_duration, slice_duration, stride_duration, temperature=1, sample_mode="gumbel", tqdm_name=None):
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        batch_size = inputs_embeds.size(0)
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)

        # Slice target durations into chunks
        slice_durations = get_slice_ranges(
            target_duration=target_duration,
            slice_duration=slice_duration,
            stride_duration=stride_duration
        )

        prev_end = 0
        output_tokens = None
        output_embeds = None
        for slice_beg, slice_end in slice_durations:
            cache_len = prev_end - slice_beg
            prev_end = slice_end

            # Get input slice
            if input_framerate is None: # Keep whole input if no framerate (e.g. mulan->semantic)
                input_beg, input_end = 0, "end"
                input_slice = inputs_embeds
            else:
                input_beg, input_end = duration_to_framerate((slice_beg, slice_end), input_framerate)
                input_slice = inputs_embeds[:, input_beg:input_end]
            
            # Get prefix slice
            output_beg, output_end = duration_to_framerate((slice_beg, slice_end), output_framerate)
            prefix_beg, prefix_end = duration_to_framerate((slice_beg, slice_beg+cache_len), output_framerate)
            prefix_slice = output_embeds[:, prefix_beg:prefix_end] if output_embeds is not None else torch.zeros((batch_size, 0, sos_embeds.size(-1)), device=sos_embeds.device)

            # Full prompt
            if self.use_cross_attn:
                model_input = { 
                    "inputs_embeds": torch.cat([sos_embeds, prefix_slice], dim=1),
                    "encoder_hidden_states": input_slice
                }
            else:
                model_input = { "inputs_embeds": torch.cat([input_slice, sos_embeds, prefix_slice], dim=1) }

            past_key_values = None
            num_tokens = output_end - prefix_end
            pbar = tqdm(range(num_tokens))
            for i in pbar:
                pbar.set_description(
                    f"{tqdm_name} [{output_beg} - {output_end}] [{input_beg} - {input_end}]"
                )

                # Get predictions
                model_output = self.model(
                    **model_input, past_key_values=past_key_values, use_cache=True
                )
                logits = model_output["logits"]
                logits = logits[:, -1:, :] # only predicting on last logit.

                # Sample logits
                predict_token = self.sample_logits(i, logits, temperature, sample_mode)
                predict_embed = self.target_embedder.embedder(predict_token)

                # Update model input selection.
                model_input["inputs_embeds"] = predict_embed
                past_key_values = model_output["past_key_values"]

                # Save outputs
                output_embeds = torch.cat([output_embeds, predict_embed], dim=1) if output_embeds is not None else predict_embed
                output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token
        return output_tokens

def delete_embedding_module(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Embedding):
            print('Found existing embedding module. Deleting:', name)
            delattr(model, name)
            break
        else:
            delete_embedding_module(module)

def get_slice_ranges(slice_duration:float, target_duration:float, stride_duration:int):
    """slice_duration=target duration model was trained on. target_duration=final duration for resulting audio, stride_duration=duration overlap"""
    slice_ranges = []
    beg = 0
    while True:
        end = beg + slice_duration
        if end >= target_duration:
            end = target_duration
            beg = end - slice_duration
            slice_ranges.append([beg, end])
            break
        else:
            slice_ranges.append([beg, end])
        beg += stride_duration

    return slice_ranges

def duration_to_framerate(range, framerate):
    beg, end = range
    return int(beg * framerate), int(end * framerate)

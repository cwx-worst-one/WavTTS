import os
import random
from copy import deepcopy
from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torchaudio.functional import resample

from samantha.criterion.masked_loss import sequence_mask
from samantha.utils.hparams import DotDict

from .embedding_modules import (
    BestRQTokenEmbedder,
    MulanEmbedder,
    BaseEmbedder,
)


class SemanticEmbModule(torch.nn.Module):
    def __init__(
        self,
        required_modules: dict,
        input_embedders: dict[str, BaseEmbedder],
        target_embedder: BestRQTokenEmbedder,
        extra_params: Optional[dict] = None,
        stage: str = "fit",
    ):
        super().__init__()

        # inherit from recipes.bigmusic.lightning.base_modules.BaseContinuousEmbedModule
        self.input_embedders = nn.ModuleDict(input_embedders)
        self.target_embedder = target_embedder

        # TODO (shuo): apply weight init here
        # self.input_embedders.apply(self.model._init_weights)
        # self.target_embedder.apply(self.model._init_weights)

        # inherit from recipes.bigmusic.lightning.base_modules.BaseModule
        self.extra_params = DotDict({"varlen_lyrics_prefix": True} if extra_params is None else extra_params)
        self.requires = {}
        self.val_outputs = dict()

        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f"cuda:{self.local_rank}"
        self.required_modules = required_modules
        self.stage = stage
        self.text_codebook_size = extra_params.get('text_codebook_size', 0)  # Yilin: what's the use of this parameter?
        # will call it in CruiseModule/PLModule's setup, should not in class init
        # self.setup(stage)
 
    def setup(self, stage: str) -> None:
        # setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        # mfu metric
        # self.metric = ModelMetric(
        #     precision=self.trainer.precision,
        #     model_obj_or_objs=self.model,
        # )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

        pretrained_path = self.extra_params.get('pretrained_path')
        if stage == "fit" and pretrained_path is not None:
            self.load_from_pretrained(pretrained_path)

    def load_required_modules(self, ignore=()):
        for name, item in self.required_modules.items():
            if name in ignore: continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))       

    def prepare_audio_prompt_token_inputs(self, batch):
        """This function works in inference when you need a audio_prompt as prompt for audio continuation."""
        frame_rate = 25
        tokenizer_sample_rate = 24000

        # channel & sample rate
        audio_prompt = batch["audio_prompt"] 
        audio_sample_rate = batch["audio_sample_rate"]
        if audio_prompt.shape[1] == 2:
            audio_prompt = audio_prompt.mean(dim=1, keepdim=True)  # take average
        if audio_sample_rate != tokenizer_sample_rate:
            audio_prompt = resample(audio_prompt, audio_sample_rate, tokenizer_sample_rate)

        # NOTE: when bs > 1, there could be zero padding at the end of audio_prompt
        token_ids = self.target_embedder.tokenize(
            requires=self.requires,
            batch=audio_prompt,
        )
        token_ids += self.text_codebook_size

        # Exclude the last token, update the audio_prompt to reflect the actual audio duration
        n_tokens = int(batch["target_tokens_length"].max())  # +1 SOS, -1 end token
        token_ids = token_ids[..., :n_tokens]  # exclude the last token
        n_samples_batch_audio = round((n_tokens - 1) * audio_sample_rate / frame_rate)  # do not count the SOS token
        batch["audio_prompt"] = batch["audio_prompt"][..., :n_samples_batch_audio]  # in-place modification
        batch["target_tokens_length"] = batch["target_tokens_length"] - 1  # -1 end token => actual num audio tokens
        batch["audio_prompt_token_ids"] = token_ids[:, 1:]  # remove SOS, add tokens to batch for reconstruction
        batch["audio_prompt_n_samples"] = batch["target_tokens_length"] // frame_rate * audio_sample_rate  # add a key for prompt concat

        embeds = self.target_embedder.embedder(token_ids)
        embed_dict = {
            "token_embeds": embeds,
            "token_length": batch["target_tokens_length"] + 1,  # +1 SOS
        }
        batch["audio_prompt_token_ids"] = token_ids[:, 1:]  # remove SOS, add tokens to batch for reconstruction
        return embed_dict

    def prepare_prefix_inputs(self, batch):
        st_idx = 0
        prefix_inputs = []
        batch["inputs_embeds_span"] = {}
        for emb_type, embedder in self.input_embedders.items():
            if isinstance(embedder, MulanEmbedder):
                emb_inputs = embedder.prepare_embed(
                    requires=self.requires,
                    batch=batch,
                    data_type='text'
                )
            else:
                emb_inputs = embedder.prepare_embed(batch)
            prefix_inputs.append(emb_inputs)

            batch["inputs_embeds_span"][emb_type] = []
            for emb in emb_inputs['token_embeds']:
                en_idx = st_idx + emb.shape[0]
                batch["inputs_embeds_span"][emb_type].append((st_idx, en_idx))
                st_idx = en_idx

            #print('-------')
            #print(emb_type, emb_inputs['token_embeds'])
    
        # Append audio_prompt to prefix_inputs for audio continuation
        if (not self.training and "audio_prompt_token" in batch):
            # Key "audio_prompt" has to be in the batch to make "audio_prompt_token" work.
            # Audio prompting only works for varlen with bs=1. When audio_prompt is not given,
            # batch["audio_prompt"] is [None], otherwise it is a tensor of shape [1, 1, n]            
            if isinstance(batch["audio_prompt"], list):
                batch["audio_prompt"] = batch["audio_prompt"][0][0]
            emb_inputs = self.prepare_audio_prompt_token_inputs(batch)
            prefix_inputs.append(emb_inputs)
        return prefix_inputs

    def prepare_training_inputs(self, batch):
        # Concate Prefix and Target tokens
        prefix_inputs = self.prepare_prefix_inputs(batch)
        target_inputs = self.target_embedder.prepare_embed(self.requires, batch)
        # Concat prefix and target in to a sequence
        token_ids = zip(*[i['token_ids'] for i in prefix_inputs + target_inputs])
        token_ids = [torch.cat(t, dim=0) for t in token_ids]
        token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs + target_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0)
        target_length = torch.vstack([i['token_length'] for i in target_inputs]).sum(dim=0)

        # Padding the sequences across the batch
        token_ids_pad = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        token_embeds_pad = pad_sequence(token_embeds, batch_first=True, padding_value=0)
        batch_seq_length = token_ids_pad.shape[1]

        # Get masks
        input_loss_mask = sequence_mask(prefix_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = sequence_mask(prefix_length + target_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = target_loss_mask.bool() ^ input_loss_mask.bool() # remove input targets from target mask

        return {
            'token_embeds': token_embeds_pad,
            'token_ids': token_ids_pad,
            'input_loss_mask': input_loss_mask,
            'target_loss_mask': target_loss_mask,
            'prefix_inputs': prefix_inputs,
            'target_inputs': target_inputs,
            'prefix_length': prefix_length,
            'target_length': target_length,
            'token_length': prefix_length + target_length,
        }

    def load_from_pretrained(self, pretrained_path=None):
        """
        Patch the parent method to translate single tag to multitag embeddings.
        Correctly handle SOS and EOS for different sizes of categorical embeddings.
        TODO: Remove this method after we completely switch to unified vocab.
        """
        from pathlib import Path

        from recipes.diffusion.utils.utils import download_checkpoint
        from recipes.musiclm.utils.dist import local_zero_first

        print('Loading pre-trained model from checkpoint', pretrained_path)
        with local_zero_first():
            cache_dir = Path(self.extra_params.get('cache_dir', '.pretrain_cache'))
            pretrained_path = download_checkpoint(pretrained_path, cache_dir=cache_dir)
        state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        model_state_dict = self.state_dict()

        for k in state_dict:
            if k not in model_state_dict:
                print(f"Dropping parameter {k}")
                continue
            # skip checking over non-tensor items. i.e. embedding_modules->set_extra_state
            if not torch.is_tensor(state_dict[k]): continue

            if state_dict[k].shape != model_state_dict[k].shape:
                # special case for embedder weights
                if k.endswith('embedder.weight') and state_dict[k].shape[1:] == model_state_dict[k].shape[1:]:
                    print(f"Embedding module found with different vocab sizes. Copying subset of weights",
                           k, state_dict[k].shape[0], model_state_dict[k].shape[0])
                    min_vocab_size = min(state_dict[k].shape[0], model_state_dict[k].shape[0])
                    model_state_dict[k][:min_vocab_size] = state_dict[k][:min_vocab_size]
                    state_dict[k] = model_state_dict[k]
                else:
                    print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]

        self.load_state_dict(state_dict, strict=False)
    
    def prepare_group_cfg_batch(self, batch, hp):
        """
        Separate all the key/value pairs whose keys end with _cfg into a CFG batch (plus "app_type" and "conditions"),
        then remove these key/value pairs from the batch.
        This method assumes that the dataloader has prepared the CFG related items in the input batch.
        """
        def is_key_cfg(key: str) -> bool: return key.endswith("_cfg")
        def rm_cfg_suffix(key: str) -> str: return key[:-4]
        cfg_group = hp.get("group_controller_cfg_label", 
                           [['style_text', 'freeform_text', 'speaker_id', 'mir'],
                            ['lyrics']])
        group_dict = {
            'lyrics': ['lyrics', 'symbols', 'lyrics_tokens', 'lyrics_coffs', 'lyrics_tokens_length',
                       'max_phone_len', 'lyrics_encoding_mode', 'lyrics_eval', 'lyrics_display', 'slice_duration',],
            'style_text': ['style_text', 'style_text_display'],
            'freeform_text': ['freeform_text'],
            'speaker_id': ['speaker_id'],
            'mir': ['tempo', 'tempo_label', 'key', 'key_root', 'key_mode', 'time_signature', 
                    'instrument', 'instrument_length', 'section_instruments'],
        }
        n_group = len(cfg_group)
        batch_group_cfg = []
        for i in range(n_group):
            this_group_cfg = {}
            for cond in cfg_group[i]:
                this_group_cfg.update({rm_cfg_suffix(k): v for k, v in batch.items() if is_key_cfg(k) and rm_cfg_suffix(k) in group_dict[cond]})
            this_group_cfg_additional = {k: deepcopy(v) for k, v in batch.items() if k not in this_group_cfg}
            this_group_cfg = this_group_cfg | this_group_cfg_additional
            batch_group_cfg.append(this_group_cfg)
        
        if self.extra_params.varlen_lyrics_prefix:
            for i in range(n_group):
                batch_group_cfg[i]["lyrics_tokens_length"] = batch["lyrics_tokens_length"]
                if "instrument_length" in batch:
                    batch_group_cfg[i]["instrument_length"] = batch["instrument_length"]
        
        return batch_group_cfg

    # TODO: make this into a static method (vibertthio)
    def infer_batch_size(self, batch):
        batch_size = [len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)][0]
        return batch_size

    def forward(self, batch):
        if random.random() < self.extra_params.get("group_dropout_rate", 0):
            group_cfg_label = self.extra_params.get("group_controller_cfg_label",
            [["style_text","freeform_text","speaker_id"],["lyrics"]])
            group_cfg_batch = self.prepare_group_cfg_batch(batch, {"group_controller_cfg_label": group_cfg_label})
            # randomly replace training batch to group cfg batch
            batch = random.choice(group_cfg_batch)

        # with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
        training_inputs = self.prepare_training_inputs(batch)
        return training_inputs

    @torch.no_grad()
    def _prepare_cfg_batch(self, batch, hp):
        """
        Separate all the key/value pairs whose keys end with _cfg into a CFG batch (plus "app_type" and "conditions"),
        then remove these key/value pairs from the batch.

        This method assumes that the dataloader has prepared the CFG related items in the input batch.

        NOTE: This method will modify the input batch in-place.
        """
        def is_key_cfg(key: str) -> bool: return key.endswith("_cfg")
        def rm_cfg_suffix(key: str) -> str: return key[:-4]
        batch_cfg = {rm_cfg_suffix(k): v for k, v in batch.items() if is_key_cfg(k)}  # create cfg batch
        batch_cfg_additional = {k: deepcopy(v) for k, v in batch.items() if k not in batch_cfg and not is_key_cfg(k)}
        batch_cfg = batch_cfg | batch_cfg_additional
        if self.extra_params.varlen_lyrics_prefix:
            # TODO (qq) temp fix: pad to same length to ensure var len inference cfg.
            batch_cfg["lyrics_tokens_length"] = batch["lyrics_tokens_length"]
            if "instrument_length" in batch:
                batch_cfg["instrument_length"] = batch["instrument_length"]
        # cfg_keys = [k for k in batch if is_key_cfg(k)]  # remove cfg keys in original batch
        # for k in cfg_keys:
        #     batch.pop(k, None)
        return batch_cfg

    # Temporary hack to make it compatible with the old module
    def is_token_input(self) -> bool:
        return False

    def predict_emb(self, batch: Dict, hp: Dict, beam: int = 1) -> Dict:
        return self.predict(batch, hp, beam)

    def _prepare_token_embeds(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_inputs = self.prepare_prefix_inputs(batch)
        token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)
        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0).int()
        return token_embeds, prefix_length

    def prepare_cfg_batch(self, batch: Dict, hp: Dict, use_controller_cfg: bool,
                          controller_cfg_gamma: Union[float, List[float]]) -> Optional[Union[Dict, List[Dict]]]:
        if not use_controller_cfg:
            return None

        if isinstance(controller_cfg_gamma, list):
            batch_uncond = self._prepare_cfg_batch(batch, hp)
            batch_cfg = self.prepare_group_cfg_batch(batch, hp)  # list
            batch_cfg.append(batch_uncond)  # also add FULL uncond path
            return batch_cfg
        else:
            return self._prepare_cfg_batch(batch, hp)

    def _prepare_group_cfg_embeddings(self, batch_cfg_list: List[Dict],
                                     prefix_length: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        token_embeds_cfg = []
        prefix_length_cfg = []

        for group_idx, batch_cfg_group in enumerate(batch_cfg_list):
            token_embeds, _prefix_length = self._prepare_token_embeds(batch_cfg_group)

            assert torch.all(_prefix_length == prefix_length), \
                f"CFG group {group_idx} prefix length mismatch: expected {prefix_length}, got {_prefix_length}"

            print(f"cfg group={group_idx}, shape={token_embeds.shape}, prefix_length={_prefix_length}")
            token_embeds_cfg.append(token_embeds)
            prefix_length_cfg.append(_prefix_length)

        return token_embeds_cfg, prefix_length_cfg

    def _prepare_cfg_embeddings(self, batch_cfg: Dict,
                                      prefix_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_embeds_cfg, prefix_length_cfg = self._prepare_token_embeds(batch_cfg)

        assert torch.all(prefix_length_cfg == prefix_length), \
            f"CFG prefix length mismatch: expected {prefix_length}, got {prefix_length_cfg}"

        print(f"token_embeds_cfg.shape={token_embeds_cfg.shape}, prefix_length_cfg={prefix_length_cfg}")
        return token_embeds_cfg, prefix_length_cfg

    def _process_group_cfg_for_generation(self, inputs_embeds: torch.Tensor,inputs_embeds_cfg: List[torch.Tensor],
                               controller_cfg_gamma: List[float], beam: int, seq_len: int) -> Tuple[torch.Tensor, int]:
        # delete cfg samples with zero cfg gammas (to avoid extra differences from different batch sizes)
        # lambda_{s,l}, lambda_{l}, lambda_{s}, lambda
        keep_indices = [i for i, gamma in enumerate(controller_cfg_gamma[1:]) if gamma != 0]
        inputs_embeds_cfg = [inputs_embeds_cfg[i] for i in keep_indices]
        inputs_embeds_cfg = torch.cat(inputs_embeds_cfg, 0)
        controller_cfg_gamma = [gamma for gamma in controller_cfg_gamma if gamma != 0]

        cfg_batch_size = inputs_embeds_cfg.shape[0]
        inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, seq_len, -1)

        if controller_cfg_gamma[0] > 0:
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)    # cat on first-dim(batch_size)
        else:
            inputs_embeds = inputs_embeds_cfg

        return inputs_embeds, cfg_batch_size

    def _process_cfg_for_generation(self, inputs_embeds: torch.Tensor, inputs_embeds_cfg: torch.Tensor, beam: int, seq_len: int) -> tuple:
        cfg_batch_size = inputs_embeds_cfg.shape[0]
        inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, seq_len, -1)
        inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)    # cat on first-dim(batch_size)
        return inputs_embeds, cfg_batch_size

    @torch.no_grad()
    def predict(self, batch: Dict, hp: Dict, beam: int = 1) -> Dict:
        use_controller_cfg = hp.get('use_controller_cfg', False)
        controller_cfg_gamma = hp.get('controller_cfg_gamma', 1)
        skip_sos = hp.get('skip_sos', False)
        self.extra_params.debug_index = hp.get('debug_index', None)

        exclude_ids = None
        if hp.get("exclude_eos", False) and self.target_embedder.eos_id is not None:
            exclude_ids = [self.target_embedder.eos_id]
            print(f"exclude_ids: {exclude_ids}")

        batch_cfg = self.prepare_cfg_batch(
            batch, hp, use_controller_cfg, controller_cfg_gamma
        )

        inputs_embeds, prefix_length = self._prepare_token_embeds(batch)

        inputs_embeds_cfg, prefix_length_cfg = None, None
        if batch_cfg:
            if isinstance(batch_cfg, list):
                inputs_embeds_cfg, prefix_length_cfg =  self._prepare_group_cfg_embeddings(batch_cfg, prefix_length)
            else:
                inputs_embeds_cfg, prefix_length_cfg =  self._prepare_cfg_embeddings(batch_cfg, prefix_length)

        batch.update({
            "prefix_length": prefix_length,
            "prefix_length_cfg": prefix_length_cfg
        })

        batch_size, seq_len, _ = inputs_embeds.size()
        original_batch_size = batch_size

        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(batch_size * beam, seq_len, -1)

        cfg_batch_size = None
        if use_controller_cfg and inputs_embeds_cfg is not None:
            if isinstance(inputs_embeds_cfg, list) and isinstance(controller_cfg_gamma, list):
                inputs_embeds, cfg_batch_size = self._process_group_cfg_for_generation(inputs_embeds, inputs_embeds_cfg, controller_cfg_gamma, beam, seq_len)
            else:
                inputs_embeds, cfg_batch_size = self._process_cfg_for_generation(inputs_embeds, inputs_embeds_cfg, beam, seq_len)

        batch_size, seq_len, _ = inputs_embeds.size() # recalculate batch size

        prefix_length = prefix_length.repeat(batch_size // original_batch_size)
        inputs_embeds_list = unpad_sequence(inputs_embeds, prefix_length, batch_first=True)

        if not skip_sos:
            sos_embeds = self.target_embedder.get_sos_embed(batch_size)
            inputs_embeds_list = [torch.cat([e, sos_embeds[i]]) for i, e in enumerate(inputs_embeds_list)]
            seq_len += 1
            prefix_length += 1

        # pad at left side
        seq_len = max(seq_len, max(e.shape[0] for e in inputs_embeds_list))
        padded_embeds = []
        for e in inputs_embeds_list:
            pad_length = seq_len - e.shape[0]
            if pad_length > 0:
                padded_e = torch.nn.functional.pad(e, (0, 0, pad_length, 0))
            else:
                padded_e = e
            padded_embeds.append(padded_e)
        inputs_embeds = torch.stack(padded_embeds, dim=0)
        inputs_embeds_mask = (torch.arange(0, seq_len, device=prefix_length.device) < (seq_len - prefix_length)[:, None]).bitwise_not()

        return {
            'inputs_embeds': inputs_embeds,
            'inputs_embeds_mask': inputs_embeds_mask,
            'cfg_batch_size': cfg_batch_size,
            'original_batch_size': original_batch_size,
            'prefix_length': prefix_length,
            'exclude_ids': exclude_ids
        }

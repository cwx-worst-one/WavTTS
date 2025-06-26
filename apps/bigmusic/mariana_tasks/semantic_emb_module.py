import os
import random
from copy import deepcopy
from typing import Optional

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

        token_ids = self.target_embedder.tokenize(
            requires=self.requires,
            batch=audio_prompt,
        )
        token_ids += self.text_codebook_size

        # Exclude the last token, update the audio_prompt to reflect the actual audio duration
        n_tokens = token_ids.shape[-1] - 1
        token_ids = token_ids[..., :n_tokens]
        n_samples_batch_audio = round((n_tokens - 1) * audio_sample_rate / frame_rate)  # do not count the SOS token
        batch["audio_prompt"] = batch["audio_prompt"][..., :n_samples_batch_audio]  # in-place modification

        embeds = self.target_embedder.embedder(token_ids)
        embed_dict = {
            "token_embeds": embeds,
            "token_length": torch.Tensor([embeds.shape[1]]).to(self.device),
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
    def prepare_cfg_batch_v2(self, batch, hp):
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

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, rl_training=False):
        # is_varlen_prefix = self.extra_params.varlen_lyrics_prefix        
        # if is_varlen_prefix: assert self.infer_batch_size(batch) == 1

        # frame_rate = self.extra_params.semantic_frame_rate
        # num_tokens = hp.duration * frame_rate
        # temperature = hp.semantic_temperature
        # sample_mode = hp.sample_mode
        # sample_thresh = hp.get('sample_thresh', 0.9)
        use_controller_cfg = hp.get('use_controller_cfg', False)
        controller_cfg_gamma = hp.get('controller_cfg_gamma', 1)
        # use_step_out_blank = hp.get('use_step_out_blank', False)
        # step_out_blank_logic = hp.get('step_out_blank_logic', 'v2')
        # step_out_blank_max_len = hp.get('step_out_blank_max_len', 10)
        # repetition_penalty = hp.get('repetition_penalty', 1.0)
        # exclude_eos_first_secs = hp.get('exclude_eos_first_secs', 0)
        # exclude_eos_thresh_secs = hp.get('exclude_eos_thresh_secs', 0)
        # emit_eos_thresh_secs = hp.get('emit_eos_thresh_secs', 0)
        skip_sos = hp.get('skip_sos', False)
        # stop_eos = hp.get('stop_eos', False)
        self.extra_params.debug_index = hp.get('debug_index', None)
        exclude_ids = None
        if hp.get("exclude_eos", False) and self.target_embedder.eos_id is not None:
            exclude_ids = [self.target_embedder.eos_id]
            print(f"exclude_ids: {exclude_ids}")
        
        # Init batch_cfg before batch gets modified
        if isinstance(controller_cfg_gamma, list) and use_controller_cfg:
            batch_uncond = self.prepare_cfg_batch_v2(batch, hp)
            batch_cfg = self.prepare_group_cfg_batch(batch, hp) # list
            batch_cfg.append(batch_uncond)  # also add FULL uncond path
        else:
            batch_cfg = self.prepare_cfg_batch_v2(batch, hp) if use_controller_cfg else None

        prefix_inputs = self.prepare_prefix_inputs(batch)        
        token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)
        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0).int()

        if batch_cfg:
            if isinstance(batch_cfg, list):
                token_embeds_cfg, prefix_length_cfg = [], []
                for group_idx in range(len(batch_cfg)):
                    _prefix_inputs_cfg = self.prepare_prefix_inputs(batch_cfg[group_idx])
                    _token_embeds_cfg = zip(*[i['token_embeds'] for i in _prefix_inputs_cfg])
                    _token_embeds_cfg = [torch.cat(t, dim=0) for t in _token_embeds_cfg]
                    _token_embeds_cfg = pad_sequence(_token_embeds_cfg, batch_first=True, padding_value=0)
                    _prefix_length_cfg = torch.vstack([i['token_length'] for i in _prefix_inputs_cfg]).sum(dim=0).int()
                    assert torch.all(_prefix_length_cfg == prefix_length)
                    print(f"cfg group={group_idx}{_token_embeds_cfg.shape=} {_prefix_length_cfg=}")
                    token_embeds_cfg.append(_token_embeds_cfg)
                    prefix_length_cfg.append(_prefix_length_cfg)
            else:
                prefix_inputs_cfg = self.prepare_prefix_inputs(batch_cfg)
                token_embeds_cfg = zip(*[i['token_embeds'] for i in prefix_inputs_cfg])
                token_embeds_cfg = [torch.cat(t, dim=0) for t in token_embeds_cfg]
                token_embeds_cfg = pad_sequence(token_embeds_cfg, batch_first=True, padding_value=0)
                prefix_length_cfg = torch.vstack([i['token_length'] for i in prefix_inputs_cfg]).sum(dim=0).int()
                assert torch.all(prefix_length_cfg == prefix_length)
                print(f"{token_embeds_cfg.shape=} {prefix_length_cfg=}")
        else:
            token_embeds_cfg = None
            prefix_length_cfg = None

        batch["prefix_length"] = prefix_length
        batch["prefix_length_cfg"] = prefix_length_cfg


        # (shuo): copy from BaseContinuousEmbedModule.predict()

        inputs_embeds = token_embeds
        use_controller_cfg = use_controller_cfg
        inputs_embeds_cfg = token_embeds_cfg

        batch_size, seq_len, _ = inputs_embeds.size()
        original_batch_size = batch_size
        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(batch_size * beam, seq_len, -1)

        if use_controller_cfg:
            if isinstance(inputs_embeds_cfg, list) and isinstance(controller_cfg_gamma, list):
                # delete cfg samples with zero cfg gammas (to avoid extra differences from different batch sizes)
                # lambda_{s,l}, lambda_{l}, lambda_{s}, lambda
                keep_idx = [i for i, gamma in enumerate(controller_cfg_gamma[1:]) if gamma != 0]
                inputs_embeds_cfg = [inputs_embeds_cfg[i] for i in keep_idx]
                inputs_embeds_cfg = torch.cat(inputs_embeds_cfg, 0)
                controller_cfg_gamma = [gamma for gamma in controller_cfg_gamma if gamma != 0]
                cfg_batch_size = inputs_embeds_cfg.shape[0]
                inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, seq_len, -1)
                if controller_cfg_gamma[0] > 0:
                    inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)  # cat on first-dim(batch_size)
                else:
                    inputs_embeds = inputs_embeds_cfg
            else:
                cfg_batch_size = inputs_embeds_cfg.shape[0]
                inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, seq_len, -1)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)  # cat on first-dim(batch_size)
                
        batch_size, seq_len, _ = inputs_embeds.size() # recalculate batch size
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)
  
        seq_len = inputs_embeds.shape[1]
        prefix_length = prefix_length.repeat(batch_size // original_batch_size)
        inputs_embeds_list = unpad_sequence(inputs_embeds, prefix_length, batch_first=True)
        if not skip_sos:
            inputs_embeds_list = [torch.cat([e, sos_embeds[i]]) for i, e in enumerate(inputs_embeds_list)]
            seq_len += 1
            prefix_length += 1
        
        pad_embeds = []
        # pad at left side
        for e in inputs_embeds_list:
            emb = torch.nn.functional.pad(e, (0, 0, seq_len-e.shape[0], 0))
            pad_embeds.append(emb)
        inputs_embeds = torch.stack(pad_embeds, dim=0)
        inputs_embeds_mask = (torch.arange(0, seq_len, device=prefix_length.device) < (seq_len - prefix_length)[:, None]).bitwise_not()
        model_input = {}
        model_input['inputs_embeds'] = inputs_embeds
        model_input['inputs_embeds_mask'] = inputs_embeds_mask
        if use_controller_cfg:
            model_input['cfg_batch_size'] = cfg_batch_size
        else:
            model_input['cfg_batch_size'] = None

        model_input['original_batch_size'] = original_batch_size
        model_input['prefix_length'] = prefix_length
        model_input['exclude_ids'] = exclude_ids

        return model_input
    
    # Temporary hack to make it compatible with the old module
    def is_token_input(self) -> bool:
        return False

    def predict_emb(self, batch, hp, beam: int, rl_training: bool = False):
        return self.predict(batch, hp, beam, rl_training)

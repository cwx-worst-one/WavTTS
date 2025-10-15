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
from apps.bigmusic.mariana_tasks.mtp_module import MultiTokenPredictionModule, HierarchicalTokenPredictionModule
from mariana.utils.audio.audio_logger import AudioLogger

logger = AudioLogger()


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
            # batch_cfg["lyrics_tokens_length"] = batch["lyrics_tokens_length"]
            if "instrument_length" in batch:
                batch_cfg["instrument_length"] = batch["instrument_length"]
        # cfg_keys = [k for k in batch if is_key_cfg(k)]  # remove cfg keys in original batch
        # for k in cfg_keys:
        #     batch.pop(k, None)
        return batch_cfg

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

        # assert torch.all(prefix_length_cfg == prefix_length), \
        #     f"CFG prefix length mismatch: expected {prefix_length}, got {prefix_length_cfg}"

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
        seq_len = inputs_embeds_cfg.shape[1]
        inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, seq_len, -1)
        inputs_embeds_list = [inputs_embeds, inputs_embeds_cfg]
        return inputs_embeds_list, cfg_batch_size

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
            # if isinstance(batch_cfg, list):
            #     inputs_embeds_cfg, prefix_length_cfg =  self._prepare_group_cfg_embeddings(batch_cfg, prefix_length)
            # else:
            inputs_embeds_cfg, prefix_length_cfg = self._prepare_cfg_embeddings(batch_cfg, prefix_length)

        batch.update({
            "prefix_length": prefix_length,
            "prefix_length_cfg": prefix_length_cfg
        })

        batch_size, seq_len, _ = inputs_embeds.size()
        original_batch_size = batch_size

        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(batch_size * beam, seq_len, -1)

        cfg_batch_size = None
        if use_controller_cfg and inputs_embeds_cfg is not None:
            # if isinstance(inputs_embeds_cfg, list) and isinstance(controller_cfg_gamma, list):
            #     inputs_embeds, cfg_batch_size = self._process_group_cfg_for_generation(inputs_embeds, inputs_embeds_cfg, controller_cfg_gamma, beam, seq_len)
            # else:
                # inputs_embeds_list, cfg_batch_size = self._process_cfg_for_generation(inputs_embeds, inputs_embeds_cfg, beam, seq_len)

            cfg_batch_size = inputs_embeds_cfg.shape[0]
            cfg_seq_len = inputs_embeds_cfg.shape[1]
            inputs_embeds_cfg = inputs_embeds_cfg.repeat(1, beam, 1).reshape(cfg_batch_size * beam, cfg_seq_len, -1)
            inputs_embeds_list = unpad_sequence(inputs_embeds, prefix_length, batch_first=True)
            cfg_inputs_embeds_list = unpad_sequence(inputs_embeds_cfg, prefix_length_cfg, batch_first=True)
            inputs_embeds_list = inputs_embeds_list + cfg_inputs_embeds_list
            prefix_length = torch.concat([prefix_length, prefix_length_cfg], dim=0)
            # return inputs_embeds_list, cfg_batch_size
            batch_size = prefix_length.shape[0]
            seq_len = prefix_length.max()
        else:
            batch_size, seq_len, _ = inputs_embeds.size() # recalculate batch size

        # prefix_length = prefix_length.repeat(batch_size // original_batch_size)
        # inputs_embeds_list = unpad_sequence(inputs_embeds, prefix_length, batch_first=True)

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


class SemanticEmbModuleMtp(SemanticEmbModule):

    def set_mtp_module(self, mtp_module: MultiTokenPredictionModule):
        self.mtp_module = mtp_module
        logger.info(f"SemanticEmbModuleMtp set mtp module {mtp_module}")
    

    def embed_token_id(self, token_ids, frame_idx=None) -> torch.Tensor:
        # assert isinstance(self.mtp_module.group, int)
        """
        Input: 
            token_ids: [G, T//G] / [B, G, T//G]
        Return:
            token_embeds: [G, T//G, D]
        """
        if frame_idx is not None and frame_idx + 1 < self.mtp_module.group and self.mtp_module.pattern == 'delay':
            assert token_ids.shape[-1] == 1
            token_ids[..., self.mtp_module.group-(frame_idx+1):, 0] = self.mtp_module.null_id

        token_embeds = self.target_embedder.embedder(token_ids) # [*, G, T//G] => [*, G, T//G, D]
        return token_embeds
    
    
    def emb_enc(self, token_length, token_ids):
        batch_size = token_ids.shape[0]
        eos_id = self.target_embedder.eos_id

        # * Name rule *
        # **flattened** means the raw token sequence, usually with [T] length
        # **group** means the token embeds/ids have been grouped into [G] groups, usually with [G, T//G] shape
        # **grouped** means the a group of token embeds have been merged via network/pooling, usually with [T//G] length
        grouped_enc_embeds, flattend_token_ids, grouped_token_lengths, group_token_embeds, group_token_ids = [], [], [], [], []
        for b in range(batch_size):
            # [T] ==> [G, T//G], padding with eos_id if T is not divisible by G, always ends by a full group of eos_id
            group_this_token_ids = self.mtp_module.group_token(token_ids[b, 1:token_length[b]-1], eos_id)
            
            # [G, T//G] => [G, T//G, D]
            group_this_token_embeds = self.embed_token_id(group_this_token_ids)
            # [G, T//G, D] => [T//G, D']
            grouped_this_token_embeds = self.mtp_module.encode_token_emb(group_this_token_embeds)
            # manally add sos token embed
            sos_embed = self.target_embedder.get_sos_embed(1).squeeze(0)
            grouped_this_token_embeds = torch.cat([
                sos_embed,                    # SOS token, [1, D]
                grouped_this_token_embeds,             # grouped tokens + EOS tokens, [T, D]
            ], dim=0)

            flattened_this_token_ids = self.mtp_module.ungroup_token(
                group_ids = group_this_token_ids, 
                sos_id = self.target_embedder.sos_id,
                eos_id = self.target_embedder.eos_id,
                delete_null=False, add_sos=True)

            grouped_enc_embeds.append(grouped_this_token_embeds)    # [1(SOS) + T//G + 1(EOS), D]
            grouped_token_lengths.append(grouped_this_token_embeds.shape[0])   # [1(SOS) + T//G + 1(EOS)]
            group_token_embeds.append(group_this_token_embeds)    # [G, T//G + 1(EOS), D]
            flattend_token_ids.append(flattened_this_token_ids)         # [1(SOS) + T//G + 1(EOS)]
            group_token_ids.append(group_this_token_ids)    # [G, T//G + 1(EOS)]

        # group_token_embeds: list of [G, T//G + 1(EOS), D] (token_id -> embedder) wo sos token emb
        # grouped_enc_embeds: list of [1(SOS) + T//G + 1(EOS), D] (token_id -> embedder -> encoder) with sos token emb
        # flattend_token_ids: list of [G, 1(SOS) + T//G + 1(EOS)] (token_id) with sos token
        # grouped_token_lengths: list of [1(SOS) + T//G + 1(EOS)] (token_id_length) with sos token length
        return group_token_embeds, grouped_enc_embeds, flattend_token_ids, grouped_token_lengths, group_token_ids


    def prepare_training_inputs(
            self, 
            batch: dict,
        ):
        # Concate Prefix and Target tokens
        prefix_inputs = self.prepare_prefix_inputs(batch)
        assert self.target_embedder.with_sos and self.target_embedder.with_eos, "target_embedder must with_sos and with_eos"
        target_ids, target_lengths = self.target_embedder._get_target_ids_and_token_lengths(self.requires, batch)
        group_token_embeds_list, grouped_enc_embeds_list, flattend_token_ids_list, grouped_token_lengths, group_token_ids = \
            self.emb_enc(token_ids=target_ids, token_length=target_lengths)
        
        target_inputs = [{
            'token_embeds': grouped_enc_embeds_list, # B x [T//G, D]
            'group_token_embeds': group_token_embeds_list,  # B x [G, T//G, D]
            'group_token_ids': group_token_ids,  # B x [G, T//G, (R)]
            'token_ids': flattend_token_ids_list,   # B x [T, (R)] (with sos eos)
            'token_length': torch.LongTensor(grouped_token_lengths).to(self.device),
            'ori_token_length': target_lengths,
            'ori_token_ids': target_ids,   # B x [T, (R)] (no sos eos)
        }]
        
        # Concat prefix and target in to a sequence
        if isinstance(self.mtp_module, HierarchicalTokenPredictionModule):
            prefix_token_ids = zip(*[i['token_ids'] for i in prefix_inputs])
            target_token_ids = zip(*[i['token_ids'] for i in target_inputs])
            target_token_ids = [torch.cat(t, dim=0) for t in target_token_ids]  # batch_size x [T, R]
            prefix_token_ids = [torch.cat(t, dim=0).unsqueeze(-1).repeat(1, self.target_embedder.R) 
                                for t in prefix_token_ids]  # batch_size x [T_p, 1->R]
            token_ids = [torch.cat([prefix_token_ids[b], target_token_ids[b]], dim=0) for b in range(len(prefix_token_ids))]
        else:
            token_ids = zip(*[i['token_ids'] for i in prefix_inputs + target_inputs])
            token_ids = [torch.cat(t, dim=0) for t in token_ids]  # batch_size x [T_p + T]

        token_ids_pad = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        batch_seq_length = token_ids_pad.shape[1]
        target_ids = zip(*[i['token_ids'] for i in target_inputs])
        target_ids = [torch.cat(t, dim=0) for t in target_ids]  # batch_size x [T]

        token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs + target_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        token_embeds_pad = pad_sequence(token_embeds, batch_first=True, padding_value=0)

        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0)
        target_length = torch.vstack([i['token_length'] for i in target_inputs]).sum(dim=0)

        # @qinxin: * original_target_length: sos_id + target_id_length + eos_id (1+T+1)
        # * target_length: sos_id + ceil(target_id_length/group) + eos_id (1+T//G+1)
        # so (original_target_length-2) = (group_size-2) * target_length
        ori_target_length = torch.vstack([i['ori_token_length'] for i in target_inputs]).sum(dim=0)
        # original token_ids is batch-padded [B, max_target_id_length] (including sos_id, eos_id) (1+max_T+1)
        ori_target_token_ids = zip(*[i['ori_token_ids'] for i in target_inputs])
        ori_target_token_ids = [torch.cat(t, dim=0) for t in ori_target_token_ids]
    
        # Get masks
        input_loss_mask = sequence_mask(prefix_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = sequence_mask(prefix_length + target_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = target_loss_mask.bool() ^ input_loss_mask.bool() # remove input targets from target mask
        if target_loss_mask.ndim < token_ids_pad.ndim:
            target_loss_mask = target_loss_mask.unsqueeze(-1).expand_as(token_ids_pad)

        out_dict = {
            'token_ids': token_ids_pad,
            'input_loss_mask': input_loss_mask,
            'target_loss_mask': target_loss_mask,
            'prefix_length': prefix_length,
            'target_length': target_length,
            'token_length': prefix_length + target_length,
        }
        out_dict['token_embeds'] = token_embeds_pad
        out_dict['ori_target_length'] = ori_target_length
        out_dict['target_ids'] = target_ids                     # with sos id, with eos id, [1+T+1]
        out_dict['ori_target_ids'] = ori_target_token_ids
        target_group_token_embeds = zip(*[i['group_token_embeds'] for i in target_inputs])
        target_group_token_embeds = [torch.cat(t, dim=0) for t in target_group_token_embeds]
        out_dict['target_group_token_embeds'] = target_group_token_embeds   # w/o sos embed, with eos embed [G, T//G+1, D]
        target_group_token_ids = zip(*[i['group_token_ids'] for i in target_inputs])
        target_group_token_ids = [torch.cat(t, dim=0) for t in target_group_token_ids]
        out_dict['target_group_token_ids'] = target_group_token_ids   # w/o sos id, with eos id [G, T//G+1]
        return out_dict
        


class SemanticEmbModuleHierarchicalMtp(SemanticEmbModuleMtp):

    def set_mtp_module(self, mtp_module: HierarchicalTokenPredictionModule):
        self.mtp_module = mtp_module
        logger.info(f"SemanticEmbModuleMtp set mtp module {mtp_module}")
        if self.target_embedder.codebook_tie and not self.target_embedder.codebook_tie_flag:  # codebook tie
            self.target_embedder.tie_codebook(self.requires)

    def embed_token_id(self, group_token_ids, frame_idx=None) -> torch.Tensor:
        """
        Input:
            group_token_ids:    [*, G, T//G, R]
        Return: 
            group_token_embeds: [*, G, T//G, token_embed_dim]
            eos_mask:           [*, G, T//G, input_R]
        """ 
        R = min(self.mtp_module.R, self.mtp_module.input_R, group_token_ids.shape[-1])
        # truncate to input_R
        group_token_ids = group_token_ids[..., :R]   # [G, T//G, input_R]

        group_token_embeds = torch.stack([self.target_embedder.embedder[r](group_token_ids[..., r]) 
                                    for r in range(R)], dim=-2) # [*, G, T//G, input_R, token_embed_dim]
        
        eos_mask = group_token_ids != self.target_embedder.eos_id # [*, G, T//G, input_R]
        null_mask = group_token_ids != self.target_embedder.null_id   # [*, G, T//G, input_R]
        group_token_embeds = group_token_embeds * (eos_mask & null_mask).unsqueeze(-1) # [*, G, T//G, R, 32]
        group_token_embeds = group_token_embeds.cumsum(dim=-2)  # [*, G, T//G, input_R, 32]

        # operate on R: quantizer dropout [*, G, T//G, input_R, 32] => [*, G, T//G, 32]
        if self.training and self.mtp_module.quantizer_dropout > 0 and torch.rand(1) < self.mtp_module.quantizer_dropout:
            # sample-level quantizer dropout
            rand_R = torch.randint(R, size=(1,))[0]
            group_token_embeds = group_token_embeds[..., rand_R, :]
        else:
            group_token_embeds = group_token_embeds[..., -1, :]
        return group_token_embeds, eos_mask
    
    
    def emb_enc(self, token_length, token_ids):
        batch_size = token_ids.shape[0]
        eos_id = self.target_embedder.eos_id

        # * Name rule *
        # **flattened** means the raw token sequence, usually with [T] length
        # **group** means the token embeds/ids have been grouped into [G] groups, usually with [G, T//G] shape
        # **grouped** means the a group of token embeds have been merged via network/pooling, usually with [T//G] length
        grouped_enc_embeds, flattend_token_ids, grouped_token_lengths, group_token_embeds, group_token_ids = [], [], [], [], []
        for b in range(batch_size):
            # [T, R] ==> [G, T//G, R], padding with eos_id if T is not divisible by G, always ends by a full group of eos_id
            group_this_token_ids = self.mtp_module.group_token(token_ids[b, 1:token_length[b]-1,:], eos_id)
            
            # [G, T//G, R] => [G, T//G, token_embed_dim]
            group_this_token_embeds, group_eos_mask = self.embed_token_id(group_this_token_ids)
            # [G, T//G, D] => [T//G, D']
            grouped_this_token_embeds = self.mtp_module.encode_token_emb(group_this_token_embeds, group_eos_mask, 
                                                                         # [T//G, D]
                                                                         self.target_embedder.get_eos_embed(group_this_token_embeds.shape[1]))
            # manally add sos token embed
            sos_embed = self.target_embedder.get_sos_embed(1).squeeze(0)
            grouped_this_token_embeds = torch.cat([
                sos_embed,                    # SOS token, [1, D]
                grouped_this_token_embeds,    # grouped tokens + EOS tokens, [T, D]
            ], dim=0)

            flattened_this_token_ids = self.mtp_module.ungroup_token(
                group_ids = group_this_token_ids, 
                sos_id = self.target_embedder.sos_id,
                eos_id = self.target_embedder.eos_id,
                delete_null=False, add_sos=True)

            grouped_enc_embeds.append(grouped_this_token_embeds)    # [1(SOS) + T//G + 1(EOS), D]
            grouped_token_lengths.append(grouped_this_token_embeds.shape[0])   # [1(SOS) + T//G + 1(EOS)]
            group_token_embeds.append(group_this_token_embeds)    # [G, T//G + 1(EOS), token_embed_dim]
            flattend_token_ids.append(flattened_this_token_ids)         # [1(SOS) + T + 1(EOS), R]
            group_token_ids.append(group_this_token_ids)    # [G, T//G + 1(EOS), R]

        # group_token_embeds: list of [G, T//G + 1(EOS), D] (token_id -> embedder) wo sos token emb
        # grouped_enc_embeds: list of [1(SOS) + T//G + 1(EOS), D] (token_id -> embedder -> encoder) with sos token emb
        # flattend_token_ids: list of [G, 1(SOS) + T + 1(EOS)] (token_id) with sos token
        # grouped_token_lengths: list of [1(SOS) + T//G + 1(EOS)] (token_id_length) with sos token length
        # group_token_ids: list of [G, T//G + 1(EOS), R]
        return group_token_embeds, grouped_enc_embeds, flattend_token_ids, grouped_token_lengths, group_token_ids
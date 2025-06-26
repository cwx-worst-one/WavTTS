import os
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torchaudio.functional import resample
from recipes.bigmusic.lightning.embedding_modules import (
    MulanEmbedder,
    LyricsTokenEmbedder,
    LyricsTokenSectionEmbedder,
    LyricsTokenEmbedderV2,
    TagCategoricalEmbedder,
    MultiTagsCategoricalEmbedder,
    LeadsheetTokenEmbedderV2,
    MetadataT5TokenEmbedder,
    T5Embedder,
    BestRQTokenEmbedder,
    BestRQMultiTokenEmbedder,
    XValEmbedder,
    XvalDurationEmbedder,
    IntensityEmbedder,
    SpeakerEmbedder,
    KeyEmbedder,
    TempoLabelEmbedder,
    InstrumentEmbedder,
    AudioKeyEmbedder,
    BpeTokenEmbedder,
)
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
try:
    from recipes.bigmusic.utils.rewards import (
        wer_reward,
        loudness_reward,
        mulan_audio_reward,
        mulan_text_reward,
        chord_reward,
        nonvocal_reward,
        structure_reward,
        chorus_sim_reward,
        chorus_presence_reward,
        audio_metrics_reward,
        intensity_sim_reward,
        semantic_diversity_reward,
        semantic_diversity_sim_reward,
        chroma_reward,
        chroma_sim_reward,
        anchor_points_sim_reward
    )
except Exception as e:
    pass

from multiprocessing.pool import ThreadPool
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Optional
from torchaudio.transforms import Resample
from itertools import zip_longest
import samantha
from samantha.criterion.masked_loss import sequence_mask
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.hparams import DotDict
from samantha.utils.model_metric import ModelMetric
from samantha.utils.flops_profiler import FlopsProfiler
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.musiclm.transforms.audio import to_energy
from recipes.umm.utils.mss import MSSPredictor
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from recipes.mulan.inference.stats.sstk_anchor_points import load_anchor_points
from recipes.bigmusic.lightning.base_modules import TokenBuffer
from mariana.utils.audio.audio_logger import AudioLogger
from cruise.utilities.hdfs_io import hcopy

from panther.custom_ops.torch.flash_attn import unpad_input, pad_input
from samantha.utils.envs import getenv_bool, getenv_int

logger = AudioLogger()

class SemanticEmbModule(torch.nn.Module):
    def __init__(
        self,
        required_modules,
        extra_params=None,
        stage="fit",
    ):
        super().__init__()

        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = f"cuda:{self.local_rank}"

        hidden_size = extra_params['hidden_size']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        lyrics_vocab_size = extra_params.get('lyrics_codebook_size', 2000)
        speaker_vocab_size = extra_params.get('speaker_codebook_size', 10)
        key_vocab_size = extra_params.get('key_codebook_size', 25)  # 1 empty + 24 keys
        tempo_label_vocab_size = extra_params.get('tempo_label_codebook_size', 9)  # 1 empty + 8 labels
        tag_taxonomy_lang = extra_params.get('tag_taxonomy_lang', 'SA')
        tag_dropout_rate = extra_params.get('tag_dropout_rate', 0)
        text_emb_max_len = extra_params.get('text_emb_max_len', 40)
        instrument_vocab_size = extra_params.get('instrument_codebook_size', 155) # 154+1  
        input_embedders = extra_params.get("input_embedders", ["mulan", "lyrics_tokens"])
        mulan_tokenizer_path = extra_params.get('mulan_tokenizer_path', None)


        #Download mulan embeder
        self.local_mulan_tokenizer_path = "/opt/tiger/tokenizer/" #fixed

        if self.local_rank == 0:
            if mulan_tokenizer_path and self.local_mulan_tokenizer_path and (not os.path.isfile(self.local_mulan_tokenizer_path)):
                logger.info(f"Downloading {mulan_tokenizer_path} to {self.local_mulan_tokenizer_path}")
                hcopy(mulan_tokenizer_path, self.local_mulan_tokenizer_path)
                logger.info(f"local_mulan_tokenizer_path: {self.local_mulan_tokenizer_path}")
        bpe_tokenizer_path = extra_params.get('bpe_tokenizer_path', None)

        # Init MulanEmbedder, which is shared by both mulan_text and mulan_music, if it is required in input_embedders
        mulan_emb_names = ["mulan_music", "mulan_text"]
        mulan_embedder = MulanEmbedder(
            input_dim=extra_params.get('mulan_embed_dim', 512),
            embedding_dim=hidden_size,
            add_sos=True,
            mulan_crop=extra_params.get('mulan_crop', True),
            mulan_average=extra_params.get('mulan_average', True),
            dropout=tag_dropout_rate,
            add_none=extra_params.get('mulan_add_cfg', False),
            text_emb_max_len=text_emb_max_len,
            return_hidden_state=extra_params.get('mulan_use_hidden_state', False),
        ) if set(mulan_emb_names).union(input_embedders) else None

        embedder_dict = {}
        for emb_type in input_embedders:
            if emb_type in mulan_emb_names:
                # Support either Mulan audio or text embedding of style_text / on the fly tag
                embedder_dict[emb_type] = mulan_embedder
            elif emb_type == "t5_text_embedder":
                embedder_dict[emb_type] = T5Embedder(
                    embedding_dim=hidden_size,
                    add_sos=True,
                    max_length=text_emb_max_len,
                )
            elif emb_type == "tag_categorical":
                # Read ground truth tags from style_text
                embedder_dict[emb_type] = TagCategoricalEmbedder(
                    embedding_dim=hidden_size,
                    add_sos=True,
                    dropout=tag_dropout_rate,
                    vocab_type=tag_taxonomy_lang,
                )
            elif emb_type == "multitags_categorical":
                # Read ground truth tags from style_text
                embedder_dict[emb_type] = MultiTagsCategoricalEmbedder(
                    embedding_dim=hidden_size,
                    add_sos=True,
                    dropout=tag_dropout_rate,
                    vocab_type=tag_taxonomy_lang,
                )
            elif emb_type == "speaker_id":
                embedder_dict[emb_type] = SpeakerEmbedder(
                    vocab_size=speaker_vocab_size, 
                    embedding_dim=hidden_size, 
                    add_sos=True)
            elif emb_type == "instrument":
                embedder_dict[emb_type] = InstrumentEmbedder(
                    vocab_size=instrument_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True, add_eos=True, 
                    is_varlen=extra_params.get('varlen_instrument_prefix', True),
                )
            elif emb_type == 'tempo':
                embedder_dict[emb_type] = XValEmbedder(
                    embedding_dim=hidden_size,
                    max_value=200,   # the max possible bpm, may need to update
                    norm_max_value=5,
                )
            elif emb_type == "lyrics_tokens":
                embedder_dict[emb_type] = LyricsTokenEmbedder(
                    vocab_size=lyrics_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=True,
                    is_varlen=extra_params['varlen_lyrics_prefix']
                )
            elif emb_type == "lyrics_tokens_v2":
                embedder_dict[emb_type] = LyricsTokenEmbedderV2(
                    vocab_size=lyrics_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=True,
                    is_varlen=extra_params['varlen_lyrics_prefix']
                )
            elif emb_type == "bpe":
                embedder_dict[emb_type] = BpeTokenEmbedder(
                    tokenizer_path=bpe_tokenizer_path,
                    device=self.device,
                )
            elif emb_type == "lyrics_tokens_section":
                embedder_dict[emb_type] = LyricsTokenSectionEmbedder(
                    vocab_size=lyrics_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=True,
                    is_varlen=extra_params['varlen_lyrics_prefix']
                )
            elif emb_type == "duration":
                embedder_dict[emb_type] = XvalDurationEmbedder(
                    embedding_dim=hidden_size,
                    max_duration=240,   # the max possible duration, may need to update
                    max_timestamp=5,
                )
            elif emb_type == "leadsheet_tokens":
                leadsheet_vocab_size = extra_params['leadsheet_codebook_size']
                embedder_dict[emb_type] = LeadsheetTokenEmbedderV2(
                    vocab_size=leadsheet_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True
                )
            elif emb_type == "audio_key_token":
                audio_key_codebook_size = extra_params['audio_key_codebook_size']
                embedder_dict[emb_type] = AudioKeyEmbedder(
                    vocab_size=audio_key_codebook_size,
                    embedding_dim=hidden_size,
                    add_sos=True
                )
            elif emb_type == "audio_prompt_token":
                # Here we use ground truth audio as prefix, will use target_embedder as embedder
                embedder_dict[emb_type] = None
            elif emb_type == "intensity":
                embedder_dict[emb_type] = IntensityEmbedder(
                    decimals=extra_params["intensity_decimals"],
                    embedding_dim=hidden_size,
                    intensity_hz=extra_params.get("intensity_hz", 1),
                )
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
        input_embedders = nn.ModuleDict(embedder_dict)

        pattern = extra_params.get("multi_token_pattern", None)
        group = extra_params.get("multi_token_num", 1)
        if pattern:
            target_embedder = BestRQMultiTokenEmbedder(
                    vocab_size=semantic_codebook_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=True,
                    pattern=pattern,
                    group=group,
                    null_placeholder=extra_params.get("null_placeholder", True),
                    encoder=extra_params.get("multi_token_encoder", "fc"),
                    decoder=extra_params.get("multi_token_decoder", {'model_type': 'fc'},),
                    semantic_codebook_depth=extra_params.get("semantic_codebook_depth", 1),
                )
        else:
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )

        self.mulan_music_counter = 0
        self.mulan_text_counter = 0

        prediction_dict = {}
        prediction_weights = {}
        prediction_heads = extra_params.get("prediction_heads", {})
        if isinstance(prediction_heads, list):  # backward compat
            prediction_heads = {x: 1.0 for x in prediction_heads}
        for pred_type, pred_wt in prediction_heads.items():
            prediction_weights[pred_type] = pred_wt
            if pred_type == "intensity":
                prediction_dict[pred_type] = nn.Linear(
                    hidden_size,
                    self.input_embedders["intensity"].intensity_vocab_size,
                    bias=False,
                )
            elif pred_type == "beat":
                # Last dim is for magnitude, others are for beat IDs
                prediction_dict[pred_type] = nn.Linear(
                    hidden_size,
                    self.input_embedders["beat"].beat_vocab_size + 1,
                    bias=False,
                )
            else:
                raise ValueError(f"Unknown pred type: {pred_type}")
        self.prediction_heads = nn.ModuleDict(prediction_dict)
        self.prediction_weights = prediction_weights

        # inherit from recipes.bigmusic.lightning.base_modules.BaseContinuousEmbedModule
        self.input_embedders = input_embedders
        self.target_embedder = target_embedder
        # TODO (shuo): apply weight init here
        # self.input_embedders.apply(self.model._init_weights)
        # self.target_embedder.apply(self.model._init_weights)

        # inherit from recipes.bigmusic.lightning.base_modules.BaseModule
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        self.required_modules = required_modules
        self.stage = stage
        self.text_codebook_size = extra_params.get("text_codebook_size", 0)
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

    def get_dummy_token_ids_length(self, embeds):
        batch_size, seq_len, _ = embeds.shape        
        token_ids = torch.zeros((batch_size, seq_len)).long().to(self.device)
        token_length = torch.zeros((batch_size), device=self.device) + seq_len
        return {
            'token_embeds': embeds,
            'token_ids': token_ids,
            'token_length': token_length,
        }

    def prepare_mulan_music_inputs(self, batch, mulan_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)

        audio_dropout_rate = self.extra_params.get("mulan_music_dropout_rate", 0)
        enable_dropout = self.training and (random.random() < audio_dropout_rate)

        if 'style_audio' in conditions and not enable_dropout:
            target_duration = self.infer_target_duration(batch)
            if self.mulan_music_counter < 10:
                print(f"target_duration: {target_duration}")
            assert "style_audio" in batch or "target_audio" in batch
            style_audio = batch.get('style_audio', batch.get('target_audio'))
            if self.training:
                embeds = mulan_embedder.embed(
                    self.requires,
                    style_audio.to(self.device),
                    with_sos=True,
                    data_type='music'
                )
            else:  
                if batch_size == 1: 
                    # Support multiple audio prompts fusion during inference.
                    style_audio = style_audio[0]
                    if not isinstance(style_audio, list): style_audio = [style_audio]
                    embeds = [mulan_embedder.embed(
                        self.requires,
                        audio.to(self.device),
                        with_sos=True,
                        data_type='music'
                    ) for audio in style_audio]
                    embeds_stack = torch.stack(embeds)
                    embeds = torch.mean(embeds_stack, dim=0)                    
                else:
                    # This is for validation
                    embeds = mulan_embedder.embed(
                        self.requires,
                        style_audio,
                        with_sos=True,
                        data_type='music'
                    )
        else:
            # adding SOS token no matter what so that all parameters get used
            embeds = mulan_embedder.get_sos_embed(batch_size)
        if self.mulan_music_counter < 10:
            #print(f"mulan_music_emb: {embeds.shape}")
            self.mulan_music_counter += 1
        return self.get_dummy_token_ids_length(embeds)

    def prepare_mulan_text_inputs(self, batch, mulan_embedder):

        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)

        if 'freeform_text' in conditions:
            freeform_texts = batch.get("freeform_text", [None] * batch_size)
            embeds = mulan_embedder.embed(
                self.requires,
                freeform_texts,
                with_sos=True,
                data_type='text'
            )
        else:
            # adding SOS token no matter what so that all parameters get used
            embeds = mulan_embedder.get_sos_embed(batch_size)
        if self.mulan_text_counter < 10:
            #print(f"mulan_text_emb: {embeds.shape}")
            self.mulan_text_counter += 1
        return self.get_dummy_token_ids_length(embeds)
    
    def prepare_lyrics_inputs(self, batch, lyrics_embedder):
        # By default, add eos to separate lyrics from audio.
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)

        lyrics_embeds, lyrics_tokens, lyrics_token_length = lyrics_embedder.prepare_embed(batch, batch_size, conditions)

        return {
            'token_embeds': lyrics_embeds,
            'token_ids': lyrics_tokens,
            'token_length': lyrics_token_length
        }
    
    def prepare_bpe_inputs(self, batch, lyrics_embedder):
        lyrics_tokens, lyrics_token_length = lyrics_embedder.get_tokens(batch)

        # add sos and eos id
        lyrics_tokens = [torch.cat(
            [
                torch.tensor([lyrics_embedder.sos_id]).to(lyrics_token.device), 
                lyrics_token, 
                torch.tensor([lyrics_embedder.eos_id]).to(lyrics_token.device),
            ]) for lyrics_token in lyrics_tokens]
        lyrics_token_length += 2  # +2 for eos and sos

        return {
            'token_embeds': [],
            'token_ids': lyrics_tokens,
            'token_length': lyrics_token_length
        }

    def prepare_audio_key_inputs(self, batch, embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'audio_key_token' in conditions:
            embeds = embedder.embed(
                self.requires,
                batch['audio_key_token'].to(self.device),
                with_sos=True,
            )
        else:
            embeds = embedder.get_sos_embed(batch_size)
        return self.get_dummy_token_ids_length(embeds)

    def prepare_duration_inputs(self, batch, embedder, dropout=0.1):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'duration' in conditions:                       
            duration = self.infer_target_duration(batch)
            if self.training and random.random() < dropout:
                duration = torch.zeros((batch_size), device=self.device)
        else:
            duration = torch.zeros((batch_size), device=self.device)
        embeds, _, _ = embedder.embed(duration)
        return self.get_dummy_token_ids_length(embeds)

    def prepare_t5_text_inputs(self, batch, t5_text_embedder):
        # For vocal dataloader output
        if 'style_category' not in batch:
            assert 'style_text' in batch
            batch["style_category"] = batch['style_text']
        # TODO (qq) prepare text labels from instrumental data loader
        batch["style_category"] = [", ".join(x) for x in batch["style_category"]]
        embeds = t5_text_embedder.embed(
            self.requires,
            batch['style_category'],
            with_sos=True,
        )
        return self.get_dummy_token_ids_length(embeds)

    def prepare_categorical_inputs(self, batch, categorical_embedder):
        if 'style_category' not in batch:
            assert 'style_text' in batch
            batch["style_category"] = batch['style_text']
        embeds = categorical_embedder.embed(
            self.requires,
            batch['style_category'],
            with_sos=True,
        )
        return self.get_dummy_token_ids_length(embeds)

    def _prepare_one_frame_inputs(self, batch, embedder, condition: str):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)        
        if condition in conditions and condition in batch:
            if isinstance(batch[condition], list):
                batch[condition] = torch.tensor(batch[condition])
            if batch[condition].dim() == 1:
                batch[condition] = batch[condition].unsqueeze(1)
            embeds = embedder.embed(
                self.requires, 
                batch[condition].to(self.device), 
                with_sos=False)     # This ensures only one frame is used for speaker ID or placeholder
        else:
            embeds = embedder.get_sos_embed(batch_size)
        return self.get_dummy_token_ids_length(embeds)        

    def prepare_speaker_inputs(self, batch, speaker_embedder):
        return self._prepare_one_frame_inputs(batch, speaker_embedder, 'speaker_id')

    def prepare_key_inputs(self, batch, key_embedder):
        return self._prepare_one_frame_inputs(batch, key_embedder, 'key')

    def prepare_tempo_label_inputs(self, batch, tempo_label_embedder):
        return self._prepare_one_frame_inputs(batch, tempo_label_embedder, 'tempo_label')
    
    def prepare_continous_value_inputs(self, batch, embedder, key='tempo', dropout=0.1):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if key in conditions:                       
            cvalue = batch[key]
            if self.training and random.random() < dropout:
                cvalue = -1 * torch.ones((batch_size), device=self.device)
        else:
            cvalue = -1 * torch.ones((batch_size), device=self.device)
        embeds, _, _ = embedder.embed(cvalue)
        return self.get_dummy_token_ids_length(embeds)

    def prepare_instrument_inputs(self, batch, embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)        
        condition = 'instrument'

        if condition in conditions and condition in batch and condition+'_length' in batch:
            token_ids, token_embeds, token_length = embedder.prepare_embed(
                self.requires, 
                batch[condition].to(self.device),
                batch[condition+"_length"].to(self.device), 
                with_sos=True, with_eos=True)
            return {
                'token_embeds': token_embeds,
                'token_ids': token_ids,
                'token_length': token_length
            }
        else:
            token_embeds = torch.cat((embedder.get_sos_embed(batch_size), 
                                    embedder.get_eos_embed(batch_size)), 1)
            return self.get_dummy_token_ids_length(token_embeds) 

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
            with_sos=True,
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
            if emb_type == "mulan_music":
                emb_inputs = self.prepare_mulan_music_inputs(batch, embedder)
            elif emb_type == "mulan_text":
                emb_inputs = self.prepare_mulan_text_inputs(batch, embedder)
            elif emb_type == "t5_text_embedder":                
                emb_inputs = self.prepare_t5_text_inputs(batch, embedder)
            elif emb_type == "tag_categorical" or emb_type == "multitags_categorical":
                emb_inputs = self.prepare_categorical_inputs(batch, embedder)
            elif emb_type in ["lyrics_tokens", "lyrics_tokens_section", "lyrics_tokens_v2"]:
                emb_inputs = self.prepare_lyrics_inputs(batch, embedder)
            elif emb_type == "bpe":
                emb_inputs = self.prepare_bpe_inputs(batch, embedder)
            elif emb_type == "speaker_id":
                emb_inputs = self.prepare_speaker_inputs(batch, embedder)
            elif emb_type == "key":
                emb_inputs = self.prepare_key_inputs(batch, embedder)
            elif emb_type == "tempo_label":
                emb_inputs = self.prepare_tempo_label_inputs(batch, embedder)
            elif emb_type == "audio_key_token":
                emb_inputs = self.prepare_audio_key_inputs(batch, embedder)
            elif emb_type == "duration":
                emb_inputs = self.prepare_duration_inputs(batch, embedder)
            elif emb_type == "instrument":
                emb_inputs = self.prepare_instrument_inputs(batch, embedder)
            elif emb_type == "tempo":
                emb_inputs = self.prepare_continous_value_inputs(batch, embedder, key=emb_type)
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
            prefix_inputs.append(emb_inputs)

            batch["inputs_embeds_span"][emb_type] = []
            for emb in emb_inputs['token_embeds']:
                en_idx = st_idx + emb.shape[0]
                batch["inputs_embeds_span"][emb_type].append((st_idx, en_idx))
                st_idx = en_idx

            #print('-------')
            #print(emb_type, emb_inputs['token_embeds'])
    
        # Append audio_prompt to prefix_inputs for audio continuation
        if (not self.training and
            "audio_prompt_token" in self.infer_conditions(batch)):
            # Key "audio_prompt" has to be in the batch to make "audio_prompt_token" work.
            # Audio prompting only works for varlen with bs=1. When audio_prompt is not given,
            # batch["audio_prompt"] is [None], otherwise it is a tensor of shape [1, 1, n]            
            if isinstance(batch["audio_prompt"], list):
                batch["audio_prompt"] = batch["audio_prompt"][0][0]
            emb_inputs = self.prepare_audio_prompt_token_inputs(batch)
            prefix_inputs.append(emb_inputs)
        return prefix_inputs

    def prepare_target_inputs(self, batch, chunk_dur=60, sample_rate=24000, slice_method='even', get_tags_from_umm=False):
        # Prepare target ids
        enable_batch = getenv_bool("USE_BATCH_IN_PREPARE_TARGET_INPUTS", True)
        random_padding_second = getenv_int("USE_RANDOM_PADDING_SECOND_IN_PREPARE_TARGET_INPUTS", 0)
        if "target_token_ids" not in batch and not enable_batch:
            # Experiment with UMM2
            if slice_method == "UMM2_30s":
                chunk_dur = 30
                slice_method = "max"

            all_target_ids = []
            get_tags_from_umm = []
            for target_audio, audio_length in zip(batch['target_audio'], batch['audio_length']):
                target_audio = target_audio.unsqueeze(0)[..., :audio_length] # 1,[c],T
                target_audio_length = float(audio_length) / sample_rate 
                if random_padding_second > 0:
                    padding_length = round(random.uniform(0, random_padding_second), 1)
                    padding_length_in_sample = int(padding_length * sample_rate)
                    padding_length_in_token = int(padding_length * 25)
                    audio_length += padding_length_in_sample
                    target_audio_length += padding_length
                    target_audio = F.pad(target_audio, (0, padding_length_in_sample))

                if target_audio_length <= chunk_dur or slice_method not in ["even", "max"]:
                    if get_tags_from_umm:
                        target_id, tag_logits = self.target_embedder.tokenize_and_tag(self.requires, target_audio, with_sos=False, with_eos=False)
                        target_id = target_id.to(self.device)  
                    wav_length = torch.LongTensor([target_audio.shape[-1]]).to(target_audio.device)
                    target_ids = self.target_embedder.tokenize(
                        self.requires, 
                        target_audio, 
                        with_sos=False, 
                        with_eos=False,
                        wav_length=wav_length
                        ).to(self.device)
                    
                else:
                    target_ids = []
                    tag_logits = []
                    n_samples = audio_length
                    # evenly slice the audio in a way that the `chunk_size` is as close as possible to `chunk_dur`
                    if slice_method == 'even':
                        chunk_num = math.ceil(target_audio_length / chunk_dur)
                        chunk_size = math.ceil(target_audio_length / chunk_num)
                    # always slice the audio with the maximum `chunk_dur`, combine the tail audio < 1s
                    elif slice_method == 'max':
                        chunk_size = chunk_dur
                    else:
                        raise NotImplementedError(f"{slice_method} is not implemented as audio slice method")

                    st = 0
                    while st < n_samples:
                        _st, _et = int(st*sample_rate), int((st+chunk_size)*sample_rate)
                        # merge the tail if the remaining chunk is too short (< 1s)
                        if n_samples - _et < sample_rate * 1 or target_audio[...,_et:].shape[-1] < sample_rate * 1:
                            _et = n_samples
                        if get_tags_from_umm:
                            _target_id, _tag_logits = self.target_embedder.tokenize_and_tag(self.requires,target_audio[...,_st:_et])
                            _target_id = _target_id.to(self.device)
                            tag_logits.append(_tag_logits)
                        else:
                            _target_audio = target_audio[...,_st:_et]
                            wav_length = torch.LongTensor([_target_audio.shape[-1]]).to(_target_audio.device)
                            _target_id = self.target_embedder.tokenize(
                                self.requires, 
                                _target_audio, 
                                with_sos=False, 
                                with_eos=False,
                                wav_length=wav_length,
                                ).to(self.device)
                        target_ids.append(_target_id)                
                        if _et >= n_samples:
                            break
                        st += chunk_size
                    target_ids = torch.cat(target_ids, dim=-1)
                    # if random_padding_second > 0:
                    #     target_ids = target_ids[:,:-padding_length_in_token]
                    # if get_tags_from_umm:
                    #     combined_tag_logits = {}
                    #     for d in tag_logits:
                    #         for key, value in d.items():
                    #             combined_tag_logits[key] = combined_tag_logits.get(key, 0.0) + value
                    #     tag_logits = {k: v / len(combined_tag_logits) for k, v in combined_tag_logits.items()}

                all_target_ids.append(target_ids.squeeze(0))
                if get_tags_from_umm:
                    get_tags_from_umm.append(tag_logits)

            target_ids = torch.nn.utils.rnn.pad_sequence(all_target_ids, True, 0)
            # batch['target_tokens_length'] = torch.tensor([len(target_id) for target_id in all_target_ids], device=self.device, dtype=torch.long)
            # TODO: get_tags_from_umm

        elif "target_token_ids" not in batch and enable_batch:
            target_audio_length = float(batch['target_audio'].shape[-1]) / sample_rate

            # Experiment with UMM2
            if slice_method == "UMM2_30s":
                chunk_dur = 30
                slice_method = "max"

            if target_audio_length <= chunk_dur or slice_method not in ["even", "max"]:
                if get_tags_from_umm:
                    target_id, tag_logits = self.target_embedder.tokenize_and_tag(self.requires, batch['target_audio'], with_sos=False, with_eos=False)
                    target_id = target_id.to(self.device)
                wav_length = torch.squeeze(batch['audio_length'].to(batch['target_audio'].device))
                target_ids = self.target_embedder.tokenize(
                        self.requires, 
                        batch['target_audio'], 
                        with_sos=False, 
                        with_eos=False,
                        wav_length=wav_length,
                        ).to(self.device)
            else:
                target_ids = []
                tag_logits = []
                n_samples = batch['target_audio'].shape[-1]
                # evenly slice the audio in a way that the `chunk_size` is as close as possible to `chunk_dur`
                if slice_method == 'even':
                    chunk_num = math.ceil(target_audio_length / chunk_dur)
                    chunk_size = math.ceil(target_audio_length / chunk_num)
                # always slice the audio with the maximum `chunk_dur`, combine the tail audio < 1s
                elif slice_method == 'max':
                    chunk_size = chunk_dur
                else:
                    raise NotImplementedError(f"{slice_method} is not implemented as audio slice method")

                st = 0
                while st < n_samples:
                    _st, _et = int(st*sample_rate), int((st+chunk_size)*sample_rate)
                    # merge the tail if the remaining chunk is too short (< 1s)
                    if n_samples - _et < sample_rate * 1 or batch['target_audio'][...,_et:].shape[-1] < sample_rate * 1:
                        _et = n_samples
                    if get_tags_from_umm:
                        _target_id, _tag_logits = self.target_embedder.tokenize_and_tag(self.requires, batch['target_audio'][...,_st:_et])
                        _target_id = _target_id.to(self.device)
                        tag_logits.append(_tag_logits)
                    else:
                        target_audio = batch['target_audio'][...,_st:_et]
                        wav_length = []
                        for audio_length in batch['audio_length']:
                            if _st > audio_length:
                                wav_length.append(0)
                            else:
                                valid_et = min(_et, audio_length)
                                wav_length.append(valid_et - _st)
                        wav_length = torch.LongTensor(wav_length).to(target_audio.device)
                        _target_id = self.target_embedder.tokenize(
                            self.requires, 
                            target_audio, 
                            with_sos=False, 
                            with_eos=False,
                            wav_length=wav_length,
                            ).to(self.device)
                    target_ids.append(_target_id)                
                    if _et >= n_samples:
                        break
                    st += chunk_size
                target_ids = torch.cat(target_ids, dim=-1)
                if get_tags_from_umm:
                    combined_tag_logits = {}
                    for d in tag_logits:
                        for key, value in d.items():
                            combined_tag_logits[key] = combined_tag_logits.get(key, 0.0) + value
                    tag_logits = {k: v / len(combined_tag_logits) for k, v in combined_tag_logits.items()}
        else:
            target_ids = batch["target_token_ids"]
        target_lengths = batch['target_tokens_length'].to(self.device)

        # TODO: process tags_logits
        target_tags = self.target_embedder.convert_tag_logits_to_tags(tag_logits) if get_tags_from_umm else None

        # add sos and eos id
        if target_ids.ndim == 2:
            target_ids = F.pad(target_ids, (1, 1))
            eos_indices = (target_lengths + 1).unsqueeze(1) # set last index to EOS
        else:
            target_ids = F.pad(target_ids, (0, 0, 1, 1))
            eos_indices = (target_lengths + 1).unsqueeze(1).unsqueeze(1) # set last index to EOS
        target_ids[:, 0] = self.target_embedder.sos_id
        target_ids.scatter_(1, eos_indices, self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        if self.extra_params.get("multi_token_pattern", None):
            # examine the target length validity 
            # @qinxin: I don't know why there exists super-short target token length (=1), which should not be allowed in dataloader, just a quick fix
            new_target_id = []
            for b in range(target_ids.shape[0]):
                eos_index = torch.where(target_ids[b] == self.target_embedder.eos_id)[0]
                if target_lengths[b] < 10 and eos_index < 10:
                    print(f"Warning: target length is too short: sample{b}: {target_lengths[b]}/{eos_index}, skip it")
                    print(batch['target_tokens_length'][b], batch['target_audio'][b].shape)
                else:
                    new_target_id.append(target_ids[b])
            target_ids = torch.stack(new_target_id, dim=0)
            target_embeds, target_group_token_ids, target_embed_lengths = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
            target_ids_unpad = [self.target_embedder.decode_target_id(target_group_token_ids[b], delete_null=False, add_sos=True) for b in range(target_embeds.shape[0])]
            target_lengths = torch.LongTensor([target_ids_unpad[b].shape[0] for b in range(target_embeds.shape[0])]).to(self.device)
        else:
            # TODO (shuo): remove target_embedder in bpe training
            target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
            # TODO (shuo): apply text_codebook_size in MTP branch
            if self.text_codebook_size > 0:
                target_ids += self.text_codebook_size
            target_ids_unpad = unpad_sequence(target_ids, target_lengths, batch_first=True)
            target_embeds = unpad_sequence(target_embeds, target_lengths, batch_first=True)
            target_embed_lengths = target_lengths
        
        # TODO (qq) return a list. placeholder for intermediate leadsheet tokens.
        return [{
            'token_embeds': target_embeds,
            'token_ids': target_ids_unpad,
            'token_ids_batched': target_ids,
            'token_length': target_lengths,
            'token_embed_length': target_embed_lengths,
            'target_tags': target_tags
        }]

    def prepare_training_inputs(self, batch):
        # Concate Prefix and Target tokens
        prefix_inputs = self.prepare_prefix_inputs(batch)
        target_inputs = self.prepare_target_inputs(
            batch,
            self.extra_params.get("tokenizer_chunk_duration", 60),
            self.extra_params.get("sample_rate", 24000),
            self.extra_params.get("token_slice_method", "even"),
        )
        # Concat prefix and target in to a sequence
        token_ids = zip(*[i['token_ids'] for i in prefix_inputs + target_inputs])
        token_ids = [torch.cat(t, dim=0) for t in token_ids]
        token_ids_pad = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        batch_seq_length = token_ids_pad.shape[1]
        if not self.is_token_input():
            token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs + target_inputs])
            token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
            token_embeds_pad = pad_sequence(token_embeds, batch_first=True, padding_value=0)

        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0)
        target_length = torch.vstack([i['token_length'] for i in target_inputs]).sum(dim=0)
            
    
        # Get masks
        input_loss_mask = sequence_mask(prefix_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = sequence_mask(prefix_length + target_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = target_loss_mask.bool() ^ input_loss_mask.bool() # remove input targets from target mask

        out_dict = {
            'token_ids': token_ids_pad,
            'input_loss_mask': input_loss_mask,
            'target_loss_mask': target_loss_mask,
            'prefix_inputs': prefix_inputs,
            'target_inputs': target_inputs,
            'prefix_length': prefix_length,
            'target_length': target_length,
            'token_length': prefix_length + target_length,
        }
        if not self.is_token_input():
            out_dict['token_embeds'] = token_embeds_pad
        return out_dict

    def load_from_pretrained(self, pretrained_path=None):
        """
        Patch the parent method to translate single tag to multitag embeddings.
        Correctly handle SOS and EOS for different sizes of categorical embeddings.
        TODO: Remove this method after we completely switch to unified vocab.
        """
        from pathlib import Path
        from recipes.musiclm.utils.dist import local_zero_first
        from recipes.diffusion.utils.utils import download_checkpoint

        print('Loading pre-trained model from checkpoint', pretrained_path)
        with local_zero_first():
            cache_dir = Path(self.extra_params.get('cache_dir', '.pretrain_cache'))
            pretrained_path = download_checkpoint(pretrained_path, cache_dir=cache_dir)
        state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        model_state_dict = self.state_dict()

        ##########################################################
        #                          Patch
        ##########################################################
        # state_dict: Pretrained model's state
        # model_state_dict: Current model's state

        # Don't use two tag embeds at the same time (it's probably ok, but just in case)
        assert not all(emb in self.input_embedders for emb in ["tag_categorical", "multitags_categorical"])

        # Trigger implicit embedder conversion when the pretrained model uses tag_categorical or multitags_categorical,
        # and the current model uses multitags_categorical with a unified vocab type.
        emb_prefixs = [
            "input_embedders.tag_categorical",
            "input_embedders.multitags_categorical",
        ]
        for prefix in emb_prefixs:
            if any(k.startswith(prefix) for k in state_dict):
                pretrain_emb_path = prefix
                break
        else:
            pretrain_emb_path = None
        
        if (
            pretrain_emb_path is not None and
            "multitags_categorical" in self.input_embedders and
            "unified" in self.input_embedders["multitags_categorical"].vocab_type
        ):
            print("Converting categorical embedding...")
            current_emb_path = "input_embedders.multitags_categorical"
            emb_weight_path = "embedder.weight"
            pretrain_emb_weight_path = f"{pretrain_emb_path}.{emb_weight_path}" 
            current_emb_weight_path = f"{current_emb_path}.{emb_weight_path}" 
            # To prevent missing embedding information, the current model must have an equal or larger vocab
            pretrain_vocab_size, pretrain_hidden = state_dict[pretrain_emb_weight_path].shape
            current_vocab_size, current_hidden = model_state_dict[current_emb_weight_path].shape
            assert pretrain_vocab_size <= current_vocab_size
            assert pretrain_hidden == current_hidden

            # Expand pretrain's embedder size to match the current one
            expanded_emb_weights = model_state_dict[current_emb_weight_path].clone()
            expanded_emb_weights[:pretrain_vocab_size, :] = state_dict[pretrain_emb_weight_path]
            state_dict[current_emb_weight_path] = expanded_emb_weights

            # Move pretrain's sos and eos to the end by swapping the embeddings
            emb: MultiTagsCategoricalEmbedder = self.input_embedders["multitags_categorical"]
            current_sos_id = emb.sos_id
            current_eos_id = emb.eos_id
            current_seos_token_ids = list(filter(None, [current_sos_id, current_eos_id]))
            pretrain_seos_token_ids = [pretrain_vocab_size-2, pretrain_vocab_size-1][-len(current_seos_token_ids):]
            for pretrain_token_id, current_token_id in zip(pretrain_seos_token_ids, current_seos_token_ids):
                curr = state_dict[current_emb_weight_path][current_token_id].clone()
                pret = state_dict[current_emb_weight_path][pretrain_token_id]
                state_dict[current_emb_weight_path][current_token_id] = pret
                state_dict[current_emb_weight_path][pretrain_token_id] = curr

            # `_extra_state` will be dropped automatically because it's attached to tag_categorical,
            # which is not in the current model.
        ##########################################################
        #                       Patch End
        ##########################################################

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
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict =  super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # HACK: mulan_text and mulan_music use the same embedder, so we only save one of them.
        # Because alias weight would make LSDP crash.
        for k in list(state_dict.keys()):
            if k.startswith('_lsdp_wrapped.module.emb.input_embedders.mulan_text'):
                state_dict.pop(k)
        return state_dict

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

    # TODO: make this into a static method (vibertthio)
    def infer_conditions(self, batch):
        if type(batch["conditions"]) == list:
            assert (
                len(set(list(map(tuple, batch["conditions"])))) == 1
            ), "Make sure that all conditions in the batch are the same"
            conditions = batch['conditions'][0].split(',')
        else:
            conditions = batch['conditions'].split(',')
        return conditions


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

    # flag to distinguish token and emb branch
    def is_token_input(self):
        return len(self.input_embedders) == 1 and any('bpe' == emb_type for emb_type, _ in self.input_embedders.items())


    @torch.no_grad()
    def predict(self, batch, hp, beam=1, rl_training=False):
        if self.is_token_input():
            return self.predict_token(batch, hp)
        else:
            return self.predict_emb(batch, hp, beam, rl_training)
            
    @torch.no_grad()
    def predict_token(self, batch, hp, beam=1, rl_training=False):
        use_controller_cfg = hp.get('use_controller_cfg', False)
        assert use_controller_cfg == False, "predict_token not support use_controller_cfg"
        assert rl_training == False, "predict_token not support rl_training"
        assert beam == 1, "predict_token not support beam > 1"
        
        prefix_inputs = self.prepare_prefix_inputs(batch)

        token_ids = zip(*[i['token_ids'] for i in prefix_inputs])
        token_ids = [torch.cat(t, dim=0) for t in token_ids]
        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=0)
        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0).int()

        batch["prefix_length"] = prefix_length

        # (shuo): copy from BaseContinuousEmbedModule.predict()
        batch_size, seq_len = token_ids.size()
        original_batch_size = batch_size  

        sos_tokens = self.target_embedder.get_sos_token(batch_size)
        sos_tokens += self.text_codebook_size    
        model_input = { 
            "token_ids": torch.cat([token_ids, sos_tokens], dim=1)
            }  
        model_input['cfg_batch_size'] = None
        model_input['original_batch_size'] = original_batch_size
        model_input['prefix_length'] = prefix_length
        model_input['exclude_ids'] = [self.target_embedder.eos_id + self.text_codebook_size]

        return model_input
    
    @torch.no_grad()
    def predict_emb(self, batch, hp, beam=1, rl_training=False):
        is_varlen_prefix = self.extra_params.varlen_lyrics_prefix        
        # if is_varlen_prefix: assert self.infer_batch_size(batch) == 1

        frame_rate = self.extra_params.semantic_frame_rate
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
        emit_eos_thresh_secs = hp.get('emit_eos_thresh_secs', 0)
        skip_sos = hp.get('skip_sos', False)
        stop_eos = hp.get('stop_eos', False)
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

        def _init_model_input():
            if skip_sos:
                # If already have sos embedding in prefix prompt, we can skip it
                return { "inputs_embeds": torch.cat([inputs_embeds,], dim=1) }
            else:
                return { "inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1) }     

        model_input = _init_model_input()
        if rl_training:
            rl_model_input = _init_model_input()


        if use_controller_cfg:
            model_input['cfg_batch_size'] = cfg_batch_size
        else:
            model_input['cfg_batch_size'] = None

        model_input['original_batch_size'] = original_batch_size
        model_input['prefix_length'] = prefix_length
        model_input['exclude_ids'] = exclude_ids

        return model_input


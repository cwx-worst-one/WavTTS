import os
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torchaudio.functional import resample
from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
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
    InstrumentEmbedder,
    AudioKeyEmbedder,
)
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
from recipes.bigmusic.utils.duration_utils import generate_duration
from recipes.bigmusic.callbacks.wer_metrics import run_asr_lyrics_sa_online
from samantha.utils.model_metric import ModelMetric
from samantha.utils.flops_profiler import FlopsProfiler

from recipes.bigmusic.utils.rewards import (
    wer_reward,
    loudness_reward,
    mulan_audio_reward,
    mulan_text_reward,
    mir_tag_reward,
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
    chroma_temporal_reward,
    anchor_points_sim_reward,
    mulan_temporal_reward,
    umm_reward,
    upload_wavs_to_tos,
)

from multiprocessing.pool import ThreadPool
import math
import pathlib
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Dict, List, Optional
from torchaudio.transforms import Resample
from itertools import zip_longest

from samantha.criterion.masked_loss import sequence_mask
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.hparams import DotDict
from samantha.utils.model_metric import ModelMetric
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.envs import getenv_int, getenv_bool
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.musiclm.transforms.audio import to_energy
from recipes.umm.utils.mss import MSSPredictor
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from recipes.mulan.inference.stats.sstk_anchor_points import load_anchor_points
from apps.bigmusic.umm.diffusion.requires.model_initializer import run_diffusion_vocoder_batch
from recipes.bigmusic.datasets.utils.zh_meta import infer_freeform_text_from_style_text


class SemanticModule(BaseContinuousEmbedModule):
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

        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()
        self.log_counter = 0
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

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        # mfu metric
        self.metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

        pretrained_path = self.extra_params.get('pretrained_path')
        if stage == "fit" and pretrained_path is not None:
            self.load_from_pretrained(pretrained_path)

    def infer_target_duration(self, batch):
        if "duration" in batch:
            target_duration = batch["duration"]
        elif "target_audio" in batch:
            target_lengths = batch['target_tokens_length'].to(self.device)
            target_duration = target_lengths / float(self.extra_params.semantic_frame_rate)
        else:
            target_duration = None
        return target_duration

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
            # print(f"mulan_music_emb: {embeds.shape}")
            self.mulan_music_counter += 1
        return self.get_dummy_token_ids_length(embeds)

    def prepare_mulan_text_inputs(self, batch, mulan_embedder):

        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)

        if 'freeform_text' in conditions:
            freeform_texts = batch.get("freeform_text", [None] * batch_size)
            embeds = torch.cat(
                [
                    mulan_embedder.embed(
                        self.requires,
                        [freeform_text],
                        with_sos=True,
                        data_type='text'
                    ) for freeform_text in freeform_texts
                ], dim=0
            )
        else:
            # adding SOS token no matter what so that all parameters get used
            embeds = mulan_embedder.get_sos_embed(batch_size)
        if self.mulan_text_counter < 10:
            # print(f"mulan_text_emb: {embeds.shape}")
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
        if self.log_counter < 1:
            print(batch)
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
                emb_inputs = self.prepare_continous_value_inputs(batch, embedder, key=emb_type, dropout=0.0)    # @qinxin: tempo dropout will be done in dataloader
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
            if self.log_counter < 1:
                print(f"{emb_type} emb input shape: ", emb_inputs['token_embeds'][0].shape)
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
            if self.log_counter < 1:
                print(f"'audio_prompt' emb input shape: ", emb_inputs['token_embeds'][0].shape)
            prefix_inputs.append(emb_inputs)
        self.log_counter += 1
        return prefix_inputs

    def prepare_target_inputs(self, batch, chunk_dur=60, sample_rate=24000, slice_method='even', get_tags_from_umm=False):
        # Prepare target ids
        target_audio_length = float(batch['target_audio'].shape[-1]) / sample_rate

        # Experiment with UMM2
        if slice_method == "UMM2_30s":
            chunk_dur = 30
            slice_method = "max"

        if target_audio_length <= chunk_dur or slice_method not in ["even", "max"]:
            if get_tags_from_umm:
                target_id, tag_logits = self.target_embedder.tokenize_and_tag(self.requires, batch['target_audio'], with_sos=False, with_eos=False)
                target_id = target_id.to(self.device)                           
            target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False).to(self.device)
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
                    _target_id = self.target_embedder.tokenize(self.requires, batch['target_audio'][...,_st:_et], with_sos=False, with_eos=False).to(self.device)
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
            target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
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
        target_inputs = self.prepare_target_inputs(batch, self.extra_params.get("tokenizer_chunk_duration", 60), 
                                                   self.extra_params.get("sample_rate", 24000),
                                                   self.extra_params.get("token_slice_method", "even"),
                                                   self.extra_params.get("get_tags_from_umm", False))
        if self.extra_params.get("get_tags_from_umm", False):
            # TODO overwrite batch style label: genre, instruments, tempo, artist, year.
            target_tags = target_inputs[0]['target_tags']
            new_genres = target_tags['genres']
            new_instruments = target_tags['instruments']
            for i, genre in enumerate(new_genres):
                batch['style_text'][i][0] = [genre]
            for i, instruments in enumerate(new_instruments):
                batch['style_text'][i][9] = instruments.split(',')

        prefix_inputs = self.prepare_prefix_inputs(batch)
        # Concat prefix and target in to a sequence
        if self.extra_params.get("multi_token_pattern", None) and self.target_embedder.R == self.target_embedder.group:
            prefix_token_ids = zip(*[i['token_ids'] for i in prefix_inputs])
            prefix_token_ids = [torch.cat(t, dim=0).unsqueeze(-1).repeat(1, self.target_embedder.R) 
                                for t in prefix_token_ids]  # batch_size x [T_p]
            target_token_ids = zip(*[i['token_ids'] for i in target_inputs])
            target_token_ids = [torch.cat(t, dim=0) for t in target_token_ids]  # batch_size x [T, G/R]
            token_ids = [torch.cat([prefix_token_ids[b], target_token_ids[b]], dim=0) for b in range(len(prefix_token_ids))]
        else:
            token_ids = zip(*[i['token_ids'] for i in prefix_inputs + target_inputs])
            token_ids = [torch.cat(t, dim=0) for t in token_ids]  # batch_size x [T_p + T]
  
        token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs + target_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0)
        target_length = torch.vstack([i['token_length'] for i in target_inputs]).sum(dim=0)
        target_embed_length = torch.vstack([i['token_embed_length'] for i in target_inputs]).sum(dim=0)

        # Padding the sequences across the batch
        token_ids_pad = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        token_embeds_pad = pad_sequence(token_embeds, batch_first=True, padding_value=0)
        batch_seq_length = token_ids_pad.shape[1]

        # Get masks
        input_loss_mask = sequence_mask(prefix_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = sequence_mask(prefix_length + target_length, max_len=batch_seq_length, device=self.device)
        target_loss_mask = target_loss_mask.bool() ^ input_loss_mask.bool() # remove input targets from target mask
        if target_loss_mask.ndim < token_ids_pad.ndim:
            target_loss_mask = target_loss_mask.unsqueeze(-1).expand_as(token_ids_pad)

        return {
            'token_embeds': token_embeds_pad,
            'token_ids': token_ids_pad,
            'input_loss_mask': input_loss_mask,
            'target_loss_mask': target_loss_mask,
            'prefix_inputs': prefix_inputs,
            'target_inputs': target_inputs,
            'prefix_length': prefix_length,
            'target_length': target_length,
            'target_embed_length': target_embed_length,
        }

    def training_step(self, batch, batch_idx):
        if random.random() < self.extra_params.get("group_dropout_rate", 0):
            group_cfg_label = self.extra_params.get("group_controller_cfg_label", 
            [["style_text","freeform_text","speaker_id","mir"],["lyrics"]])
            group_cfg_batch = self.prepare_group_cfg_batch(batch, {"group_controller_cfg_label": group_cfg_label})
            # randomly replace training batch to group cfg batch
            batch = random.choice(group_cfg_batch)

        return super().training_step(batch, batch_idx)
    
    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)

        if update_mfu:
            if "token_embeds" in training_inputs:
                b, t, _ = training_inputs["token_embeds"].shape
                self.metric.update(
                    num_tokens=b * t,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t}
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    self.log_dict(
                        self.metric.compute(self.trainer.global_step),
                        prog_bar=True,
                        sync_dist=True,
                    )
        input_token_embeds = training_inputs['token_embeds'][:, :-1]
        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        target_ids = training_inputs['token_ids'][:, 1:] * target_loss_mask # set non-target ids to 0 to avoid OOB

        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            model_output = self.model(inputs_embeds=input_token_embeds, output_hidden_states=True)
        if isinstance(model_output, dict):
            target_logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            target_logits, last_hidden_state = model_output

        result_dict = {}
        if self.extra_params.get("multi_token_pattern", None):
            # => [B, G, T, num_logits]
            target_logits = self.target_embedder.decode_target_logits(target_logits, 
                                                                      training_inputs["prefix_length"], 
                                                                      training_inputs["target_embed_length"],
                                                                      token_embeds=input_token_embeds, target_ids=target_ids)
            if isinstance(target_logits, list) or isinstance(target_logits, tuple):
                target_logits, target_ids, target_loss_mask = target_logits
            if self.target_embedder.null_id in target_ids:
                # @qinxin: Direct use of masked_fill_ on expanded tensors is deprecated, use clone() instead
                target_loss_mask_tmp = target_loss_mask.clone()
                target_loss_mask_tmp[target_ids == self.target_embedder.null_id] = 0
                target_loss_mask = target_loss_mask_tmp
                target_ids = target_ids * target_loss_mask
            # @qinxin: cancel the loss for sos token (because sos token is not given in the form of multi-token), 
            # in this case we need to manully add sos token during inference
            sos_token_pos = training_inputs["prefix_length"].long().unsqueeze(1) - 1
            if sos_token_pos.ndim < target_loss_mask.ndim:
                for r in range(target_loss_mask.shape[-1]):
                    target_loss_mask[..., r].scatter_(1, sos_token_pos, 0)
                target_loss = []
                for r in range(target_loss_mask.shape[-1]):
                    target_loss_r = self.criterion(target_logits[..., r,:], target_ids[..., r], mask=target_loss_mask[..., r])
                    target_accu_r = (target_logits[..., r,:].argmax(dim=-1) == target_ids[..., r])[target_loss_mask[...,r]].float().mean() * 100
                    result_dict.update({
                        f'tgt_accu_r{r}': target_accu_r.item(),
                        f'tgt_loss_r{r}': target_loss_r.item(),
                    })
                    target_loss.append(target_loss_r)
                target_loss = sum(target_loss) / len(target_loss)
            else:
                target_loss_mask.scatter_(1, sos_token_pos, 0)
                target_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)
            
            if self.target_embedder.decoder is not None:
                # @qinxin (25/02/12): add null loss to avoid `ddp_find_unused_parameters` error: this error maybe due to that sos_embed of decoder has no gradients (not affect training & inference)
                for param in self.target_embedder.decoder.parameters():
                    target_loss = target_loss + param.sum() * 0
        else:
            target_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)

        target_accu = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask].float().mean() * 100
        target_mask_cumsum = torch.cumsum(target_loss_mask, dim=-1)
        target_loss_mask_seq_25 = (target_mask_cumsum <= 26) & (target_mask_cumsum > 1) & target_loss_mask # first 25 tokens after eos
        target_accu_seq_25 = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask_seq_25].float().mean() * 100

        target_eos_mask = (target_ids == self.target_embedder.eos_id)
        target_accu_eos = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask & target_eos_mask].float().mean() * 100

        result_dict.update({
            'loss': (target_loss).item(),
            'accu': (target_accu).mean().item(),
            'tgt_loss': target_loss.item(),
            'tgt_accu': target_accu.item(),
            'tgt_accu_seq_25': target_accu_seq_25.item(),
            'tgt_accu_eos': target_accu_eos.item(),
        })

        return target_loss, result_dict

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
        # batch_cfg_additional = {k: deepcopy(v) for k, v in batch.items() if k not in batch_cfg and not is_key_cfg(k)}
        batch_cfg_additional = {k: v.detach().clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in batch.items() if k not in batch_cfg and not is_key_cfg(k)}
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


    @torch.no_grad()
    def predict(self, batch, hp, beam=1, rl_training=False, sample_seed=None, return_prefix=False):
        # is_varlen_prefix = self.extra_params.varlen_lyrics_prefix
        # if is_varlen_prefix: assert self.infer_batch_size(batch) == 1

        frame_rate = self.extra_params.semantic_frame_rate
        if 'duration' in batch and rl_training:
            assert not isinstance(batch['duration'], list), "type(batch['duration']) shoule be tensor not list"
            num_tokens = batch['duration'] * frame_rate
        elif isinstance(hp.duration, int):
            num_tokens = hp.duration * frame_rate
            num_tokens = torch.tensor([num_tokens])
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
        edt_theta = hp.get('edt_theta', 1.0)
        edt_t0 = hp.get('edt_t0', 0.7)
        p_base = hp.get('p_base', 0.1)
        skip_sos = hp.get('skip_sos', False)
        stop_eos = hp.get('stop_eos', False)
        self.extra_params.debug_index = hp.get('debug_index', None)
        exclude_ids = None
        if hp.get("exclude_eos", False) and self.target_embedder.eos_id is not None:
            exclude_ids = [self.target_embedder.eos_id]
            print(f"exclude_ids: {exclude_ids}")

        # for multi-token prediction
        multi_token_pattern = hp.get("multi_token_pattern", None)
        apply_cfg_to_LM_output = hp.get("apply_cfg_to_LM_output", False)
        if multi_token_pattern is not None:
            skip_sos = False

        # Init batch_cfg before batch gets modified
        if isinstance(controller_cfg_gamma, list) and use_controller_cfg:
            batch_uncond = self.prepare_cfg_batch_v2(batch, hp)
            batch_cfg = self.prepare_group_cfg_batch(batch, hp) # list
            batch_cfg.append(batch_uncond)  # also add FULL uncond path
        else:
            batch_cfg = self.prepare_cfg_batch_v2(batch, hp) if use_controller_cfg else None

        batch_size = self.infer_batch_size(batch)
        prefix_inputs = self.prepare_prefix_inputs(batch)

        if batch_size == 1 or not self.extra_params.varlen_lyrics_prefix:
            token_embeds = zip(*[i['token_embeds'] for i in prefix_inputs])
            token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
            token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)
            prefix_length = torch.vstack([i['token_length'] for i in prefix_inputs]).sum(dim=0).int()
        else:
            token_embeds_total_list = [[] for _ in range(batch_size)]
            prefix_inputs_token_length = torch.zeros(batch_size, device=self.device)
            for prefix_input in prefix_inputs:
                if isinstance(prefix_input['token_embeds'], torch.Tensor):
                    token_embeds_list = unpad_sequence(prefix_input['token_embeds'], prefix_input['token_length'], batch_first=True)
                elif isinstance(prefix_input['token_embeds'], list):
                    token_embeds_list = prefix_input['token_embeds']
                else:
                    raise NotImplementedError(f"{type(prefix_input['token_embeds'])=}")
                for i in range(batch_size):
                    token_embeds_total_list[i].append(token_embeds_list[i])
                    assert prefix_input['token_length'][i] == len(token_embeds_list[i]), f"{prefix_input['token_length'][i]=}, {token_embeds_list[i].shape=}"
                    prefix_inputs_token_length[i] += prefix_input['token_length'][i]
            token_embeds = [torch.cat(t, dim=0) for t in token_embeds_total_list]
            token_embeds = pad_sequence(token_embeds, batch_first=True, padding_value=0)
            prefix_length = prefix_inputs_token_length.int()

        print(f"{token_embeds.shape=}")
        print(f"{prefix_length=}")
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
        
        predict_func = super().predict if self.infer_batch_size(batch) == 1 else super().predict_varlen
        if multi_token_pattern is not None:
            predict_func = self.predict_MTP
            if multi_token_pattern in ['parallel-rq', 'parallel-arq']:
                predict_func = self.predict_MTP_rq

        generate_outs = predict_func(
            batch,
            token_embeds,
            num_tokens,
            temperature=temperature,
            beam=beam,
            sample_mode=sample_mode,
            sample_thresh=sample_thresh,
            exclude_ids=exclude_ids,
            rl_training=rl_training,
            skip_sos=skip_sos,
            stop_eos=stop_eos,
            use_controller_cfg=use_controller_cfg,
            inputs_embeds_cfg=token_embeds_cfg,
            controller_cfg_gamma=controller_cfg_gamma,
            use_step_out_blank=use_step_out_blank,
            step_out_blank_logic=step_out_blank_logic,
            step_out_blank_max_len=step_out_blank_max_len,
            repetition_penalty=repetition_penalty,
            exclude_eos_first_secs=exclude_eos_first_secs,
            exclude_eos_thresh_secs=exclude_eos_thresh_secs,
            emit_eos_thresh_secs=emit_eos_thresh_secs,
            edt_theta=edt_theta,
            edt_t0=edt_t0,
            p_base=p_base,
            multi_token_pattern=multi_token_pattern,
            apply_cfg_to_LM_output=apply_cfg_to_LM_output,
        )

        if return_prefix:
            prefix_inputs = {
                # "prefix_token_ids": token_ids,
                "prefix_token_embeds": token_embeds,
                "prefix_lengths": prefix_length,
            }
            if batch_cfg:
                prefix_inputs.update(
                    {
                        # "prefix_token_ids_cfg": token_ids_cfg,
                        "prefix_token_embeds_cfg": token_embeds_cfg,
                        "prefix_lengths_cfg": prefix_length_cfg,
                    }
                )
            if rl_training:
                return *generate_outs, prefix_inputs
            else:
                return generate_outs, prefix_inputs
        return generate_outs


        from samantha.utils import groundtruth
        groundtruth.emit('semantic', data={
            'batch': batch,
            'inputs_embeds': torch.cat([token_embeds, token_embeds_cfg], dim=0) if use_controller_cfg else token_embeds,
            'semantic_tokens': semantic_tokens,
            'inference_params': dict(
                num_tokens=num_tokens,
                temperature=temperature,
                beam=beam,
                sample_mode=sample_mode,
                sample_thresh=sample_thresh,
                exclude_ids=exclude_ids,
                rl_training=rl_training,
                skip_sos=skip_sos,
                stop_eos=stop_eos,
                use_controller_cfg=use_controller_cfg,
                controller_cfg_gamma=controller_cfg_gamma,
                use_step_out_blank=use_step_out_blank,
                step_out_blank_logic=step_out_blank_logic,
                step_out_blank_max_len=step_out_blank_max_len,
                repetition_penalty=repetition_penalty,
                exclude_eos_first_secs=exclude_eos_first_secs,
            )
        })

        return semantic_tokens

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)

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


    @staticmethod
    def _init_model_input(skip_sos, inputs_embeds, sos_embeds):
        if skip_sos:    # skip sos embedding if already have one in prefix prompt
            return {"inputs_embeds": torch.cat([inputs_embeds,], dim=1) }
        else:
            return {"inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1) }
             

    @staticmethod
    def prepare_cfg_inputs(use_controller_cfg, inputs_embeds, inputs_embeds_cfg, controller_cfg_gamma):
        """
        inputs_embeds: [B, T, D]
        inputs_embeds_cfg: [B x n_cfg_path, T, D]
        controller_cfg_gamma: int or list

        """
        keep_idx = None
        if use_controller_cfg:
            if isinstance(inputs_embeds_cfg, list) and isinstance(controller_cfg_gamma, list):
                raw_controller_cfg_gamma = controller_cfg_gamma
                keep_idx = [i for i, gamma in enumerate(controller_cfg_gamma) if gamma!= 0]
                controller_cfg_gamma = [gamma for gamma in controller_cfg_gamma if gamma != 0]
                # delete cfg samples with zero cfg gammas (to avoid extra differences from different batch sizes)
                # lambda_{s,l}, lambda_{l}, lambda_{s}, lambda
                cfg_keep_idx = [i for i, gamma in enumerate(raw_controller_cfg_gamma[1:]) if gamma != 0]
                inputs_embeds_cfg = [inputs_embeds_cfg[i] for i in cfg_keep_idx]
                inputs_embeds_cfg = torch.cat(inputs_embeds_cfg, 0)
                cfg_batch_size = inputs_embeds_cfg.shape[0]
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)  
            else:
                cfg_batch_size = inputs_embeds_cfg.shape[0]
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0) 
        return inputs_embeds, cfg_batch_size, keep_idx, controller_cfg_gamma
    
    # @qinxin (25/02/12): This function is used for multi-token prediction, currently it does not support beam search, rl_samples
    @torch.no_grad()
    def predict_MTP(
        self,
        batch,
        inputs_embeds,
        num_tokens,
        temperature=1,
        sample_mode="gumbel",
        sample_thresh=0.9,
        tqdm_name=None,
        beam=1,
        ref_samples=None,
        rl_training=False,
        exclude_ids=None,
        skip_sos=False,
        use_controller_cfg=False,        
        inputs_embeds_cfg=None,
        controller_cfg_gamma=1.0,
        use_step_out_blank=False,
        step_out_blank_logic='v1',
        step_out_blank_max_len=10,
        repetition_penalty=1.0,
        exclude_eos_first_secs=0,
        exclude_eos_thresh_secs=0,
        emit_eos_thresh_secs=0,
        stop_eos=False,
        multi_token_pattern=None,
        apply_cfg_to_LM_output=True,
        edt_theta=1.0,
        edt_t0=0.7,
        p_base=0.1,
        **kwargs,
    ):
        assert multi_token_pattern is not None
        assert beam == 1 and rl_training is False and ref_samples is None

        tqdm_name = f"{self.__class__.__name__}.rank{self.global_rank}" if tqdm_name is None else tqdm_name

        original_batch_size = inputs_embeds.shape[0]
        inputs_embeds, decoder_embeds, cfg_batch_size, keep_idx, controller_cfg_gamma = \
            self.prepare_cfg_inputs(use_controller_cfg, inputs_embeds, inputs_embeds_cfg, controller_cfg_gamma)
        
        batch_size, seq_len, _ = inputs_embeds.size() # recalculate batch size
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)

        slice_dur = math.ceil(batch['slice_duration'].item())
        exclude_eos_first_secs = slice_dur - exclude_eos_thresh_secs if exclude_eos_thresh_secs > 0 else exclude_eos_first_secs
        num_tokens = (slice_dur + emit_eos_thresh_secs) * self.extra_params.semantic_frame_rate if emit_eos_thresh_secs > 0 else num_tokens

        pbar = tqdm(range(num_tokens))
        is_eos_stop = torch.zeros((original_batch_size), dtype=torch.long, device=self.device)
        num_tokens = num_tokens // self.target_embedder.group * self.target_embedder.R + self.target_embedder.group 
        print(f"{num_tokens=}, {self.target_embedder.group=}, {self.target_embedder.R=}")

        model_input = self._init_model_input(skip_sos, inputs_embeds, sos_embeds)
        output_tokens = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            gpt_max_seq_len = seq_len + num_tokens + 10
            inference_params = InferenceParams(
                max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
            )
        else:
            raise NotImplementedError(f"Unsupported model type for multi-token prediction inference: {type(self.model)}")

        if use_step_out_blank:
            previous_tokens = [[] for _ in range(original_batch_size)]
            if "audio_prompt_token_ids" in batch:
                previous_tokens = batch["audio_prompt_token_ids"].tolist()
                buffer_len = min(min(len(pt) for pt in previous_tokens), step_out_blank_max_len)
                previous_tokens = [pt[-buffer_len:] for pt in previous_tokens]
        
        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")
            if self.extra_params.get("debug_index") is not None and i < self.extra_params.get("debug_index"):
                if i == 0:
                    print("[Debug] using ground truth token n=", self.extra_params.get("debug_index"))
                predict_token = batch["remi_leadsheet_tokens"][:, i].view(-1, 1)
                offset = self.extra_params.audio_codebook_size
                predict_token = predict_token + offset
                predict_token_emb = self.target_embedder.embedder(predict_token)
                model_input['inputs_embeds'] = predict_token_emb
                output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token
                continue

            # to enable inference without a trainer, we simply cast the inputs to the expected model type
            # which is either torch.float16 or torch.bfloat16
            model_input["inputs_embeds"] = model_input["inputs_embeds"].to(self.model.lm_head.weight.dtype)
                
            # pre allocate rotary cos/sin to reduce recompute them per step
            if i == 0:
                for layer in self.model.transformer.layers:
                    if hasattr(layer.mixer, "rotary_emb"):
                        layer.mixer.rotary_emb._update_cos_sin_cache(model_input["inputs_embeds"], gpt_max_seq_len)
            logits = self.model(
                **model_input,
                inference_params=inference_params,
                position_ids=None,
                last_token_only=False,
            ).logits
            # cast back to full precision
            logits = logits.float()

            inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            logits = logits[:, -1:, :] # only predicting on last logit.

            if apply_cfg_to_LM_output and use_controller_cfg:
                logits = self.apply_cfg(logits, controller_cfg_gamma, batch_size, cfg_batch_size)
            
            logits = self.target_embedder.decode_hidden_state_inference(logits, predict_token_emb, frame_idx=i)
            if logits.shape[0] > original_batch_size and use_controller_cfg:    # in this case apply_cfg_to_LM_output=False
                logits = self.apply_cfg(logits, controller_cfg_gamma, batch_size, cfg_batch_size)

            logits_batch_size = original_batch_size * self.target_embedder.group
            # logits: [B, 1, G*D] => [B, G, 1, D] => [B*G, 1, D] => [B, G, D]
            logits = torch.stack(torch.chunk(logits, chunks=self.target_embedder.group, dim=-1), dim=1)
            logits = logits.reshape(logits_batch_size, 1, logits.shape[-1])
                
            
            if use_step_out_blank and step_out_blank_logic == 'v4': # by default we use v4
                previous_output_tokens = torch.tensor(previous_tokens, dtype=torch.long, device='cuda').reshape(original_batch_size, -1)
                bin_counts = torch.zeros([original_batch_size, logits.size(-1)+1], dtype=torch.long, device='cuda')
                bin_counts.scatter_add_(1, previous_output_tokens, torch.ones_like(previous_output_tokens))

                bin_counts = bin_counts[:, 0:logits.size(-1)]

                mask = (bin_counts > 0).expand(logits.shape)
                negative_mask = logits < 0

                logits = torch.where(mask.logical_and(negative_mask), logits * repetition_penalty, logits)
                logits = torch.where(mask.logical_and(negative_mask.logical_not()), logits / repetition_penalty, logits)

            exclude_eos_first_n_tokens = self.extra_params.semantic_frame_rate * exclude_eos_first_secs
            exclude_eos_first_n_tokens = exclude_eos_first_n_tokens // self.target_embedder.group * self.target_embedder.R
            if i < exclude_eos_first_n_tokens:
                # Ensure that it generates at least 30s music.
                logits[:original_batch_size, 0, self.target_embedder.eos_id] = -float('Inf')

            temp = temperature
            if r_idx is not None and isinstance(temperature, list):
                temp = temperature[r_idx]
            predict_token = self.sample_logits(
                i, logits, temp, sample_mode, sample_thresh, exclude_ids,
                edt_theta=edt_theta, edt_t0=edt_t0, p_base=p_base,
            )

            if use_step_out_blank and step_out_blank_logic == 'v4':
                predict_token_cpu = predict_token.cpu().numpy()
                # predict_token_cpu: (b)
                # previous_tokens: (b, queue_len)
                for j in range(original_batch_size):
                    if len(previous_tokens[j]) < step_out_blank_max_len:
                        previous_tokens[j].append(predict_token_cpu[j])
                    else:
                        previous_tokens[j] = previous_tokens[j][-step_out_blank_max_len:]
                        previous_tokens[j].append(predict_token_cpu[j])

            predict_token = predict_token.reshape(original_batch_size, self.target_embedder.group, 1) # [B, G, 1]
            predict_token_emb = self.target_embedder.embed_token_id(predict_token, frame_idx=i)    # [B, 1, D]

            if use_controller_cfg:
                # If unconditioned path uses the predict token history from the conditioned path.
                predict_token_emb = predict_token_emb.repeat(cfg_batch_size + 1, 1, 1)                

            model_input['inputs_embeds'] = predict_token_emb
            output_tokens = torch.cat([output_tokens, predict_token], dim=-1) if output_tokens is not None else predict_token
            
            if stop_eos:
                eos_mask = (predict_token.squeeze(-1) == self.target_embedder.eos_id)  # (B, G)
                is_eos_stop += (eos_mask.sum(-1) > 0)
                if torch.all(is_eos_stop > 0):
                    break 

        output_tokens = torch.stack([self.target_embedder.decode_target_id(output_tokens[b], delete_null=False, 
                                                                           add_sos=False, delay_back=True) 
                                    for b in range(output_tokens.shape[0])], 0)
        return output_tokens
        


    @torch.no_grad()
    def predict_MTP_rq(
        self,
        batch,
        inputs_embeds,
        num_tokens,
        temperature=1,
        sample_mode="gumbel",
        sample_thresh=0.9,
        tqdm_name=None,
        beam=1,
        ref_samples=None,
        rl_training=False,
        exclude_ids=None,
        skip_sos=False,
        use_controller_cfg=False,
        inputs_embeds_cfg=None,
        controller_cfg_gamma=1.0,
        use_step_out_blank=False,
        step_out_blank_logic='v1',
        step_out_blank_max_len=10,
        repetition_penalty=1.0,
        exclude_eos_first_secs=0,
        exclude_eos_thresh_secs=0,
        emit_eos_thresh_secs=0,
        stop_eos=False,
        multi_token_pattern=None,
        apply_cfg_to_LM_output=True,
        edt_theta=1.0,
        edt_t0=0.7,
        p_base=0.1,
        **kwargs,
    ):
        assert multi_token_pattern is not None
        assert beam == 1 and rl_training is False and ref_samples is None
        tqdm_name = f"{self.__class__.__name__}.rank{self.global_rank}" if tqdm_name is None else tqdm_name

        original_batch_size = inputs_embeds.shape[0]
        inputs_embeds, cfg_batch_size, keep_idx, controller_cfg_gamma = \
            self.prepare_cfg_inputs(use_controller_cfg, inputs_embeds, inputs_embeds_cfg, controller_cfg_gamma)
        
        batch_size, seq_len, _ = inputs_embeds.size() # recalculate batch size
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)

        slice_dur = math.ceil(batch['slice_duration'].item())
        exclude_eos_first_secs = slice_dur - exclude_eos_thresh_secs if exclude_eos_thresh_secs > 0 else exclude_eos_first_secs
        num_tokens = (slice_dur + emit_eos_thresh_secs) * self.extra_params.semantic_frame_rate if emit_eos_thresh_secs > 0 else num_tokens
        num_tokens = num_tokens // self.target_embedder.group * self.target_embedder.R + self.target_embedder.group 

        pbar = tqdm(range(num_tokens))
        is_eos_stop = torch.zeros((original_batch_size), dtype=torch.long, device=self.device)
        exclude_eos_first_n_tokens = self.extra_params.semantic_frame_rate * exclude_eos_first_secs
        exclude_eos_first_n_tokens = exclude_eos_first_n_tokens // self.target_embedder.group * self.target_embedder.R


        def sample_single_token(logits, r_idx=None):
            logits = self.apply_cfg(logits, controller_cfg_gamma, batch_size, cfg_batch_size)
            if use_step_out_blank:
                bin_counts = torch.zeros([original_batch_size, logits.size(-1)+1], dtype=torch.long, device='cuda')
                if r_idx is not None:
                    previous_output_tokens = torch.tensor(previous_tokens[r_idx], dtype=torch.long, device='cuda').reshape(original_batch_size, -1)
                else:
                    previous_output_tokens = torch.tensor(previous_tokens, dtype=torch.long, device='cuda').reshape(original_batch_size, -1)
                bin_counts.scatter_add_(1, previous_output_tokens, torch.ones_like(previous_output_tokens))

                bin_counts = bin_counts[:, 0:logits.size(-1)]

                mask = (bin_counts > 0).expand(logits.shape)
                negative_mask = logits < 0

                logits = torch.where(mask.logical_and(negative_mask), logits * repetition_penalty, logits)
                logits = torch.where(mask.logical_and(negative_mask.logical_not()), logits / repetition_penalty, logits)
            if i < exclude_eos_first_n_tokens:
                logits[:original_batch_size, 0, self.target_embedder.eos_id] = -float('Inf')

            temp = temperature
            if r_idx is not None and isinstance(temperature, list):
                temp = temperature[r_idx]
            predict_token = self.sample_logits(
                i, logits, temp, sample_mode, sample_thresh, exclude_ids, 
                edt_theta=edt_theta, edt_t0=edt_t0, p_base=p_base)
            
            if use_step_out_blank:
                predict_token_cpu = predict_token.cpu().numpy()
                if r_idx is not None:
                    for j in range(original_batch_size):
                        if len(previous_tokens[r_idx][j] >= step_out_blank_max_len):
                            previous_tokens[r_idx][j] = previous_tokens[r_idx][j][-step_out_blank_max_len:]
                        previous_tokens[r_idx][j].append(predict_token_cpu[j])
                else:
                    for j in range(original_batch_size):
                        if len(previous_tokens[j]) >= step_out_blank_max_len:
                            previous_tokens[j] = previous_tokens[j][-step_out_blank_max_len:]
                        previous_tokens[j].append(predict_token_cpu[j])
            predict_token = predict_token.reshape(original_batch_size, 1) # [B, 1]
            return predict_token


        def RQ_transformer(model_input, hidden_state, use_controller_cfg):
            batch_size = hidden_state.shape[0]  # (cfg_batch_size + 1)
            self.target_embedder.inference_params = InferenceParams(max_sequence_len=self.target_embedder.group+1, max_batch_size=batch_size)
            last_predict_token_emb = self.target_embedder.get_sos_embed(batch_size)
            pred_tokens = []

            for g in range(self.target_embedder.group):
                # [B, T=1, 2*D]
                model_input["inputs_embeds"] = torch.cat((hidden_state, last_predict_token_emb), dim=-1).reshape(batch_size*1, -1).unsqueeze(1)
                decoder_output = self.target_embedder.decoder(**model_input, inference_params=self.target_embedder.inference_params,
                                             position_ids=None, last_token_only=False,)
                self.target_embedder.inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
                target_logits = decoder_output.logits
                # [B, T=1, N] => [B, T=1]
                if self.target_embedder.R == 1:
                    predict_token = sample_single_token(target_logits)
                    last_predict_token_emb = self.target_embedder.embedder(predict_token)
                else:
                    predict_token = sample_single_token(target_logits, r_idx=g)
                    last_predict_token_emb = self.target_embedder.embedder[g](predict_token)
                if use_controller_cfg:
                    last_predict_token_emb = last_predict_token_emb.repeat(batch_size, 1, 1)
                pred_tokens.append(predict_token)
            return torch.cat(pred_tokens, -1)   # [B, T=G]
        
        def ARQ_transformer(model_input, hidden_state, use_controller_cfg):
            # cond_{1...G}, token_1, token_2, ..., token_G
            batch_size = hidden_state.shape[0]
            last_predict_token_emb = hidden_state
            pred_tokens = []
            for g in range(self.target_embedder.group + 1):
                # [B, T=1, 2*D]
                model_input["inputs_embeds"] = last_predict_token_emb   # [B, T=1, D]
                decoder_output = self.target_embedder.decoder(**model_input, inference_params=self.target_embedder.inference_params,
                                             position_ids=None, last_token_only=False,)
                self.target_embedder.inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
                if g == self.target_embedder.group: # only inputs the last predicted token (token_G) without obtaining its output
                    break
                target_logits = decoder_output.logits
                # [B, T=1, N] => [B, T=1]
                predict_token = sample_single_token(target_logits)
                last_predict_token_emb = self.target_embedder.embedder(predict_token)
                if use_controller_cfg:
                    last_predict_token_emb = last_predict_token_emb.repeat(batch_size, 1, 1)
                pred_tokens.append(predict_token)

            return torch.cat(pred_tokens, -1)   # [B, T=G]

        model_input = self._init_model_input(skip_sos, inputs_embeds, sos_embeds)
        output_tokens = None
        predict_token_emb = None
        gpt_max_seq_len = seq_len + num_tokens + 10
        inference_params = InferenceParams(
            max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
        )
        if use_step_out_blank:
            previous_tokens = [[] for _ in range(original_batch_size)]
            if self.target_embedder.R > 1:
                previous_tokens = [[[] for _ in range(original_batch_size)] for _ in range(self.target_embedder.R)]

        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")
            # to enable inference without a trainer, we simply cast the inputs to the expected model type
            # which is either torch.float16 or torch.bfloat16
            model_input["inputs_embeds"] = model_input["inputs_embeds"].to(self.model.lm_head.weight.dtype)
            
            # pre allocate rotary cos/sin to reduce recompute them per step
            if i == 0:
                for layer in self.model.transformer.layers:
                    if hasattr(layer.mixer, "rotary_emb"):
                        layer.mixer.rotary_emb._update_cos_sin_cache(model_input["inputs_embeds"], gpt_max_seq_len)
            logits = self.model(
                **model_input,
                inference_params=inference_params,
                position_ids=None,
                last_token_only=False,
            ).logits
            # cast back to full precision
            logits = logits.float()

            inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            logits = logits[:, -1:, :] # only predicting on last logit.
    

            if apply_cfg_to_LM_output and use_controller_cfg:
                logits = self.apply_cfg(logits, controller_cfg_gamma, batch_size, cfg_batch_size)
                
            if i == 0:
                if multi_token_pattern == 'parallel-arq':
                    # func init_decoder only activates the inference params, but not pass the sos_embed to decoder
                    self.target_embedder.init_decoder(batch_size=logits.shape[0], max_seq_len=8000)
                    # pass sos_embed to decoder
                    init_model_input = {
                        "inputs_embeds": self.target_embedder.get_sos_embed(logits.shape[0]).to(self.target_embedder.decoder.lm_head.weight.dtype)
                        }
                    decoder_out = self.target_embedder.decoder(**init_model_input, inference_params=self.target_embedder.inference_params,
                                                                position_ids=None, last_token_only=False,)
                    self.target_embedder.inference_params.sequence_len_offset += init_model_input['inputs_embeds'].size(1)

                    
            if multi_token_pattern == 'parallel-arq':
                pred_tokens = ARQ_transformer(model_input, logits, use_controller_cfg)
            elif multi_token_pattern == 'parallel-rq':
                pred_tokens = RQ_transformer(model_input, logits, use_controller_cfg)

            predict_token = pred_tokens.unsqueeze(-1) # [B, G, 1]
            predict_token_emb = self.target_embedder.embed_token_id(predict_token, frame_idx=i)    # [B, 1, D]
            
            if use_controller_cfg:
                predict_token_emb = predict_token_emb.repeat(cfg_batch_size + 1, 1, 1)                

            model_input['inputs_embeds'] = predict_token_emb
            output_tokens = torch.cat([output_tokens, predict_token], dim=-1) if output_tokens is not None else predict_token
            
            if stop_eos:
                eos_mask = (predict_token.squeeze(-1) == self.target_embedder.eos_id)  # (B, G)
                is_eos_stop += (eos_mask.sum(-1) > 0)
                if torch.all(is_eos_stop > 0):
                    break 

        output_tokens = torch.stack([self.target_embedder.decode_target_id(output_tokens[b], delete_null=False, 
                                                                           add_sos=False, delay_back=True) 
                                    for b in range(output_tokens.shape[0])], 0)
        return output_tokens
    
class SemanticRLModule(SemanticModule):
    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=MaskedCrossEntropy,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.decoder_type = self.extra_params.get("decoder_type", "diffusion")
        if self.decoder_type == "diffusion":
            self.decoder_fn = self.run_diffusion
        elif self.decoder_type == 'ar-diffusion-vocoder':
            # this is the tts token2wav
            self.decoder_fn = run_diffusion_vocoder_batch
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")
        self.val_outputs = dict()
        self.positive_qualitative_emb = None
        self.negative_qualitative_emb = None
        self.resampler = None
        if self.extra_params.get("token2wav_sample_rate", self.extra_params.sample_rate) != self.extra_params.sample_rate:
            self.resampler = Resample(
                orig_freq=self.extra_params.token2wav_sample_rate,
                new_freq=self.extra_params.sample_rate,
            )
        self.reward_pool = ThreadPool(self.extra_params.get('reward_num_thread', 4))

    def set_requires_grad(self, requires_grad):
        for n, p in self.named_parameters():
            p.requires_grad = requires_grad

    def _shared_step(self, batch, mode, update_mfu=False):
        wavs_gt = batch["target_audio"]
        if wavs_gt.dim() == 2:
            wavs_gt = wavs_gt.unsqueeze(1)
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)

        input_token_embeds = training_inputs['token_embeds'][:, :-1]
        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        target_ids = training_inputs['token_ids'][:, 1:] * target_loss_mask # set non-target ids to 0 to avoid OOB
        B, T, _ = training_inputs["token_embeds"].shape

        beam = self.extra_params.beam_size

        if update_mfu:
            if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                log_dict = {
                    # "loss": loss.item(),
                    "bsz": B,
                    "seqlen": T,
                    }
                self.metric.update(
                    num_tokens=B * T,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": B, "seq_len": T}
                )
                metric = self.metric.compute(self.trainer.global_step)
                log_dict.update(metric)
                self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        # CE loss
        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            print(f"1st round forward model input shape: {input_token_embeds.shape}")
            model_output = self.model(inputs_embeds=input_token_embeds, output_hidden_states=True)
            input_token_embeds_length = input_token_embeds.shape[1]
        if isinstance(model_output, dict):
            logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            target_logits, last_hidden_state = model_output
        # target_logits = logits[:, -T:, :]
        ce_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)
        accu = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask].float().mean() * 100
        # aux_loss, _ = self.aux_loss(batch, target_logits, last_hidden_state, T)
        # ce += aux_loss

        def batch_data(data_list, padding_value=0):
            sequences = []
            sequences_len = []
            for data in data_list:
                for seq in data:
                    sequences.append(seq)
                    sequences_len.append(seq.shape[0])
            batch_data = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
            return batch_data, sequences_len

        def create_mask(lengths):
            if isinstance(lengths, list):
                lengths = torch.LongTensor(lengths)
            batch_size = lengths.size(0)
            max_length = lengths.max()
            mask = torch.arange(max_length).expand(batch_size, max_length) < lengths.unsqueeze(1)
            return mask

        def adapt_slice_duration(duration, range=20):
            return int(duration + random.randrange(-range // 2, range))

        # Sequence loss
        if self.training and self.trainer.global_step < self.extra_params.get("ce_only_steps", 0):
            seq_loss = 0
            wavs_gt = None
            sampled_audio = None
            rewards = None
            reward_breakdown = None
            seq_probs3 = None
            skip = False
        else:
            # Inference
            batch["inputs_embeds"] = input_token_embeds

            if "duration" not in batch:
                if isinstance(self.extra_params.duration, (list, tuple)):
                    batch["duration"] = self.extra_params.duration[-1]
                else:
                    batch["duration"] = self.extra_params.duration

                if self.extra_params.get("use_adaptive_duration", False):
                    assert len(batch['slice_type']) == len(batch['lyrics']) == len(batch['style_category']) == len(batch['slice_duration'])
                    adptive_duration_mode = self.extra_params.get("adaptive_duration_mode", "predict")
                    if adptive_duration_mode == "predict":
                        batch["duration"] = [generate_duration(lyrics, sc[0]) if st == "vocal" else adapt_slice_duration(sd.item()) for lyrics, sc, st, sd in zip(batch['lyrics'], batch['style_category'], batch['slice_type'], batch['slice_duration'])]
                    elif adptive_duration_mode == "slice":
                        batch["duration"] = [adapt_slice_duration(sd.item()) for sd in batch['slice_duration']]
                    else:
                        raise NotImplementedError

            batch["duration"] = torch.tensor(batch["duration"], device=self.device)  # (bsz,), list to tensor
            num_tokens = batch["duration"] * self.extra_params.semantic_frame_rate

            self.set_requires_grad(False) # fix nested autocast bug: https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/2
            self.model.eval()

            use_batch = int(os.environ.get("MUSIC_RL_DEBUG_USE_BATCH", "1"))

            if use_batch:
                sampled_semantic_tokens, model_inputs = super().predict(
                    batch,
                    self.extra_params,
                    beam=beam,
                    rl_training=True,
                )

                sampled_semantic_token_processed, eos_index = process_eos_indexes(
                    sampled_semantic_tokens,
                    self,
                    sample_rate=self.extra_params.sample_rate,
                )
                eos_index_list = eos_index
                # Compute rewards
                rewards, sampled_audios, rewards_breakdown = self.get_reward({
                    "target_semantic_tokens": target_ids,
                    "sampled_semantic_tokens": sampled_semantic_token_processed,
                    "eos_index_list": eos_index,
                    "target_audio": wavs_gt,
                    "batch_size": B,
                    "beam_size": beam,
                    "batch": batch,
                })
            else:
                rewards = []
                sampled_audios = []
                rewards_breakdown = {}
                model_inputs = []
                sampled_semantic_tokens = []
                eos_index_list = []

                for idx in range(B):
                    single_batch = {}
                    for key in batch:
                        if torch.is_tensor(batch[key]) or isinstance(batch[key], List):
                            single_batch[key] = batch[key][idx:idx+1]
                        else:
                            single_batch[key] = batch[key]
                    sampled_semantic_token, single_model_inputs = super().predict(
                        single_batch,
                        self.extra_params,
                        beam=beam,
                        # ref_samples=ref_samples,
                        rl_training=True,
                    )
                    sampled_semantic_token_processed, eos_index = process_eos_indexes(
                        sampled_semantic_token,
                        self,
                        sample_rate=self.extra_params.sample_rate,
                    )
                    # Compute rewards
                    reward, sampled_audio, reward_breakdown = self.get_reward({
                        "target_semantic_tokens": target_ids[idx:idx+1],
                        "sampled_semantic_tokens": sampled_semantic_token_processed,
                        "eos_index_list": eos_index,
                        "target_audio": wavs_gt[idx:idx+1],
                        "batch_size": 1,
                        "beam_size": beam,
                        "batch": single_batch,
                    })
                    rewards.append(reward)
                    sampled_audios.append(sampled_audio)
                    model_inputs.append(single_model_inputs)
                    sampled_semantic_tokens.append(sampled_semantic_token)
                    eos_index_list.append(eos_index)
                    if len(rewards_breakdown) != 0:
                        for key in reward_breakdown:
                            rewards_breakdown[key] = torch.cat([rewards_breakdown[key], reward_breakdown[key]])
                    else:
                        rewards_breakdown = reward_breakdown
                eos_index_list = torch.cat(eos_index_list, dim=0)
                model_inputs = [item["inputs_embeds"] for item in model_inputs]
                model_inputs = [item for item in model_inputs]
                # concat sequences with padding
                model_inputs, sequence_lengths = batch_data(model_inputs)
                # sequence_length = token_length + inputs_length
                sampled_semantic_tokens, token_lengths = batch_data(sampled_semantic_tokens)

                rewards = torch.cat(rewards, dim=0).to(sampled_semantic_tokens.device)
                model_inputs = {
                    "inputs_embeds"  : model_inputs,
                    "attention_mask" : create_mask(sequence_lengths).to(sampled_semantic_tokens.device),
                    }

            self.set_requires_grad(True)
            self.model.train()
            # Compute sequence probs
            print(f"2nd round forward model input shape: {model_inputs['inputs_embeds'].shape}")
            prediction_ratio = (model_inputs['inputs_embeds'].shape[1]-input_token_embeds_length)*1.0/input_token_embeds_length
            if prediction_ratio > self.extra_params.get("max_allowed_prediction_ratio", 1.):
                print(f"too long predict tokens {input_token_embeds_length}->{model_inputs['inputs_embeds'].shape[1]} {prediction_ratio=}")

            model_outputs = self.model(**model_inputs)
            if isinstance(model_outputs, dict):
                seq_logits = model_outputs["logits"]
            elif isinstance(model_outputs, tuple):
                seq_logits = model_outputs[0]

            token2wav_rate = self.extra_params.sample_rate // self.extra_params.semantic_frame_rate
            seq_len = eos_index_list // token2wav_rate + 1
            seq_probs2_list = []
            for i in range(len(eos_index_list)):
                seq_probs1 = F.log_softmax(seq_logits[i:i+1, -seq_len[i]:, :], dim=-1)
                _seq_probs2 = torch.gather(
                    seq_probs1, -1, sampled_semantic_tokens[i:i+1,-seq_len[i]:].unsqueeze(2)
                ).squeeze(2)    # (B * beam, T)
                _seq_probs2 = (_seq_probs2.sum(dim=-1) / seq_len[i])
                seq_probs2_list.append(_seq_probs2)
            seq_probs2 = torch.stack(seq_probs2_list).reshape(B, beam)
            seq_probs3 = F.softmax(seq_probs2, dim=-1)
            seq_loss = -1 * (rewards * seq_probs3).sum(dim=-1).mean()

            # Only add up non-eos probs
            # token2wav_rate = self.extra_params.sample_rate // self.extra_params.semantic_frame_rate
            # seq_len = eos_index_list // token2wav_rate + 1  # need to add 1 to include <eos>
            # seq_probs2_list = []
            # for i in range(len(eos_index_list)):
            #     seq_probs1 = F.log_softmax(seq_logits[i:i+1, -seq_len[i]:, :], dim=-1)
            #     _seq_probs2 = torch.gather(
            #         seq_probs1, -1, sampled_semantic_tokens[i:i+1,-seq_len[i]:].unsqueeze(2)
            #     ).squeeze(2)    # (B * beam, T)
            #     _seq_probs2 = (_seq_probs2.sum(dim=-1) / seq_len[i])
            #     seq_probs2_list.append(_seq_probs2)

            # seq_probs2 = torch.stack(seq_probs2_list).reshape(B, beam)
            # seq_probs3 = F.softmax(seq_probs2, dim=-1)
            # seq_loss = -1 * (rewards * seq_probs3).sum(dim=-1).mean()
            skip = torch.any(torch.isnan(seq_probs1)) or torch.any(torch.isnan(seq_probs3))
            if skip:
                print(f"Skipped samples: {sampled_semantic_tokens}")
        return (
            ce_loss,
            accu,
            seq_loss,
            wavs_gt,
            sampled_audios,
            rewards,
            rewards_breakdown,
            seq_probs3,
            skip,
        )

    def training_step(self, batch, batch_idx):
        if 'embedding_pretrain_steps' in self.extra_params:
            embedding_pretrain_steps = self.extra_params['embedding_pretrain_steps']
            if batch_idx < embedding_pretrain_steps:
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.parameters():
                    param.requires_grad = True
        ce_loss, accu, seq_loss, _, _, rewards, reward_breakdown, seq_probs, skip = self._shared_step(
            batch=batch,
            mode="training",
            update_mfu=True,
        )
        stats = {
            "ce_loss/train": ce_loss,
            "accuracy/train": accu,
            "seq_loss/train": seq_loss,
        }
        if seq_probs is not None:
            stats.update(
                {
                    "seq_probs/train_max_mean": seq_probs.max(dim=-1).values.mean(),
                    "seq_probs/train_max_std": seq_probs.max(dim=-1).values.std(),
                }
            )
        if reward_breakdown is not None:
            for rw_type, rw in reward_breakdown.items():
                stats.update(
                    {
                        f"reward_{rw_type}/train_avg_mean": rw.mean(dim=-1).mean(),
                        f"reward_{rw_type}/train_avg_std": rw.mean(dim=-1).std(),
                        f"reward_{rw_type}/train_intra_beam_std": rw.std(dim=-1).mean(),
                        f"reward_{rw_type}/train_max_mean": rw.max(dim=-1).values.mean(),
                        f"reward_{rw_type}/train_max_std": rw.max(dim=-1).values.std(),
                    }
                )
        if rewards is not None:
            stats.update(
                {
                    "rewards_all/train_avg_mean": rewards.mean(dim=-1).mean(),
                    "rewards_all/train_avg_std": rewards.mean(dim=-1).std(),
                    "rewards_all/train_intra_beam_std": rewards.std(dim=-1).mean(),
                    "rewards_all/train_max_mean": rewards.max(dim=-1).values.mean(),
                }
            )
        if skip:
            print("Skipping update due to NaN...")
            loss = None
        else:
            ce_weight = self.extra_params.ce_weight
            seq_weight = self.extra_params.seq_weight
            loss = ce_loss * ce_weight + seq_loss * seq_weight
        stats.update({
            "tr_loss": loss,
            "training/loss": loss,
            })
        self.log_dict(stats, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(
            self._shared_step(
                batch=batch,
                mode="validation",
            )
        )

    def on_validation_epoch_end(self):
        max_log_samples = self.extra_params.get("log_num_val_samples", None)
        max_log_beam_samples = self.extra_params.get("log_num_val_beam_samples", None)
        sample_count = 0
        for dataloader_idx, outputs in self.val_outputs.items():
            prefix = f"val_{dataloader_idx}"
            stats = defaultdict(int)
            for batch_idx, (ce_l, a, seq_l, wavs_gt, wavs_sampled, rewards, reward_breakdown, seq_probs, _) in enumerate(outputs):
                stats[f"ce_loss/{prefix}"] += ce_l
                stats[f"accuracy/{prefix}"] += a
                stats[f"accu_{prefix}"] += a    # remove `/` to make it can be used in path, like `accu_val_0`
                stats[f"seq_loss/{prefix}"] += seq_l
                stats[f"seq_probs/{prefix}_max_mean"] += seq_probs.max(dim=-1).values.mean()
                stats[f"seq_probs/{prefix}_max_std"] += seq_probs.max(dim=-1).values.std()
                for rw_type, rw in reward_breakdown.items():
                    stats[f"reward_{rw_type}/{prefix}_avg_mean"] += rw.mean(dim=-1).mean()
                    stats[f"reward_{rw_type}/{prefix}_avg_std"] += rw.mean(dim=-1).std()
                    stats[f"reward_{rw_type}/{prefix}_intra_beam_std"] += rw.std(dim=-1).mean()
                    stats[f"reward_{rw_type}/{prefix}_max_mean"] += rw.max(dim=-1).values.mean()
                    stats[f"reward_{rw_type}/{prefix}_max_std"] += rw.max(dim=-1).values.std()
                for i in range(len(wavs_gt)):
                    if max_log_samples and sample_count >= max_log_samples:
                        break
                    sample_count += 1

                    # workaround add_audio() do not support stereo tracing
                    if len(wavs_gt[i].shape) == 2 and wavs_gt[i].shape[0] == 2:
                        wavs_gt[i] = wavs_gt[i].mean(0)

                    self.logger.experiment.add_audio(
                        f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_target",
                        wavs_gt[i],
                        self.global_step,
                        sample_rate=self.extra_params.sample_rate,
                    )
                    # Log highest reward first
                    indices = torch.argsort(rewards[i], descending=True).cpu().tolist()
                    for j in range(len(indices)):
                        idx = indices[j]
                        if max_log_beam_samples and j >= max_log_beam_samples:
                            break
                        # workaround add_audio() do not support stereo tracing
                        if len(wavs_sampled[i][idx].shape) == 2 and wavs_sampled[i][idx].shape[0] == 2:
                            wavs_sampled[i][idx] = wavs_sampled[i][idx].mean(0)

                        self.logger.experiment.add_audio(
                            f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_{j}",
                            wavs_sampled[i][idx],
                            self.global_step,
                            sample_rate=self.extra_params.sample_rate,
                        )
                        log_text = ""
                        for rw_type, rw in reward_breakdown.items():
                            log_text += f"{rw_type}={rw[i][idx].item():.2f} "
                        self.logger.experiment.add_text(
                            f"validation/sampled_{dataloader_idx}_{batch_idx}_{i}_{j}",
                            log_text,
                            self.global_step,
                        )
            for key in stats:
                stats[key] /= len(outputs)
            self.log_dict(stats, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    @torch.no_grad()
    def run_diffusion(self, items):
        diffusion_params = self.extra_params.get("diffusion_params", {})
        sampler = self.requires["sampler"]
        diffusion = self.requires["diffusion"]
        vocoder = self.requires["vocoder"]
        semantic_tokens = items["sampled_semantic_tokens"]

        # diffusion sampling
        pred_emb = sampler(
            model=diffusion,
            semantic_context=semantic_tokens,
            num_items=semantic_tokens.shape[0],
            num_chunks=1,
            num_steps=self.extra_params.diffusion_steps,
            bf16_portion=self.extra_params.bf16_portion,
            #start=None,
            #show_progress=False,
            angle_schedule="linear",
            schdeule_slope=diffusion_params.get("schedule_slope", 2.5),
            classifier_free_guidance=diffusion_params.get("guidance_scale", 2.5),
        ).detach().float()
        # torch.interpolate causes OOM for large batch sizes > 24. chunking to batch of 8 instead.
        # vocoder upsample requires a lot of memory. Chunk vocode instead
        duration = semantic_tokens.shape[-1] / self.extra_params.get('semantic_frame_rate', 25)
        if duration >= 100:
            wavs = vocode_in_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=4)
        else:
            wavs = vocode_in_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=1)
        return wavs

    @torch.no_grad()
    def get_reward(self, items):
        b = items["batch_size"]
        beam = items["beam_size"]
        batch = items["batch"]
        eos_index_list = items["eos_index_list"]

        if self.decoder_type == "diffusion":
            sampled_audio = self.decoder_fn(items).float()
        elif self.decoder_type == 'ar-diffusion-vocoder':
            sampled_audio = self.decoder_fn(
                self.requires,
                items["sampled_semantic_tokens"])

        # Resample and convert to mono if necessary
        if self.resampler is not None:
            sampled_audio = self.resampler(sampled_audio.to(self.device)).mean(dim=1, keepdim=True)

        # Zero out samples after <eos>
        if len(eos_index_list) > 0:
            for i in range(len(eos_index_list)):
                sampled_audio[i, :, eos_index_list[i]:] = 0

        if "duration" in batch:
            max_samples = batch["duration"] * self.extra_params.sample_rate
            # sampled_audio = sampled_audio[..., :max_samples]
            max_samples = max_samples[:,None].repeat(1, beam).reshape(-1)
            # sampled_audio: (b * beam, 1, n)  bs0beam0, bs0beam1, ...
            sampled_audio = sampled_audio[..., :max_samples.max()]
            items["sampled_audio_lens"] = max_samples
            if len(eos_index_list) > 0:
                assert len(eos_index_list) == len(max_samples), f"{max_samples=}, {eos_index_list=}"
                max_samples = eos_index_list
            sampled_audio_list = [s[..., :n] for s, n in zip(sampled_audio, max_samples)]
        else:
            sampled_audio_list = sampled_audio
        items["sampled_audio"] = sampled_audio
        reward = 0.0
        reward_breakdown = {}
        rws, wav_urls = [], []
        # Upload wav to tos once while any kind of mir tag rewards
        # if any(i in ['genre_tag', 'mood_tag', 'lang_tag'] for i in self.extra_params.rewards.keys()):
        #     wav_urls = upload_wavs_to_tos(sampled_audio_list, self.extra_params.sample_rate)
        if any(i in ['genre_tag', 'mood_tag', 'lang_tag', 'wer'] for i in self.extra_params.rewards.keys()):
           wav_paths, wav_urls = upload_wavs_to_tos(sampled_audio_list, self.extra_params.sample_rate)
        for rw_type, rw_weight in self.extra_params.rewards.items():
            if float(rw_weight) == 0.:
                continue
            # TODO(hang): check mulan reward
            if rw_type == "style_sim":
                conditions = self.infer_conditions(batch)
                rw_type = "style_text_sim" if "style_text" in conditions else "mulan_sim"
            rws.append((rw_type, rw_weight, self.reward_pool.apply_async(self._get_reward, args=(items, rw_type, wav_urls, wav_paths))))
        for rw_type, rw_weight, handler in rws:
            rw = handler.get().reshape(b, beam)
            reward += rw_weight * rw
            reward_breakdown[rw_type] = rw
        return reward, sampled_audio.reshape(b, beam, -1), reward_breakdown

    @torch.no_grad()
    def _get_reward(self, items, reward_type, wav_urls=None, wav_paths=None):
        sampled_audio = items["sampled_audio"]
        target_audio = items["target_audio"]
        batch_size = items["batch_size"]
        beam = items["beam_size"]
        batch = items["batch"]
        if reward_type == "mulan_sim":
            (
                mulan_sim,
                items["sampled_mulan_embeds"],
                items["target_mulan_embeds"],
            ) = mulan_audio_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                target_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
                sampled_embeds=items.get("sampled_mulan_embeds"),
                target_embeds=items.get("target_mulan_embeds"),
            )
            return mulan_sim
        elif reward_type == "genre_tag":
            items = items['batch']
            target_genres = [style_text[0] for style_text in items["style_category"]]
            return mir_tag_reward(
                wav_urls,
                mir_tag_type='genre',
                target_tags=target_genres,
                beam_size=beam,
                device=sampled_audio.device)
        elif reward_type == "lang_tag":
            batch_items = items['batch']
            target_langs = [style_text[7] for style_text in batch_items["style_category"]]
            assert len(batch_items["lyrics"]) == len(batch_items["slice_type"]), "num of lyrics should be equal to slices"
            style_text_pairs = [(lyrics, slice_type) for lyrics, slice_type in zip(batch_items["lyrics"], batch_items["slice_type"])]
            return mir_tag_reward(
                wav_urls,
                mir_tag_type='lang',
                target_tags=target_langs,
                beam_size=beam,
                device=sampled_audio.device,
                style_text=style_text_pairs,
                )
        elif reward_type == "mood_tag":
            batch_items = items['batch']
            target_mood = [style_text[3] for style_text in batch_items["style_category"]]
            return mir_tag_reward(
                wav_urls,
                mir_tag_type='mood',
                target_tags=target_mood,
                beam_size=beam,
                device=sampled_audio.device)
        elif reward_type == "mulan_temporal":
            mulan_temporal = mulan_temporal_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                device=sampled_audio.device,
                sample_rate=self.extra_params.sample_rate,
            )
            return mulan_temporal
        elif reward_type == "chroma_temporal":
            chroma_temporal = chroma_temporal_reward(
                sampled_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
            return chroma_temporal
        elif reward_type == "wer":
            if "sampled_lyrics" not in items:
                use_online_asr = self.extra_params.get("use_online_asr", True)
                if not use_online_asr:
                    eos_index_list = items["eos_index_list"]
                    items["sampled_lyrics"] = asr_transcribe_lyrics(
                        self.requires,
                        sampled_audio,
                        sample_rate=self.extra_params.sample_rate,
                        sample_lengths=None if len(eos_index_list) == 0 else eos_index_list
                    )
                else:
                    batch_items = items['batch']
                    target_langs = [style_text[7] for style_text in batch_items["style_category"]]
                    sampled_lyrics = []
                    for i in range(batch_size):
                        tgt_lang = target_langs[i]
                        if "Chinese" in tgt_lang or "Cantonese" in tgt_lang or "Instrumental/Non-vocal" in tgt_lang or "Non-vocal" in tgt_lang:
                            langid = "zh-CN"
                        elif "Japanese" in tgt_lang:
                            langid = "ja-JP"
                        elif "English" in tgt_lang:
                            langid = "en-US"
                        else:
                            langid = "zh-CN"
                            print(f"Unsupported language {tgt_lang}")
                        for b in range(beam):
                            wav_path = wav_paths[i*beam+b]
                            sampled_lyrics.append(run_asr_lyrics_sa_online(wav_path, langid)[0])
                    items["sampled_lyrics"] = sampled_lyrics
            if len(sampled_lyrics) != sampled_audio.size(0):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(sampled_lyrics)} (expected {sampled_audio.size(0)}), content={items['sampled_lyrics']}")
                return 0
            return wer_reward(
                items["sampled_lyrics"],
                batch["lyrics"] if "lyrics" in batch else batch["lyrics_text"],
                sampled_audio.device,
            )
        elif reward_type == "melody_flow":
            melody_reward = umm_reward(
                self.requires[reward_type + "_rm"],
                sampled_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
                truncate_len_in_sec=80,
                rm_type=reward_type,
            )
            return melody_reward
        elif reward_type == "arrangement":
            arrangement_reward = umm_reward(
                self.requires[reward_type + "_rm"],
                sampled_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
                rm_type=reward_type,
            )
            return arrangement_reward
        elif reward_type == "style_text_sim":
            if not batch.get("freeform_text", None):
                batch["freeform_text"] = [infer_freeform_text_from_style_text(st, freeform_dropout=0.1) for st in batch["style_text"]]
            (
                style_text_sim,
                items["sampled_mulan_embeds"],
                items["style_text_embeds"],
            ) = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                batch["freeform_text"],
                device=sampled_audio.device,
                sampled_embeds=items.get("sampled_mulan_embeds"),
                target_embeds=items.get("style_text_embeds"),
            )
            return style_text_sim
        elif reward_type == "loudness_sim":
            return loudness_reward(
                sampled_audio,
                target_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "qualitative_sim":
            qualitative_reward = 0
            if self.extra_params.positive_phrase is not None:
                (
                    positive_sim,
                    items["sampled_mulan_embeds"],
                    self.positive_qualitative_emb,
                ) = mulan_text_reward(
                    self.requires["mulan_infer_fn"],
                    self.requires["mulan"],
                    sampled_audio.squeeze(1),
                    [self.extra_params.positive_phrase],
                    device=sampled_audio.device,
                    sampled_embeds=items.get("sampled_mulan_embeds"),
                    target_embeds=self.positive_qualitative_emb,
                )
                qualitative_reward += positive_sim
            if self.extra_params.negative_phrase is not None:
                (
                    negative_sim,
                    items["sampled_mulan_embeds"],
                    self.negative_qualitative_emb,
                ) = mulan_text_reward(
                    self.requires["mulan_infer_fn"],
                    self.requires["mulan"],
                    sampled_audio.squeeze(1),
                    [self.extra_params.negative_phrase],
                    device=sampled_audio.device,
                    sampled_embeds=items.get("sampled_mulan_embeds"),
                    target_embeds=self.negative_qualitative_emb,
                )
                qualitative_reward -= negative_sim
            return qualitative_reward
        elif reward_type == "nonvocal":
            if "sampled_lyrics" not in items:
                eos_index_list = items["eos_index_list"]
                items["sampled_lyrics"] = asr_transcribe_lyrics(
                    self.requires,
                    sampled_audio,
                    sample_rate=self.extra_params.sample_rate,
                    sample_lengths=None if len(eos_index_list) == 0 else eos_index_list
                )
            sampled_lyrics = items["sampled_lyrics"]
            if len(sampled_lyrics) != sampled_audio.size(0):
                # This sometimes happens, not sure why
                print(f"lyrics: len={len(sampled_lyrics)} (expected {sampled_audio.size(0)}), content={sampled_lyrics}")
                return 0
            return nonvocal_reward(sampled_lyrics, device=sampled_audio.device)
        elif reward_type == "chord":
            # Use genre-specific chord LM if possible, otherwise fall back to default LM
            chord_lm_keys = []
            for i in range(sampled_audio.size(0)):
                style_metadata = batch.get("style_metadata")
                genres = []
                if style_metadata is not None:
                    genres = style_metadata[i // beam].get("genres", [])
                    genres = ["_".join(g.lower().split()) for g in genres]
                    genres = [g for g in genres if g in self.requires["chord_lms"]]
                if len(genres) == 0 and "default" in self.requires["chord_lms"]:
                    genres = ["default"]
                # Final filter
                genres = [g for g in genres if self.requires["chord_lms"][g] is not None]
                chord_lm_keys.append(genres)
            return chord_reward(
                self.requires["chord"],
                self.requires["chord_lms"],
                sampled_audio,
                chord_lm_keys=chord_lm_keys,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "structure":
            return structure_reward(
                self.requires["structure"],
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "chorus_sim":
            return chorus_sim_reward(
                sampled_audio,
                batch["structure"],
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "chorus_presence":
            return chorus_presence_reward(
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "audio_metrics":
            return audio_metrics_reward(
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "intensity_sim":
            return intensity_sim_reward(
                sampled_audio,
                batch["intensity"],
                calculation_mode=self.extra_params.intensity_calculation,
                intensity_hz=self.extra_params.intensity_hz,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "semantic_diversity":
            return semantic_diversity_reward(
                items["sampled_semantic_tokens"],
                device=sampled_audio.device,
            )
        elif reward_type == "semantic_diversity_sim":
            return semantic_diversity_sim_reward(
                items["sampled_semantic_tokens"],
                items["target_semantic_tokens"],
                device=sampled_audio.device,
            )
        elif reward_type == "chroma":
            return chroma_reward(
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "chroma_sim":
            return chroma_sim_reward(
                sampled_audio,
                target_audio,
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        elif reward_type == "anchor_points_sim":
            return anchor_points_sim_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                sample_rate=self.extra_params.sample_rate,
                device=sampled_audio.device,
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


def process_eos_indexes(semantic_samples, semantic_module: SemanticModule, sample_rate=24000):
    semantic_samples = semantic_samples.clone()
    semantic_frame_rate = semantic_module.extra_params.semantic_frame_rate
    eos_id = semantic_module.target_embedder.eos_id
    sos_id = semantic_module.target_embedder.sos_id
    eos_index_list = []

    if (semantic_samples == sos_id).any():
        # Hacky fix: sometimes model can predict SOS token. Here, we set it to EOS to discourage output. Diffusion has no concept of SOS
        print('Found SOS in semantic tokens. Setting to 0:', semantic_samples.shape, (semantic_samples == sos_id).sum())
        semantic_samples[semantic_samples == sos_id] = 0

    if eos_id is not None:
        """
        @renyi 08/02/2024: if we set eos_padding_id to 0, the "semantic_samples == eos_padding_id" will also include the real acoustic code 0, 
        leading to end of sentence when the real acoustic code 0 appears.
        So we set a EOS padding id (-10000) to a placeholder which is impossibly shown in the acoustic tokens. 
        """
        eos_padding_id = -10000
        eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
        semantic_samples[eos_mask] = eos_padding_id
        token2wav_rate = int(sample_rate / semantic_frame_rate)
        eos_index_list = ((semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0).bool().sum(
            axis=1) * token2wav_rate
        # @qinxin: temp fix, currently tokenizer_pad_id = eos_id - 1
        semantic_samples[eos_mask] = eos_id - 1
    return semantic_samples, eos_index_list


def truncate_wav_to_eos(wavs, eos_index_list):
    if torch.is_tensor(wavs) and wavs.ndim == 1:
        wavs = [wavs]
    truncated_wavs = []
    for i, (eos, wav) in enumerate(zip_longest(eos_index_list, wavs)):
        if eos is not None:
            wav = wav[..., :eos]
        truncated_wavs.append(wav)
    return truncated_wavs

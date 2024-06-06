from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import (
    MulanEmbedder,
    MulanCategoricalEmbedder,
    LyricsTokenEmbedder,
    TagCategoricalEmbedder,
    MultiTagsCategoricalEmbedder,
    LeadsheetTokenEmbedderV2,
    REMILeadsheetTokenEmbedder,
    WavToVecTokenEmbedder,
    MetadataT5TokenEmbedder,
    BestRQTokenEmbedder,
    DurationEmbedder,
    StructureEmbedder,
    IntensityEmbedder,
    SpeakerEmbedder,
    KeyEmbedder,
    TempoLabelEmbedder,
    BeatEmbedder,
    IntEmbedder,
    AudioKeyEmbedder,
    ChordSeqEmbedder,
)
from recipes.bigmusic.datasets.mir_data_util import convert_m1_tag_to_style_text
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics
from recipes.bigmusic.utils.mulan_tag import get_mulan_tags
from samantha.utils import groundtruth
try:
    from recipes.bigmusic.utils.rewards import (
        mulan_audio_reward,
        mulan_text_reward,
        wer_reward,
        mir_tag_reward,
        loudness_reward,
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
        anchor_points_sim_reward,
        mulan_temporal_reward,
        chroma_temporal_reward,
    )
except Exception as e:
    pass
import numpy as np
import random
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F

from recipes.umm.utils.mss import MSSPredictor
from samantha.criterion.masked_loss import sequence_mask
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.hparams import DotDict
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.musiclm.transforms.audio import to_energy
from collections import defaultdict
from copy import deepcopy
from itertools import zip_longest
from typing import Optional
from recipes.mi1.models.music_sft import get_m1_tags
from torchaudio.transforms import Resample
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from recipes.mulan.inference.stats.sstk_anchor_points import load_anchor_points

from recipes.audio_quality_classifier.models.audio_quality_model.utils import aq_classifier_inference

DEFAULT_REWARDS = {"mulan_sim": 1.0, "wer": 1.0}


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
        style_category_vocab_size = extra_params.get('style_category_vocab_size', 256)
        speaker_vocab_size = extra_params.get('speaker_codebook_size', 10)
        key_vocab_size = extra_params.get('key_codebook_size', 25)  # 1 empty + 24 keys
        tempo_label_vocab_size = extra_params.get('tempo_label_codebook_size', 9)  # 1 empty + 8 labels
        offset_codebook_size = extra_params.get('offset_codebook_size', 512)
        tag_taxonomy_lang = extra_params.get('tag_taxonomy_lang', 'SA')
        tag_dropout_rate = extra_params.get('tag_dropout_rate', 0)
        mulan_add_cfg = extra_params.get('mulan_add_cfg', False)
        mulan_embed_dim = extra_params.get('mulan_embed_dim', 512)
        mulan_crop = extra_params.get('mulan_crop', True)
        mulan_average = extra_params.get('mulan_average', True)
        style_category_vocab_path = extra_params.get('style_category_vocab_path')
        chord_vocab_size = extra_params.get('chord_vocab_size', 73)  # 1 empty + 12 key * (maj, min, sus2, sus4, dim, aug)

        self.augmented_noise_gain = extra_params.get('augmented_noise_gain', 0.01)
        self.prepare_input_types = extra_params.get("prepare_input_types", []) # m1_tagger, mulan_tagger
        embedder_dict = {}
        for emb_type in extra_params.get("input_embedders", ["mulan", "lyrics_tokens"]):
            if emb_type == "mulan":
                # Support either Mulan audio or text embedding of style_text / on the fly tag
                embedder_dict[emb_type] = MulanEmbedder(
                    input_dim=mulan_embed_dim,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    mulan_crop=mulan_crop,
                    mulan_average=mulan_average,
                    dropout=tag_dropout_rate,
                    add_none=mulan_add_cfg
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
            elif emb_type == "mulan_categorical":
                # Read ground truth tags from style_text
                embedder_dict[emb_type] = MulanCategoricalEmbedder(
                    input_dim=mulan_embed_dim,
                    vocab_size=style_category_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    dropout=tag_dropout_rate,
                    vocab_path=style_category_vocab_path,
                )
            elif emb_type == "speaker_id":
                embedder_dict[emb_type] = SpeakerEmbedder(
                    vocab_size=speaker_vocab_size, 
                    embedding_dim=hidden_size, 
                    add_sos=True)
            elif emb_type == "key":
                embedder_dict[emb_type] = KeyEmbedder(
                    vocab_size=key_vocab_size, 
                    embedding_dim=hidden_size, 
                )
            elif emb_type == "tempo_label":
                embedder_dict[emb_type] = TempoLabelEmbedder(
                    vocab_size=tempo_label_vocab_size, 
                    embedding_dim=hidden_size, 
                )
            elif emb_type == "lyrics_tokens":
                embedder_dict[emb_type] = LyricsTokenEmbedder(
                    vocab_size=lyrics_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=False,
                )
            elif emb_type == "chord_seq":
                embedder_dict[emb_type] = ChordSeqEmbedder(
                    vocab_size=chord_vocab_size, 
                    embedding_dim=hidden_size,
                    add_sos=True,
                    add_eos=False,
                )
            elif emb_type == "duration":
                embedder_dict[emb_type] = DurationEmbedder(
                    durations=extra_params["duration"],
                    embedding_dim=hidden_size,
                )
            elif emb_type == "structure":
                embedder_dict[emb_type] = StructureEmbedder(
                    durations=extra_params["duration"],
                    embedding_dim=hidden_size,
                    structure_labels=extra_params["structure_labels"],
                    granularity_in_secs=extra_params["granularity_in_secs"],
                )
            elif emb_type == "leadsheet_tokens":
                leadsheet_vocab_size = extra_params['leadsheet_codebook_size']
                embedder_dict[emb_type] = LeadsheetTokenEmbedderV2(
                    vocab_size=leadsheet_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True
                )
            elif emb_type == "remi_leadsheet_tokens":
                remi_leadsheet_vocab_size = extra_params['remi_leadsheet_codebook_size']
                embedder_dict[emb_type] = REMILeadsheetTokenEmbedder(
                    vocab_size=remi_leadsheet_vocab_size,
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
            elif emb_type == "offset_token":
                embedder_dict[emb_type] = IntEmbedder(
                    vocab_size=offset_codebook_size,
                    embedding_dim=hidden_size,
                    add_sos=True
                )
            elif emb_type in ["acc_audio", "vocal_audio"]:
                embedder_dict[emb_type] = BestRQTokenEmbedder(
                    vocab_size=semantic_codebook_size, 
                    embedding_dim=hidden_size, 
                    add_sos=True,
                    add_eos=False
                )
            elif emb_type in ["prefix_audio"]:
                # Here we use ground truth audio as prefix, will use target_embedder as embedder
                embedder_dict[emb_type] = None
            elif emb_type == "intensity":
                embedder_dict[emb_type] = IntensityEmbedder(
                    decimals=extra_params["intensity_decimals"],
                    embedding_dim=hidden_size,
                    intensity_hz=extra_params.get("intensity_hz", 1),
                )
            elif emb_type == "beat":
                duration = extra_params["duration"]
                max_duration = duration[-1] if isinstance(duration, (list, tuple)) else duration
                embedder_dict[emb_type] = BeatEmbedder(
                    beat_labels=extra_params["beat_labels"],
                    embedding_dim=hidden_size,
                    max_duration=max_duration,
                )
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
        input_embedders = nn.ModuleDict(embedder_dict)

        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            chunk_size = extra_params.get("semantic_chunk_size", None)
            if chunk_size is not None:
                chunk_size = extra_params["sample_rate"] * chunk_size
            store_hidden_states = "m1_tags" in self.prepare_input_types
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
                chunk_size=chunk_size,
                store_hidden_states=store_hidden_states
                
            )
        else:
            raise NotImplementedError

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
        self.mulan_counter = 0

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

    def infer_target_duration(self, batch):
        if "duration" in batch:
            target_duration = batch["duration"]
        elif "target_audio" in batch:
            target_duration = batch["target_audio"].shape[-1] // self.extra_params.sample_rate
        else:
            target_duration = None
        return target_duration

    def prepare_mulan_inputs(self, batch, mulan_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        target_duration = self.infer_target_duration(batch)
        target_samples_length = target_duration * self.extra_params.sample_rate
        if 'style_text' in conditions:
            assert "style_text" in batch
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_text'],
                with_sos=True,
                data_type='text',                
            )
        elif 'style_audio' in conditions:
            assert "style_audio" in batch or "target_audio" in batch
            embeds = mulan_embedder.embed(
                self.requires,
                batch.get('style_audio', batch.get('target_audio')).to(self.device),
                with_sos=True,
                data_type='music',
                target_samples_length=target_samples_length,
            )
        elif 'style_embedding' in conditions:
            assert "style_embedding" in batch
            embeds = mulan_embedder.embed(
                self.requires,
                batch.get('style_embedding').to(self.device),
                with_sos=True,
                data_type='embed',
                target_samples_length=target_samples_length,
            )
        elif 'style_category' in conditions:
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_category'],
                with_sos=True,
                data_type='category'
            )
        elif 'style_tag' in conditions: # using Mulan for on-the-fly MIR tagging
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_audio'].to(self.device),
                mcc_style_text=batch.get('style_text'),
                with_sos=True,
                data_type='tag',
            )
        elif 'style_none' in conditions: # using Mulan for on-the-fly MIR tagging
            embeds = mulan_embedder.embed(
                self.requires,
                batch['style_text'],
                with_sos=True,
                data_type='none',
            )
        else:
            # adding SOS token no matter what so that all parameters get used
            embeds = mulan_embedder.get_sos_embed(batch_size)
        if self.mulan_counter < 10:
            print(f"target_duration: {target_duration}, mulan_emb: {embeds.shape}")
            self.mulan_counter += 1
        return embeds

    def prepare_lyrics_inputs(self, batch, lyrics_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'lyrics_tokens' in conditions:
            embeds = lyrics_embedder.embed(
                self.requires,
                batch['lyrics_tokens'].to(self.device),
                with_sos=True,
            )
        else:
            embeds = lyrics_embedder.get_sos_embed(batch_size)
        return embeds
    
    def prepare_chord_seq_inputs(self, batch, chord_seq_embedder, add_eos=False):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        if "chord_seq" in conditions:
            chord_seq_tokens = batch['chord_seq_tokens'].to(self.device)
        else:
            chord_seq_tokens = torch.zeros((batch_size,  0)).long().to(self.device)
        embeds = chord_seq_embedder.embed(token_ids=chord_seq_tokens, with_sos=True).to(self.device)
        return embeds

    def prepare_remi_leadsheet_inputs(self, batch, embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'remi_leadsheet_tokens' in conditions:
            embeds = embedder.embed(
                self.requires,
                batch['remi_leadsheet_tokens'].to(self.device),
                with_sos=True,
            )
        else:
            embeds = embedder.get_sos_embed(batch_size)
        return embeds

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
        return embeds

    def prepare_offset_inputs(self, batch, embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'offset_token' in conditions:
            if 'offset_token' in batch:
                offset_token = batch['offset_token']
            else:
                offset_token = [0] * batch_size
                offset_token = torch.as_tensor(offset_token)
                
            embeds = embedder.embed(
                self.requires,
                offset_token.to(self.device),
                with_sos=True,
            )
        else:
            embeds = embedder.get_sos_embed(batch_size)
        return embeds


    def prepare_leadsheet_inputs(self, batch, embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'leadsheet_tokens' in conditions:
            embeds = embedder.embed(self.requires, batch['leadsheet_tokens'].to(
                self.device), batch['leadsheet_tokens_coff'].to(self.device), with_sos=True)
        else:
            embeds = embedder.get_sos_embed(batch_size)
        return embeds
    def prepare_duration_inputs(self, batch, duration_embedder):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        if 'duration' in conditions:
            duration = self.infer_target_duration(batch)
            embeds = duration_embedder.embed(duration, batch_size)
        else:
            embeds = duration_embedder.empty_embed(batch_size)
        return embeds

    def prepare_structure_inputs(self, batch, structure_embedder):
        batch_size = self.infer_batch_size(batch)
        structure_labels = batch["structure"]
        assert batch_size == len(structure_labels)
        # Dropout if needed
        structure_dropout = self.extra_params.get("structure_dropout", 0.0)
        if self.training and structure_dropout > 0:
            all_keep = [x >= structure_dropout for x in np.random.rand(batch_size)]
            for i, keep in enumerate(all_keep):
                if not keep:
                    structure_labels[i] = None
        # Find target duration
        target_duration = self.infer_target_duration(batch)
        embeds = structure_embedder.embed(structure_labels, target_duration)
        return embeds
    
    def prepare_categorical_inputs(self, batch, categorical_embedder):
        if 'style_category' not in batch:
            assert 'style_text' in batch
            batch["style_category"] = batch['style_text']
        embeds = categorical_embedder.embed(
            self.requires,
            batch['style_category'],
            with_sos=True,
        )
        return embeds

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
        return embeds

    def prepare_speaker_inputs(self, batch, speaker_embedder):
        return self._prepare_one_frame_inputs(batch, speaker_embedder, 'speaker_id')

    def prepare_key_inputs(self, batch, key_embedder):
        return self._prepare_one_frame_inputs(batch, key_embedder, 'key')

    def prepare_tempo_label_inputs(self, batch, tempo_label_embedder):
        return self._prepare_one_frame_inputs(batch, tempo_label_embedder, 'tempo_label')

    # TODO: make this into a static method (vibertthio)
    def prepare_acc_audio_inputs(self, batch, acc_embedder: BestRQTokenEmbedder):
        conditions = self.infer_conditions(batch)
        batch_size = self.infer_batch_size(batch)
        if "acc_audio" in conditions:
            embeds = acc_embedder.embed(self.requires, batch['acc_audio'], with_sos=True)
        else:
            embeds = acc_embedder.get_sos_embed(batch_size)
        return embeds

    # TODO: make this into a static method (vibertthio)
    def prepare_vocal_audio_inputs(self, batch, vocal_embedder: BestRQTokenEmbedder):
        conditions = self.infer_conditions(batch)
        batch_size = self.infer_batch_size(batch)
        
        if "noisy_vocal_audio" in conditions:
            vocal_audio = batch['vocal_audio']
            noise = (torch.rand_like(vocal_audio) - 0.5) * self.augmented_noise_gain
            vocal_audio = vocal_audio + noise
            embeds = vocal_embedder.embed(self.requires, vocal_audio, with_sos=True)
        elif "vocal_audio" in conditions:
            embeds = vocal_embedder.embed(self.requires, batch['vocal_audio'], with_sos=True)
        else:
            embeds = vocal_embedder.get_sos_embed(batch_size)
        return embeds

    def prepare_prefix_audio_inputs(self, batch):
        # The prefix_audio inputs is for audio continuation
        # extract embedding from known audio using target embedder, and let model predict the rest
        # So we want SOS token from target embedder before audio embedding
        # And skip adding SOS token in model predict
        conditions = self.infer_conditions(batch)
        batch_size = self.infer_batch_size(batch)
        if "prefix_audio" in conditions: 
            embeds = self.target_embedder.embed(self.requires, batch['prefix_audio'], with_sos=True)
            #temp fixed by add sos token to makeup the gap between training and inference
            embeds = torch.cat([self.target_embedder.get_sos_embed(batch_size), embeds], dim=1)
        else:
            embeds = self.target_embedder.get_sos_embed(batch_size)
        return embeds

    def prepare_prefix_audio_woembedder(self, inputs_embeds, batch):
        # this function works in inference when you need a prefix_audio as prompt, while
        # there is no prefix_audio embedder in training
        assert 'prefix_audio' in batch
        emb_inputs = self.target_embedder.embed(self.requires, batch['prefix_audio'], with_sos=True)
        if self.log_counter < 1:
            print(f"{'prefix_audio'} emb input shape: ", emb_inputs.shape)
        inputs_embeds.append(emb_inputs)
        # en_idx = st_idx + emb_inputs.shape[1]
        # batch["inputs_embeds_span"]['prefix_audio'] = (st_idx, en_idx)
        # st_idx = en_idx
        return inputs_embeds


    def prepare_intensity_inputs(self, batch, intensity_embedder):
        batch_size = self.infer_batch_size(batch)
        intensity_labels = batch["intensity"]
        assert batch_size == len(intensity_labels)
        target_duration = self.infer_target_duration(batch)
        embeds = intensity_embedder.embed(intensity_labels, target_duration)
        return embeds

    def prepare_beat_inputs(self, batch, beat_embedder):
        batch_size = self.infer_batch_size(batch)
        if self.training and random.random() < self.extra_params.get("beat_dropout", 0.0):
            batch["beat"] = [[] for _ in range(batch_size)]
        elif "beat" not in batch:
            target_audio = None
            if "beat_audio" in batch:
                target_audio = batch["beat_audio"]
            elif "target_audio" in batch:
                target_audio = batch["target_audio"]
            elif "style_audio" in batch:
                target_audio = batch["style_audio"]
            if target_audio is None:
                batch["beat"] = [[] for _ in range(batch_size)]
            else:
                batch["beat"] = self.requires["beat"].predict_step({"target_audio": target_audio}, 0)
        assert batch_size == len(batch["beat"])
        target_duration = self.infer_target_duration(batch)
        embeds, beat_ids, beat_timestamps = beat_embedder.embed(batch["beat"], target_duration)
        batch["beat_ids"] = beat_ids
        batch["beat_timestamps"] = beat_timestamps
        return embeds

    def prepare_batch_inputs(self, batch):
        """For extra batch preparation that requires GPU"""
        if "m1_tagger" in self.prepare_input_types:
            assert isinstance(self.target_embedder, BestRQTokenEmbedder), "m1 requires target_hidden_states to predict categories"
            assert self.target_embedder.hidden_states is not None, "m1 requires target_hidden_states to predict categories"
            assert 'target_audio' in batch, "m1 requires target_audio to predict categories"
            target_audio = batch["target_audio"]
            target_hidden_states = self.target_embedder.hidden_states
            m1_tags = get_m1_tags(self.requires, target_hidden_states, target_audio)
            batch["style_category"] = convert_m1_tag_to_style_text(m1_tags)

        if "mulan_tagger" in self.prepare_input_types:
            mulan_tags = get_mulan_tags(self.requires, batch["target_audio"])
            batch["style_category"] = mulan_tags
        return batch

    def prepare_inputs_embeddings(self, batch, input_embedders=None):
        if input_embedders is None:
            input_embedders = self.input_embedders.items()

        batch = self.prepare_batch_inputs(batch)
        if self.log_counter < 1:
            print(batch)

        st_idx = 0
        inputs_embeds = []
        batch["inputs_embeds_span"] = {}
        for emb_type, embedder in input_embedders:
            if (emb_type == "mulan") or (emb_type == "mulan_categorical"):
                emb_inputs = self.prepare_mulan_inputs(batch, embedder)
            elif emb_type == "tag_categorical":            
                emb_inputs = self.prepare_categorical_inputs(batch, embedder)
            elif emb_type == "multitags_categorical":
                emb_inputs = self.prepare_categorical_inputs(batch, embedder)
            elif emb_type == "lyrics_tokens":
                emb_inputs = self.prepare_lyrics_inputs(batch, embedder)
            elif emb_type == "speaker_id":
                emb_inputs = self.prepare_speaker_inputs(batch, embedder)
            elif emb_type == "key":
                emb_inputs = self.prepare_key_inputs(batch, embedder)
            elif emb_type == "tempo_label":
                emb_inputs = self.prepare_tempo_label_inputs(batch, embedder)
            elif emb_type == "audio_key_token":
                emb_inputs = self.prepare_audio_key_inputs(batch, embedder)
            elif emb_type == "offset_token":
                emb_inputs = self.prepare_offset_inputs(batch, embedder)
            elif emb_type == "leadsheet_tokens":
                emb_inputs = self.prepare_leadsheet_inputs(batch, embedder)
            elif emb_type == "remi_leadsheet_tokens":
                emb_inputs = self.prepare_remi_leadsheet_inputs(batch, embedder)
            elif emb_type == "duration":
                emb_inputs = self.prepare_duration_inputs(batch, embedder)
            elif emb_type == "structure":
                emb_inputs = self.prepare_structure_inputs(batch, embedder)
            elif emb_type == "acc_audio":
                emb_inputs = self.prepare_acc_audio_inputs(batch, embedder)
            elif emb_type == "vocal_audio":
                emb_inputs = self.prepare_vocal_audio_inputs(batch, embedder)
            elif emb_type == "intensity":
                emb_inputs = self.prepare_intensity_inputs(batch, embedder)
            elif emb_type == "beat":
                emb_inputs = self.prepare_beat_inputs(batch, embedder)
            elif emb_type == "prefix_audio":
                emb_inputs = self.prepare_prefix_audio_inputs(batch)
            elif emb_type == "chord_seq":
                emb_inputs = self.prepare_chord_seq_inputs(batch, embedder)
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
            if self.log_counter < 1:
                print(f"{emb_type} emb input shape: ", emb_inputs.shape)
            inputs_embeds.append(emb_inputs)
            en_idx = st_idx + emb_inputs.shape[1]
            batch["inputs_embeds_span"][emb_type] = (st_idx, en_idx)
            st_idx = en_idx

        if not self.training and 'prefix_audio' in self.infer_conditions(batch):
            inputs_embeds = self.prepare_prefix_audio_woembedder(inputs_embeds, batch)

        self.log_counter += 1
        return torch.cat(inputs_embeds, dim=1)

    def aux_loss(self, batch, logits, last_hidden_state, target_length):
        assert (
            "inputs_embeds_span" in batch
        ), f"Check `prepare_inputs_embeddings` to make sure `inputs_embeds_span` is inserted in `batch`"
        aux_loss = 0
        aux_dict = {}
        for pred_type, pred_head in self.prediction_heads.items():
            st_idx, en_idx = batch["inputs_embeds_span"][pred_type]
            pred_wt = self.prediction_weights[pred_type]
            if pred_type == "intensity":
                target_intensity = self.input_embedders["intensity"].quantize(batch["intensity"])
                intensity_length = target_intensity.shape[1]
                assert (
                    intensity_length == en_idx - st_idx
                ), f"{intensity_length} != {en_idx} - {st_idx}"
                # Offset by one
                st = st_idx - 1
                intensity_embed = last_hidden_state[:, st:st + intensity_length, :]
                intensity_logits = pred_head(intensity_embed)
                pred_loss = self.criterion(intensity_logits, target_intensity)
                pred_accu = (intensity_logits.argmax(dim=-1) == target_intensity).float().mean() * 100
                pred_dict = {
                    "intensity_loss": pred_loss.item(),
                    "intensity_accu": pred_accu.item(),
                }
            elif pred_type == "beat":
                target_beat_ids = batch["beat_ids"]
                target_timestamps = batch["beat_timestamps"]
                assert (
                    target_beat_ids.shape[-1] == en_idx - st_idx
                ), f"{target_beat_ids.shape[-1]} != {en_idx} - {st_idx}"
                # Offset by one
                st = st_idx - 1
                beat_embed = last_hidden_state[:, st:st + target_beat_ids.shape[-1], :]
                beat_logits = pred_head(beat_embed)
                beat_ids_logits = beat_logits[..., :-1]
                beat_timestamps = torch.clamp(
                    beat_logits[..., -1],
                    min=-self.input_embedders["beat"].max_timestamp,
                    max=self.input_embedders["beat"].max_timestamp,
                )
                beat_ids_loss = self.criterion(beat_ids_logits, target_beat_ids)
                beat_ids_accu = (beat_ids_logits.argmax(dim=-1) == target_beat_ids).float().mean() * 100
                beat_timestamps_loss = ((beat_timestamps - target_timestamps) ** 2).mean()
                pred_loss = beat_ids_loss + beat_timestamps_loss
                pred_dict = {
                    "beat_ids_loss": beat_ids_loss.item(),
                    "beat_ids_accu": beat_ids_accu.item(),
                    "beat_timestamps_loss": beat_timestamps_loss.item(),
                }
            else:
                raise ValueError(f"Unknown pred type: {pred_type}")
            aux_loss += pred_loss * pred_wt
            aux_dict.update(pred_dict)
        return aux_loss, aux_dict

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)
            model_inputs = training_inputs['model_inputs']  # If use_cross_attn is False { "input_embedds": (B, T_input+T_target, hidden_size) }
            target_ids = training_inputs['target_ids']  # (B, T_target)

        if update_mfu:
            if "inputs_embeds" in model_inputs:
                b, t, _ = model_inputs["inputs_embeds"].shape
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

        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            model_output = self.model(**model_inputs, output_hidden_states=True)
        if isinstance(model_output, dict):
            logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            logits, last_hidden_state = model_output
        target_length = target_ids.size(1)
        target_logits = logits[:, -target_length:, :]
        loss_mask = None
        if 'target_lengths' in training_inputs:
            loss_mask = sequence_mask(
                training_inputs['target_lengths'], max_len=target_ids.shape[1], device=target_ids.device)
        loss = self.criterion(target_logits, target_ids, loss_mask)
        if loss_mask is None:
            loss_mask = torch.ones_like(target_ids)

        accu = ((target_logits.argmax(dim=-1) == target_ids).float() * loss_mask).sum() / loss_mask.sum() * 100
        # measure accuracy of first 10 tokens as a measurement for style
        accu_seq_25 = ((target_logits.argmax(dim=-1)[..., :25] == target_ids[..., :25])
                       * loss_mask[..., :25]).sum() / loss_mask[..., :25].sum() * 100
        result_dict = {
            'loss': loss.item(),
            'accu': accu.item(),
            'accu_seq_25': accu_seq_25.item()
        }
        # Auxiliary loss
        aux_loss, aux_dict = self.aux_loss(batch, logits, last_hidden_state, target_length)
        loss += aux_loss
        result_dict.update(aux_dict)

        return loss, result_dict

    @torch.no_grad()
    def _predict_intensity(
        self,
        batch,
        intensity_embedder,
        intensity_head,
        intensity_temperature=1.0,
        tqdm_name="Intensity Prediction",
    ):
        target_duration = self.infer_target_duration(batch)
        target_length = int(target_duration * intensity_embedder.intensity_hz)
        input_embedders = list(self.input_embedders.items())
        cutoff = None
        for i, embedder in enumerate(input_embedders):
            if embedder[0] == "intensity":
                cutoff = i
                break
        assert cutoff is not None, "Can't find cutoff, this shouldn't happen"

        inputs_embeds = self.prepare_inputs_embeddings(batch, input_embedders[:cutoff])
        batch_size, seq_len, _ = inputs_embeds.size()
        model_input = {"inputs_embeds": inputs_embeds}

        output_tokens = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            gpt_max_seq_len = 4000 if target_length < 2500 else 8000
            inference_params = InferenceParams(
                max_sequence_len=4000, max_batch_size=batch_size
            )
        else:
            past_key_values = None
        pbar = tqdm(range(target_length))
        previous_inputs_embeds = model_input["inputs_embeds"]
        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {target_length}]")

            if isinstance(self.model, gpt.GPTLMHeadModel):
                model_output = self.model(
                    **model_input,
                    inference_params=inference_params,
                    position_ids=None,
                    last_token_only=False,
                    output_hidden_states=True
                )
                last_hidden_state = model_output.hidden_states
                inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            else:
                model_output = self.model(
                    **model_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = model_output["past_key_values"]
                last_hidden_state = model_output['hidden_states'][-1]

            target_logits = intensity_head(last_hidden_state[:, -1:]) # only predict for last logit
            predict_token = self.sample_logits(i, target_logits, intensity_temperature, "top_p")
            predict_token_emb = intensity_embedder.embedder(predict_token)
            model_input = {"inputs_embeds": predict_token_emb}
            previous_inputs_embeds = torch.cat([previous_inputs_embeds, predict_token_emb], dim=1)
            output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token
        return intensity_embedder.unquantize(output_tokens)
    
    @torch.no_grad()
    def prepare_cfg_batch(self, batch, hp):
        controller_cfg_label = hp.get('controller_cfg_label', "genre")
        rewrite_target = hp.get("rewrite_target", "multi_tag")
        batch_cfg = deepcopy(batch)
        
        # instrumental use case
        if 'mulan' in self.input_embedders:
            if controller_cfg_label == 'negative_anchor':
                batch_cfg['conditions'] = ['style_embedding']
                batch_size = self.infer_batch_size(batch)
                good, bad = load_anchor_points()
                bad_t = torch.tensor(bad, device=self.device).repeat(batch_size, 1)
                batch_cfg['style_embedding'] = bad_t.unsqueeze(1)
            else:
                batch_cfg['conditions'] = ['style_none']
                batch_size = self.infer_batch_size(batch)
                batch_cfg['style_text'] = [''] * batch_size
                batch_cfg['style_audio'] = [''] * batch_size
            return batch_cfg

        # vocal use case
        cfg_style_text = []
        for x in batch['style_text']:
            x = x.split('|')
            print(x, ' apply cfg to : ', controller_cfg_label)
            if rewrite_target in ['multi_tag', 'multi_tag_v3', 'multi_tag_combo_v3', 'multi_tag_combo_v3_5']:
                x[0] = '' if 'genre' in controller_cfg_label else x[0]
                x[1] = '' if 'mood' in controller_cfg_label else x[1]
                x[2] = '' if 'scene' in controller_cfg_label else x[2]
                x[3] = '' if 'speaker' in controller_cfg_label else x[3]
                x[4] = '' if 'voice' in controller_cfg_label else x[4]
            else:
                # Default to using sa_tag
                x[0] = '' if 'genre' in controller_cfg_label else x[0]
                x[1] = '' if 'mood' in controller_cfg_label else x[1]
                x[2] = '' if 'scene' in controller_cfg_label else x[2]
                x[3] = '' if 'sinking' in controller_cfg_label else x[3]
                x[4] = '' if 'lang' in controller_cfg_label else x[4]
            print(x)
            cfg_style_text.append('|'.join(x))
        batch_cfg['style_text'] = cfg_style_text
        batch_cfg['style_category'] = cfg_style_text
        if "speaker" in controller_cfg_label:                
            print(batch_cfg['speaker_id'], ' apply cfg to speaker')
            batch_cfg['speaker_id'] = torch.as_tensor([0] * len(batch['style_category']))
            print(batch_cfg['speaker_id']) 
        if "section_tag" in controller_cfg_label:
            print('apply cfg to section_tag')
            batch_cfg["lyrics_tokens"] = batch["lyrics_tokens_cfg"]
            batch_cfg["lyrics_normalized_text"] = batch["lyrics_normalized_text_cfg"]
            batch_cfg["lyrics_tokens_length"] = batch["lyrics_tokens_length_cfg"]
            # delete cfg fields
            for b in [batch_cfg, batch]:
                for k in ["lyrics_tokens_cfg", "lyrics_normalized_text_cfg", "lyrics_tokens_length_cfg"]:
                    b.pop(k, None)

        return batch_cfg


    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None, rl_training=False):
        frame_rate = self.extra_params.semantic_frame_rate
        if "duration" not in batch:
            batch["duration"] = hp.duration
        duration = batch["duration"]
        num_tokens = duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode
        sample_thresh = hp.get('sample_thresh', 0.9)
        use_controller_cfg = hp.get('use_controller_cfg', False)
        controller_cfg_gamma = hp.get('controller_cfg_gamma', 1)
        use_step_out_blank = hp.get('use_step_out_blank', False)
        step_out_blank_logic = hp.get('step_out_blank_logic', 'v2')
        step_out_blank_max_len = hp.get('step_out_blank_max_len', 10)
        repetition_penalty = hp.get('repetition_penalty', 1.0)
        skip_sos = hp.get('skip_sos', False)
        self.extra_params.debug_index = hp.get('debug_index', None)
        exclude_ids = None
        if hp.get("exclude_eos", False) and self.target_embedder.eos_id is not None:
            exclude_ids = [self.target_embedder.eos_id]
            print(f"exclude_ids: {exclude_ids}")
        
        # Predict intensity if it's not available
        if "intensity" in self.input_embedders and "intensity" not in batch:
            assert not self.training, "intensity should be provided in training!"
            batch["intensity"] = self._predict_intensity(
                batch,
                self.input_embedders["intensity"],
                self.prediction_heads["intensity"],
                hp.get("intensity_temperature", 1.0),
            )

        if "inputs_embeds" in batch:
            inputs_embeds = batch["inputs_embeds"]
        else:
            inputs_embeds = self.prepare_inputs_embeddings(batch)

        inputs_emb_cfg = None
        if use_controller_cfg:
            batch_cfg = self.prepare_cfg_batch(batch, hp)
            inputs_emb_cfg = self.prepare_inputs_embeddings(batch_cfg)

        semantic_tokens = super().predict(
            batch,
            inputs_embeds,
            num_tokens,
            temperature=temperature,
            beam=beam,
            sample_mode=sample_mode,
            sample_thresh=sample_thresh,
            ref_samples=ref_samples,
            exclude_ids=exclude_ids,
            rl_training=rl_training,
            skip_sos=skip_sos,
            use_controller_cfg=use_controller_cfg,
            inputs_embeds_cfg=inputs_emb_cfg,
            controller_cfg_gamma=controller_cfg_gamma,
            use_step_out_blank=use_step_out_blank,
            step_out_blank_logic=step_out_blank_logic,
            step_out_blank_max_len=step_out_blank_max_len,
            repetition_penalty=repetition_penalty,
        )

        groundtruth.emit('semantic', data={
            'batch': batch,
            'inputs_embeds': torch.cat([inputs_embeds, inputs_emb_cfg], dim=0) if use_controller_cfg else inputs_embeds,
            'semantic_tokens': semantic_tokens,
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


class SemanticModuleExtendedTarget(SemanticModule):
    def prepare_target_inputs(self, batch):
        # Prepare audio target ids
        audio_target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False).to(self.device)
        if 'target_tokens_length' in batch:
            target_lengths = batch['target_tokens_length'].to(self.device)
            batch_size = self.infer_batch_size(batch)
        else: # set to default batch length. Silence will happen before EOS
            batch_size, seq_len = audio_target_ids.shape[:2]
            target_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=self.device)

        # Extract remi_leadsheet_ids with sos
        remi_leadsheet_ids = batch['remi_leadsheet_tokens'].to(self.device)
        # Offset remi_leadsheet_ids to avoid vocab conflicts
        offset = self.extra_params.audio_codebook_size
        remi_leadsheet_ids = remi_leadsheet_ids + offset
        # Concat remi_leadsheet_ids on target 
        target_ids = torch.cat([remi_leadsheet_ids, audio_target_ids], dim=1)

        # Move audio token to the end of leadsheet, and update target_lengths accordingly
        remi_token_length = batch['remi_token_length']
        for i in range(batch_size):
            remi_token_offset = remi_token_length[i].item()
            target_ids[i, remi_token_offset: remi_token_offset + len(audio_target_ids[i])] = audio_target_ids[i]
            target_lengths[i] += remi_token_offset

        target_ids = F.pad(target_ids, (1, 1)) # pad for extra sos/eos ids
        target_ids[:, 0] = self.target_embedder.sos_id # add SOS
        eos_indices = (target_lengths+1).unsqueeze(1) # set last index to EOS
        target_ids.scatter_(dim=1, index=eos_indices, value=self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
            'token_seq_lengths': target_lengths
        }

    def predict(self, batch, hp, beam=1, ref_samples=None, rl_training=False):
        offset = self.extra_params.audio_codebook_size
        # exp_index of first target sos of first target token
        exp_index = self.extra_params.segment_max_leadsheet_len + 1
        frame_rate = self.extra_params.semantic_frame_rate

        # Expand generation target length
        if "duration" not in batch:
            batch["duration"] = hp.duration
        duration = batch["duration"]
        num_audio_tokens = duration * frame_rate
        num_all_tokens = exp_index + num_audio_tokens
        batch["duration"] = num_all_tokens//frame_rate

        # Predict all target
        output_tokens = super().predict(batch, hp, beam, ref_samples, rl_training)

        # Slice first target
        generated_leadsheet_tokens = []
        generated_audio_tokens = []
        max_audio_len = 0
        eos_id = self.extra_params.leadsheet_codec.indexer["eos"] + offset

        for output_token in output_tokens.clone():
            output_token = output_token.flatten().cpu().tolist()
            if eos_id in output_token:
                index = output_token.index(eos_id)+1
                leadsheet_token = np.array(output_token[:index])
                audio_token = output_token[index:]
                # remove OOV tokens
                audio_token = np.array(output_token[index:])
                audio_token = audio_token[audio_token<offset]
                leadsheet_token = leadsheet_token[leadsheet_token>=offset]
                leadsheet_token = leadsheet_token - offset
                
                generated_leadsheet_tokens.append(leadsheet_token)
                generated_audio_tokens.append(audio_token)
                max_audio_len = max(max_audio_len, len(audio_token))
            else:
                print("[Warning][SemanticModuleExtendedTarget.predict] no leadsheet EOS in the prediction, \
                       will skip this sample")

        # Pad audio token to same length
        new_generated_audio_tokens = []
        for i in generated_audio_tokens:
            padlen = max_audio_len-len(i)
            padding = [self.target_embedder.eos_id] * padlen
            i = list(i)
            i.extend(padding)
            new_generated_audio_tokens.append(torch.LongTensor(i))
        generated_audio_tokens = torch.stack(new_generated_audio_tokens).to(output_tokens.device)

        # This should not happen in a well trained model, but if happen might crash infer
        # So we in this case we use naive slicing to avoid zero-shape and raise an warning
        if len(generated_leadsheet_tokens) == 0 or len(generated_audio_tokens) == 0:
            generated_leadsheet_tokens = output_token[:, :exp_index]
            generated_audio_tokens = output_token[:, exp_index:]
            print(f"[Warning][SemanticModuleExtendedTarget.predict] no leadsheet EOS in the batch. \
                    generated_leadsheet_tokens={len(generated_leadsheet_tokens)} \
                    generated_audio_tokens={len(generated_audio_tokens)}")


        batch["generated_leadsheet_tokens"] = generated_leadsheet_tokens
        return generated_audio_tokens


class SemanticModuleDualUMMFullTrack(SemanticModule):
    def load_required_modules(self, ignore=()):
        super().load_required_modules(ignore=list(ignore) + ['mss'])
        if "mss" in self.hparams.required_modules:
            mss_config = self.hparams.required_modules["mss"]
            self.requires['mss'] = MSSPredictor(mss_config['ckpt_path'], sr=self.extra_params['sample_rate'],
                                                cache_dir=mss_config['cache_dir']).to(f'cuda:{self.local_rank}')
        # self.requires['mss'] = MSSPredictor('hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/sunyakun.king/sami_models/mss-checkpoint-epoch=235-val_median_sdr_0=12.49.ckpt', sr=self.extra_params['sample_rate'],
        #                                     cache_dir='.module_cache').to(f'cuda:{self.local_rank}')

    def prepare_target_inputs(self, batch):
        # Prepare target ids
        use_full_input = self.requires['Stage3'].config.get('use_full_input', False)
        print("self.requires['Stage3'].config.get('use_full_input'", self.requires['Stage3'].config.get('use_full_input', "no"))
        if not use_full_input and 'audio_vocal' not in batch:
            batch['audio_vocal'], batch['audio_acc'] = self.requires['mss'](batch['target_audio'])
            batch['audio_vocal'], batch['audio_acc'] = batch['audio_vocal'][:, None], batch['audio_acc'][:, None]
        target_ids_vocal = self.target_embedder.tokenize(
            self.requires,
            batch['target_audio'] if use_full_input else batch['audio_vocal'],
            with_sos=False, with_eos=False, wav_type='vocal')
        target_ids_acc = self.target_embedder.tokenize(
            self.requires, batch['target_audio'] if use_full_input else batch['audio_acc'],
            with_sos=False, with_eos=False, wav_type='acc')
        target_ids = torch.stack([target_ids_vocal, target_ids_acc], -1)
        target_ids = torch.flatten(target_ids, 1)

        tr = target_ids[0].cpu().detach().tolist()
        open("train_id.txt", "w").write(str(tr))

        if 'target_tokens_length' in batch:
            target_lengths = batch['target_tokens_length'].to(self.device)
            # DualUMM full track has double token length
            target_lengths = target_lengths * 2
        else: # set to default batch length. Silence will happen before EOS
            batch_size, seq_len = target_ids.shape[:2]
            target_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=self.device)
        target_ids = F.pad(target_ids, (1, 1)) # pad for extra sos/eos ids
        target_ids[:, 0] = self.target_embedder.sos_id # add SOS
        eos_indices = (target_lengths+1).unsqueeze(1) # set last index to EOS
        target_ids.scatter_(1, eos_indices, self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
            'token_seq_lengths': target_lengths
        }

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None, rl_training=False):
        if "duration" not in batch:
            batch["duration"] = hp.duration * 2
        return super().predict(batch, hp, beam, ref_samples, rl_training)


class SemanticModuleDualUMMFullTrack2(SemanticModuleDualUMMFullTrack):
    def prepare_target_inputs(self, batch):
        # Prepare target ids
        use_full_input = self.requires['Stage3'].config.get('use_full_input', False)
        if not use_full_input and 'audio_vocal' not in batch:
            batch['audio_vocal'], batch['audio_acc'] = self.requires['mss'](batch['target_audio'])
            batch['audio_vocal'], batch['audio_acc'] = batch['audio_vocal'][:, None], batch['audio_acc'][:, None]
        target_ids_vocal = self.target_embedder.tokenize(
            self.requires,
            batch['target_audio'] if use_full_input else batch['audio_vocal'],
            with_sos=False, with_eos=False, wav_type='vocal')
        target_ids_acc = self.target_embedder.tokenize(
            self.requires, batch['target_audio'] if use_full_input else batch['audio_acc'],
            with_sos=False, with_eos=False, wav_type='acc')
        cb_size_track = self.extra_params.semantic_codebook_size // 2
        target_ids = torch.stack([target_ids_vocal, target_ids_acc + cb_size_track], -1)
        target_ids = torch.flatten(target_ids, 1)

        if 'target_tokens_length' in batch:
            target_lengths = batch['target_tokens_length'].to(self.device)
            # DualUMM full track has double token length
            target_lengths = target_lengths * 2
        else: # set to default batch length. Silence will happen before EOS
            batch_size, seq_len = target_ids.shape[:2]
            target_lengths = torch.full((batch_size,), fill_value=seq_len, dtype=torch.long, device=self.device)
        target_ids = F.pad(target_ids, (1, 1)) # pad for extra sos/eos ids
        target_ids[:, 0] = self.target_embedder.sos_id # add SOS
        eos_indices = (target_lengths+1).unsqueeze(1) # set last index to EOS
        target_ids.scatter_(1, eos_indices, self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
            'token_seq_lengths': target_lengths
        }

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None, rl_training=False):
        output_tokens = super().predict(batch, hp, beam, ref_samples, rl_training)
        cb_size_half = self.extra_params.semantic_codebook_size // 2
        m = (((output_tokens - cb_size_half) >= 0) & (output_tokens != self.target_embedder.eos_id)).long()
        output_tokens = (output_tokens - cb_size_half) * m + output_tokens * (1 - m)
        return output_tokens


class SemanticModuleDualUMMFullTrackParaPrediction(SemanticModuleDualUMMFullTrack):
    def prepare_target_inputs(self, batch):
        # Prepare target ids
        use_full_input = self.requires['Stage3'].config.get('use_full_input', False)
        if not use_full_input and 'audio_vocal' not in batch:
            batch['audio_vocal'], batch['audio_acc'] = self.requires['mss'](batch['target_audio'])
            batch['audio_vocal'], batch['audio_acc'] = batch['audio_vocal'][:, None], batch['audio_acc'][:, None]
        target_ids_vocal = self.target_embedder.tokenize(
            self.requires,
            batch['target_audio'] if use_full_input else batch['audio_vocal'],
            with_sos=False, with_eos=False, wav_type='vocal')
        target_ids_acc = self.target_embedder.tokenize(
            self.requires, batch['target_audio'] if use_full_input else batch['audio_acc'],
            with_sos=False, with_eos=False, wav_type='acc')

        if self.extra_params.get('mask_acc_tokens', False):
            target_ids_acc = torch.zeros_like(target_ids_acc)

        target_lengths = batch['target_tokens_length'].to(self.device)

        cb_size_half = self.extra_params.semantic_codebook_size // 2

        def set_sos_eos(target_ids):
            target_ids = F.pad(target_ids, (1, 1))  # pad for extra sos/eos ids
            target_ids[:, 0] = self.target_embedder.sos_id  # add SOS
            eos_indices = (target_lengths + 1).unsqueeze(1)  # set last index to EOS
            target_ids.scatter_(1, eos_indices, self.target_embedder.eos_id)
            return target_ids

        # for input
        target_ids_vocal = set_sos_eos(target_ids_vocal)
        target_ids_acc = set_sos_eos(target_ids_acc + cb_size_half)
        target_ids = torch.stack([target_ids_vocal, target_ids_acc], -1)  # [B, T, 2]
        target_embeds = self.target_embedder.embed(token_ids=target_ids[:, :, 0], with_sos=False, with_eos=False) + \
                        self.target_embedder.embed(token_ids=target_ids[:, :, 1], with_sos=False, with_eos=False)
        target_lengths = target_lengths + 2  # +2 for eos and sos

        m = ((target_ids - cb_size_half) >= 0).long()
        target_ids = (target_ids - cb_size_half) * m + target_ids * (1 - m)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
            'token_seq_lengths': target_lengths
        }

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)
            input_ids = training_inputs['model_inputs']
            target_ids = training_inputs['target_ids']

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
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

        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            model_output = self.model(**input_ids, output_hidden_states=True)
        if isinstance(model_output, dict):
            logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            logits, last_hidden_state = model_output
        target_length = target_ids.size(1)
        target_logits = logits[:, -target_length:, :]
        loss_mask = None
        if 'target_lengths' in training_inputs:
            loss_mask = sequence_mask(
                training_inputs['target_lengths'], max_len=target_ids.shape[1], device=target_ids.device)
        H = target_logits.shape[-1] // 2
        loss = self.criterion(target_logits[:, :, :H], target_ids[..., 0], loss_mask) + \
               self.criterion(target_logits[:, :, H:], target_ids[..., 1], loss_mask)
        loss = loss / 2
        result_dict = {
            'loss': loss.item(),
        }
        return loss, result_dict

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None, rl_training=False):
        frame_rate = self.extra_params.semantic_frame_rate
        if "duration" not in batch:
            batch["duration"] = hp.duration
        duration = batch["duration"]
        num_tokens = duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode
        sample_thresh = hp.get('sample_thresh', 0.9)
        use_controller_cfg = hp.get('use_controller_cfg', False)
        controller_cfg_gamma = hp.get('controller_cfg_gamma', 1)
        skip_sos = hp.get('skip_sos', False)
        exclude_ids = None
        if hp.get("exclude_eos", False) and self.target_embedder.eos_id is not None:
            exclude_ids = [self.target_embedder.eos_id]
            print(f"exclude_ids: {exclude_ids}")

        # Predict intensity if it's not available
        if "intensity" in self.input_embedders and "intensity" not in batch:
            assert not self.training, "intensity should be provided in training!"
            batch["intensity"] = self._predict_intensity(
                batch,
                self.input_embedders["intensity"],
                self.prediction_heads["intensity"],
                hp.get("intensity_temperature", 1.0),
            )

        if "inputs_embeds" in batch:
            inputs_embeds = batch["inputs_embeds"]
        else:
            inputs_embeds = self.prepare_inputs_embeddings(batch)

        inputs_emb_cfg = None
        if use_controller_cfg:
            assert beam == 1  # TODO(qq) support beam > 1 with CFG.
            batch_cfg = deepcopy(batch)
            batch_cfg['style_text'] = [''] * len(batch['style_text'])
            batch_cfg['style_category'] = [''] * len(batch['style_category'])
            inputs_emb_cfg = self.prepare_inputs_embeddings(batch_cfg)

        inputs_embeds_cfg = inputs_emb_cfg
        batch_size, seq_len, _ = inputs_embeds.size()
        if use_controller_cfg:
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_cfg], dim=0)  # cat on first-dim(batch_size)
            batch_size = 2 * batch_size
        if ref_samples is not None:
            assert ref_samples.size(0) == batch_size
            assert ref_samples.size(1) == num_tokens
            assert beam > 1, "Can't use beam size 1 with ref_samples!"
            beam = beam - 1
        # (b, s, d) --> (b * beam, s, d)
        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(batch_size * beam, seq_len, -1)
        sos_embeds = self.target_embedder.get_sos_embed(batch_size * beam)
        batch_size, seq_len, _ = inputs_embeds.size()  # recalculate batch size

        def _init_model_input():
            if self.use_cross_attn:
                if skip_sos:
                    raise NotImplementedError
                return {
                    "inputs_embeds": sos_embeds,
                    "encoder_hidden_states": inputs_embeds
                }
            else:
                if skip_sos:
                    # If already have sos embedding in prefix prompt, we can skip it
                    return {"inputs_embeds": torch.cat([inputs_embeds, ], dim=1)}
                else:
                    return {"inputs_embeds": torch.cat([inputs_embeds, sos_embeds + sos_embeds], dim=1)}

        model_input = _init_model_input()
        if rl_training:
            rl_model_input = _init_model_input()

        output_tokens = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            gpt_max_seq_len = 4000 if num_tokens < 2500 else 8000
            inference_params = InferenceParams(
                max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
            )
        else:
            past_key_values = None
        pbar = tqdm(range(num_tokens))

        for i in pbar:
            pbar.set_description(f"[0 - {num_tokens}]")

            if isinstance(self.model, gpt.GPTLMHeadModel):
                # to enable inference without a trainer, we simply cast the inputs to the expected model type
                # which is either torch.float16 or torch.bfloat16
                model_input["inputs_embeds"] = model_input["inputs_embeds"].to(self.model.lm_head.weight.dtype)
                logits = self.model(
                    **model_input,
                    inference_params=inference_params,
                    position_ids=None,
                    last_token_only=False,
                ).logits
                # cast back to full precision
                logits = logits.float()

                if use_controller_cfg:
                    logits_cfg = logits[batch_size // 2:]  # unconditioned path
                    logits = controller_cfg_gamma * logits[0:batch_size // 2] + (1 - controller_cfg_gamma) * logits[
                                                                                                             batch_size // 2:]

                inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
                logits = logits[:, -1:, :]  # only predicting on last logit.
                H = logits.shape[-1] // 2
                predict_token_voc = self.sample_logits(
                    i, logits[:, :, :H], temperature, sample_mode, sample_thresh, exclude_ids
                )
                predict_token_inst = self.sample_logits(
                    i, logits[:, :, H:], temperature, sample_mode, sample_thresh, exclude_ids
                )
                predict_token = torch.stack([predict_token_voc, predict_token_inst], -1)
                predict_token_emb = self.target_embedder.embedder(predict_token_voc) + \
                                    self.target_embedder.embedder(predict_token_inst + H)

                if use_controller_cfg:
                    # If unconditioned path uses the predict token history from the conditioned path.
                    predict_token_emb = torch.cat([predict_token_emb, predict_token_emb], dim=0)

                    # If unconditioned path always uses the predict token history from the unconditioned path.
                    # logits_cfg = logits_cfg[:, -1:, :]
                    # predict_token_cfg = self.sample_logits(
                    #     i, logits_cfg, temperature, sample_mode, sample_thresh, exclude_ids)
                    # predict_token_cfg_emb = self.target_embedder.embedder(predict_token_cfg)
                    # predict_token_emb = torch.cat([predict_token_emb, predict_token_cfg_emb], dim=0)

            else:
                raise NotImplementedError

            model_input['inputs_embeds'] = predict_token_emb
            if rl_training and i < num_tokens - 1:
                rl_model_input["inputs_embeds"] = torch.cat(
                    [rl_model_input["inputs_embeds"], predict_token_emb],
                    dim=1,
                )
            output_tokens = torch.cat([output_tokens, predict_token],
                                      dim=1) if output_tokens is not None else predict_token
        output_tokens = output_tokens.flatten(1, 2)
        # Add ref_samples to generation beam
        if ref_samples is not None:
            raise NotImplementedError

        if rl_training:
            return output_tokens, rl_model_input
        else:
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
        decoder_type = self.extra_params.get("decoder_type", "diffusion")
        if decoder_type == "diffusion":
            self.decoder_fn = self.run_diffusion
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        self.val_outputs = dict()
        self.positive_qualitative_emb = None
        self.negative_qualitative_emb = None
        self.resampler = None
        if self.extra_params.get("token2wav_sample_rate", self.extra_params.sample_rate) != self.extra_params.sample_rate:
            self.resampler = Resample(
                orig_freq=self.extra_params.token2wav_sample_rate,
                new_freq=self.extra_params.sample_rate,
            )

    def set_requires_grad(self, requires_grad):
        for n, p in self.named_parameters():
            p.requires_grad = requires_grad

    def _shared_step(self, batch, mode):
        wavs_gt = batch["target_audio"]
        if wavs_gt.dim() == 2:
            wavs_gt = wavs_gt.unsqueeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            training_inputs = self.prepare_training_inputs(batch)
            model_inputs = training_inputs['model_inputs']
            target_ids = training_inputs['target_ids']
            inputs_embeds = training_inputs['inputs_embeds']
        B, T = target_ids.size()
        beam = self.extra_params.beam_size

        # CE loss
        model_output = self.model(**model_inputs, output_hidden_states=True)
        if isinstance(model_output, dict):
            logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            logits, last_hidden_state = model_output
        target_logits = logits[:, -T:, :]
        ce_loss = self.criterion(target_logits, target_ids)
        accu = (target_logits.argmax(dim=-1) == target_ids).float().mean() * 100
        # Auxiliary loss (TODO: separate these losses from CE loss)
        aux_loss, _ = self.aux_loss(batch, logits, last_hidden_state, T)
        ce_loss += aux_loss

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
            batch["inputs_embeds"] = inputs_embeds
            ref_samples = None
            if self.extra_params.add_ref_to_beam and mode == "training":
                ref_samples = target_ids
            if "duration" not in batch:
                if isinstance(self.extra_params.duration, (list, tuple)):
                    batch["duration"] = self.extra_params.duration[-1]
                else:
                    batch["duration"] = self.extra_params.duration
            num_tokens = batch["duration"] * self.extra_params.semantic_frame_rate

            self.set_requires_grad(False) # fix nested autocast bug: https://discuss.pytorch.org/t/autocast-and-torch-no-grad-unexpected-behaviour/93475/2
            sampled_semantic_tokens, model_inputs = super().predict(
                batch,
                self.extra_params,
                beam=beam,
                ref_samples=ref_samples,
                rl_training=True,
            )
            sampled_semantic_tokens_processed, eos_index_list = process_eos_indexes(
                sampled_semantic_tokens,
                self,
                sample_rate=self.extra_params.sample_rate,
            )
            # Compute rewards
            rewards, sampled_audio, reward_breakdown = self.get_reward({
                "target_semantic_tokens": target_ids,
                "sampled_semantic_tokens": sampled_semantic_tokens_processed,
                "eos_index_list": eos_index_list,
                "target_audio": wavs_gt,
                "batch_size": B,
                "beam_size": beam,
                "batch": batch,
            })
            self.set_requires_grad(True)
            # Compute sequence probs
            seq_logits = self.model(**model_inputs)
            if isinstance(seq_logits, dict):
                seq_logits = seq_logits["logits"]
            elif isinstance(seq_logits, tuple):
                seq_logits = seq_logits[0]
            seq_probs1 = F.log_softmax(seq_logits[:, -num_tokens:, :], dim=-1)
            seq_probs2 = torch.gather(
                seq_probs1, -1, sampled_semantic_tokens.unsqueeze(2)
            ).squeeze(2)    # (B * beam, T)
            # Only add up non-eos probs
            if len(eos_index_list) > 0:
                token2wav_rate = self.extra_params.sample_rate // self.extra_params.semantic_frame_rate
                seq_len = eos_index_list // token2wav_rate + 1  # need to add 1 to include <eos>
                for i in range(len(eos_index_list)):
                    seq_probs2[i, seq_len[i]:] = 0
                seq_probs2 = (seq_probs2.sum(dim=-1) / seq_len).reshape(B, beam)
            else:
                seq_probs2 = seq_probs2.reshape(B, beam, -1).mean(dim=-1)
            seq_probs3 = F.softmax(seq_probs2, dim=-1)
            seq_loss = -1 * (rewards * seq_probs3).sum(dim=-1).mean()
            skip = torch.any(torch.isnan(seq_probs1)) or torch.any(torch.isnan(seq_probs3))
            if skip:
                print(f"Skipped samples: {sampled_semantic_tokens}")
        return (
            ce_loss,
            accu,
            seq_loss,
            wavs_gt,
            sampled_audio,
            rewards,
            reward_breakdown,
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
        ce_loss, accu, seq_loss, _, _, _, reward_breakdown, seq_probs, skip = self._shared_step(
            batch=batch,
            mode="training",
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
        self.log_dict(stats, prog_bar=True, sync_dist=True)
        if skip:
            print("Skipping update due to NaN...")
            return None
        else:
            ce_weight = self.extra_params.ce_weight
            seq_weight = self.extra_params.seq_weight
            return ce_loss * ce_weight + seq_loss * seq_weight

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
        eos_index_list = items["eos_index_list"]
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

        # Resample and convert to mono if necessary
        if self.resampler is not None:
            wavs = self.resampler(wavs).mean(dim=1, keepdim=True)
        # Zero out samples after <eos>
        if len(eos_index_list) > 0:
            for i in range(len(eos_index_list)):
                wavs[i, :, eos_index_list[i]:] = 0
        return wavs

    @torch.no_grad()
    def get_reward(self, items):
        b = items["batch_size"]
        beam = items["beam_size"]
        batch = items["batch"]
        sampled_audio = self.decoder_fn(items).float()
        if "duration" in batch:
            max_samples = batch["duration"] * self.extra_params.sample_rate
            sampled_audio = sampled_audio[..., :max_samples]
        items["sampled_audio"] = sampled_audio
        reward = 0.0
        reward_breakdown = {}
        for rw_type, rw_weight in self.extra_params.rewards.items():
            if rw_weight == 0:
                continue
            if rw_type == "style_sim":
                conditions = self.infer_conditions(batch)
                rw_type = "style_text_sim" if "style_text" in conditions else "mulan_sim"
            rw = self._get_reward(items, rw_type).reshape(b, beam)
            reward += rw_weight * rw
            reward_breakdown[rw_type] = rw
        return reward, sampled_audio.reshape(b, beam, -1), reward_breakdown

    @torch.no_grad()
    def _get_reward(self, items, reward_type):
        sampled_audio = items["sampled_audio"]
        target_audio = items["target_audio"]
        b = items["batch_size"]
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
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                mir_tag_type='genre',
                target_tags=target_genres,
                device=sampled_audio.device)
        elif reward_type == "mood_tag":            
            target_mood = [style_text[1] for style_text in items["style_category"]]            
            return mir_tag_reward(
                sampled_audio,
                sample_rate=self.extra_params.sample_rate,
                mir_tag_type='mood',
                target_tags=target_mood,
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
            return wer_reward(
                sampled_lyrics,
                batch["lyrics"] if "lyrics" in batch else batch["lyrics_text"],
                sampled_audio.device,
            )
        elif reward_type == "style_text_sim":
            (
                style_text_sim,
                items["sampled_mulan_embeds"],
                items["style_text_embeds"],
            ) = mulan_text_reward(
                self.requires["mulan_infer_fn"],
                self.requires["mulan"],
                sampled_audio.squeeze(1),
                batch["style_text"],
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


class SemanticT5Module(BaseContinuousEmbedModule):
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
        lyrics_vocab_size = extra_params['lyrics_codebook_size']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            'style_tokens': MetadataT5TokenEmbedder(embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        else:
            raise NotImplementedError
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

    def prepare_inputs_embeddings(self, batch):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        with_sos=True
        # convert inputs to conditions
        inputs_embeds = []
        if 'lyrics_tokens' in conditions:
            embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['lyrics_tokens'].get_sos_embed(batch_size))
        if 'style_tokens' in conditions:
            embeds = self.input_embedders['style_tokens'].embed(self.requires, batch['style_tokens'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(self.input_embedders['style_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

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
        semantic_samples[eos_mask] = 0
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

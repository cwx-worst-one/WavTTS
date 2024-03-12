
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import (
    LyricsTokenEmbedder,
    MulanCategoricalEmbedder,
    TagCategoricalEmbedder,
    WavToVecTokenEmbedder,
    BestRQTokenEmbedder,
    MulanEmbedder,
    DurationEmbedder,
    StructureEmbedder,
)
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams
from recipes.bigmusic.lightning.semantic_modules import process_eos_indexes, truncate_wav_to_eos

# RL
from recipes.bigmusic.utils.rewards import (
    mulan_audio_reward,
    mulan_text_reward,
    wer_reward,
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
)
from torchaudio.transforms import Resample
from recipes.musiclm.lightning.modules import MaskedCrossEntropy
from recipes.bigmusic.utils.metrics_asr import asr_transcribe_lyrics


class SemanticModuleVarlen(BaseContinuousEmbedModule):
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
        mulan_embed_dim = extra_params['mulan_embed_dim']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        style_category_vocab_size = extra_params.get('style_category_vocab_size', 256)
        tag_dropout_rate = extra_params.get('tag_dropout_rate', 0)
        embedder_dict = {}
        for emb_type in extra_params.get("input_embedders", ["mulan", "lyrics_tokens"]):
            if emb_type == "mulan":
                embedder_dict[emb_type] = MulanEmbedder(
                    input_dim=mulan_embed_dim,
                    embedding_dim=hidden_size,
                    add_sos=True,
                )
            elif emb_type == "mulan_categorical":
                # Read ground truth tags from style_text
                embedder_dict[emb_type] = MulanCategoricalEmbedder(
                    input_dim=mulan_embed_dim,
                    vocab_size=style_category_vocab_size,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    dropout=tag_dropout_rate
                )
            elif emb_type == "lyrics_tokens":
                embedder_dict[emb_type] = LyricsTokenEmbedder(
                    vocab_size=lyrics_vocab_size,
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
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
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

        prediction_dict = {}
        for pred_type in extra_params.get("prediction_heads", ["input"]):
            if pred_type == "input":
                prediction_dict[pred_type] = nn.Linear(
                    hidden_size,
                    lyrics_vocab_size + 2,
                    bias=False,
                )
            else:
                raise ValueError(f"Unknown pred type: {pred_type}")
        self.prediction_heads = nn.ModuleDict(prediction_dict)
    
    def prepare_mulan_inputs(self, batch, mulan_embedder):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)

        # Mulan
        if 'style_text' in conditions:
            mulan_embeds = mulan_embedder.embed(self.requires, batch['style_text'], with_sos=True, data_type='text')
        elif 'style_audio' in conditions:
            mulan_embeds = mulan_embedder.embed(self.requires, batch['style_audio'].to(self.device), with_sos=True, data_type='music')
        elif 'style_category' in conditions:
            mulan_embeds = mulan_embedder.embed(self.requires, batch['style_category'], with_sos=True, data_type='category')
        elif 'style_tag' in conditions: # using Mulan for on-the-fly MIR tagging
            mulan_embeds = mulan_embedder.embed(self.requires, batch['style_audio'].to(self.device), with_sos=True, data_type='tag')
        else: # adding SOS token no matter what so that all parameters get used
            mulan_embeds = mulan_embedder.get_sos_embed(batch_size)
        
        batch_size, seq_len, emb_dim = mulan_embeds.shape
        # for now, create dummy token ids. we won't be predicting them anyways
        token_ids = torch.zeros((batch_size, seq_len)).long().to(mulan_embeds.device)
        batch_seq_lengths = torch.zeros((batch_size), device=mulan_embeds.device) + seq_len

        return {
            'token_embeds': mulan_embeds,
            'token_ids': token_ids,
            'token_seq_lengths': batch_seq_lengths
        }

    def prepare_lyrics_inputs(self, batch, lyrics_embedder, add_eos=False):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)

        # Lyrics
        if 'lyrics_tokens' in conditions:
            lyrics_tokens = batch['lyrics_tokens'].to(self.device)
            lyrics_lengths = batch['lyrics_tokens_length'].to(self.device)
        else:
            lyrics_tokens = torch.zeros((batch_size, 0)).long().to(self.device)
            lyrics_lengths = torch.zeros((batch_size), device=self.device)

        if add_eos:
            lyrics_tokens = lyrics_embedder.tokenize(token_ids=lyrics_tokens, with_sos=False)
            # add sos and eos id
            lyrics_tokens = F.pad(lyrics_tokens, (1, 1), 'constant', 0)
            lyrics_tokens[:, 0] = lyrics_embedder.sos_id
            eos_indices = (lyrics_lengths+1).unsqueeze(1) # set last index to EOS
            lyrics_tokens.scatter_(1, eos_indices, lyrics_embedder.eos_id)
            lyrics_lengths = lyrics_lengths + 2 # +2 for eos and sos
        else:
            lyrics_tokens = lyrics_embedder.tokenize(token_ids=lyrics_tokens, with_sos=True)
            lyrics_lengths = lyrics_lengths + 1 # +1 for sos

        # embed
        lyrics_embeds = lyrics_embedder.embed(token_ids=lyrics_tokens, with_sos=False).to(self.device)
        # unpad 
        lyrics_tokens = unpad_sequence(lyrics_tokens, lyrics_lengths, batch_first=True)
        lyrics_embeds = unpad_sequence(lyrics_embeds, lyrics_lengths, batch_first=True)
        # save
        return {
            'token_embeds': lyrics_embeds,
            'token_ids': lyrics_tokens,
            'token_seq_lengths': lyrics_lengths
        }

    def prepare_duration_inputs(self, batch, duration_embedder):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)

        if 'duration' in conditions:
            duration = batch["duration"]
            duration_embeds = duration_embedder.embed(duration, batch_size)
        else:
            duration_embeds = duration_embedder.empty_embed(batch_size)
        
        batch_size, seq_len, _ = duration_embeds.shape
        # for now, create dummy token ids. we won't be predicting them anyways
        token_ids = torch.zeros((batch_size, seq_len)).long().to(self.device)
        batch_seq_lengths = torch.zeros((batch_size), device=self.device) + seq_len

        return {
            "token_embeds": duration_embeds,
            "token_ids": token_ids,
            "token_seq_lengths": batch_seq_lengths,
        }

    def prepare_structure_inputs(self, batch, structure_embedder):
        batch_size = self.infer_batch_size(batch)
        structure_labels = batch["structure"]
        assert batch_size == len(structure_labels)
        # Dropout if needed
        if self.training and self.extra_params.structure_dropout > 0:
            dropout = self.extra_params.structure_dropout
            all_keep = [x >= dropout for x in np.random.rand(batch_size)]
            for i, keep in enumerate(all_keep):
                if not keep:
                    structure_labels[i] = None

        if "duration" in batch:
            target_duration = batch["duration"]
        else:
            target_duration = batch["target_audio"].shape[-1] // self.extra_params.sample_rate

        structure_embeds = structure_embedder.embed(structure_labels, target_duration)
        batch_size, seq_len, _ = structure_embeds.shape
        # for now, create dummy token ids. we won't be predicting them anyways
        token_ids = torch.zeros((batch_size, seq_len)).long().to(self.device)
        batch_seq_lengths = torch.zeros((batch_size), device=self.device) + seq_len

        return {
            "token_embeds": structure_embeds,
            "token_ids": token_ids,
            "token_seq_lengths": batch_seq_lengths,
        }

    def prepare_model_inputs(self, batch):
        model_inputs = []
        for emb_type, embedder in self.input_embedders.items():
            if (emb_type == "mulan") or (emb_type == "mulan_categorical"):
                emb_inputs = self.prepare_mulan_inputs(batch, embedder)
            elif emb_type == "lyrics_tokens":
                emb_inputs = self.prepare_lyrics_inputs(batch, embedder)
            elif emb_type == "duration":
                emb_inputs = self.prepare_duration_inputs(batch, embedder)
            elif emb_type == "structure":
                emb_inputs = self.prepare_structure_inputs(batch, embedder)
            else:
                raise ValueError(f"Unknown emb type: {emb_type}")
            model_inputs.append(emb_inputs)
        return model_inputs

    def prepare_target_inputs(self, batch):
        # Prepare target ids
        target_lengths = batch['target_tokens_length'].to(self.device)
        target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False).to(self.device)
        # add sos and eos id
        target_ids = F.pad(target_ids, (1, 1))
        target_ids[:, 0] = self.target_embedder.sos_id
        eos_indices = (target_lengths+1).unsqueeze(1) # set last index to EOS
        target_ids.scatter_(1, eos_indices, self.target_embedder.eos_id)
        target_lengths = target_lengths + 2 # +2 for eos and sos

        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        target_ids_unpad = unpad_sequence(target_ids, target_lengths, batch_first=True)
        target_embeds = unpad_sequence(target_embeds, target_lengths, batch_first=True)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids_unpad,
            'token_ids_batched': target_ids,
            'token_seq_lengths': target_lengths
        }

    def prepare_training_inputs(self, batch):
        model_inputs = self.prepare_model_inputs(batch)
        target_inputs = self.prepare_target_inputs(batch)

        # zip inputs and concat
        token_ids = zip(*[i['token_ids'] for i in model_inputs + [target_inputs]])
        token_ids = [torch.cat(t, dim=0) for t in token_ids]
        token_embeds = zip(*[i['token_embeds'] for i in model_inputs + [target_inputs]])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        input_seq_lengths = torch.vstack([i['token_seq_lengths'] for i in model_inputs]).sum(dim=0)
        total_seq_lengths = input_seq_lengths + target_inputs['token_seq_lengths']

        # Padding step
        token_ids_pad = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        token_embeds_pad = pad_sequence(token_embeds, batch_first=True, padding_value=0)

        # get masks
        input_loss_mask = sequence_mask(input_seq_lengths, max_len=token_ids_pad.shape[1], device=self.device)
        target_loss_mask = sequence_mask(total_seq_lengths, max_len=token_ids_pad.shape[1], device=self.device)
        target_loss_mask = target_loss_mask ^ input_loss_mask # remove input targets from target mask

        return {
            'token_embeds': token_embeds_pad,
            'token_ids': token_ids_pad,
            'input_loss_mask': input_loss_mask,
            'target_loss_mask': target_loss_mask,
            'model_inputs': model_inputs,
            'target_inputs': target_inputs
        }

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)

        token_embeds = training_inputs['token_embeds'][:, :-1]
        token_ids = training_inputs['token_ids'][:, 1:]
        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        
        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            model_output = self.model(inputs_embeds=token_embeds, output_hidden_states=True)

        
        if isinstance(model_output, dict):
            target_logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            target_logits, last_hidden_state = model_output
        target_ids = token_ids * target_loss_mask.long() # set non-target ids to 0 to avoid OOB
        target_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)
        target_accu = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask].float().mean() * 100
        # measure accuracy of first 25 tokens as a measurement for style
        target_mask_cumsum = torch.cumsum(target_loss_mask, dim=-1)
        target_loss_mask_seq_25 = (target_mask_cumsum <= 26) & (target_mask_cumsum > 1) & target_loss_mask # first 25 tokens after eos
        target_accu_seq_25 = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask_seq_25].float().mean() * 100

        loss = target_loss
        results_dict = {
            'loss': (target_loss).item(),
            'accu': (target_accu).mean().item(),
            'tgt_loss': target_loss.item(),
            'tgt_accu': target_accu.item(),
            'tgt_accu_seq_25': target_accu_seq_25.item()
        }

        for pred_type, pred_head in self.prediction_heads.items():
            if pred_type == "input":
                input_loss_mask = training_inputs['input_loss_mask'][:, 1:]
                inputs_logits = pred_head(last_hidden_state)
                inputs_ids = token_ids * input_loss_mask.long() # offset by one
                pred_loss = self.criterion(inputs_logits, inputs_ids, mask=input_loss_mask)
                inputs_accu = (inputs_logits.argmax(dim=-1) == inputs_ids)[input_loss_mask].float().mean() * 100
                pred_dict = {
                    'inp_loss': pred_loss.item(),
                    'inp_accu': inputs_accu.item(),
                }
            else:
                raise ValueError(f"Unknown pred type: {pred_type}")
            loss += pred_loss
            results_dict.update(pred_dict)

        # del training_inputs
        return loss, results_dict
    
    def training_step(self, batch, batch_idx):
        loss, results_dict = self._shared_step(batch, update_mfu=True)
        log_dict = { 'tr_' + key: value for key, value in results_dict.items() }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, result_dict = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(result_dict)

    def predict_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        for dataloader_idx, results_list in self.val_outputs.items():
            accum = defaultdict(list)
            for result_dict in results_list:
                for k, v in result_dict.items():
                    accum[k].append(v)
            
            log_dict = { f'val_{k}_{str(dataloader_idx)}': torch.tensor(v).mean().item() for k,v in accum.items()}

            self.log_dict(
                log_dict,
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []


    @torch.no_grad()
    def predict(self, batch, hp, beam=None, return_inputs_embeds=False):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        predict_lyrics = hp.get("predict_lyrics", False)
        sample_thresh = hp.get('sample_thresh', 0.9)

        return self._predict(
            batch,
            num_tokens,
            temperature=temperature,
            beam=beam,
            predict_lyrics=predict_lyrics,
            sample_thresh=sample_thresh,
            return_inputs_embeds=return_inputs_embeds
        )

    @torch.no_grad()
    def _predict(
        self,
        batch,
        num_tokens,
        temperature=1,
        beam=None,
        sample_mode="gumbel",
        tqdm_name=None,
        return_inputs_embeds=False,
        predict_lyrics=False,
        sample_thresh=0.9
    ):
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        if predict_lyrics:
            lyrics_tokens = batch['lyrics_tokens']
            lyrics_lengths = batch['lyrics_tokens_length']
            lyrics_seqs = unpad_sequence(lyrics_tokens, lyrics_lengths, batch_first=True)
            pred_lyric_ids = self._predict_lyrics(batch, num_tokens=1500, sample_mode=sample_mode)
            new_seqs = []
            for idx, (lyrics_seq, pred_lyric) in enumerate(zip(lyrics_seqs, pred_lyric_ids)):
                new_seq = torch.cat([lyrics_seq, pred_lyric], dim=-1)
                new_seqs.append(new_seq)
            new_lens = [len(s) for s in new_seqs]
            batch['lyrics_tokens'] = pad_sequence(new_seqs, batch_first=True)
            batch['lyrics_tokens_length'] = torch.tensor(new_lens, device=self.device)

        if 'model_inputs' in batch:
            model_inputs = batch['model_inputs']
        else:
            model_inputs = self.prepare_model_inputs(batch)

        # zip inputs and concat
        token_embeds = zip(*[i['token_embeds'] for i in model_inputs])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        input_seq_lengths = torch.vstack([i['token_seq_lengths'] for i in model_inputs]).sum(dim=0)

        # pad front
        token_embeds_reversed = [embeds.flip(dims=(0,)) for embeds in token_embeds]
        token_embeds_pad = pad_sequence(token_embeds_reversed, batch_first=True, padding_value=0).flip(dims=(1,))

        # add SOS
        batch_size, seq_len, _ = token_embeds_pad.shape
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)
        input_seq_lengths += 1
        token_embeds_pad = torch.cat([token_embeds_pad, sos_embeds], dim=1)

        if beam is not None:
            batch_size, seq_len, _ = token_embeds_pad.shape
            token_embeds_pad = token_embeds_pad.repeat(1, beam, 1).reshape(batch_size * beam, seq_len, -1)
            batch_size, seq_len, _ = token_embeds_pad.shape
        
        # attention_mask = ~sequence_mask(input_seq_lengths, device=self.device) # TODO: enable once attention mask works.
        model_input = { 
            "inputs_embeds": token_embeds_pad,
            # "attention_mask": attention_mask
        }
        
        output_tokens = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            gpt_max_seq_len = 4000 if num_tokens < 2500 else 8000
            inference_params = InferenceParams(
                max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
            )
        else:
            past_key_values = None
        pbar = tqdm(range(num_tokens))
        previous_inputs_embeds = model_input["inputs_embeds"]
        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")

            if isinstance(self.model, gpt.GPTLMHeadModel):
                logits = self.model(
                    **model_input,
                    inference_params=inference_params,
                    position_ids=None,
                    last_token_only=False,
                ).logits
                inference_params.sequence_len_offset += model_input['inputs_embeds'].size(1)
            else:
                model_output = self.model(
                    **model_input,
                    past_key_values=past_key_values, use_cache=True
                )
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]

            logits = logits[:, -1:, :] # only predicting on last logit.
            predict_token = self.sample_logits(i, logits, temperature, sample_mode, thresh=sample_thresh)
            predict_token_emb = self.target_embedder.embedder(predict_token)
            model_input = { 'inputs_embeds': predict_token_emb }
            previous_inputs_embeds = torch.cat([previous_inputs_embeds, predict_token_emb], dim=1)
            output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token

        if return_inputs_embeds:
            return output_tokens, previous_inputs_embeds

        return output_tokens

    @torch.no_grad()
    def _predict_lyrics(
        self,
        batch,
        num_tokens,
        temperature=1,
        sample_mode="gumbel",
        tqdm_name="Lyrics Prediction",
    ):
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        mulan_inputs = self.prepare_mulan_inputs(batch)
        lyrics_inputs = self.prepare_lyrics_inputs(batch, add_eos=False)

        # zip inputs and concat
        token_embeds = zip(*[i['token_embeds'] for i in [mulan_inputs, lyrics_inputs]])
        token_embeds = [torch.cat(t, dim=0) for t in token_embeds]
        input_seq_lengths = mulan_inputs['token_seq_lengths'] + lyrics_inputs['token_seq_lengths']

        # pad front
        token_embeds_reversed = [embeds.flip(dims=(0,)) for embeds in token_embeds]
        token_embeds_pad = pad_sequence(token_embeds_reversed, batch_first=True, padding_value=0).flip(dims=(1,))
        batch_size, seq_len, _ = token_embeds_pad.shape

        # attention_mask = ~sequence_mask(input_seq_lengths, device=self.device)
        lyrics_embedder: LyricsTokenEmbedder = self.input_embedders['lyrics_tokens']
        model_input = { 
            "inputs_embeds": token_embeds_pad,
            # "attention_mask": attention_mask # TODO: (AS) enable one attention mask works
        }

        output_tokens = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            gpt_max_seq_len = 4000 if num_tokens < 2500 else 8000
            inference_params = InferenceParams(
                max_sequence_len=gpt_max_seq_len, max_batch_size=batch_size
            )
        else:
            past_key_values = None
        pbar = tqdm(range(num_tokens))
        previous_inputs_embeds = model_input["inputs_embeds"]
        for i in pbar:
            pbar.set_description(f"{tqdm_name} [0 - {num_tokens}]")

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
                    past_key_values=past_key_values, use_cache=True
                )
                past_key_values = model_output["past_key_values"]
                last_hidden_state = model_output['hidden_states'][-1]

            target_logits = self.lm_input_head(last_hidden_state[:, -1:]) # only predicting for last logit.

            predict_token = self.sample_logits(i, target_logits, temperature, sample_mode)
            predict_token_emb = lyrics_embedder.embedder(predict_token)
            model_input = { 'inputs_embeds': predict_token_emb }
            previous_inputs_embeds = torch.cat([previous_inputs_embeds, predict_token_emb], dim=1)
            output_tokens = torch.cat([output_tokens, predict_token], dim=1) if output_tokens is not None else predict_token
        batch_size = output_tokens.shape[0]
        return output_tokens
        output_tokens = torch.cat([output_tokens, lyrics_embedder.get_eos_token(batch_size)], dim=1)
        previous_inputs_embeds = torch.cat([previous_inputs_embeds, lyrics_embedder.get_eos_embed(batch_size)], dim=1)

        first_indexes = first_index(output_tokens, axis=-1, value=lyrics_embedder.eos_id)
        output_tokens = unpad_sequence(output_tokens, first_indexes, batch_first=True)
        previous_inputs_embeds = unpad_sequence(previous_inputs_embeds, first_indexes, batch_first=True)
        return output_tokens

def first_index(x, axis=0, value=0):
    nonz = (x == value)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)[-1]

def sequence_mask(seq_lens, max_len=None, device='cpu'):
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    return mask


class SemanticRLModule(SemanticModuleVarlen):
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

    def _shared_step_rl(self, batch, training_inputs, mode):
        ##### RL #####
        beam = self.extra_params.beam_size
        wavs_gt = batch["target_audio"]
        if wavs_gt.dim() == 2:
            wavs_gt = wavs_gt.unsqueeze(1)
        target_ids_batched = training_inputs['target_inputs']['token_ids_batched'] # needed for predict
        B, T = target_ids_batched.size()
        batch["model_inputs"] = training_inputs['model_inputs'] # needed for predict

        # Inference
        sampled_semantic_tokens, model_inputs = super().predict(
            batch,
            self.extra_params,
            beam=beam,
            return_inputs_embeds=True
        )
        del batch['model_inputs']
        sampled_semantic_tokens_processed, eos_index_list = process_eos_indexes(
            sampled_semantic_tokens,
            self,
            sample_rate=self.extra_params.sample_rate,
        )
        # Compute rewards
        rewards, sampled_audio, reward_breakdown = self.get_reward({
            "target_semantic_tokens": target_ids_batched,
            "sampled_semantic_tokens": sampled_semantic_tokens_processed,
            "eos_index_list": eos_index_list,
            "target_audio": wavs_gt,
            "batch_size": B,
            "beam_size": beam,
            "batch": batch,
        })
        # Compute sequence probs
        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            seq_logits = self.model(inputs_embeds=model_inputs, output_hidden_states=False)
        if isinstance(seq_logits, dict):
            seq_logits = seq_logits["logits"]
        elif isinstance(seq_logits, tuple):
            seq_logits = seq_logits[0]
        num_tokens = sampled_semantic_tokens.shape[-1]
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

        return (
            seq_loss,
            wavs_gt,
            sampled_audio,
            rewards,
            reward_breakdown,
            seq_probs3,
            skip,
        )
    
    def _shared_step_ce(self, batch, training_inputs, mode):
        token_embeds = training_inputs['token_embeds'][:, :-1]
        token_ids = training_inputs['token_ids'][:, 1:]
        target_loss_mask = training_inputs['target_loss_mask'][:, 1:]
        
        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            model_output = self.model(inputs_embeds=token_embeds, output_hidden_states=True)

        
        if isinstance(model_output, dict):
            target_logits = model_output["logits"]
            last_hidden_state = model_output['hidden_states'][-1]
        elif isinstance(model_output, tuple):
            target_logits, last_hidden_state = model_output
        target_ids = token_ids * target_loss_mask.long() # set non-target ids to 0 to avoid OOB
        target_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)
        target_accu = (target_logits.argmax(dim=-1) == target_ids)[target_loss_mask].float().mean() * 100

        if 'input' in self.prediction_heads:
            pred_head = self.prediction_heads['input']
            input_loss_mask = training_inputs['input_loss_mask'][:, 1:]
            inputs_logits = pred_head(last_hidden_state)
            inputs_ids = token_ids * input_loss_mask.long() # offset by one
            inputs_loss = self.criterion(inputs_logits, inputs_ids, mask=input_loss_mask)
            inputs_accu = (inputs_logits.argmax(dim=-1) == inputs_ids)[input_loss_mask].float().mean() * 100
        else:
            inputs_loss = torch.zeros_like(target_loss)
            inputs_accu = torch.zeros_like(target_accu)

        return (
            target_loss,
            target_accu,
            inputs_loss,
            inputs_accu,
        )

    def _shared_step(self, batch, mode):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            training_inputs = self.prepare_training_inputs(batch)

        ##### RL #####
        (
            seq_loss,
            wavs_gt,
            sampled_audio,
            rewards,
            reward_breakdown,
            seq_probs3,
            skip,
        ) = self._shared_step_rl(batch, training_inputs, mode)
        
        ##### CE Loss #####
        (
            target_loss,
            target_accu,
            inputs_loss,
            inputs_accu
        ) = self._shared_step_ce(batch, training_inputs, mode)

        return (
            target_loss,
            target_accu,
            inputs_loss,
            inputs_accu,
            seq_loss,
            wavs_gt,
            sampled_audio,
            rewards,
            reward_breakdown,
            seq_probs3,
            skip,
        )

    def training_step(self, batch, batch_idx):
        target_loss, target_accu, input_loss, input_accu, seq_loss, _, _, _, reward_breakdown, seq_probs, skip = self._shared_step(
            batch=batch,
            mode="training",
        )
        stats = {
            "ce_loss/train": target_loss.item(),
            "accuracy/train": target_accu.item(),
            "input_loss/train": input_loss.item(),
            "input_accuracy/train": input_accu.item(),
            "seq_loss/train": seq_loss.item(),
            "seq_probs/train_max_mean": seq_probs.max(dim=-1).values.mean(),
            "seq_probs/train_max_std": seq_probs.max(dim=-1).values.std(),
        }
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
        ce_weight = self.extra_params.ce_weight
        seq_weight = self.extra_params.seq_weight
        if skip:
            print("Skipping update due to NaN...")
            seq_weight = 0
        return target_loss * ce_weight + input_loss + seq_loss * seq_weight

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
            for batch_idx, (ce_l, a, inp_l, inp_a, seq_l, wavs_gt, wavs_sampled, rewards, reward_breakdown, seq_probs, _) in enumerate(outputs):
                stats[f"ce_loss/{prefix}"] += ce_l.item()
                stats[f"accuracy/{prefix}"] += a.item()
                stats[f"inp_loss/{prefix}"] += inp_l.item()
                stats[f"inp_accuracy/{prefix}"] += inp_a.item()
                stats[f"seq_loss/{prefix}"] += seq_l.item()
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
        # If you see this error, lower batch size:
        # RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.
        wavs = torch.cat([vocoder.decode(c).detach() for c in torch.split(pred_emb, 1)])
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
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import (
    LyricsTokenEmbedder,
    WavToVecTokenEmbedder,
    BestRQTokenEmbedder, 
    MulanTagEmbedder,
    DurationEmbedder,
)
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams


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
        mulan_OTF_tag_type = extra_params.get('mulan_tag_type', 'mulan_genres')
        embedder_dict = {}
        for emb_type in extra_params.get("input_embedders", ["mulan", "lyrics_tokens"]):
            if emb_type == "mulan":
                embedder_dict[emb_type] = MulanTagEmbedder(
                    input_dim=mulan_embed_dim,
                    embedding_dim=hidden_size,
                    add_sos=True,
                    mulan_tag_type=mulan_OTF_tag_type,
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
        lyrics_embeds = lyrics_embedder.embed(token_ids=lyrics_tokens, with_sos=False)
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

    def prepare_model_inputs(self, batch):
        model_inputs = []
        for emb_type, embedder in self.input_embedders.items():
            if emb_type == "mulan":
                emb_inputs = self.prepare_mulan_inputs(batch, embedder)
            elif emb_type == "lyrics_tokens":
                emb_inputs = self.prepare_lyrics_inputs(batch, embedder)
            elif emb_type == "duration":
                emb_inputs = self.prepare_duration_inputs(batch, embedder)
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
        target_ids = unpad_sequence(target_ids, target_lengths, batch_first=True)
        target_embeds = unpad_sequence(target_embeds, target_lengths, batch_first=True)
        return {
            'token_embeds': target_embeds,
            'token_ids': target_ids,
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
            'target_loss_mask': target_loss_mask
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
        # target_logits = model_output['logits']
        target_ids = token_ids * target_loss_mask.long() # set non-target ids to 0 to avoid OOB
        target_loss = self.criterion(target_logits, target_ids, mask=target_loss_mask)
        target_accu = (target_logits.argmax(dim=-1) == token_ids)[target_loss_mask].float().mean() * 100

        loss = target_loss
        results_dict = {
            'loss': (target_loss).item(),
            'accu': (target_accu).mean().item(),
            'target_loss': target_loss.item(),
            'target_accu': target_accu.item()
        }

        for pred_type, pred_head in self.prediction_heads.items():
            if pred_type == "input":
                input_loss_mask = training_inputs['input_loss_mask'][:, 1:]
                inputs_logits = pred_head(last_hidden_state)
                inputs_ids = token_ids * input_loss_mask.long() # offset by one
                pred_loss = self.criterion(inputs_logits, inputs_ids, mask=input_loss_mask)
                inputs_accu = (inputs_logits.argmax(dim=-1) == inputs_ids)[input_loss_mask].float().mean() * 100
                pred_dict = {
                    'input_loss': pred_loss.item(),
                    'input_accu': inputs_accu.item(),
                }
            else:
                raise ValueError(f"Unknown pred type: {pred_type}")
            loss += pred_loss
            results_dict.update(pred_dict)

        # del training_inputs
        return loss, results_dict
    
    def training_step(self, batch, batch_idx):
        loss, results_dict = self._shared_step(batch, update_mfu=True)
        log_dict = { 'train_' + key: value for key, value in results_dict.items() }
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
    def predict(self, batch, hp, beam=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        predict_lyrics = hp.get("predict_lyrics", False)

        return self._predict(
            batch,
            num_tokens,
            temperature=temperature,
            beam=beam,
            predict_lyrics=predict_lyrics
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
            predict_token = self.sample_logits(i, logits, temperature, sample_mode)
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

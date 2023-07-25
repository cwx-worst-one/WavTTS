from recipes.musiclm.lightning.modules import BaseModule
import torch
from tqdm.auto import tqdm
from recipes.musiclm.inference.utils import sample
import random

class LyricsSemanticEmbedModule(BaseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()
        semantic_type = self.extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            self.semantic_embed_fn = self.get_wav2vec_embeds
        elif semantic_type == "best_rq" or semantic_type == "best_rq_minz":
            self.semantic_embed_fn = self.get_best_rq_embeds
        else:
            raise KeyError(f"Invalid semantic_type, got {semantic_type}")
        
        mulan_dim = self.extra_params.mulan_embed_dim
        emb_dim = self.model.config.hidden_size
        self.mulan_lin_embed = torch.nn.Linear(mulan_dim, emb_dim, bias=False)

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            model_input, target_embeds = self.prepare_feature(batch)
        logits = self.model(**model_input)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -target_embeds.size(1) :, :]
        loss = self.criterion(x, target_embeds)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log_dict({"tr_loss": loss}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            for l in outputs:
                loss += l
            loss /= len(outputs)

            self.log_dict(
                {f"val_loss_{dataloader_idx}": loss}, prog_bar=True, sync_dist=True
            )
            self.val_outputs[dataloader_idx] = []

    # @torch.no_grad() # need grad for input embeddings
    def prepare_feature(self, batch):
        with torch.no_grad():
            mulan_audio, target_audio, lyrics_tokens = batch.get("mulan_audio"), batch.get("target_audio"), batch.get("lyrics_tokens")

            wav2vec_embeds = self.semantic_embed_fn(target_audio)
            mulan_embeds = self.get_mulan_embeds(mulan_audio)
            lyrics_ids = lyrics_tokens

        b = target_audio.shape[0]
        sos_id = 80
        sos_ids = torch.full(size=(b, 1), fill_value=sos_id, dtype=torch.long, device=self.device)
        from samantha.models.flash_llama import LlamaForCausalLM
        model: LlamaForCausalLM = self.model
        mulan_embeds = self.mulan_lin_embed(mulan_embeds)[:, None, :]
        lyrics_embeds = model.model.embed_tokens(lyrics_ids)
        sos_embeds = model.model.embed_tokens(sos_ids)
        inputs_embeds = torch.cat([mulan_embeds, lyrics_embeds, sos_embeds, wav2vec_embeds[:, :-1, :]], dim=1)
        inputs = {
            "inputs_embeds": inputs_embeds
        }

        return inputs, wav2vec_embeds[:, 1:, :]
            
    @torch.no_grad()
    def predict(self, mulan_embeds, hp, lyrics_tokens):
        b = mulan_embeds.size(0)

        sos_id = 80
        sos_ids = torch.full(size=(b, 1), fill_value=sos_id, dtype=torch.long, device=self.device)
        from samantha.models.flash_llama import LlamaForCausalLM
        model: LlamaForCausalLM = self.model
        mulan_embeds = self.mulan_lin_embed(mulan_embeds)[:, None, :]
        lyrics_embeds = model.model.embed_tokens(lyrics_tokens)
        sos_embeds = model.model.embed_tokens(sos_ids)

        semantic_embeds = None
        num_tokens = hp.duration * hp.wav2vec_frame_rate
        inputs_embeds = torch.cat([mulan_embeds, lyrics_embeds, sos_embeds], dim=1)
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        for _ in pbar:
            pbar.set_description(f"Semantic [0 - {num_tokens}]")
            model_output = self.model(
                inputs_embeds=inputs_embeds,
                # encoder_hidden_states=mulan_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = model_output["past_key_values"]
            logits = model_output["logits"]
            last_embed = logits[:, -1:]
            inputs_embeds = last_embed
            if semantic_embeds is None:
                semantic_embeds = last_embed
            else:
                semantic_embeds = torch.cat([semantic_embeds, last_embed], dim=1)
        return semantic_embeds

class ConditionalMulanPhonemeCoarseModule(BaseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()
        
        if 'pretrained_path' in self.extra_params and self.extra_params.pretrained_path is not None:
            print('Loading checkpoint', self.extra_params.pretrained_path)
            state = torch.load(self.extra_params.pretrained_path)
            self.load_state_dict(state['state_dict'], strict=False)

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch)
        logits = self.model(input_ids=input_ids)
        if isinstance(logits, dict):
            logits = logits["logits"]
        x = logits[:, -target_ids.size(1):, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, batch):
        target_wavs, mulan_wavs, vocal_wavs, lyrics_tokens = batch.get("target_audio"), batch.get("mulan_audio"), batch.get("vocal_audio"), batch.get("lyrics_tokens")
        vocal_chroma = batch.get("vocal_chroma")
        device = self._device
        b, _ = target_wavs.size()

        soundstream_ids = self.get_soundstream_tokens(target_wavs)
        mulan_ids = None
        lyrics_ids = None
        vocal_mulan_ids = None
        chroma_ids = None

        input_ids = torch.zeros((b, 0), dtype=torch.long, device=device)
        if mulan_wavs is not None:
            mulan_ids = self.get_mulan_tokens(mulan_wavs) 
        if lyrics_tokens is not None:
            lyrics_ids = lyrics_tokens
        if vocal_wavs is not None and random.random() > self.extra_params.vocal_condition_p:
            vocal_mulan_ids = self.get_mulan_tokens(vocal_wavs)
        if vocal_wavs is not None and random.random() > self.extra_params.chroma_condition_p:
            chroma_ids = vocal_chroma

        soundstream_ids, *condition_ids = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids, vocal_mulan_ids, chroma_ids)
        input_ids = torch.cat(condition_ids + [soundstream_ids[:, :-1]], dim=1)
        target_ids = soundstream_ids[:, 1:]

        return input_ids, target_ids

    @torch.no_grad()
    def prepare_conditions(self, soundstream_ids, mulan_ids, lyrics_ids, vocal_mulan_ids=None, chroma_ids=None):
        device = self.device
        num_coarse = self.extra_params.num_coarse
        b = [ids.size(0) for ids in [soundstream_ids, mulan_ids, lyrics_ids] if ids is not None][0]

        mulan_vocab_size = self.extra_params.mulan_vocab_size
        soundstream_vocab_size = self.extra_params.soundstream_vocab_size
        lyrics_vocab_size = self.extra_params.lyrics_codebook_size
        special_tokens_vocab_size = self.extra_params.special_tokens_size
        chroma_vocab_size = self.extra_params.chroma_vocab_size
        mulan_vocab_offset = soundstream_vocab_size
        lyrics_vocab_offset = mulan_vocab_size + soundstream_vocab_size
        extra_tokens_offset = mulan_vocab_size + soundstream_vocab_size + lyrics_vocab_size
        chroma_vocab_offset = mulan_vocab_size + soundstream_vocab_size + lyrics_vocab_size + special_tokens_vocab_size
        # vocal_vocab_offset = uses either mulan or soundstream vocab

        soundstream_sos_id = extra_tokens_offset
        mulan_sos_id = extra_tokens_offset + 1
        lyrics_sos_id = extra_tokens_offset + 2
        vocal_sos_id = extra_tokens_offset + 3
        chroma_sos_id = extra_tokens_offset + 4

        if mulan_ids is not None:
            mulan_ids = (
                mulan_ids
                + torch.arange(self.extra_params.mulan_num_rvq, device=device) * self.extra_params.mulan_codebook_size
                + mulan_vocab_offset
            )
            mulan_sos_ids = torch.full(size=(b, 1), fill_value=mulan_sos_id, dtype=torch.long, device=device)
            mulan_ids = torch.cat([mulan_sos_ids, mulan_ids], dim=1)
        else:
            mulan_ids = torch.zeros((b, 0), dtype=torch.long, device=device)

        if lyrics_ids is not None:
            lyrics_ids = lyrics_ids + lyrics_vocab_offset
            lyrics_sos_ids = torch.full(size=(b, 1), fill_value=lyrics_sos_id, dtype=torch.long, device=device)
            lyrics_ids = torch.cat([lyrics_sos_ids, lyrics_ids], dim=1)
        else:
            lyrics_ids = torch.zeros((b, 0), dtype=torch.long, device=device)

        if vocal_mulan_ids is not None:
            vocal_mulan_ids = (
                vocal_mulan_ids
                + torch.arange(self.extra_params.mulan_num_rvq, device=device) * self.extra_params.mulan_codebook_size
                + mulan_vocab_offset
            )
            vocal_sos_ids = torch.full(size=(b, 1), fill_value=vocal_sos_id, dtype=torch.long, device=device)
            vocal_mulan_ids = torch.cat([vocal_sos_ids, vocal_mulan_ids], dim=1)
        else:
            vocal_mulan_ids = torch.zeros((b, 0), dtype=torch.long, device=device)

        if chroma_ids is not None:
            chroma_ids = chroma_ids + chroma_vocab_offset
            chroma_sos_ids = torch.full(size=(b, 1), fill_value=chroma_sos_id, dtype=torch.long, device=device)
            chroma_ids = torch.cat([chroma_sos_ids, chroma_ids], dim=1)
        else: 
            chroma_ids = torch.zeros((b, 0), dtype=torch.long, device=device)

        if soundstream_ids is not None:
            soundstream_ids = (
                soundstream_ids[:, :, 0 : num_coarse]
                + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
            )
            soundstream_ids = torch.reshape(soundstream_ids, [b, -1])
            soundstream_sos_ids = torch.full(size=(b, 1), fill_value=soundstream_sos_id, dtype=torch.long, device=device)
            soundstream_ids = torch.cat([soundstream_sos_ids, soundstream_ids], dim=1)
        else:
            # return soundstream sos ids for inference
            soundstream_ids = torch.full(size=(b, 1), fill_value=soundstream_sos_id, dtype=torch.long, device=device)

        return soundstream_ids, mulan_ids, lyrics_ids, vocal_mulan_ids, chroma_ids
            
    
    @torch.no_grad()
    def get_mulan_prompt_embeds(self, texts, device="cuda"):
        text_embs = []
        for text in texts:
            text_emb = self.requires["mulan_infer_fn"](
                self.requires["mulan"], text=text, device=device
            )
            text_embs.append(text_emb)

        mulan_embeds = torch.cat(text_embs, dim=0)
        return mulan_embeds

    @torch.no_grad()
    def get_mulan_prompt_tokens(self, texts, device="cuda"):
        mulan_embeds = self.get_mulan_prompt_embeds(texts, device)
        mulan_ids, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        return mulan_ids

    @torch.no_grad()
    def predict_with_inputs(self, input_ids, hp):
        soundstream_frame_rate = self.extra_params.soundstream_frame_rate
        soundstream_codebook_size = self.extra_params.soundstream_codebook_size
        num_coarse = self.extra_params.num_coarse
        num_tokens = self.extra_params.duration * soundstream_frame_rate * num_coarse
        
        coarse_samples = None
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        for i in pbar:
            pbar.set_description(f"Coarse [0 - {num_tokens}]")
            model_output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = model_output["past_key_values"]
            logits = model_output["logits"]
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1:, layer_idx * soundstream_codebook_size : (layer_idx + 1) * soundstream_codebook_size]
            samples = sample(predict_logits, temp=hp.coarse_temperature, mode=hp.sample_mode)
            samples = samples + layer_idx * soundstream_codebook_size
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples

    @torch.no_grad()
    def predict(self, batch, hp, conditions):

        lyrics_tokens = batch.get('lyrics_tokens')
        mulan_text = batch.get('mulan_text')
        mulan_audio = batch.get('mulan_audio')
        vocal_audio = batch.get('vocal_audio')
        chroma = batch.get("vocal_chroma")

        if 'text_prompt' in conditions:
            mulan_ids = self.get_mulan_prompt_tokens(mulan_text)
        elif 'audio_prompt' in conditions:
            mulan_ids = self.get_mulan_tokens(mulan_audio.float())
        else:
            mulan_ids = None

        soundstream_ids = None
        lyrics_ids = lyrics_tokens.to(self.device) if 'lyrics' in conditions else None
        vocal_mulan_ids = self.get_mulan_tokens(vocal_audio) if 'mulan_vocals' in conditions else None
        chroma_ids = chroma if 'vocal_chroma' in conditions else None

        soundstream_ids, *condition_ids = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids, vocal_mulan_ids, chroma_ids)

        input_ids = torch.cat(condition_ids + [soundstream_ids], dim=1)
        return self.predict_with_inputs(input_ids, hp)

class MulanPhonemeCoarseModule(ConditionalMulanPhonemeCoarseModule):
    @torch.no_grad()
    def prepare_feature(self, batch):
        mulan_audio, target_audio, lyrics_tokens = batch.get("mulan_audio"), batch.get("target_audio"), batch.get("lyrics_tokens")

        soundstream_ids = self.get_soundstream_tokens(target_audio)
        mulan_ids = self.get_mulan_tokens(mulan_audio)
        lyrics_ids = lyrics_tokens

        soundstream_ids, mulan_ids, lyrics_ids, *_ = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids)
        input_ids = torch.cat([mulan_ids, lyrics_ids, soundstream_ids[:, :-1]], dim=1)
        target_ids = soundstream_ids[:, 1:]

        return input_ids, target_ids
    
    @torch.no_grad()
    def predict(self, batch, hp, conditions):
        lyrics_tokens = batch.get('lyrics_tokens')
        mulan_text = batch.get('mulan_text')
        mulan_audio = batch.get('mulan_audio')

        if 'text_prompt' in conditions:
            mulan_ids = self.get_mulan_prompt_tokens(mulan_text)
        elif 'audio_prompt' in conditions:
            mulan_ids = self.get_mulan_tokens(mulan_audio.float())
        else:
            mulan_ids = None

        soundstream_ids = None
        lyrics_ids = lyrics_tokens.to(self.device)
        soundstream_ids, mulan_ids, lyrics_ids, *_ = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids)
        input_ids = torch.cat([mulan_ids, lyrics_ids, soundstream_ids], dim=1)
        return self.predict_with_inputs(input_ids, hp)

class EmbedMulanPhonemeCoarseModule(ConditionalMulanPhonemeCoarseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        # mulan_dim = self.extra_params.mulan_codebook_size
        mulan_dim = 512
        emb_dim = self.model.config.hidden_size
        self.mulan_lin_embed = torch.nn.Linear(mulan_dim, emb_dim, bias=False)

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch)
        logits = self.model(**input_ids)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        x = logits[:, -target_ids.size(1):, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    # @torch.no_grad() # need grad for input embeddings
    def prepare_feature(self, batch):
        mulan_audio, target_audio, lyrics_tokens = batch.get("mulan_audio"), batch.get("target_audio"), batch.get("lyrics_tokens")
        vocal_audio, vocal_chroma = batch.get("vocal_audio"), batch.get("vocal_chroma")

        soundstream_ids = None
        mulan_ids = None
        mulan_embeds = None
        vocal_mulan_embeds = None
        chroma_ids = None
        lyrics_ids = lyrics_tokens
            
        with torch.no_grad():
            soundstream_ids = self.get_soundstream_tokens(target_audio)

            if mulan_audio is not None:
                mulan_embeds = self.get_mulan_embeds(mulan_audio)

            if vocal_audio is not None and random.random() > self.extra_params.vocal_condition_p:
                vocal_mulan_embeds = self.get_mulan_embeds(vocal_audio)
            if vocal_audio is not None and random.random() > self.extra_params.chroma_condition_p:
                chroma_ids = vocal_chroma

            soundstream_ids, mulan_ids, lyrics_ids, _, chroma_ids = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids, chroma_ids=chroma_ids)

        from samantha.models.flash_llama import LlamaForCausalLM
        model: LlamaForCausalLM = self.model
        soundstream_embeds = model.model.embed_tokens(soundstream_ids)
        b, sl, d = soundstream_embeds.shape
        empty_embeds = input_ids = torch.zeros((b, 0, d), dtype=soundstream_embeds.dtype, device=self.device)

        mulan_embeds = self.mulan_lin_embed(mulan_embeds)[:, None, :] if mulan_embeds is not None else empty_embeds
        lyrics_embeds = model.model.embed_tokens(lyrics_ids) if lyrics_ids is not None else empty_embeds
        vocal_mulan_embeds = self.mulan_lin_embed(vocal_mulan_embeds)[:, None, :] if vocal_mulan_embeds is not None else empty_embeds
        chroma_embeds = model.model.embed_tokens(chroma_ids) if chroma_ids is not None else empty_embeds
        inputs_embeds = torch.cat([mulan_embeds, lyrics_embeds, vocal_mulan_embeds, chroma_embeds, soundstream_embeds[:, :-1, :]], dim=1)
        target_ids = soundstream_ids[:, 1:]
        inputs = {
            "inputs_embeds": inputs_embeds
        }

        return inputs, target_ids
            

    @torch.no_grad()
    def predict(self, batch, hp, conditions):

        lyrics_tokens = batch.get('lyrics_tokens')
        mulan_text = batch.get('mulan_text')
        mulan_audio = batch.get('mulan_audio')
        vocal_audio = batch.get('vocal_audio')
        vocal_chroma = batch.get("vocal_chroma")

        if 'text_prompt' in conditions:
            mulan_embeds = self.get_mulan_prompt_embeds(mulan_text)
        elif 'audio_prompt' in conditions:
            mulan_embeds = self.get_mulan_embeds(mulan_audio.float())
        else:
            mulan_embeds = None

        if 'mulan_vocals' in conditions:
            vocal_mulan_embeds = self.get_mulan_embeds(vocal_audio)
        else:
            vocal_mulan_embeds = None


        soundstream_ids = None
        lyrics_ids = lyrics_tokens.to(self.device) if 'lyrics' in conditions else None
        chroma_ids = vocal_chroma if 'vocal_chroma' in conditions else None
        mulan_ids = None
        soundstream_ids, mulan_ids, lyrics_ids, _, chroma_ids = self.prepare_conditions(soundstream_ids, mulan_ids, lyrics_ids, chroma_ids=chroma_ids)



        from samantha.models.flash_llama import LlamaForCausalLM
        model: LlamaForCausalLM = self.model
        soundstream_embeds = model.model.embed_tokens(soundstream_ids)

        b, sl, d = soundstream_embeds.shape
        empty_embeds = input_ids = torch.zeros((b, 0, d), dtype=soundstream_embeds.dtype, device=self.device)

        mulan_embeds = self.mulan_lin_embed(mulan_embeds)[:, None, :] if mulan_embeds is not None else empty_embeds
        lyrics_embeds = model.model.embed_tokens(lyrics_ids) if lyrics_ids is not None else empty_embeds
        vocal_mulan_embeds = self.mulan_lin_embed(vocal_mulan_embeds)[:, None, :] if vocal_mulan_embeds is not None else empty_embeds
        chroma_embeds = model.model.embed_tokens(chroma_ids) if chroma_ids is not None else empty_embeds
        inputs_embeds = torch.cat([mulan_embeds, lyrics_embeds, vocal_mulan_embeds, chroma_embeds, soundstream_embeds[:, :-1, :]], dim=1)

        soundstream_frame_rate = self.extra_params.soundstream_frame_rate
        soundstream_codebook_size = self.extra_params.soundstream_codebook_size
        num_coarse = self.extra_params.num_coarse
        num_tokens = self.extra_params.duration * soundstream_frame_rate * num_coarse
        
        coarse_samples = None
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        for i in pbar:
            pbar.set_description(f"Coarse [0 - {num_tokens}]")
            model_output = self.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
            past_key_values = model_output["past_key_values"]
            logits = model_output["logits"]
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1:, layer_idx * soundstream_codebook_size : (layer_idx + 1) * soundstream_codebook_size]
            samples = sample(predict_logits, temp=hp.coarse_temperature, mode=hp.sample_mode)
            samples = samples + layer_idx * soundstream_codebook_size
            inputs_embeds = model.model.embed_tokens(samples)
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return coarse_samples

from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import (
    MulanEmbedder, LyricsTokenEmbedder, WavToVecTokenEmbedder, 
    MetadataT5TokenEmbedder, SpeakerEmbedder, BestRQTokenEmbedder
)
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from samantha.utils.hparams import DotDict

# from recipes.umm.models.bestrq import BestRQMelCTC
from recipes.umm.modules.lit_module import (
    BestRQMelCTC, Stage3)

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
        lyrics_vocab_size = extra_params['lyrics_codebook_size']
        mulan_embed_dim = extra_params['mulan_embed_dim']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            'mulan': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True)
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
        if 'style_text' in conditions:
            embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_text'], with_sos=with_sos, data_type='text')
            inputs_embeds.append(embeds)
        elif 'style_audio' in conditions:
            embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'], with_sos=with_sos, data_type='music')
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(self.input_embedders['mulan'].get_sos_embed(batch_size))
        if 'lyrics_tokens' in conditions:
            embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['lyrics_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature):
        return super().predict(inputs_embeds, num_tokens, temperature)


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
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=False)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=False)
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


class SpeechSemanticModule(BaseContinuousEmbedModule):
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
        # speaker_vocab_size = extra_params['speaker_codebook_size']
        # For now we use mulan audio tower as speaker embedder
        mulan_embed_dim = extra_params['mulan_embed_dim']
        embedder_dict = {
            'mulan': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        semantic_vocab_size = extra_params['semantic_codebook_size']
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params['semantic_type']
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_vocab_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_vocab_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
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
        with_sos=True
        # convert inputs to conditions
        # speaker_embeds = self.input_embedders['speaker_id'].embed(self.requires, batch['speaker_id'], with_sos=with_sos)
        speaker_embeds = self.input_embedders['mulan'].embed(
            self.requires, 
            batch['target_audio'][:,:,0:10*24000].squeeze(), 
            with_sos=with_sos, 
            data_type='music')
        lyrics_embeds = self.input_embedders['lyrics_tokens'].embed(
            self.requires, 
            batch['lyrics_tokens'], 
            with_sos=with_sos)
        inputs_embeds = [speaker_embeds, lyrics_embeds]
        return torch.cat(inputs_embeds, dim=1)


    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

    
    @torch.no_grad()
    def get_mel(self, batch):
        # predict tokens
        hp = {
            **self.extra_params,
            'semantic_temperature': 1,
            'duration': 10
        }
        hp = DotDict(hp)
        predict_tokens = self.predict(batch, hp)
        
        bestrq_model: BestRQMelCTC = self.requires['semantic']
        predict_mel = bestrq_model.model.token_to_mel(predict_tokens)

        target_mel, _ = bestrq_model.prepare_feature({'audio': batch['target_audio']})
        return predict_mel.transpose(1, 2), target_mel.transpose(1, 2)


class MixSemanticModule(BaseContinuousEmbedModule):
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
        mulan_embed_dim = extra_params['mulan_embed_dim']
        lyrics_vocab_size = extra_params['lyrics_codebook_size']
        semantic_vocab_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            # TODO: (AS) rename mulan_text to style_text
            'mulan_text': MulanEmbedder(data_type='text', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'lyrics': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params['semantic_type']
        assert semantic_type == 'bestrq'
        target_embedder = BestRQTokenEmbedder(vocab_size=semantic_vocab_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        
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
        with_sos=True
        # convert inputs to conditions
        style_embeds = self.input_embedders['mulan_text'].embed(self.requires, batch['style_text'], with_sos=with_sos)
        lyrics_embeds = self.input_embedders['lyrics'].embed(self.requires, batch['lyrics_tokens'], with_sos=with_sos)
        # Lyrics at last so it is ok to truncate at prefix_max_len
        inputs_embeds = [style_embeds, lyrics_embeds]
        return torch.cat(inputs_embeds, dim=1)


    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature):
        return super().predict(inputs_embeds, num_tokens, temperature)

    @torch.no_grad()
    def get_mel(self, batch):
        # predict tokens
        hp = {
            **self.extra_params,
            'semantic_temperature': 1,
            'duration': 10
        }
        hp = DotDict(hp)
        predict_tokens = self.predict(batch, hp)
        
        bestrq_model: Stage3 = self.requires['Stage3']
        predict_mel = bestrq_model.model.token_to_mel(predict_tokens)

        target_mel, _ = bestrq_model.prepare_feature({'audio': batch['target_audio']})
        return predict_mel.transpose(1, 2), target_mel.transpose(1, 2)


class SingSongSemanticModule(BaseContinuousEmbedModule):
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
        mulan_embed_dim = extra_params['mulan_embed_dim']
        semantic_vocab_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            'mulan': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params['semantic_type']
        assert semantic_type == 'bestrq'
        target_embedder = BestRQTokenEmbedder(vocab_size=semantic_vocab_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        
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
        with_sos=True
        style_embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'], with_sos=with_sos)        
        vocal_embeds = self.target_embedder.embed(self.requires, batch['vocal_audio'], with_sos=True, with_eos=False)
        inputs_embeds = [style_embeds, vocal_embeds]
        return torch.cat(inputs_embeds, dim=1)

    def prepare_training_inputs(self, batch):        
        if "target_audio" in batch:
            target_audio = batch["target_audio"]
        else:
            assert "style_audio" in batch and "vocal_audio" in batch
            target_audio = batch["style_audio"] + batch["vocal_audio"]
        target_ids = self.target_embedder.tokenize(self.requires, target_audio, with_sos=False, with_eos=False)
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(batch)
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)
        eos_ids = self.target_embedder.get_eos_token(batch_size)
        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)
        if self.use_cross_attn:
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds
            }, torch.cat([target_ids, eos_ids], dim=1)
        input_embeds = torch.cat([inputs_embeds, sos_embeds, target_embeds], dim=1)
        target_ids = torch.cat([target_ids, eos_ids], dim=1)        
        return {"inputs_embeds":  input_embeds}, target_ids
    
    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)
        


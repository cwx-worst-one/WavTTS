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

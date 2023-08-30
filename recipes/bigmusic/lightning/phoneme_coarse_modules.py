from recipes.bigmusic.lightning.base_modules import BaseContinuousEmbedModule
from recipes.bigmusic.lightning.embedding_modules import MulanEmbedder, LyricsTokenEmbedder, VocalChromaEmbedder, SoundstreamTokenEmbedder, MetadataT5TokenEmbedder
import torch
from tqdm.auto import tqdm
from recipes.musiclm.inference.utils import sample
import torch.nn as nn

class ConditionalCoarseModule(BaseContinuousEmbedModule):
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
        embedder_dict = {
            'mulan': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
            'vocal_audio': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'vocal_chroma': VocalChromaEmbedder(13, hidden_size, add_sos=True)
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        target_embedder = SoundstreamTokenEmbedder(layer_range=(0,4), embedding_dim=hidden_size, add_sos=True)

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
        if 'vocal_audio' in conditions:
            embeds = self.input_embedders['vocal_audio'].embed(self.requires, batch['vocal_audio'], with_sos=with_sos, data_type='music')
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(self.input_embedders['vocal_audio'].get_sos_embed(batch_size))

        if 'vocal_chroma' in conditions:
            embeds = self.input_embedders['vocal_chroma'].embed(self.requires, batch['vocal_chroma'], with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['vocal_chroma'].get_sos_embed(batch_size))

        print('Shapes', [i.shape for i in inputs_embeds])
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        soundstream_frame_rate = self.extra_params.soundstream_frame_rate
        num_coarse = self.extra_params.num_coarse
        num_tokens = self.extra_params.duration * soundstream_frame_rate * num_coarse
        temperature = hp.coarse_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

class LyricsCoarseModule(BaseContinuousEmbedModule):
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
        embedder_dict = {
            'mulan': MulanEmbedder(data_type='music', input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        target_embedder = SoundstreamTokenEmbedder(layer_range=(0,4), embedding_dim=hidden_size, add_sos=True)

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

    def prepare_inputs_embeddings(self, batch):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        with_sos=True # TODO: (AS) clean this up
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
        print('Shapes', conditions, [i.shape for i in inputs_embeds])
        return torch.cat(inputs_embeds, dim=1)

    def sample_logits(self, i, logits, temp, mode):
        soundstream_codebook_size = self.extra_params.soundstream_codebook_size
        num_coarse = self.extra_params.num_coarse

        layer_idx = i % num_coarse
        predict_logits = logits[
            :,
            -1:,
            layer_idx
            * soundstream_codebook_size : (layer_idx + 1)
            * soundstream_codebook_size,
        ]
        samples = sample(
            predict_logits, temp=temp, mode=mode
        )
        samples = samples + layer_idx * soundstream_codebook_size
        return samples

    @torch.no_grad()
    def predict(self, batch, hp):
        soundstream_frame_rate = self.extra_params.soundstream_frame_rate
        num_coarse = self.extra_params.num_coarse
        num_tokens = self.extra_params.duration * soundstream_frame_rate * num_coarse
        temperature = hp.coarse_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

class T55LyricsCoarseModule(BaseContinuousEmbedModule):
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
        embedder_dict = {
            'style_tokens': MetadataT5TokenEmbedder(embedding_dim=hidden_size, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        target_embedder = SoundstreamTokenEmbedder(layer_range=(0,4), embedding_dim=hidden_size, add_sos=True)

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
            inputs_embeds.append(self.input_embedders['style_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        soundstream_frame_rate = self.extra_params.soundstream_frame_rate
        num_coarse = self.extra_params.num_coarse
        num_tokens = self.extra_params.duration * soundstream_frame_rate * num_coarse
        temperature = hp.coarse_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)

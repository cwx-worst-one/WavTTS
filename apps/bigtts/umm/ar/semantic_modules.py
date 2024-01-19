

import torch
import torch.nn as nn
from apps.bigtts.umm.ar.base_modules import BaseContinuousEmbedModule

from samantha.components.embedder import (BestRQTokenEmbedder,
                                          LyricsTokenEmbedder,
                                          WavToVecTokenEmbedder)


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.int()
    return mask


class SemanticModule_Valle(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules={},
        checkpointing=False,
        extra_params=None,
    ):
        hidden_size = extra_params['hidden_size']
        lyrics_vocab_size = extra_params['lyrics_codebook_size']
        # mulan_embed_dim = extra_params['mulan_embed_dim']
        semantic_codebook_size = extra_params['semantic_codebook_size']
        embedder_dict = {
            # 'mulan': MulanTagEmbedder(input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            # 'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
            "lyrics_phones": LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
            "lyrics_tones": LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size // 2, add_sos=True),
            "lyrics_wordsegs": LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size // 2, add_sos=True),
            "lyrics_joints": torch.nn.Linear(2*hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get('semantic_type', 'wav2vec')
        if semantic_type  == 'wav2vec':
            target_embedder = WavToVecTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        elif semantic_type == 'bestrq':
            target_embedder = BestRQTokenEmbedder(vocab_size=semantic_codebook_size, embedding_dim=hidden_size, add_sos=True, add_eos=True)
        else:
            raise NotImplementedError

        print(f"lang_embeddings size: [256, {hidden_size}]")
        lang_embeddings = nn.Embedding(256, hidden_size)

        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            lang_embeddings=lang_embeddings,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch, update_mfu=True)
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_dict({"training/loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(f"bigmusic.prepare_training_inputs{self.trainer.global_step}"):
            input_ids, target_ids, input_lens, target_lens = self.prepare_training_inputs(batch)
            # input_ids['inputs_embeds']: [b, t = 1+tp + 1 + t_u, c]
            # target_ids: [b, t_u+1]
            # input_lens: token lengths, max = t_p
            # target_lens: target length, max = t_u

            seq_lens = input_lens + 1 + target_lens+1   # input + sos + target + eos
            loss_mask = sequence_mask(seq_lens, max_len=input_ids['inputs_embeds'].shape[1], device=input_ids['inputs_embeds'].device)
            text_loss_mask = sequence_mask(input_lens+1, max_len=input_ids['inputs_embeds'].shape[1], device=input_ids['inputs_embeds'].device)
            if True:    # do not cal text_loss
                loss_mask = loss_mask - text_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t*valid_token_ratio}
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    self.log_dict(
                        self.metric.compute(self.trainer.global_step),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    self.log_dict({"batch": b, "token_num": b*t, "valid_token_ratio": valid_token_ratio}, prog_bar=True, sync_dist=True)

        with self.profiler.profile(f"bigmusic.forward.step{self.trainer.global_step}"):
            logits = self.model(**input_ids)    # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):        # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        # x = logits[:, -target_ids.size(1):, :]  # [b, t, c]
        # loss = self.criterion(x, target_ids)
        # accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        with self.profiler.profile(f"bigmusic.post.step{self.trainer.global_step}"):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)    # [b, 1]
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                h[i, : input_lens[i] + 1 + target_lens[i]+1] = torch.cat((batch['lyrics_tokens'][i, :input_lens[i]], sos_ids[i, :], target_ids[i, :target_lens[i]+1]))    # [b, t_p + 1 + t_u+1]
            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum() / loss_mask.sum() * 100

        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(self.requires, batch['target_audio'], with_sos=False, with_eos=False)    # [b, 1, t_w] --> [b, t_u]
            target_lens = (batch['audio_lengths'] / (self.extra_params['sample_rate'] / self.extra_params['semantic_frame_rate'])).ceil().long()   # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch['target_ids']
            target_lens = batch['target_ids_length']

        land_ids = batch['lang']
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(batch)           # [b, t_p] --> [b, 1+t_p, h]
        input_lens = 1 + batch['lyrics_token_length']

        sos_embeds = self.target_embedder.get_sos_embed(batch_size)     # [b, 1, h]
        target_embeds = self.target_embedder.embed(token_ids=target_ids, with_sos=False, with_eos=False)    # [b, t_u] --> [b, t_u, h]
        target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:     # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)    # [b, 1]
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(target_ids.device)
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
        if self.use_cross_attn:     # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds
            }, torch.cat([target_ids, eos_ids], dim=1)

        h = torch.zeros([batch_size, inputs_embeds.shape[1] + sos_embeds.shape[1] + target_embeds.shape[1], inputs_embeds.shape[-1]], device=inputs_embeds.device)
        target_ids_  = torch.zeros([batch_size, target_ids.shape[1] + eos_ids.shape[1]], device=inputs_embeds.device).long()
        for i in range(batch_size):
            h[i, :input_lens[i], :] = inputs_embeds[i, :input_lens[i], :]
            h[i, input_lens[i]:input_lens[i]+1, :] = sos_embeds[i, :, :]
            try:
                h[i, input_lens[i]+1:input_lens[i]+1+target_lens[i], :] = target_embeds[i, :target_lens[i], :]
                target_ids_[i, :target_lens[i]] = target_ids[i, :target_lens[i]]
                target_ids_[i, target_lens[i]:target_lens[i]+1] = eos_ids[i, :]
            except:
                import pdb; pdb.set_trace()
                h[i, input_lens[i]+1:input_lens[i]+1+target_lens[i]-1, :] = target_embeds[i, :target_lens[i], :]
                target_ids_[i, :target_lens[i]] = target_ids[i, :target_lens[i]]
                target_ids_[i, target_lens[i]:target_lens[i]+1] = eos_ids[i, :]
                print(f">>> mismatch length")

        model_inputs = {
            # "inputs_embeds": torch.cat([inputs_embeds, sos_embeds, target_embeds], dim=1)
            "inputs_embeds": h          # [b, 1+t_p + 1 + t_u]
        }
        # target_ids = torch.cat([target_ids, eos_ids], dim=1)
        target_ids = target_ids_        # [b, t_u + 1]
        target_lens = target_lens + 1

        if return_all:
            return model_inputs, target_ids, inputs_embeds, sos_embeds, target_embeds
        else:
            return model_inputs, target_ids, input_lens - 1, target_lens - 1

    def prepare_inputs_embeddings(self, batch):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        with_sos=True
        # convert inputs to conditions
        inputs_embeds = []
        # if 'style_text' in conditions:
        #     embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_text'], with_sos=with_sos, data_type='text')
        #     inputs_embeds.append(embeds)
        # elif 'style_audio' in conditions:
        #     embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'].to(self.device), with_sos=with_sos, data_type='music')
        #     inputs_embeds.append(embeds)
        # elif 'style_tag' in conditions: # using Mulan for on-the-fly MIR tagging
        #     embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'].to(self.device), with_sos=with_sos, data_type='tag')
        #     inputs_embeds.append(embeds)
        # else:
        #     # adding SOS token no matter what so that all parameters get used
        #     inputs_embeds.append(self.input_embedders['mulan'].get_sos_embed(batch_size))
        if 'lyrics_tokens' in conditions:
            # embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'].to(self.device), with_sos=with_sos)
            phone_embeds = self.input_embedders['lyrics_phones'].embed(self.requires, batch['phones'].to(self.device), with_sos=with_sos)
            tone_embeds = self.input_embedders['lyrics_tones'].embed(self.requires, batch['tones'].to(self.device), with_sos=with_sos)
            wordseg_embeds = self.input_embedders['lyrics_wordsegs'].embed(self.requires, batch['wordsegs'].to(self.device), with_sos=with_sos)
            embeds = self.input_embedders['lyrics_joints'](torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1))
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['lyrics_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(self.requires, batch['audio_prompt'].to(self.device), with_sos=False, with_eos=False)
        print(f"target_embeds: {target_embeds}")

        return super().predict(
            inputs_embeds,
            target_embeds,
            num_tokens,
            sample_mode=sample_mode,
            temperature=temperature,
            beam=beam,
            ref_samples=ref_samples,
        )

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)

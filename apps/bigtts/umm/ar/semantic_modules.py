import logging
from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

from samantha.components.embedder import (
    BestRQTokenEmbedder,
    LyricsTokenEmbedder,
    MetadataT5TokenEmbedder,
    WavToVecTokenEmbedder,
)
from samantha.components.ops import masked_cat2d, masked_cat3d
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams

from .base_modules import BaseContinuousEmbedModule
from .utils import TokenBuffer, get_split_emb, reorder_attr, sequence_mask

logger = logging.getLogger(__name__)


class SemanticModule(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):

        if required_modules is None:
            required_modules = {}
        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "lyrics_tokens": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            )
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
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

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            embeds = self.input_embedders["lyrics_tokens"].embed(
                self.requires, batch["lyrics_tokens"].to(self.device), with_sos=with_sos
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )
        logger.info(f"target_embeds: {target_embeds}")
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

        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "style_tokens": MetadataT5TokenEmbedder(
                embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tokens": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
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

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            embeds = self.input_embedders["lyrics_tokens"].embed(
                self.requires, batch["lyrics_tokens"], with_sos=with_sos
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        if "style_tokens" in conditions:
            embeds = self.input_embedders["style_tokens"].embed(
                self.requires, batch["style_tokens"], with_sos=with_sos
            )
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(
                self.input_embedders["style_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        return super().predict(inputs_embeds, num_tokens, temperature)


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
        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        # mulan_embed_dim = extra_params['mulan_embed_dim']
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "lyrics_phones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_wordsegs": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_joints": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        else:
            raise NotImplementedError

        self.use_lang_embedding = use_lang_embedding = extra_params.get(
            "use_lang_embedding", True
        )
        self.use_text_lang_embedding = extra_params.get(
            "use_text_lang_embedding", False
        )
        lang_embeddings = None
        if use_lang_embedding:
            logger.info(f"lang_embeddings size: [256, {hidden_size}]")
            logger.info(f"use_text_lang_embedding: {self.use_text_lang_embedding}")
            lang_embeddings = nn.Embedding(256, hidden_size)
        else:
            logger.info(f"======= do not use lang_embeddings...")

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
            self.log_dict(
                {"training/loss": loss, "training/accu": accu},
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(
            f"bigtts.umm.ar.prepare_training_inputs{self.trainer.global_step}"
        ):
            (
                input_ids,
                target_ids,
                input_lens,
                target_lens,
            ) = self.prepare_training_inputs(batch)
            # input_ids['inputs_embeds']: [b, t = 1+tp + 1 + t_u, c]
            # target_ids: [b, t_u+1]
            # input_lens: token lengths, max = t_p
            # target_lens: target length, max = t_u

            seq_lens = input_lens + 1 + target_lens + 1  # input + sos + target + eos
            loss_mask = sequence_mask(
                seq_lens,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            text_loss_mask = sequence_mask(
                input_lens + 1,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            if True:  # do not cal text_loss
                loss_mask = loss_mask - text_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t * valid_token_ratio},
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    logs = {
                        "batch": b,
                        "token_num": b * t,
                        "valid_token_ratio": valid_token_ratio,
                    }
                    logs.update(self.metric.compute(self.trainer.global_step))
                    self.log_dict(logs, prog_bar=True, sync_dist=True)

        with self.profiler.profile(
            f"bigtts.umm.ar.forward.step-{self.trainer.global_step}"
        ):
            logits = self.model(**input_ids)  # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):  # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        with self.profiler.profile(
            f"bigtts.umm.ar.post.step-{self.trainer.global_step}"
        ):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)  # [b, 1]
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                h[i, : input_lens[i] + 1 + target_lens[i] + 1] = torch.cat(
                    (
                        batch["lyrics_tokens"][i, : input_lens[i]],
                        sos_ids[i, :],
                        target_ids[i, : target_lens[i] + 1],
                    )
                )  # [b, t_p + 1 + t_u+1]
            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = (
                ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum()
                / loss_mask.sum()
                * 100
            )

        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(
                self.requires, batch["target_audio"], with_sos=False, with_eos=False
            )  # [b, 1, t_w] --> [b, t_u]
            target_lens = (
                (
                    batch["audio_lengths"]
                    / (
                        self.extra_params["sample_rate"]
                        / self.extra_params["semantic_frame_rate"]
                    )
                )
                .ceil()
                .long()
            )  # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch["target_ids"]
            target_lens = batch["target_ids_length"]

        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(
            batch
        )  # [b, t_p] --> [b, 1+t_p, h]
        input_lens = 1 + batch["lyrics_token_length"]

        sos_embeds = self.target_embedder.get_sos_embed(batch_size)  # [b, 1, h]
        target_embeds = self.target_embedder.embed(
            token_ids=target_ids, with_sos=False, with_eos=False
        )  # [b, t_u] --> [b, t_u, h]
        if self.lang_embeddings is not None:
            land_ids = batch["lang"]
            target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

            if self.use_text_lang_embedding:
                inputs_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:  # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)  # [b, 1]
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(
                target_ids.device
            )
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
        if self.use_cross_attn:  # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds,
            }, torch.cat([target_ids, eos_ids], dim=1)

        eos_len = torch.ones(
            batch_size, dtype=input_lens.dtype, device=input_lens.device
        )
        # customized op
        # concat inputs_embeds, sos_embes and target_embeds with correspond valid length
        # basically like below:
        #
        # h: [b, max(sum(valid_length)), ...]
        # h[i] = Concat(
        #           inputs_embeds[i, :input_lens[i]],
        #           sos_embeds[i, :eos_len[i]],
        #           target_embeds[i, :target_lens]
        #        )
        h = masked_cat3d(
            inputs_embeds, sos_embeds, target_embeds, input_lens, eos_len, target_lens
        )
        target_ids_ = masked_cat2d(target_ids, eos_ids, target_lens, eos_len)

        model_inputs = {"inputs_embeds": h}  # [b, 1+t_p + 1 + t_u]
        # target_ids = torch.cat([target_ids, eos_ids], dim=1)
        target_ids = target_ids_  # [b, t_u + 1]
        target_lens = target_lens + 1  # no need cause return target_lens -1 ?

        if return_all:
            return model_inputs, target_ids, inputs_embeds, sos_embeds, target_embeds
        else:
            return model_inputs, target_ids, input_lens - 1, target_lens - 1

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            phone_embeds = self.input_embedders["lyrics_phones"].embed(
                self.requires, batch["phones"].to(self.device), with_sos=with_sos
            )
            tone_embeds = self.input_embedders["lyrics_tones"].embed(
                self.requires, batch["tones"].to(self.device), with_sos=with_sos
            )
            wordseg_embeds = self.input_embedders["lyrics_wordsegs"].embed(
                self.requires, batch["wordsegs"].to(self.device), with_sos=with_sos
            )
            embeds = self.input_embedders["lyrics_joints"](
                torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1)
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        thresh = hp.semantic_thresh
        sample_mode = hp.sample_mode
        max_blank_length = hp.max_blank_length
        step_out_blank = hp.step_out_blank

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )

        src_lang_embed = None
        tgt_lang_embed = None
        if self.lang_embeddings is not None:
            src_land_ids = batch["src_lang"].to(self.device)
            src_lang_embed = self.lang_embeddings(src_land_ids).unsqueeze(1)
            target_embeds += src_lang_embed

            tgt_land_ids = batch["tgt_lang"].to(self.device)
            tgt_lang_embed = self.lang_embeddings(tgt_land_ids).unsqueeze(1)

            if self.use_text_lang_embedding:
                prompt_text_lens = batch["prompt_text_lens"][0]
                inputs_embeds[:, :prompt_text_lens] += src_lang_embed
                inputs_embeds[:, prompt_text_lens:] += tgt_lang_embed

        else:
            logger.info("======= do not use lang_embeddings...")

        logger.info(f"target_embeds: {target_embeds}")

        return super().predict(
            inputs_embeds,
            target_embeds,
            num_tokens,
            tgt_lang_embed,
            sample_mode=sample_mode,
            temperature=temperature,
            thresh=thresh,
            beam=beam,
            ref_samples=ref_samples,
            max_blank_length=max_blank_length,
            step_out_blank=step_out_blank,
        )

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)


class SemanticModule_SpkidLLMV3(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):

        if required_modules is None:
            required_modules = {}
        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "lyrics_phones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_wordsegs": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_joints": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
        else:
            raise NotImplementedError

        logger.info(f"lang_embeddings size: [256, {hidden_size}]")
        lang_embeddings = nn.Embedding(256, hidden_size)

        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            prompt_embedder=prompt_embedder,
            lang_embeddings=lang_embeddings,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch, update_mfu=True)
        self.log_dict(
            {"training/loss": loss, "training/accu": accu},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(
            f"bigtts.umm.ar.prepare_training_inputs{self.trainer.global_step}"
        ):
            (
                input_ids,
                target_ids,
                input_lens,
                target_lens,
                prompt_lens,
            ) = self.prepare_training_inputs(batch)

            seq_lens = input_lens + 1 + prompt_lens + 1 + target_lens + 1
            loss_mask = sequence_mask(
                seq_lens,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            text_query_loss_mask = sequence_mask(
                input_lens + 1 + prompt_lens + 1,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )

            if True:  # do not cal text_loss
                loss_mask = loss_mask - text_query_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t * valid_token_ratio},
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    logs = {
                        "batch": b,
                        "token_num": b * t,
                        "valid_token_ratio": valid_token_ratio,
                    }
                    logs.update(self.metric.compute(self.trainer.global_step))
                    self.log_dict(logs, prog_bar=True, sync_dist=True)

        with self.profiler.profile(
            f"bigtts.umm.ar.forward.step-{self.trainer.global_step}"
        ):
            logits = self.model(**input_ids)  # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):  # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        with self.profiler.profile(
            f"bigtts.umm.ar.post.step-{self.trainer.global_step}"
        ):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)  # [b, 1]
            prompt_sos_ids = self.prompt_embedder.get_sos_token(bsz)
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                offset = input_lens[i] + 1 + prompt_lens[i] + 1 + target_lens[i] + 1
                h[i, :offset] = torch.cat(
                    (
                        batch["lyrics_tokens"][i, : input_lens[i]],
                        prompt_sos_ids[i, :],
                        # query placeholder
                        torch.zeros([prompt_lens[i]]).to(sos_ids.device),
                        sos_ids[i, :],
                        target_ids[i, : target_lens[i] + 1],
                    )
                )

            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = (
                ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum()
                / loss_mask.sum()
                * 100
            )

        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(
                self.requires, batch["target_audio"], with_sos=False, with_eos=False
            )  # [b, 1, t_w] --> [b, t_u]
            sample_rate, semantic_frame_rate = (
                self.extra_params["sample_rate"],
                self.extra_params["semantic_frame_rate"],
            )
            target_lens = (
                (batch["audio_lengths"] / (sample_rate / semantic_frame_rate))
                .ceil()
                .long()
            )  # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch["target_ids"]
            target_lens = batch["target_ids_length"]

        assert "prompt_ids" in batch
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_ids_length"]

        land_ids = batch["lang"]
        batch_size = target_ids.size(0)
        # [b, t_p] --> [b, 1+t_p, h]
        inputs_embeds = self.prepare_inputs_embeddings(batch)
        inputs_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        input_lens = 1 + batch["lyrics_token_length"]

        # [b, 1, h]
        sos_embeds = self.target_embedder.get_sos_embed(batch_size)
        # [b, t_u] --> [b, t_u, h]
        target_embeds = self.target_embedder.embed(
            token_ids=target_ids, with_sos=False, with_eos=False
        )
        sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size)
        prompt_embeds = self.prompt_embedder.embed(
            token_ids=prompt_ids, with_sos=False, with_eos=False
        )
        prompt_sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        prompt_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:  # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)  # [b, 1]
        else:
            eos_ids = torch.zeros(
                (batch_size, 0), dtype=target_ids.dtype, device=target_ids.device
            )
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
            prompt_embeds = prompt_embeds[:, :-1, :]  # 保持和target一样长度，用同一套target len

        if self.use_cross_attn:  # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds,
            }, torch.cat([target_ids, eos_ids], dim=1)

        max_len = torch.max(input_lens + prompt_lens + target_lens + 2).item()
        h = torch.zeros(
            [batch_size, max_len, inputs_embeds.shape[-1]], device=inputs_embeds.device
        )
        target_ids_ = torch.zeros(
            [batch_size, target_ids.shape[1] + eos_ids.shape[1]],
            device=inputs_embeds.device,
        ).long()

        # TODO: @colin optimize via maskedcat or indexing-mask
        for i in range(batch_size):
            start, end = 0, input_lens[i]
            h[i, :end, :] = inputs_embeds[i, :end, :]

            start, end = end, end + 1
            h[i, start:end, :] = prompt_sos_embeds[i, :, :]

            start, end = end, end + prompt_lens[i]
            h[i, start:end, :] = prompt_embeds[i, : prompt_lens[i]]

            start, end = end, end + 1
            h[i, start:end, :] = sos_embeds[i, :, :]
            try:
                start, end = end, end + target_lens[i]
                h[i, start:end, :] = target_embeds[i, : target_lens[i], :]
                target_ids_[i, : target_lens[i]] = target_ids[i, : target_lens[i]]
                target_ids_[i, target_lens[i] : target_lens[i] + 1] = eos_ids[i, :]
            except:
                breakpoint()
                logger.warning(">>> mismatch length")

        model_inputs = {"inputs_embeds": h}  # [b, 1+t_p + 1 + t_u]
        # target_ids = torch.cat([target_ids, eos_ids], dim=1)
        target_ids = target_ids_  # [b, t_u + 1]
        target_lens = target_lens + 1

        if return_all:
            return model_inputs, target_ids, inputs_embeds, sos_embeds, target_embeds
        else:
            return (
                model_inputs,
                target_ids,
                input_lens - 1,
                target_lens - 1,
                prompt_lens,
            )

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            phone_embeds = self.input_embedders["lyrics_phones"].embed(
                self.requires, batch["phones"].to(self.device), with_sos=with_sos
            )
            tone_embeds = self.input_embedders["lyrics_tones"].embed(
                self.requires, batch["tones"].to(self.device), with_sos=with_sos
            )
            wordseg_embeds = self.input_embedders["lyrics_wordsegs"].embed(
                self.requires, batch["wordsegs"].to(self.device), with_sos=with_sos
            )
            embeds = self.input_embedders["lyrics_joints"](
                torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1)
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )
        logger.info(f"target_embeds: {target_embeds}")
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


## Merge 版本
class SemanticModule_Valle_V2(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        spkenc_model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules={},
        checkpointing=False,
        extra_params=None,
    ):

        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "lyrics_phones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_wordsegs": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_joints": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        else:
            raise NotImplementedError

        logger.info(f"lang_embeddings size: [256, {hidden_size}]")
        lang_embeddings = nn.Embedding(256, hidden_size)

        self.spkenc_croplen = extra_params["spkenc_croplen"]
        self.spkenc_minlen = extra_params["spkenc_minlen"]

        super().__init__(
            model_cls=model_cls,
            spkenc_model_cls=spkenc_model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            prompt_embedder=prompt_embedder,
            lang_embeddings=lang_embeddings,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch, update_mfu=True)
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_dict(
                {"training/loss": loss, "training/accu": accu},
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(
            f"bigtts.umm.ar.prepare_training_inputs{self.trainer.global_step}"
        ):
            (
                input_ids,
                target_ids,
                input_lens,
                target_lens,
                prompt_lens,
            ) = self.prepare_training_inputs(batch)
            # input_ids['inputs_embeds']: [b, t = 1+tp + 1 + t_u, c]
            # target_ids: [b, t_u+1]
            # input_lens: token lengths, max = t_p
            # target_lens: target length, max = t_u

            seq_lens = input_lens + 1 + target_lens + 1
            loss_mask = sequence_mask(
                seq_lens,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            text_query_loss_mask = sequence_mask(
                input_lens + 1,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )

            if True:  # do not cal text_loss
                loss_mask = loss_mask - text_query_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t * valid_token_ratio},
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    logs = {
                        "batch": b,
                        "token_num": b * t,
                        "valid_token_ratio": valid_token_ratio,
                    }

                    logs.update(self.metric.compute(self.trainer.global_step))
                    self.log_dict(logs, prog_bar=True, sync_dist=True)

        with self.profiler.profile(
            f"bigtts.umm.ar.forward.step-{self.trainer.global_step}"
        ):
            logits = self.model(**input_ids)  # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):  # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        with self.profiler.profile(
            f"bigtts.umm.ar.post.step-{self.trainer.global_step}"
        ):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)  # [b, 1]
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                h[i, : input_lens[i] + 1 + target_lens[i] + 1] = torch.cat(
                    (
                        batch["lyrics_tokens"][i, : input_lens[i]],
                        sos_ids[i, :],
                        target_ids[i, : target_lens[i] + 1],
                    )
                )

            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = (
                ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum()
                / loss_mask.sum()
                * 100
            )

        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(
                self.requires, batch["target_audio"], with_sos=False, with_eos=False
            )  # [b, 1, t_w] --> [b, t_u]
            target_lens = (
                (
                    batch["audio_lengths"]
                    / (
                        self.extra_params["sample_rate"]
                        / self.extra_params["semantic_frame_rate"]
                    )
                )
                .ceil()
                .long()
            )  # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch["target_ids"]
            target_lens = batch["target_ids_length"]

        assert "prompt_ids" in batch
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_ids_length"]

        land_ids = batch["lang"]
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(
            batch
        )  # [b, t_p] --> [b, 1+t_p, h]
        inputs_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        input_lens = 1 + batch["lyrics_token_length"]

        sos_embeds = self.target_embedder.get_sos_embed(batch_size)  # [b, 1, h]
        target_embeds = self.target_embedder.embed(
            token_ids=target_ids, with_sos=False, with_eos=False
        )  # [b, t_u] --> [b, t_u, h]
        sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        # attention_mask
        # prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size)
        prompt_embeds = self.prompt_embedder.embed(
            token_ids=prompt_ids, with_sos=False, with_eos=False
        )

        map_list = []
        spkemb_input = []
        spkemb_index = 0

        for i in range(batch_size):
            start_pos = 0
            emb_index_list = []
            # import pdb;pdb.set_trace()
            while start_pos <= self.spkenc_minlen:
                end_pos = start_pos + self.spkenc_croplen
                if end_pos > prompt_lens[i]:
                    end_pos = prompt_lens[i]
                if end_pos - start_pos <= self.spkenc_minlen:
                    if len(emb_index_list) == 0:
                        emb_index_list.append(None)
                    break
                elif (
                    end_pos - start_pos < self.spkenc_croplen
                ):  # 不足的部分，用之前的补齐。 若超出则循环补齐
                    # 如果总长度够大，start_pos 往左延伸一些即可，即有一定overlap
                    copy_time = self.spkenc_croplen // (end_pos - start_pos) + 1
                    repeat_emb = prompt_embeds[i].repeat(copy_time, 1)
                    spkemb_input.append(repeat_emb[-self.spkenc_croplen :, :])
                    emb_index_list.append(spkemb_index)
                    spkemb_index += 1
                    break
                else:
                    spkemb_input.append(prompt_embeds[i, start_pos:end_pos])
                    emb_index_list.append(spkemb_index)
                    spkemb_index += 1
                start_pos = end_pos
            map_list.append(emb_index_list)  # 第i个embedding从spkemb_output的哪些index里去取
            assert len(map_list) == (i + 1)

        if len(spkemb_input) > 0:
            spkemb_input = torch.stack(spkemb_input, dim=0)
            spkemb_output = self.spkenc_model(spkemb_input)

        zero_emb = torch.zeros([self.extra_params["hidden_size"]]).to(
            target_embeds.device
        )
        # recover 回去，同个speaker的不同句求平均：
        prompt_embeds = []
        for i, emb_index_list in enumerate(map_list):
            fm = len(emb_index_list)
            assert fm > 0
            if None in emb_index_list:
                avg_emb = zero_emb
            else:
                emb_sum = None
                for j in emb_index_list:
                    if emb_sum is None:
                        emb_sum = spkemb_output[j]
                    else:
                        emb_sum += spkemb_output[j]
                avg_emb = emb_sum / fm
            prompt_embeds.append(avg_emb)

        prompt_embeds = torch.stack(prompt_embeds, dim=0).unsqueeze(1)

        # if prompt_ids.shape[1] < self.spkenc_down_rate:
        #     # 16 随便填的
        #     prompt_embeds = torch.zeros([prompt_embeds.shape[0], 16, prompt_embeds.shape[2]]).to(prompt_embeds.device)
        # else:
        #     # prompt_mask = sequence_mask(prompt_lens, max_len=prompt_ids.shape[1], device=prompt_ids.device)
        #     # prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds, attention_mask=prompt_mask)
        #     # prompt_embeds = self.spkenc_avgpool(prompt_embeds.transpose(1, 2)).transpose(1, 2)
        #     prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds)
        #     prompt_embeds = prompt_embeds[:, self.spkenc_down_rate-1::self.spkenc_down_rate, :].clone()
        # prompt_lens = prompt_lens // self.spkenc_down_rate

        # prompt_sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:  # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)  # [b, 1]
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(
                target_ids.device
            )
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
            prompt_embeds = prompt_embeds[:, :-1, :]  # 保持和target一样长度，用同一套target len

        if self.use_cross_attn:  # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds,
            }, torch.cat([target_ids, eos_ids], dim=1)

        max_len = torch.max(input_lens + target_lens + 2).item()
        # h = torch.zeros([batch_size, inputs_embeds.shape[1] + sos_embeds.shape[1] + prompt_embeds.shape[1] + sos_embeds.shape[1] + target_embeds.shape[1], inputs_embeds.shape[-1]], device=inputs_embeds.device)
        h = torch.zeros(
            [batch_size, max_len, inputs_embeds.shape[-1]], device=inputs_embeds.device
        )
        target_ids_ = torch.zeros(
            [batch_size, target_ids.shape[1] + eos_ids.shape[1]],
            device=inputs_embeds.device,
        ).long()

        for i in range(batch_size):
            h[i, : input_lens[i], :] = inputs_embeds[i, : input_lens[i], :]
            h[i, input_lens[i] : input_lens[i] + 1, :] = sos_embeds[i, :, :]

            try:
                h[
                    i, input_lens[i] + 1 : input_lens[i] + 1 + target_lens[i], :
                ] = target_embeds[i, : target_lens[i], :]
                target_ids_[i, : target_lens[i]] = target_ids[i, : target_lens[i]]
                target_ids_[i, target_lens[i] : target_lens[i] + 1] = eos_ids[i, :]
            except:
                logger.warning(">>> mismatch length")

        h += prompt_embeds

        model_inputs = {"inputs_embeds": h}  # [b, 1+t_p + 1 + t_u]
        target_ids = target_ids_  # [b, t_u + 1]
        target_lens = target_lens + 1

        if return_all:
            return model_inputs, target_ids, inputs_embeds, sos_embeds, target_embeds
        else:
            return (
                model_inputs,
                target_ids,
                input_lens - 1,
                target_lens - 1,
                prompt_lens,
            )

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            phone_embeds = self.input_embedders["lyrics_phones"].embed(
                self.requires, batch["phones"].to(self.device), with_sos=with_sos
            )
            tone_embeds = self.input_embedders["lyrics_tones"].embed(
                self.requires, batch["tones"].to(self.device), with_sos=with_sos
            )
            wordseg_embeds = self.input_embedders["lyrics_wordsegs"].embed(
                self.requires, batch["wordsegs"].to(self.device), with_sos=with_sos
            )
            embeds = self.input_embedders["lyrics_joints"](
                torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1)
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        sample_mode = hp.sample_mode

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )
        logger.info(f"target_embeds: {target_embeds}")

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


class SemanticModule_SpkidLLMV3_3(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        spkenc_model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules={},
        checkpointing=False,
        extra_params=None,
    ):

        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            "lyrics_phones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_wordsegs": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_joints": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        else:
            raise NotImplementedError

        logger.info(f"lang_embeddings size: [256, {hidden_size}]")
        lang_embeddings = nn.Embedding(256, hidden_size)

        self.spkenc_croplen = extra_params["spkenc_croplen"]
        self.spkenc_minlen = extra_params["spkenc_minlen"]

        super().__init__(
            model_cls=model_cls,
            spkenc_model_cls=spkenc_model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            prompt_embedder=prompt_embedder,
            lang_embeddings=lang_embeddings,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        # reorder model for correct DDP bucket order
        reorder_attr(
            self,
            [
                "target_embedder",
                "input_embedders",
                "lang_embeddings",
                "prompt_embedder",
                "spkenc_model",
                "model",
                "criterion",
            ],
        )

        if extra_params.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch, update_mfu=True)
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_dict(
                {"training/loss": loss, "training/accu": accu},
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(
            f"bigtts.umm.ar.prepare_training_inputs{self.trainer.global_step}"
        ):
            (
                input_ids,
                target_ids,
                input_lens,
                target_lens,
                prompt_lens,
                spkenc_flops_kwargs,
            ) = self.prepare_training_inputs(batch)
            # input_ids['inputs_embeds']: [b, t = 1+tp + 1 + t_u, c]
            # target_ids: [b, t_u+1]
            # input_lens: token lengths, max = t_p
            # target_lens: target length, max = t_u

            seq_lens = input_lens + 1 + 1 + target_lens + 1
            loss_mask = sequence_mask(
                seq_lens,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            text_query_loss_mask = sequence_mask(
                input_lens + 1 + 1,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )

            if True:  # do not cal text_loss
                loss_mask = loss_mask - text_query_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={
                        "model": {"batch_size": b, "seq_len": t * valid_token_ratio},
                        "spkenc": spkenc_flops_kwargs,
                    },
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    logs = {
                        "batch": b,
                        "token_num": b * t,
                        "seq_len": t,
                        "valid_token_ratio": valid_token_ratio,
                    }
                    logs.update(self.metric.compute(self.trainer.global_step))
                    self.log_dict(logs, prog_bar=True, sync_dist=False)

        with self.profiler.profile(
            f"bigtts.umm.ar.forward.step-{self.trainer.global_step}"
        ):
            logits = self.model(**input_ids)  # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):  # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        with self.profiler.profile(
            f"bigtts.umm.ar.post.step-{self.trainer.global_step}"
        ):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)  # [b, 1]
            # prompt_sos_ids = self.prompt_embedder.get_sos_token(bsz)
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                h[i, : input_lens[i] + 1 + 1 + target_lens[i] + 1] = torch.cat(
                    (
                        torch.zeros([1]).to(sos_ids.device),  # placeholder
                        batch["lyrics_tokens"][i, : input_lens[i]],
                        # torch.zeros([prompt_lens[i]]).to(sos_ids.device), # query placeholder
                        sos_ids[i, :],
                        target_ids[i, : target_lens[i] + 1],
                    )
                )

            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = (
                ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum()
                / loss_mask.sum()
                * 100
            )
        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(
                self.requires, batch["target_audio"], with_sos=False, with_eos=False
            )  # [b, 1, t_w] --> [b, t_u]
            target_lens = (
                (
                    batch["audio_lengths"]
                    / (
                        self.extra_params["sample_rate"]
                        / self.extra_params["semantic_frame_rate"]
                    )
                )
                .ceil()
                .long()
            )  # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch["target_ids"]
            target_lens = batch["target_ids_length"]

        assert "prompt_ids" in batch
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_ids_length"]

        land_ids = batch["lang"]
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(
            batch
        )  # [b, t_p] --> [b, 1+t_p, h]
        inputs_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        input_lens = 1 + batch["lyrics_token_length"]

        sos_embeds = self.target_embedder.get_sos_embed(batch_size)  # [b, 1, h]
        target_embeds = self.target_embedder.embed(
            token_ids=target_ids, with_sos=False, with_eos=False
        )  # [b, t_u] --> [b, t_u, h]
        sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        # attention_mask
        # prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size)
        prompt_embeds = self.prompt_embedder.embed(
            token_ids=prompt_ids, with_sos=False, with_eos=False
        )
        # prompt_embeds: B x T x emd_dim

        with self.profiler.profile(
            f"bigtts.umm.ar.get_split_emb{self.trainer.global_step}"
        ):
            spkemb_input, scatter_index = get_split_emb(
                prompt_embeds, prompt_lens, self.spkenc_croplen, self.spkenc_minlen
            )

        with self.profiler.profile(
            f"bigtts.umm.ar.spkenc_model{self.trainer.global_step}"
        ):
            if spkemb_input is not None:
                spkemb_output = self.spkenc_model(spkemb_input)
            else:
                spkemb_output = None

        with self.profiler.profile(
            f"bigtts.umm.ar.sactter_reduce{self.trainer.global_step}"
        ):
            if spkemb_output is not None:
                out_prompt = torch.zeros(
                    batch_size, spkemb_output.size(-1), device=spkemb_output.device
                )
                scatter_index = scatter_index[:, 0][:, None].repeat(
                    1, spkemb_output.size(-1)
                )
                out_prompt = torch.scatter_reduce(
                    out_prompt,
                    0,
                    scatter_index,
                    spkemb_output.float(),
                    reduce="mean",
                    include_self=False,
                )
                out_prompt = out_prompt.unsqueeze(1)
            else:
                zero_emb = torch.zeros([self.extra_params["hidden_size"]]).to(
                    target_embeds.device
                )
                out_prompt = torch.stack([zero_emb] * batch_size, dim=0).unsqueeze(1)
            prompt_embeds = out_prompt

        spkenc_flops_kwargs = {"b": 1, "t": 1}

        prompt_embeds = prompt_embeds + self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:  # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)  # [b, 1]
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(
                target_ids.device
            )
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
            prompt_embeds = prompt_embeds[:, :-1, :]  # 保持和target一样长度，用同一套target len

        if self.use_cross_attn:  # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds,
            }, torch.cat([target_ids, eos_ids], dim=1)

        max_len = torch.max(input_lens + target_lens + 2).item()
        eos_len = torch.ones(
            batch_size, dtype=input_lens.dtype, device=input_lens.device
        )
        # customized op
        h = masked_cat3d(
            inputs_embeds, sos_embeds, target_embeds, input_lens, eos_len, target_lens
        )
        target_ids_ = masked_cat2d(target_ids, eos_ids, target_lens, eos_len)
        h = torch.cat([prompt_embeds, h], dim=1)

        model_inputs = {"inputs_embeds": h}  # [b, 1+t_p + 1 + t_u]
        target_ids = target_ids_  # [b, t_u + 1]
        target_lens = target_lens + 1

        if return_all:
            return (
                model_inputs,
                target_ids,
                inputs_embeds,
                sos_embeds,
                target_embeds,
                spkenc_flops_kwargs,
            )
        else:
            return (
                model_inputs,
                target_ids,
                input_lens - 1,
                target_lens - 1,
                prompt_lens,
                spkenc_flops_kwargs,
            )

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            phone_embeds = self.input_embedders["lyrics_phones"].embed(
                self.requires, batch["phones"].to(self.device), with_sos=with_sos
            )
            tone_embeds = self.input_embedders["lyrics_tones"].embed(
                self.requires, batch["tones"].to(self.device), with_sos=with_sos
            )
            wordseg_embeds = self.input_embedders["lyrics_wordsegs"].embed(
                self.requires, batch["wordsegs"].to(self.device), with_sos=with_sos
            )
            embeds = self.input_embedders["lyrics_joints"](
                torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1)
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        thresh = hp.semantic_thresh
        sample_mode = hp.sample_mode
        step_out_blank = hp.step_out_blank
        max_blank_length = hp.max_blank_length
        icl_mode = hp.icl_mode  # [continuation, non-continuation]

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )
        prompt_embeds = self.prompt_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )

        src_land_ids = batch["src_lang"].to(self.device)
        src_lang_embed = self.lang_embeddings(src_land_ids).unsqueeze(1)
        tgt_land_ids = batch["tgt_lang"].to(self.device)
        tgt_lang_embed = self.lang_embeddings(tgt_land_ids).unsqueeze(1)

        prompt_text_lens = (
            batch["prompt_text_lens"][0] + 1
        )  # Adding 1 is because when embedding input, one frame will be added at the beginning of the sequence.

        return self.icl_predict(
            inputs_embeds,
            target_embeds,
            prompt_embeds,
            prompt_text_lens,
            num_tokens,
            src_lang_embed,
            tgt_lang_embed,
            sample_mode=sample_mode,
            temperature=temperature,
            thresh=thresh,
            beam=beam,
            ref_samples=ref_samples,
            step_out_blank=step_out_blank,
            max_blank_length=max_blank_length,
            icl_mode=icl_mode,
        )

    @torch.no_grad()
    def icl_predict(
        self,
        inputs_embeds,
        target_embeds,
        prompt_embeds,
        prompt_text_lens,
        num_tokens,
        src_lang_embed=None,
        tgt_lang_embed=None,
        temperature=1,
        thresh=1,
        sample_mode="gumbel",
        tqdm_name=None,
        beam=1,
        ref_samples=None,
        rl_training=False,
        step_out_blank=False,
        max_blank_length=5,
        icl_mode="continuation",
    ):
        """
        Input:
            beam: inference beam
            [optional] ref_samples: (batch_size, seq_len)
                If specified, add these samples to the beam. Beam size during generation
                will be reduced by 1 so the final returned shape stays unchanged.

        Return a tuple of:
            output_tokens: (batch_size * beam, seq_len)
            [if beam > 1]
                inputs_embeds: (batch_size * beam, seq_len, dim)
                sos_embeds: (batch_size * beam, 1, dim)
        """
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        batch_size, seq_len, _ = inputs_embeds.size()
        if ref_samples is not None:
            assert ref_samples.size(0) == batch_size
            assert ref_samples.size(1) == num_tokens
            assert beam > 1, "Can't use beam size 1 with ref_samples!"
            beam = beam - 1
        # (b, s, d) --> (b * beam, s, d)
        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(
            batch_size * beam, seq_len, -1
        )
        sos_embeds = self.target_embedder.get_sos_embed(batch_size * beam)
        # prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size * beam)

        # prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds)
        # prompt_embeds = prompt_embeds[:, self.spkenc_down_rate-1::self.spkenc_down_rate, :].clone()
        # spkenc_out = prompt_embeds

        spkemb_input = []
        assert prompt_embeds.shape[0] == 1
        prompt_len = prompt_embeds.shape[1]

        start_pos = 0
        while start_pos < prompt_len:
            end_pos = start_pos + self.spkenc_croplen
            if end_pos > prompt_len:
                end_pos = prompt_len
            if end_pos - start_pos <= self.spkenc_minlen:
                break
            elif end_pos - start_pos < self.spkenc_croplen:  # 不足的部分，用之前的补齐。 若超出则循环补齐
                # 如果总长度够大，start_pos 往左延伸一些即可，即有一定overlap
                if prompt_len < self.spkenc_croplen:
                    copy_time = self.spkenc_croplen // prompt_len + 1
                    repeat_emb = prompt_embeds[0].repeat(copy_time, 1)
                    spkemb_input.append(repeat_emb[-self.spkenc_croplen :, :])
                else:
                    spkemb_input.append(prompt_embeds[0, -self.spkenc_croplen :, :])
                break
            else:
                spkemb_input.append(prompt_embeds[0, start_pos:end_pos])
            start_pos = end_pos
        if len(spkemb_input) > 0:
            spkemb_input = torch.stack(spkemb_input, dim=0)
            spkemb_output = self.spkenc_model(spkemb_input)

        fm = spkemb_output.shape[0]
        assert fm > 0

        emb_sum = None
        for j in range(fm):
            if emb_sum is None:
                emb_sum = spkemb_output[j]
            else:
                emb_sum += spkemb_output[j]
        avg_emb = emb_sum / fm

        spkenc_out = avg_emb.unsqueeze(0).unsqueeze(1)

        ## 3.3 与 3.5 的区别
        spkenc_out += src_lang_embed

        ## 非续写, given [target_text, sos], predict [target_token]
        if icl_mode == "non-continuation":
            inputs_embeds += tgt_lang_embed
            sos_embeds += tgt_lang_embed

            def _init_model_input():
                return {
                    "inputs_embeds": torch.cat(
                        [spkenc_out, inputs_embeds, sos_embeds], dim=1
                    )
                }

        ## 续写, given [prompt_text, target_text, sos, prompt_token], predict [target_token]
        elif icl_mode == "continuation":
            assert prompt_text_lens is not None
            inputs_embeds[:, :prompt_text_lens] += src_lang_embed
            inputs_embeds[:, prompt_text_lens:] += tgt_lang_embed
            sos_embeds += tgt_lang_embed
            target_embeds += src_lang_embed

            # 第一帧置 0
            # spkenc_out = spkenc_out.new_zeros(spkenc_out.shape) + src_lang_embed

            def _init_model_input():
                return {
                    "inputs_embeds": torch.cat(
                        [spkenc_out, inputs_embeds, sos_embeds, target_embeds], dim=1
                    )
                }

        else:
            raise NotImplementedError(f"icl_mode={icl_mode} is not supported!")

        model_input = _init_model_input()

        if rl_training:
            rl_model_input = _init_model_input()

        output_tokens = None
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        if not isinstance(self.model, gpt.GPTLMHeadModel):
            raise NotImplementedError("Only support GPT(ctiga) model now.")

        infer_params = InferenceParams(
            max_sequence_len=4096, max_batch_size=batch_size, fused_ft_kernel=False
        )
        position_ids = None
        token_buffer = TokenBuffer(
            max_length=max_blank_length,
            init_temperature=temperature,
            max_temperature=1.5,
            max_retry_times=5,
        )
        for i in pbar:
            pbar.set_description(
                f"{tqdm_name} [0 - {num_tokens}], mode: {sample_mode}, temp: {temperature}, icl: {icl_mode}"
            )
            input_embeds = model_input["inputs_embeds"]
            logits = self.model(
                inputs_embeds=input_embeds.half(),
                inference_params=infer_params,
                position_ids=position_ids,
                last_token_only=True,
            ).logits  # b,1,num_logits

            if i < 10:
                logits = logits[..., :-2]

            predict_token = self.sample_logits(
                logits.float(), temperature, thresh, sample_mode
            )
            predict_token = predict_token[:, None]
            if predict_token[0][0] == self.target_embedder.eos_id:
                break

            if step_out_blank:
                token_buffer.put(predict_token.item())  # v2: temperature 不断增加
                new_predict_token = token_buffer.process_duplicate(
                    i,
                    self.sample_logits,
                    logits=logits,
                    thresh=thresh,
                    mode=sample_mode,
                )
                if new_predict_token is not None:
                    predict_token = new_predict_token

            predict_token_emb = self.target_embedder.embedder(predict_token)
            predict_token_emb += tgt_lang_embed
            model_input["inputs_embeds"] = predict_token_emb
            if rl_training and i < num_tokens - 1:
                rl_model_input["inputs_embeds"] = torch.cat(
                    [rl_model_input["inputs_embeds"], predict_token_emb], dim=1
                )

            if i == 0:
                infer_params.sequence_len_offset = input_embeds.shape[-2]
            else:
                infer_params.sequence_len_offset += 1
            position_ids = torch.full(
                (batch_size, 1),
                infer_params.sequence_len_offset,
                dtype=torch.long,
                device=predict_token.device,
            )
            output_tokens = (
                torch.cat([output_tokens, predict_token], dim=1)
                if output_tokens is not None
                else predict_token
            )

        # Add ref_samples to generation beam
        if ref_samples is not None:
            # (b * beam, s, d) -> (b, s, d) -> (b * (beam + 1), s, d)
            inputs_embeds = (
                inputs_embeds.reshape(batch_size, beam, seq_len, -1)[:, 0, :, :]
                .repeat(1, beam + 1, 1)
                .reshape(batch_size * (beam + 1), seq_len, -1)
            )
            sos_embeds = self.target_embedder.get_sos_embed(batch_size * (beam + 1))
            # (b * beam, s) -> (b * (beam + 1), s)
            output_tokens = torch.cat(
                [ref_samples.unsqueeze(1), output_tokens.reshape(batch_size, beam, -1)],
                dim=1,
            ).reshape(batch_size * (beam + 1), -1)
        if rl_training:
            return output_tokens, rl_model_input
        else:
            return output_tokens

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)


class SemanticModule_SpkidLLMV3_5(BaseContinuousEmbedModule):
    def __init__(
        self,
        model_cls,
        spkenc_model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules={},
        checkpointing=False,
        extra_params=None,
    ):

        hidden_size = extra_params["hidden_size"]
        lyrics_vocab_size = extra_params["lyrics_codebook_size"]
        semantic_codebook_size = extra_params["semantic_codebook_size"]
        embedder_dict = {
            # 'mulan': MulanTagEmbedder(input_dim=mulan_embed_dim, embedding_dim=hidden_size, add_sos=True),
            # 'lyrics_tokens': LyricsTokenEmbedder(vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True),
            "lyrics_phones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size, embedding_dim=hidden_size, add_sos=True
            ),
            "lyrics_tones": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_wordsegs": LyricsTokenEmbedder(
                vocab_size=lyrics_vocab_size,
                embedding_dim=hidden_size // 2,
                add_sos=True,
            ),
            "lyrics_joints": torch.nn.Linear(2 * hidden_size, hidden_size, bias=False),
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            target_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        elif semantic_type == "bestrq":
            target_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=True,
                add_eos=True,
            )
            prompt_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=extra_params["spkenc_hidden_size"],
                add_sos=True,
                add_eos=True,
            )
        else:
            raise NotImplementedError

        logger.info(f"lang_embeddings size: [256, {hidden_size}]")
        lang_embeddings = nn.Embedding(256, hidden_size)

        self.spkenc_croplen = extra_params["spkenc_croplen"]
        self.spkenc_minlen = extra_params["spkenc_minlen"]

        super().__init__(
            model_cls=model_cls,
            spkenc_model_cls=spkenc_model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            input_embedders=input_embedders,
            target_embedder=target_embedder,
            prompt_embedder=prompt_embedder,
            lang_embeddings=lang_embeddings,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch, update_mfu=True)
        self.log_dict(
            {"training/loss": loss, "accu": accu}, prog_bar=True, sync_dist=True
        )
        return loss

    def _shared_step(self, batch, update_mfu=False):
        with self.profiler.profile(
            f"bigtts.umm.ar.prepare_training_inputs{self.trainer.global_step}"
        ):
            (
                input_ids,
                target_ids,
                input_lens,
                target_lens,
                prompt_lens,
            ) = self.prepare_training_inputs(batch)
            # input_ids['inputs_embeds']: [b, t = 1+tp + 1 + t_u, c]
            # target_ids: [b, t_u+1]
            # input_lens: token lengths, max = t_p
            # target_lens: target length, max = t_u

            seq_lens = input_lens + 1 + target_lens + 1
            loss_mask = sequence_mask(
                seq_lens,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )
            text_query_loss_mask = sequence_mask(
                input_lens + 1,
                max_len=input_ids["inputs_embeds"].shape[1],
                device=input_ids["inputs_embeds"].device,
            )

            if True:  # do not cal text_loss
                loss_mask = loss_mask - text_query_loss_mask

        if update_mfu:
            if "inputs_embeds" in input_ids:
                b, t, _ = input_ids["inputs_embeds"].shape
                valid_token_ratio = 1.0
                self.metric.update(
                    num_tokens=b * t * valid_token_ratio,
                    stage=self.trainer.state.stage,
                    model_kwargs={"batch_size": b, "seq_len": t * valid_token_ratio},
                )
                if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
                    self.log_dict(
                        self.metric.compute(self.trainer.global_step),
                        prog_bar=True,
                        sync_dist=True,
                    )
                    self.log_dict(
                        {
                            "batch": b,
                            "token_num": b * t,
                            "valid_token_ratio": valid_token_ratio,
                        },
                        prog_bar=True,
                        sync_dist=True,
                    )

        with self.profiler.profile(
            f"bigtts.umm.ar.forward.step-{self.trainer.global_step}"
        ):
            logits = self.model(**input_ids)  # [b, t, h] --> [b, t, c]
        if isinstance(logits, dict):  # true
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]

        # x = logits[:, -target_ids.size(1):, :]  # [b, t, c]
        # loss = self.criterion(x, target_ids)
        # accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100

        with self.profiler.profile(
            f"bigtts.umm.ar.post.step-{self.trainer.global_step}"
        ):
            x = logits
            bsz, t, c = logits.shape
            sos_ids = self.target_embedder.get_sos_token(bsz)  # [b, 1]
            h = torch.zeros([bsz, t], device=logits.device).long()
            for i in range(bsz):
                h[i, : input_lens[i] + 1 + target_lens[i] + 1] = torch.cat(
                    (
                        batch["lyrics_tokens"][i, : input_lens[i]],
                        # torch.zeros([prompt_lens[i]]).to(sos_ids.device), # query placeholder
                        sos_ids[i, :],
                        target_ids[i, : target_lens[i] + 1],
                    )
                )

            target_ids = h
            loss = self.criterion(x, target_ids, mask=loss_mask)
            accu = (
                ((x.argmax(dim=-1) == target_ids).float() * loss_mask).sum()
                / loss_mask.sum()
                * 100
            )

        return loss, accu

    def prepare_training_inputs(self, batch, return_all=False):
        assert "target_audio" in batch or "target_ids" in batch
        if "target_audio" in batch:
            target_ids = self.target_embedder.tokenize(
                self.requires, batch["target_audio"], with_sos=False, with_eos=False
            )  # [b, 1, t_w] --> [b, t_u]
            target_lens = (
                (
                    batch["audio_lengths"]
                    / (
                        self.extra_params["sample_rate"]
                        / self.extra_params["semantic_frame_rate"]
                    )
                )
                .ceil()
                .long()
            )  # semantic_len = ceil(audio_len / hop_size)
            target_lens = torch.clamp(target_lens, max=target_ids.shape[1])
        else:
            target_ids = batch["target_ids"]
            target_lens = batch["target_ids_length"]

        assert "prompt_ids" in batch
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_ids_length"]

        # import pdb;pdb.set_trace()

        land_ids = batch["lang"]
        batch_size = target_ids.size(0)
        inputs_embeds = self.prepare_inputs_embeddings(
            batch
        )  # [b, t_p] --> [b, 1+t_p, h]
        inputs_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        input_lens = 1 + batch["lyrics_token_length"]

        sos_embeds = self.target_embedder.get_sos_embed(batch_size)  # [b, 1, h]
        target_embeds = self.target_embedder.embed(
            token_ids=target_ids, with_sos=False, with_eos=False
        )  # [b, t_u] --> [b, t_u, h]
        sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)
        target_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        # attention_mask
        # prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size)
        prompt_embeds = self.prompt_embedder.embed(
            token_ids=prompt_ids, with_sos=False, with_eos=False
        )

        map_list = []
        spkemb_input = []
        spkemb_index = 0

        for i in range(batch_size):
            start_pos = 0
            emb_index_list = []
            # import pdb;pdb.set_trace()
            while start_pos <= self.spkenc_minlen:
                end_pos = start_pos + self.spkenc_croplen
                if end_pos > prompt_lens[i]:
                    end_pos = prompt_lens[i]
                if end_pos - start_pos <= self.spkenc_minlen:
                    if len(emb_index_list) == 0:
                        emb_index_list.append(None)
                    break
                elif (
                    end_pos - start_pos < self.spkenc_croplen
                ):  # 不足的部分，用之前的补齐。 若超出则循环补齐
                    # 如果总长度够大，start_pos 往左延伸一些即可，即有一定overlap
                    copy_time = self.spkenc_croplen // (end_pos - start_pos) + 1
                    repeat_emb = prompt_embeds[i].repeat(copy_time, 1)
                    spkemb_input.append(repeat_emb[-self.spkenc_croplen :, :])
                    emb_index_list.append(spkemb_index)
                    spkemb_index += 1
                    break
                else:
                    spkemb_input.append(prompt_embeds[i, start_pos:end_pos])
                    emb_index_list.append(spkemb_index)
                    spkemb_index += 1
                start_pos = end_pos
            map_list.append(emb_index_list)  # 第i个embedding从spkemb_output的哪些index里去取
            assert len(map_list) == (i + 1)

        if len(spkemb_input) > 0:
            spkemb_input = torch.stack(spkemb_input, dim=0)  # [b_new, 150=6s, h_new]
            spkemb_output = self.spkenc_model(spkemb_input)  # [b_new, h]

        zero_emb = torch.zeros([self.extra_params["hidden_size"]]).to(
            target_embeds.device
        )
        # recover 回去，同个speaker的不同句求平均：
        prompt_embeds = []
        for i, emb_index_list in enumerate(map_list):
            fm = len(emb_index_list)
            assert fm > 0
            if None in emb_index_list:
                avg_emb = zero_emb
            else:
                emb_sum = None
                for j in emb_index_list:
                    if emb_sum is None:
                        emb_sum = spkemb_output[j]
                    else:
                        emb_sum += spkemb_output[j]
                avg_emb = emb_sum / fm
            prompt_embeds.append(avg_emb)

        prompt_embeds = torch.stack(prompt_embeds, dim=0).unsqueeze(1)  # [b, h]

        # if prompt_ids.shape[1] < self.spkenc_down_rate:
        #     # 16 随便填的
        #     prompt_embeds = torch.zeros([prompt_embeds.shape[0], 16, prompt_embeds.shape[2]]).to(prompt_embeds.device)
        # else:
        #     # prompt_mask = sequence_mask(prompt_lens, max_len=prompt_ids.shape[1], device=prompt_ids.device)
        #     # prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds, attention_mask=prompt_mask)
        #     # prompt_embeds = self.spkenc_avgpool(prompt_embeds.transpose(1, 2)).transpose(1, 2)
        #     prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds)
        #     prompt_embeds = prompt_embeds[:, self.spkenc_down_rate-1::self.spkenc_down_rate, :].clone()
        # prompt_lens = prompt_lens // self.spkenc_down_rate

        # prompt_sos_embeds += self.lang_embeddings(land_ids).unsqueeze(1)

        if self.target_embedder.eos_id is not None:  # true
            eos_ids = self.target_embedder.get_eos_token(batch_size)  # [b, 1]
        else:
            eos_ids = torch.zeros((batch_size, 0), dtype=target_ids.dtype).to(
                target_ids.device
            )
            # targets must be offset by one if no eos id added
            target_embeds = target_embeds[:, :-1, :]
            prompt_embeds = prompt_embeds[:, :-1, :]  # 保持和target一样长度，用同一套target len

        if self.use_cross_attn:  # false
            return {
                "inputs_embeds": torch.cat([sos_embeds, target_embeds], dim=1),
                "encoder_hidden_states": inputs_embeds,
            }, torch.cat([target_ids, eos_ids], dim=1)

        max_len = torch.max(input_lens + target_lens + 2).item()
        # h = torch.zeros([batch_size, inputs_embeds.shape[1] + sos_embeds.shape[1] + prompt_embeds.shape[1] + sos_embeds.shape[1] + target_embeds.shape[1], inputs_embeds.shape[-1]], device=inputs_embeds.device)
        h = torch.zeros(
            [batch_size, max_len, inputs_embeds.shape[-1]], device=inputs_embeds.device
        )
        target_ids_ = torch.zeros(
            [batch_size, target_ids.shape[1] + eos_ids.shape[1]],
            device=inputs_embeds.device,
        ).long()

        for i in range(batch_size):
            h[i, : input_lens[i], :] = inputs_embeds[i, : input_lens[i], :]
            h[i, input_lens[i] : input_lens[i] + 1, :] = sos_embeds[i, :, :]

            try:
                h[
                    i, input_lens[i] + 1 : input_lens[i] + 1 + target_lens[i], :
                ] = target_embeds[i, : target_lens[i], :]
                target_ids_[i, : target_lens[i]] = target_ids[i, : target_lens[i]]
                target_ids_[i, target_lens[i] : target_lens[i] + 1] = eos_ids[i, :]
            except:
                logger.warning(">>> mismatch length")

        h += prompt_embeds

        model_inputs = {"inputs_embeds": h}  # [b, 1+t_p + 1 + t_u]
        target_ids = target_ids_  # [b, t_u + 1]
        target_lens = target_lens + 1

        if return_all:
            return model_inputs, target_ids, inputs_embeds, sos_embeds, target_embeds
        else:
            return (
                model_inputs,
                target_ids,
                input_lens - 1,
                target_lens - 1,
                prompt_lens,
            )

    def prepare_inputs_embeddings(self, batch):
        conditions = batch["conditions"].split(",")
        batch_size = self.infer_batch_size(batch)
        with_sos = True
        # convert inputs to conditions
        inputs_embeds = []
        if "lyrics_tokens" in conditions:
            # embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'].to(self.device), with_sos=with_sos)
            phone_embeds = self.input_embedders["lyrics_phones"].embed(
                self.requires, batch["phones"].to(self.device), with_sos=with_sos
            )
            tone_embeds = self.input_embedders["lyrics_tones"].embed(
                self.requires, batch["tones"].to(self.device), with_sos=with_sos
            )
            wordseg_embeds = self.input_embedders["lyrics_wordsegs"].embed(
                self.requires, batch["wordsegs"].to(self.device), with_sos=with_sos
            )
            embeds = self.input_embedders["lyrics_joints"](
                torch.cat([phone_embeds, tone_embeds, wordseg_embeds], dim=-1)
            )
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(
                self.input_embedders["lyrics_tokens"].get_sos_embed(batch_size)
            )
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def predict(self, batch, hp, beam=1, ref_samples=None):
        frame_rate = self.extra_params.semantic_frame_rate
        num_tokens = hp.duration * frame_rate
        temperature = hp.semantic_temperature
        thresh = hp.semantic_thresh
        sample_mode = hp.sample_mode
        step_out_blank = hp.step_out_blank
        max_blank_length = hp.max_blank_length
        icl_mode = hp.icl_mode  # [continuation, non-continuation]

        inputs_embeds = self.prepare_inputs_embeddings(batch)
        target_embeds = self.target_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )
        prompt_embeds = self.prompt_embedder.embed(
            self.requires,
            token_ids=batch["audio_prompt"].to(self.device),
            with_sos=False,
            with_eos=False,
        )

        src_land_ids = batch["src_lang"].to(self.device)
        src_lang_embed = self.lang_embeddings(src_land_ids).unsqueeze(1)
        tgt_land_ids = batch["tgt_lang"].to(self.device)
        tgt_lang_embed = self.lang_embeddings(tgt_land_ids).unsqueeze(1)

        prompt_text_lens = (
            batch["prompt_text_lens"][0] + 1
        )  # Adding 1 is because when embedding input, one frame will be added at the beginning of the sequence.

        return self.icl_predict(
            inputs_embeds,
            target_embeds,
            prompt_embeds,
            prompt_text_lens,
            num_tokens,
            src_lang_embed,
            tgt_lang_embed,
            sample_mode=sample_mode,
            temperature=temperature,
            thresh=thresh,
            beam=beam,
            ref_samples=ref_samples,
            step_out_blank=step_out_blank,
            max_blank_length=max_blank_length,
            icl_mode=icl_mode,
        )

    @torch.no_grad()
    def icl_predict(
        self,
        inputs_embeds,
        target_embeds,
        prompt_embeds,
        prompt_text_lens,
        num_tokens,
        src_lang_embed=None,
        tgt_lang_embed=None,
        temperature=1,
        thresh=1,
        sample_mode="gumbel",
        tqdm_name=None,
        beam=1,
        ref_samples=None,
        rl_training=False,
        step_out_blank=False,
        max_blank_length=5,
        icl_mode="continuation",
    ):
        """
        Input:
            beam: inference beam
            [optional] ref_samples: (batch_size, seq_len)
                If specified, add these samples to the beam. Beam size during generation
                will be reduced by 1 so the final returned shape stays unchanged.

        Return a tuple of:
            output_tokens: (batch_size * beam, seq_len)
            [if beam > 1]
                inputs_embeds: (batch_size * beam, seq_len, dim)
                sos_embeds: (batch_size * beam, 1, dim)
        """
        tqdm_name = self.__class__.__name__ if tqdm_name is None else tqdm_name
        batch_size, seq_len, _ = inputs_embeds.size()
        if ref_samples is not None:
            assert ref_samples.size(0) == batch_size
            assert ref_samples.size(1) == num_tokens
            assert beam > 1, "Can't use beam size 1 with ref_samples!"
            beam = beam - 1
        # (b, s, d) --> (b * beam, s, d)
        inputs_embeds = inputs_embeds.repeat(1, beam, 1).reshape(
            batch_size * beam, seq_len, -1
        )
        sos_embeds = self.target_embedder.get_sos_embed(batch_size * beam)
        # prompt_sos_embeds = self.prompt_embedder.get_sos_embed(batch_size * beam)

        # prompt_embeds = self.spkenc_model(inputs_embeds=prompt_embeds)
        # prompt_embeds = prompt_embeds[:, self.spkenc_down_rate-1::self.spkenc_down_rate, :].clone()
        # spkenc_out = prompt_embeds

        spkemb_input = []
        assert prompt_embeds.shape[0] == 1
        prompt_len = prompt_embeds.shape[1]

        start_pos = 0
        while start_pos <= self.spkenc_minlen:
            end_pos = start_pos + self.spkenc_croplen
            if end_pos > prompt_len:
                end_pos = prompt_len
            if end_pos - start_pos <= self.spkenc_minlen:
                break
            elif end_pos - start_pos < self.spkenc_croplen:  # 不足的部分，用之前的补齐。 若超出则循环补齐
                # 如果总长度够大，start_pos 往左延伸一些即可，即有一定overlap
                copy_time = self.spkenc_croplen // (end_pos - start_pos) + 1
                repeat_emb = prompt_embeds[0].repeat(copy_time, 1)
                spkemb_input.append(repeat_emb[-self.spkenc_croplen :, :])
                break
            else:
                spkemb_input.append(prompt_embeds[0, start_pos:end_pos])
            start_pos = end_pos

        if len(spkemb_input) > 0:
            # import pdb; pdb.set_trace()
            spkemb_input = torch.stack(spkemb_input, dim=0)
            spkemb_output = self.spkenc_model(spkemb_input)  # [1, 150, 1536]

        fm = spkemb_output.shape[0]
        assert fm > 0

        emb_sum = None
        for j in range(fm):
            if emb_sum is None:
                emb_sum = spkemb_output[j]
            else:
                emb_sum += spkemb_output[j]
        avg_emb = emb_sum / fm

        spkenc_out = avg_emb.unsqueeze(0).unsqueeze(1)

        ## 非续写, given [target_text, sos], predict [target_token]
        if icl_mode == "non-continuation":
            inputs_embeds += tgt_lang_embed
            sos_embeds += tgt_lang_embed

            def _init_model_input():
                return {
                    "inputs_embeds": torch.cat([inputs_embeds, sos_embeds], dim=1)
                    + spkenc_out
                }

        ## 续写, given [prompt_text, target_text, sos, prompt_token], predict [target_token]
        elif icl_mode == "continuation":
            assert prompt_text_lens is not None
            inputs_embeds[:, :prompt_text_lens] += src_lang_embed
            inputs_embeds[:, prompt_text_lens:] += tgt_lang_embed
            sos_embeds += src_lang_embed
            target_embeds += src_lang_embed

            def _init_model_input():
                return {
                    "inputs_embeds": torch.cat(
                        [inputs_embeds, sos_embeds, target_embeds], dim=1
                    )
                    + spkenc_out
                }

        else:
            raise NotImplementedError(f"icl_mode={icl_mode} is not supported!")

        model_input = _init_model_input()

        if rl_training:
            rl_model_input = _init_model_input()

        output_tokens = None
        past_key_values = None
        pbar = tqdm(range(num_tokens))
        if not isinstance(self.model, gpt.GPTLMHeadModel):
            raise NotImplementedError("Only support GPT(ctiga) model now.")

        infer_params = InferenceParams(
            max_sequence_len=4096, max_batch_size=batch_size, fused_ft_kernel=False
        )
        position_ids = None
        token_buffer = TokenBuffer(
            max_length=max_blank_length,
            init_temperature=temperature,
            max_temperature=1.5,
            max_retry_times=5,
        )
        for i in pbar:
            pbar.set_description(
                f"{tqdm_name} [0 - {num_tokens}], mode: {sample_mode}, temp: {temperature}, icl: {icl_mode}"
            )
            input_embeds = model_input["inputs_embeds"]
            logits = self.model(
                inputs_embeds=input_embeds.half(),
                inference_params=infer_params,
                position_ids=position_ids,
                last_token_only=True,
            ).logits  # b,1,num_logits

            if i < 10:
                logits = logits[..., :-2]

            predict_token = self.sample_logits(
                logits.float(), temperature, thresh, sample_mode
            )
            predict_token = predict_token[:, None]
            if predict_token[0][0] == self.target_embedder.eos_id:
                break

            if step_out_blank:
                token_buffer.put(predict_token.item())
                new_predict_token = token_buffer.process_duplicate(
                    i,
                    self.sample_logits,
                    logits=logits,
                    thresh=thresh,
                    mode=sample_mode,
                )
                if new_predict_token is not None:
                    predict_token = new_predict_token

            predict_token_emb = self.target_embedder.embedder(predict_token)
            predict_token_emb += tgt_lang_embed
            predict_token_emb += spkenc_out

            model_input["inputs_embeds"] = predict_token_emb
            if rl_training and i < num_tokens - 1:
                rl_model_input["inputs_embeds"] = torch.cat(
                    [rl_model_input["inputs_embeds"], predict_token_emb], dim=1
                )

            if i == 0:
                infer_params.sequence_len_offset = input_embeds.shape[-2]
            else:
                infer_params.sequence_len_offset += 1
            position_ids = torch.full(
                (batch_size, 1),
                infer_params.sequence_len_offset,
                dtype=torch.long,
                device=predict_token.device,
            )
            output_tokens = (
                torch.cat([output_tokens, predict_token], dim=1)
                if output_tokens is not None
                else predict_token
            )

        # Add ref_samples to generation beam
        if ref_samples is not None:
            # (b * beam, s, d) -> (b, s, d) -> (b * (beam + 1), s, d)
            inputs_embeds = (
                inputs_embeds.reshape(batch_size, beam, seq_len, -1)[:, 0, :, :]
                .repeat(1, beam + 1, 1)
                .reshape(batch_size * (beam + 1), seq_len, -1)
            )
            sos_embeds = self.target_embedder.get_sos_embed(batch_size * (beam + 1))
            # (b * beam, s) -> (b * (beam + 1), s)
            output_tokens = torch.cat(
                [ref_samples.unsqueeze(1), output_tokens.reshape(batch_size, beam, -1)],
                dim=1,
            ).reshape(batch_size * (beam + 1), -1)
        if rl_training:
            return output_tokens, rl_model_input
        else:
            return output_tokens

    @torch.no_grad()
    def super_predict(self, inputs_embeds, num_tokens, temperature, **kwargs):
        return super().predict(inputs_embeds, num_tokens, temperature, **kwargs)

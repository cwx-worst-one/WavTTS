import torch
import torch.nn as nn

from apps.bigmusic.umm.ar.utils.utils import sample
from apps.bigmusic.umm.ar.lightning.base_modules import BaseContinuousEmbedModule
from apps.bigmusic.umm.ar.lightning.embedding_modules import (
    BestRQEmbedder,
    BestRQTokenEmbedder,
    SoundstreamTokenEmbedder,
    WavToVecTokenEmbedder,
    get_soundstream_tokens,
)


class FineModule(BaseContinuousEmbedModule):
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
        embedder_dict = {
            "coarse": SoundstreamTokenEmbedder(
                layer_range=(0, 4), embedding_dim=hidden_size, add_sos=False
            )
        }
        input_embedders = nn.ModuleDict(embedder_dict)
        target_embedder = SoundstreamTokenEmbedder(
            layer_range=(4, 12), embedding_dim=hidden_size, add_sos=True
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

    def prepare_training_inputs(self, batch):
        # Manually encode tokens we only run soundstream once
        b = batch["target_audio"].shape[0]
        soundstream_ids = get_soundstream_tokens(self.requires, batch["target_audio"])
        num_coarse, num_fine = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
        )
        device = self.device
        # get coarse ids
        coarse_ids = (
            soundstream_ids[:, :, 0:num_coarse]
            + (torch.arange(num_coarse, device=device))
            * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            soundstream_ids[:, :, num_coarse : num_coarse + num_fine]
            + torch.arange(num_fine, device=device)
            * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))

        # If we are predicting tokens, must get tokens first, before converting to embeddings
        inputs_embeds = self.input_embedders["coarse"].embed(token_ids=coarse_ids)
        sos_embeds = self.target_embedder.get_sos_embed(b)
        target_embeds = self.target_embedder.embed(token_ids=fine_ids)[:, :-1, :]
        return {
            "inputs_embeds": torch.cat(
                [inputs_embeds, sos_embeds, target_embeds], dim=1
            )
        }, fine_ids

    def sample_logits(self, i, logits, temp, mode, thresh=0.9):
        soundstream_codebook_size = self.extra_params.soundstream_codebook_size
        num_fine = self.extra_params.num_fine

        layer_idx = i % num_fine
        predict_logits = logits[
            :,
            -1:,
            layer_idx
            * soundstream_codebook_size : (layer_idx + 1)
            * soundstream_codebook_size,
        ]
        samples = sample(predict_logits, temp=temp, mode=mode, thresh=thresh)
        samples = samples + layer_idx * soundstream_codebook_size
        return samples

    @torch.no_grad()
    def predict(self, coarse_samples, hp):
        input_embeds = self.input_embedders["coarse"].embed(token_ids=coarse_samples)

        num_coarse = self.extra_params.num_coarse
        num_fine = self.extra_params.num_fine
        soundstream_codebook_size = self.extra_params.soundstream_codebook_size
        soundstream_frame_rate = self.extra_params.soundstream_frame_rate

        input_framerate = soundstream_frame_rate * num_coarse
        output_framerate = soundstream_frame_rate * num_fine

        target_duration = hp.duration
        slice_duration = hp.fine_duration
        stride_duration = hp.fine_stride

        temperature = hp.fine_temperature
        sample_mode = hp.get("ar_sample_mode", hp.sample_mode)

        fine_samples = self.predict_slice(
            input_embeds,
            input_framerate,
            output_framerate,
            target_duration,
            slice_duration,
            stride_duration,
            temperature=temperature,
            sample_mode=sample_mode,
        )
        batch_size = coarse_samples.size(0)
        # return (
        #     fine_samples.reshape((batch_size, -1, num_fine))
        #     - torch.arange(num_fine, device=coarse_samples.device)
        #     * self.extra_params.soundstream_codebook_size
        # )
        return fine_samples + num_coarse * soundstream_codebook_size


class CoarseModule(BaseContinuousEmbedModule):
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
        semantic_codebook_size = extra_params["semantic_codebook_size"]

        semantic_type = extra_params.get("semantic_type", "wav2vec")
        if semantic_type == "wav2vec":
            semantic_embedder = WavToVecTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=False,
            )
        elif semantic_type == "bestrq":
            semantic_embedder = BestRQTokenEmbedder(
                vocab_size=semantic_codebook_size,
                embedding_dim=hidden_size,
                add_sos=False,
            )
        elif semantic_type == "bestrq_embeds":
            semantic_embedder = BestRQEmbedder(
                input_dim=1024, embedding_dim=hidden_size, add_sos=False
            )
        else:
            raise NotImplementedError
        embedder_dict = {"semantic": semantic_embedder}
        input_embedders = nn.ModuleDict(embedder_dict)
        target_embedder = SoundstreamTokenEmbedder(
            layer_range=(0, 4), embedding_dim=hidden_size, add_sos=True
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

    def prepare_inputs_embeddings(self, batch):
        # convert inputs to conditions
        return self.input_embedders["semantic"].embed(
            self.requires, batch["target_audio"], with_sos=False
        )

    def sample_logits(self, i, logits, temp, mode, thresh=0.9):
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
        samples = sample(predict_logits, temp=temp, mode=mode, thresh=thresh)
        samples = samples + layer_idx * soundstream_codebook_size
        return samples

    @torch.no_grad()
    def predict(self, semantic_samples, hp, semantic_embeds=None):
        if semantic_embeds is not None:
            input_embeds = semantic_embeds
        else:
            input_embeds = self.input_embedders["semantic"].embed(
                token_ids=semantic_samples
            )

        input_framerate = self.extra_params.semantic_frame_rate
        output_framerate = (
            self.extra_params.soundstream_frame_rate * self.extra_params.num_coarse
        )

        target_duration = hp.duration
        slice_duration = hp.coarse_duration
        stride_duration = hp.coarse_stride

        temperature = hp.coarse_temperature
        sample_mode = hp.get("ar_sample_mode", hp.sample_mode)

        coarse_samples = self.predict_slice(
            input_embeds,
            input_framerate,
            output_framerate,
            target_duration,
            slice_duration,
            stride_duration,
            temperature=temperature,
            sample_mode=sample_mode,
        )
        return coarse_samples
        # return (
        #     coarse_samples.reshape((coarse_samples.shape(0), -1, num_coarse))
        #     - torch.arange(num_coarse, device=coarse_samples.device)
        #     * self.extra_params.soundstream_codebook_size
        # )

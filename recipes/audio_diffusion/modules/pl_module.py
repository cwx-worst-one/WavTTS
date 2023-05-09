from cmath import inf

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from einops import rearrange

# from frechet_audio_distance import FrechetAudioDistance
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.distributed import rank_zero_only


def print_model_params(model):
    total_params = 0
    trainable_params = 0
    for _, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
    print(f"Total Params: {total_params:,}, Trainable Params: {trainable_params:,}")


class MusicLMText2SemanticModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        semantic_token_model,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.semantic_token_model = semantic_token_model
        print_model_params(semantic_token_model)
        # self.semantic_token_model = semantic_token_model
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.min_loss = inf
        self.save_hyperparameters(ignore=["semantic_token_model", "cuda_transforms"])

    def step(self, batch):
        semantic, text_embeddings = batch

        (
            semantic_token_model_loss,
            semantic_token_accuracy,
        ) = self.semantic_token_model.loss(semantic, text_embeddings)

        return semantic_token_model_loss, semantic_token_accuracy

    def training_step(self, batch, batch_idx):
        semantic_token_model_loss, semantic_token_accuracy = self.step(batch)

        self.log("loss/train", semantic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/train",
            semantic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
        )
        return semantic_token_model_loss

    def validation_step(self, batch, batch_idx):
        semantic_token_model_loss, semantic_token_accuracy = self.step(batch)

        self.log(
            "loss/valid", semantic_token_model_loss, rank_zero_only=True, sync_dist=True
        )
        self.log(
            "accuracy/valid",
            semantic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return semantic_token_model_loss

    def generate_samples(self, text_input, temperature, sequence_length):
        return self.semantic_token_model.sample(
            text_input=text_input,
            temperature=temperature,
            sequence_length=sequence_length,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.semantic_token_model.parameters())
        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


class MusicLMCoarseModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        acoustic_token_model,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        seed_model=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acoustic_token_model = acoustic_token_model
        print_model_params(acoustic_token_model)
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.min_loss = inf
        self.save_hyperparameters(
            ignore=["acoustic_token_model", "cuda_transforms", "seed_model"]
        )
        if seed_model is not None:
            print(f"Loading seed model from {seed_model}")
            state_dict = torch.load(seed_model, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            self.load_state_dict(state_dict=state_dict)

    def step(self, batch):
        audio, semantic_tokens, text_input = batch
        if self.cuda_transforms:
            with torch.no_grad():
                acoustic_tokens = self.cuda_transforms(audio)
        (
            acoustic_token_model_loss,
            acoustic_token_accuracy,
        ) = self.acoustic_token_model.loss(acoustic_tokens, semantic_tokens, text_input)

        return acoustic_token_model_loss, acoustic_token_accuracy

    def training_step(self, batch, batch_idx):
        acoustic_token_model_loss, acoustic_token_accuracy = self.step(batch)

        self.log("loss/train", acoustic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/train",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
        )
        return acoustic_token_model_loss

    def validation_step(self, batch, batch_idx):
        acoustic_token_model_loss, acoustic_token_accuracy = self.step(batch)

        self.log(
            "loss/valid", acoustic_token_model_loss, rank_zero_only=True, sync_dist=True
        )
        self.log(
            "accuracy/valid",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return acoustic_token_model_loss

    def generate_samples(
        self, text_input, semantic_tokens, temperature, sequence_length
    ):
        return self.acoustic_token_model.sample(
            text_input=text_input,
            semantic_tokens=semantic_tokens,
            temperature=temperature,
            sequence_length=sequence_length,
        )

    def configure_optimizers(self):
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        else:
            optimizer = self.optimizer_class(self.acoustic_token_model.parameters())

        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False


class MusicLMFineModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        acoustic_token_model,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acoustic_token_model = acoustic_token_model
        print_model_params(acoustic_token_model)
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.min_loss = inf
        self.save_hyperparameters(ignore=["acoustic_token_model", "cuda_transforms"])

    def step(self, batch):
        audio = batch[0]
        if self.cuda_transforms:
            with torch.no_grad():
                acoustic_tokens = self.cuda_transforms(audio)
        (
            acoustic_token_model_loss,
            acoustic_token_accuracy,
        ) = self.acoustic_token_model.loss(acoustic_tokens)

        return acoustic_token_model_loss, acoustic_token_accuracy

    def training_step(self, batch, batch_idx):
        acoustic_token_model_loss, acoustic_token_accuracy = self.step(batch)

        self.log("loss/train", acoustic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/train",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
        )
        return acoustic_token_model_loss

    def validation_step(self, batch, batch_idx):
        acoustic_token_model_loss, acoustic_token_accuracy = self.step(batch)

        self.log(
            "loss/valid", acoustic_token_model_loss, rank_zero_only=True, sync_dist=True
        )
        self.log(
            "accuracy/valid",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return acoustic_token_model_loss

    def generate_samples(
        self, coarse_tokens, temperature, sequence_length, seed_sequence=None
    ):
        return self.acoustic_token_model.sample(
            coarse_tokens=coarse_tokens,
            temperature=temperature,
            sequence_length=sequence_length,
            seed_sequence=None,
        )

    def configure_optimizers(self):
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        else:
            optimizer = self.optimizer_class(self.acoustic_token_model.parameters())

        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False


class MusicLMFineModuleLegacy(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        acoustic_token_model,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        seed_model=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acoustic_token_model = acoustic_token_model
        print_model_params(acoustic_token_model)
        # self.semantic_token_model = semantic_token_model
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.min_loss = inf
        self.save_hyperparameters(
            ignore=["acoustic_token_model", "cuda_transforms", "seed_model"]
        )
        if seed_model is not None:
            print(f"Loading seed model from {seed_model}")
            state_dict = torch.load(seed_model, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            self.load_state_dict(state_dict=state_dict)

    def step(self, batch):
        audio, semantic, cond_embeddings = batch

        if self.cuda_transforms:
            with torch.no_grad():
                target_acoustic_tokens = self.cuda_transforms(audio)

        (
            acoustic_token_model_loss,
            acoustic_token_accuracy,
        ) = self.acoustic_token_model.loss(
            target_acoustic_tokens, semantic, cond_embeddings
        )

        return acoustic_token_model_loss, acoustic_token_accuracy

    def training_step(self, batch, batch_idx):
        (
            target_acoustic_tokens,
            cond_embeddings,
            semantic_acc,
            semantic_vocal,
        ) = self.extract_tokens_from_batch(batch)
        acoustic_token_model_loss, acoustic_token_accuracy, pred_logits = self.step(
            target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal
        )

        # TODO: when trained using ddp, is accumulation of those metrics happen correct?
        self.log("loss/train", acoustic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/train",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
        )
        # self.generate_extreme_demos(batch, acoustic_token_model_loss)
        return acoustic_token_model_loss

    # This was used to test our hypothesis about silent input leading super low loss
    @rank_zero_only
    def generate_extreme_demos(self, batch, loss):
        if loss < self.min_loss:
            self.logger.experiment.add_audio(
                f"low_loss_demos/{self.global_step}_{loss}",
                batch[0],
                self.global_step,
                sample_rate=24000,
            )
            self.min_loss = loss

    def validation_step(self, batch, batch_idx):
        (
            target_acoustic_tokens,
            cond_embeddings,
            semantic_acc,
            semantic_vocal,
        ) = self.extract_tokens_from_batch(batch)
        acoustic_token_model_loss, acoustic_token_accuracy, pred_logits = self.step(
            target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal
        )

        if (
            self.acoustic_token_model.use_t5
            and self.acoustic_token_model.use_semantic_tokens
        ):
            (
                audio_accompaniment,
                audio_vocal,
                semantic_acc,
                semantic_vocal,
                audio_mix,
            ) = batch

            # calculate the token id's for logits
            pred_tokens = pred_logits.argmax(dim=2)

            pred_coarse_tokens = pred_tokens[
                :, self.acoustic_token_model.semantic_tokens_length :
            ]
            pred_coarse_tokens = rearrange(
                pred_coarse_tokens,
                "b (s q) -> b q s",
                q=self.acoustic_token_model.n_channels,
            )

            # fill in the predicted accompaniment coarse tokens
            modeled_tokens = target_acoustic_tokens.clone()
            modeled_tokens[
                :, : self.acoustic_token_model.n_channels, :
            ] = pred_coarse_tokens

            # decode predicted coarse + ground truth accompaniment tokens
            with torch.no_grad():
                pred_audio_accompaniment = self.cuda_transforms.decode(modeled_tokens)

            pred_mix = (pred_audio_accompaniment + audio_vocal) / 2

            if batch_idx % 5 == 0:
                self.logger.experiment.add_audio(
                    f"audio/valid_reconstructed_coarse_accompaniment-{batch_idx}",
                    pred_audio_accompaniment.cpu(),
                    self.global_step,
                    sample_rate=24000,
                )
                self.logger.experiment.add_audio(
                    f"audio/valid_reconstructed_coarse_mix-{batch_idx}",
                    pred_mix.cpu(),
                    self.global_step,
                    sample_rate=24000,
                )
        self.log("loss/valid", acoustic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/valid",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return acoustic_token_model_loss

    def generate_samples(
        self, coarse_tokens, temperature, sequence_length, seed_sequence=None
    ):
        return self.acoustic_token_model.sample(
            coarse_tokens=coarse_tokens,
            temperature=temperature,
            sequence_length=sequence_length,
            seed_sequence=seed_sequence,
        )

    def configure_optimizers(self):
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        else:
            optimizer = self.optimizer_class(self.acoustic_token_model.parameters())

        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False


class SingSongLitModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        acoustic_token_model,
        # music_embedding_model,
        # semantic_token_model,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acoustic_token_model = acoustic_token_model
        print_model_params(acoustic_token_model)
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.min_loss = inf
        # self.fad = FrechetAudioDistance()
        self.save_hyperparameters(ignore=["acoustic_token_model", "cuda_transforms"])

    def calculate_fad_score(self, pred_audios, target_audios):
        self.fad.model.device = "cpu"
        self.fad.model = self.fad.model.to("cpu")

        def frechet_input_transform(tensor):
            return tensor.cpu().numpy().transpose(), self.cuda_transforms.sample_rate

        pred_audios = list(map(frechet_input_transform, pred_audios))
        target_audios = list(map(frechet_input_transform, target_audios))

        emb_pred = self.fad.get_embeddings(pred_audios)
        emb_target = self.fad.get_embeddings(target_audios)

        mu_pred, sigma_pred = self.fad.calculate_embd_statistics(emb_pred)
        mu_target, sigma_target = self.fad.calculate_embd_statistics(emb_target)
        return self.fad.calculate_frechet_distance(
            mu_pred, sigma_pred, mu_target, sigma_target
        )

    def extract_tokens_from_batch(self, batch):
        semantic_acc = None
        semantic_vocal = None
        (
            audio_accompaniment,
            audio_vocal,
            semantic_acc,
            semantic_vocal,
            audio_mix,
        ) = batch
        with torch.no_grad():
            cond_embeddings = self.cuda_transforms(audio_vocal)
            target_acoustic_tokens = self.cuda_transforms(audio_accompaniment)
        return target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal

    def step(
        self, target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal
    ):
        return self.acoustic_token_model.loss(
            target_acoustic_tokens,
            cond_embeddings,
            accompaniment_semantic_token_ids=semantic_acc,
            vocal_semantic_token_ids=semantic_vocal,
        )

    def training_step(self, batch, batch_idx):
        (
            target_acoustic_tokens,
            cond_embeddings,
            semantic_acc,
            semantic_vocal,
        ) = self.extract_tokens_from_batch(batch)
        acoustic_token_model_loss, acoustic_token_accuracy, pred_logits = self.step(
            target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal
        )

        self.log("loss/train", acoustic_token_model_loss, rank_zero_only=True)
        self.log(
            "accuracy/train",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
        )
        # self.generate_extreme_demos(batch, acoustic_token_model_loss)
        return acoustic_token_model_loss

    # This was used to test our hypothesis about silent input leading super low loss
    @rank_zero_only
    def generate_extreme_demos(self, batch, loss):
        if loss < self.min_loss:
            self.logger.experiment.add_audio(
                f"low_loss_demos/{self.global_step}_{loss}",
                batch[0],
                self.global_step,
                sample_rate=24000,
            )
            self.min_loss = loss

    def validation_step(self, batch, batch_idx):
        (
            target_acoustic_tokens,
            cond_embeddings,
            semantic_acc,
            semantic_vocal,
        ) = self.extract_tokens_from_batch(batch)
        acoustic_token_model_loss, acoustic_token_accuracy, pred_logits = self.step(
            target_acoustic_tokens, cond_embeddings, semantic_acc, semantic_vocal
        )

        if (
            self.acoustic_token_model.use_t5
            and self.acoustic_token_model.use_semantic_tokens
        ):
            (
                audio_accompaniment,
                audio_vocal,
                semantic_acc,
                semantic_vocal,
                audio_mix,
            ) = batch

            # calculate the token id's for logits
            pred_tokens = pred_logits.argmax(dim=2)

            # remove predicted accompaniment semantic token id's,
            # and rearrange into shape (batch, quantizer, sequence)
            pred_coarse_tokens = pred_tokens[
                :, self.acoustic_token_model.semantic_tokens_length :
            ]
            pred_coarse_tokens = rearrange(
                pred_coarse_tokens,
                "b (s q) -> b q s",
                q=self.acoustic_token_model.n_channels,
            )

            # fill in the predicted accompaniment coarse tokens
            modeled_tokens = target_acoustic_tokens.clone()
            modeled_tokens[
                :, : self.acoustic_token_model.n_channels, :
            ] = pred_coarse_tokens

            # decode predicted coarse + ground truth accompaniment tokens
            with torch.no_grad():
                pred_audio_accompaniment = self.cuda_transforms.decode(modeled_tokens)

            pred_mix = (pred_audio_accompaniment + audio_vocal) / 2

            if batch_idx % 5 == 0:
                self.logger.experiment.add_audio(
                    f"audio/valid_reconstructed_coarse_accompaniment-{batch_idx}",
                    pred_audio_accompaniment.cpu(),
                    self.global_step,
                    sample_rate=24000,
                )
                self.logger.experiment.add_audio(
                    f"audio/valid_reconstructed_coarse_mix-{batch_idx}",
                    pred_mix.cpu(),
                    self.global_step,
                    sample_rate=24000,
                )

        # TODO: when trained using ddp, is accum of those metrics happen correctly?
        self.log(
            "loss/valid", acoustic_token_model_loss, rank_zero_only=True, sync_dist=True
        )
        self.log(
            "accuracy/valid",
            acoustic_token_accuracy,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return acoustic_token_model_loss

    def generate_samples(
        self,
        cond_embeddings,
        labels,
        temperature,
        num_outputs,
        sequence_length,
        seed_sequence=None,
    ):
        # TODO: temperature is 0.85 in the SingSong paper
        return self.acoustic_token_model.sample(
            cond_embeddings,
            labels,
            temperature=temperature,
            num_outputs=num_outputs,
            sequence_length=sequence_length,
            device=self.device,
        )

    def configure_optimizers(self):
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.acoustic_token_model.parameters(),
                lr=self.optimizer_class.keywords["lr"],
                weight_decay=self.optimizer_class.keywords["weight_decay"],
            )
        else:
            optimizer = self.optimizer_class(self.acoustic_token_model.parameters())

        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False


class DiffusionLitModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self,
        model,
        diffusion_scheme,
        optimizer_class,
        scheduler_class,
        cuda_transforms=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.diffusion_scheme = diffusion_scheme
        print_model_params(model)
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.cuda_transforms = cuda_transforms
        self.save_hyperparameters(ignore=["model", "cuda_transforms"])

    def step(self, batch):
        audio = batch[0]
        cond_embeddings = batch[1]

        if self.cuda_transforms:
            with torch.no_grad():
                audio_latents = self.cuda_transforms(audio)
        else:
            audio_latents = audio

        loss = self.diffusion_scheme.train(
            audio_latents, self.model, text_cond=cond_embeddings
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log("log_loss/train", loss, rank_zero_only=True)
        self.log("loss/train", torch.pow(10.0, loss), rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("log_loss/valid", loss, rank_zero_only=True)
        self.log("loss/valid", torch.pow(10.0, loss), rank_zero_only=True)
        return loss

    @rank_zero_only
    def generate_samples(self, latents, cond_embeddings=None, **kwargs):
        return self.diffusion_scheme.sample(
            latents, self.model, text_cond=cond_embeddings, **kwargs
        )

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters())
        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

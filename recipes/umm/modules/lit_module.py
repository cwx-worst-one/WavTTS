from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
from transformers import BertTokenizer

from recipes.musiclm.inference.utils import sample
from recipes.umm.models.utils import clip_grad_value_, mel_spectrogram_torch
from recipes.umm.modules.criterion_vocoder import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from samantha.dataio.webdataset import ShardWriter
from samantha.utils.hparams import DotDict


class BaseModule(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> None:
        pass

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["tr_loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(loss_dict)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            val_loss_dict = {}
            for loss in outputs:
                for k, v in loss.items():
                    if k == "tr_loss":
                        k = "loss"
                    k = f"val_{k}_{dataloader_idx}"
                    if k not in val_loss_dict:
                        val_loss_dict[k] = v
                    else:
                        val_loss_dict[k] = val_loss_dict[k] + v
            for k, v in val_loss_dict.items():
                val_loss_dict[k] = v / len(outputs)
            self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_nuc(self, target_tokens):
        if target_tokens.dim() == 3:
            nuc = (
                sum(
                    [
                        len(target_tokens[i, j, :].unique())
                        for i in range(target_tokens.size(0))
                        for j in range(target_tokens.size(1))
                    ]
                )
                / target_tokens.size(0)
                / target_tokens.size(1)
                / target_tokens.size(2)
            )
        elif target_tokens.dim() == 2:
            nuc = (
                sum(
                    [
                        len(target_tokens[i, :].unique())
                        for i in range(target_tokens.size(0))
                    ]
                )
                / target_tokens.size(0)
                / target_tokens.size(1)
            )
        else:
            raise ValueError(f"Input shape wrong, got {target_tokens.size()}")
        return nuc

    def get_quant_rate(self, quant_index, quant_token_num):
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = 100 * one_hot.sum() / quant_token_num
        return quant_rate


class BestRQ(BaseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.len_masking_raw = int(
            self.model.frontend_config.sample_rate * self.model.frontend_config.mask_hop
        )
        self.len_masking_token = int(
            self.model.frontend_config.sample_rate
            * self.model.frontend_config.mask_hop
            / self.model.frontend_config.hop_length
            / pow(2, len(self.model.frontend_config.conv_dim))
        )

    def load_required_modules(self):
        bestrq = self.hparams.required_modules["pretrained_bestrq"]
        state_dict = bestrq["init_fn"](
            bestrq["ckpt_path"], self.local_rank, bestrq["cache_dir"]
        )["pretrained_bestrq_state"]
        print(f'Loading pretrained bestrq from {bestrq["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        # Check if weights are properly loaded
        assert len(unexpected_keys) == 0
        assert all("model.vq" in name for name in missing_keys)

    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.len_masking_raw, device=device)
            < self.model.frontend_config.mask_prob
        )
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), dtype=x.dtype, device=device)
            * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x):
        return self.model.preprocessing(x)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        if isinstance(batch, list):
            wav = batch[0]
        elif isinstance(batch, dict):
            wav = batch["audio"]

        wav = wav.squeeze(dim=1)
        wav = wav.float()
        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)
        return masked_feature, masked_indices, feature

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        return self.model.get_latent(x, layer_idx=layer_idx)

    def _shared_step(self, batch, return_loss: bool = True):
        masked_feature, masked_indices, feature = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(masked_feature, masked_indices, feature)
        else:
            logits, masked_logits, target_tokens, masked_target_tokens = self.model(
                masked_feature, masked_indices, feature
            )

        if return_loss:
            loss_dict = self.criterion(masked_logits, masked_target_tokens)
            accu = (
                masked_logits.argmax(1) == masked_target_tokens
            ).float().mean() * 100

            # target_tokens: [batch, time, codebook_idx]
            rq_nuc = self.get_nuc(target_tokens.transpose(1, 2))
            rq_quant_rate = 0
            for i in range(self.model.codebook_config.n_softmax):
                rq_quant_rate += self.get_quant_rate(
                    target_tokens[:, :, i], self.model.codebook_config.codebook_size
                )
            loss_dict["tr_loss"] = loss_dict["rq_loss"]
            if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
                vq_quant_rate = self.get_quant_rate(
                    quant_idx, self.model.vq_config.codebook_size
                )
                vq_nuc = self.get_nuc(quant_idx)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["tr_loss"] = loss_dict["tr_loss"] + quant_loss.sum()
                loss_dict["vq_quant_loss"] = quant_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
            loss_dict["accu"] = accu
            loss_dict["rq_nuc"] = rq_nuc
            loss_dict["frames"] = feature.size(1)
            loss_dict["feature_mean"] = feature.mean()
            loss_dict["feature_std"] = feature.std()
            loss_dict["rq_quant_rate"] = (
                rq_quant_rate / self.model.codebook_config.n_softmax
            )
            return loss_dict
        return logits


class BestRQInference(BestRQ):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.embeds_bucket = dict()
        self.bucket_idx = dict()

        assert hasattr(self.extra_params, "layer_idx")
        pattern = f"hdfs://haruna/home/byte_speech_sv/data/speech/librilight/embeddings/medium/16020565-step=110000/rank-{self.global_rank}-%05d.tar"
        maxsize = (1 << 32) * 2  # 8GiB, maximum size of each shard
        self.writer = ShardWriter(pattern, maxsize=maxsize)

    def predict_step(self, batch, batch_idx: int):
        masked_feature, masked_indices, feature = self.prepare_feature(batch)
        # ! TODO NOTE: inject feature / masked_feature
        # embeds = self.get_latent(masked_feature, layer_idx=self.extra_params.layer_idx)
        embeds = self.get_latent(feature, layer_idx=self.extra_params.layer_idx)
        # embeds = embeds.reshape((-1, embeds.size(-1))).cpu()
        for batch_idx, e in enumerate(embeds):
            id = f"{batch['speaker_id'][batch_idx]}-{batch['chapter_id'][batch_idx]}-{batch['utterance_id'][batch_idx]}-{batch['utterance_sub_id'][batch_idx]}"
            obj = {
                "__key__": id,
                "embedding.npy": e.cpu().numpy(),
                "metadata.json": {
                    "speaker_id": batch["speaker_id"],
                    # "book_id": batch["book_id"],
                    "chapter_id": batch["chapter_id"],
                    "utterance_id": batch["utterance_id"],
                    "utterance_sub_id": batch["utterance_sub_id"],
                },
            }
            self.writer.write(obj)


class BestRQMel(BestRQ):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    def load_required_modules(self):
        bestrq = self.hparams.required_modules["pretrained_bestrq"]
        state_dict = bestrq["init_fn"](
            bestrq["ckpt_path"], self.local_rank, bestrq["cache_dir"]
        )["pretrained_bestrq_state"]
        print(f'Loading pretrained bestrq from {bestrq["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        # Check if weights are properly loaded
        assert len(unexpected_keys) == 0
        assert all(
            any(
                name.startswith(prefix_key)
                for prefix_key in ["model.vq", "model.spec_reconstructor"]
            )
            for name in missing_keys
        )

    def _shared_step(self, batch, return_loss: bool = True):
        masked_feature, masked_indices, feature = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(masked_feature, masked_indices, feature)
        else:
            (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
            ) = self.model(masked_feature, masked_indices, feature)

        if return_loss:
            loss_dict = self.criterion(
                masked_logits, masked_target_tokens, recon_feature, feature
            )
            accu = (
                masked_logits.argmax(1) == masked_target_tokens
            ).float().mean() * 100

            # target_tokens: [batch, time, codebook_idx]
            rq_nuc = self.get_nuc(target_tokens.transpose(1, 2))
            rq_quant_rate = 0
            for i in range(self.model.codebook_config.n_softmax):
                rq_quant_rate += self.get_quant_rate(
                    target_tokens[:, :, i], self.model.codebook_config.codebook_size
                )
            loss_dict["tr_loss"] = loss_dict["rq_loss"] + loss_dict["stft_loss"]
            if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
                vq_quant_rate = self.get_quant_rate(
                    quant_idx, self.model.vq_config.codebook_size
                )
                vq_nuc = self.get_nuc(quant_idx)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["tr_loss"] = loss_dict["tr_loss"] + quant_loss.sum()
                loss_dict["vq_quant_loss"] = quant_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
            loss_dict["accu"] = accu
            loss_dict["rq_nuc"] = rq_nuc
            loss_dict["frames"] = feature.size(1)
            loss_dict["feature_mean"] = feature.mean()
            loss_dict["feature_std"] = feature.std()
            loss_dict["rq_quant_rate"] = (
                rq_quant_rate / self.model.codebook_config.n_softmax
            )
            return loss_dict
        return logits

    @torch.no_grad()
    def get_mel(self, batch):
        masked_feature, masked_indices, feature = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(masked_feature, masked_indices, feature)
        else:
            (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
            ) = self.model(masked_feature, masked_indices, feature)
        return (
            recon_feature.transpose(1, 2),
            masked_feature.transpose(1, 2),
            feature.transpose(1, 2),
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_quant_idx(self, wav):
        # wav.size(): (batch, time)
        wav = wav.float()
        feature = self.preprocessing(wav)
        encoded_feature = self.model.frontend(feature)
        model_output = self.model.encoder(encoded_feature, vq=self.model.vq)
        hidden_state = model_output["last_hidden_state"]
        quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        return quant_encoder_out, quant_idx, quant_loss


class BestRQMelCTC(BestRQ):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    def setup(self, stage: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        bestrq = self.hparams.required_modules["pretrained_bestrq"]
        state_dict = bestrq["init_fn"](
            bestrq["ckpt_path"], self.local_rank, bestrq["cache_dir"]
        )["pretrained_bestrq_state"]
        print(f'Loading pretrained bestrq from {bestrq["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        # Check if weights are properly loaded
        assert all(
            any(name.startswith(prefix_key) for prefix_key in ["model.codebook_head"])
            for name in unexpected_keys
        )
        assert all(
            any(
                name.startswith(prefix_key)
                for prefix_key in [
                    "model.vq",
                    "model.lm_head",
                    "model.spec_reconstructor",
                ]
            )
            for name in missing_keys
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        feature = self.preprocessing(audio)
        if "normalized_text" in batch:
            encoded_text = self.tokenizer(
                batch["normalized_text"],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            text_ids = encoded_text["input_ids"].to(audio.device)
        else:
            text_ids = None
        return feature, text_ids

    def _shared_step(self, batch, return_loss: bool = True):
        feature, text_ids = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(feature)
        else:
            logits, recon_feature = self.model(feature)

        if return_loss:
            loss_dict = self.criterion(recon_feature, feature, logits, text_ids)
            loss_dict["tr_loss"] = loss_dict["ctc_loss"] + loss_dict["stft_loss"]
            if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
                vq_quant_rate = self.get_quant_rate(
                    quant_idx, self.model.vq_config.codebook_size
                )
                vq_nuc = self.get_nuc(quant_idx)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["tr_loss"] = loss_dict["tr_loss"] + quant_loss.sum()
                loss_dict["vq_quant_loss"] = quant_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
            loss_dict["num_tokens"] = text_ids.size(1)
            loss_dict["num_frames"] = feature.size(1)
            loss_dict["feature_mean"] = feature.mean()
            loss_dict["feature_std"] = feature.std()
            return loss_dict
        return logits, recon_feature

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Union[torch.LongTensor, int],
        add_adapter: Optional[bool] = None,
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (
                torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            )

        for kernel_size, stride in zip(
            self.config.conv_kernel, self.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, 1, self.config.adapter_stride
                )

        return input_lengths

    @torch.no_grad()
    def get_mel(self, batch):
        feature, text_ids = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(feature)
        else:
            logits, recon_feature = self.model(feature)
        return recon_feature.transpose(1, 2), feature.transpose(1, 2)

    @torch.no_grad()
    def get_logits(self, batch):
        feature, text_ids = self.prepare_feature(batch)
        if hasattr(self.model, "vq_config") and self.model.vq_config is not None:
            (
                logits,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            ) = self.model(feature)
        else:
            logits, recon_feature = self.model(feature)
        return logits

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_quant_idx(self, wav):
        # wav.size(): (batch, time)
        wav = wav.float()
        feature = self.preprocessing(wav)
        encoded_feature = self.model.frontend(feature)
        model_output = self.model.encoder(encoded_feature, vq=self.model.vq)
        hidden_state = model_output["last_hidden_state"]
        quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        return quant_encoder_out, quant_idx, quant_loss


class BaseStage(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> None:
        pass

    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(loss_dict)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            val_loss_dict = {}
            for loss in outputs:
                for k, v in loss.items():
                    k = f"val_{k}_{dataloader_idx}"
                    if k not in val_loss_dict:
                        val_loss_dict[k] = v
                    else:
                        val_loss_dict[k] = val_loss_dict[k] + v
            for k, v in val_loss_dict.items():
                val_loss_dict[k] = v / len(outputs)
            self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.model.config.len_masking_raw, device=device)
            < self.model.config.mask_prob
        )
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.model.config.len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.model.config.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), dtype=x.dtype, device=device)
            * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices

    @torch.no_grad()
    def pad_audio(self, x):
        return self.model.pad_audio(x)

    @torch.no_grad()
    def preprocessing(self, x):
        return self.model.preprocessing(x)

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        return self.model.get_latent(x, layer_idx=layer_idx)

    def get_nuc(self, target_tokens):
        if target_tokens.dim() == 3:
            nuc = (
                sum(
                    [
                        len(target_tokens[i, j, :].unique())
                        for i in range(target_tokens.size(0))
                        for j in range(target_tokens.size(1))
                    ]
                )
                / target_tokens.size(0)
                / target_tokens.size(1)
                / target_tokens.size(2)
            )
        elif target_tokens.dim() == 2:
            nuc = (
                sum(
                    [
                        len(target_tokens[i, :].unique())
                        for i in range(target_tokens.size(0))
                    ]
                )
                / target_tokens.size(0)
                / target_tokens.size(1)
            )
        else:
            raise ValueError(f"Input shape wrong, got {target_tokens.size()}")
        return nuc

    def get_quant_rate(self, quant_index, quant_token_num):
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = 100 * one_hot.sum() / quant_token_num
        return quant_rate

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_mulan_embeds(self, x, data_type="music"):
        if data_type == "music":
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"], music=x.float(), device=x.device
            )
        elif data_type == "text":
            # x should be a list of strings
            mulan_embeds = self.requires["mulan_infer_fn"](
                model=self.requires["mulan"],
                text=x,
                device=self.requires["mulan"].device,
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        return mulan_embeds


class Stage1(BaseStage):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        wav = batch["audio"].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)
        return {
            "masked_feature": masked_feature,
            "masked_indices": masked_indices,
            "feature": feature,
        }

    def _shared_step(self, batch, return_loss: bool = True):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        rq_logits = output_dict["rq_logits"]
        rq_masked_logits = output_dict["rq_masked_logits"]
        rq_target = output_dict["rq_target"]
        rq_masked_target = output_dict["rq_masked_target"]
        if return_loss:
            loss_dict = self.criterion(rq_masked_logits, rq_masked_target)
            accu = (rq_masked_logits.argmax(1) == rq_masked_target).float().mean() * 100
            # target_tokens: [batch, time, codebook_idx]
            rq_nuc = self.get_nuc(rq_target.transpose(1, 2))
            rq_quant_rate = 0
            for i in range(self.model.config.rq_codebook_num):
                rq_quant_rate += self.get_quant_rate(
                    rq_target[:, :, i], self.model.config.rq_codebook_size
                )
            rq_quant_rate = rq_quant_rate / self.model.config.rq_codebook_num
            loss_dict["loss"] = loss_dict["rq_loss"]
            if self.model.config.add_vq:
                vq_states = output_dict["vq_states"]
                vq_ids = output_dict["vq_ids"]
                vq_loss = output_dict["vq_loss"]
                vq_quant_rate = self.get_quant_rate(
                    vq_ids, self.model.config.vq_codebook_size
                )
                vq_nuc = self.get_nuc(vq_ids)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["loss"] = loss_dict["loss"] + vq_loss.sum()
                loss_dict["vq_loss"] = vq_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
            loss_dict["accu"] = accu
            loss_dict["rq_nuc"] = rq_nuc
            loss_dict["num_frames"] = input_dict["feature"].size(1)
            loss_dict["feature_mean"] = input_dict["feature"].mean()
            loss_dict["feature_std"] = input_dict["feature"].std()
            loss_dict["rq_quant_rate"] = rq_quant_rate
            return loss_dict

        return output_dict

    @torch.no_grad()
    def get_mel(self, batch):
        input_dict = self.prepare_feature(batch)
        masked_feature = input_dict["masked_feature"]
        feature = input_dict["feature"]
        return {
            "Masked": masked_feature.transpose(1, 2),
            "Original": feature.transpose(1, 2),
        }


class Stage2(BaseStage):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    def setup(self, stage: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        state_dict = pretrained["init_fn"](
            pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"]
        )["state_dict"]
        print(f'Loading pretrained model from {pretrained["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        if self.model.config.add_mulan:
            print(f'Loading mulan from {mulan["ckpt_path"]}')
            mulan = self.hparams.required_modules["mulan"]
            mulan = mulan["init_fn"](
                mulan["ckpt_path"], self.local_rank, mulan["cache_dir"]
            )
            self.requires.update(mulan)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        input_dict = self.preprocessing(audio)
        encoded_text = self.tokenizer(
            batch["text"],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        text_ids = encoded_text["input_ids"].to(audio.device)
        input_dict.update({"text_ids": text_ids, "wav": audio})
        if self.model.config.add_mulan:
            mulan_embeds = self.get_mulan_embeds(audio, data_type="music")
            input_dict["mulan_embeds"] = mulan_embeds
        return input_dict

    def _shared_step(self, batch, return_loss: bool = True):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        if return_loss:
            feature = input_dict["feature"]
            text_ids = input_dict["text_ids"]
            logits = output_dict["logits"]
            recon_feature = output_dict["recon_feature"]
            if self.model.config.get("add_chroma", False):
                chroma = input_dict["chroma"]
                recon_chroma = output_dict["recon_chroma"]
            # if self.model.config.add_mulan:
            #     mulan_embeds = input_dict["mulan_embeds"]
            #     vq_embeds = output_dict["vq_embeds"]
            # else:
            #     mulan_embeds = None
            #     vq_embeds = None
            loss_dict = self.criterion(
                recon_feature=recon_feature,
                feature=feature,
                logits=logits,
                text_ids=text_ids,
                recon_chroma=recon_chroma
                if self.model.config.get("add_chroma", False)
                else None,
                chroma=chroma if self.model.config.get("add_chroma", False) else None,
            )
            loss_dict["loss"] = (
                loss_dict["ctc_loss"] * self.model.config.w_ctc_loss
                + loss_dict["stft_loss"] * self.model.config.w_stft_loss
            )
            if self.model.config.get("add_chroma", False):
                loss_dict["loss"] = (
                    loss_dict["loss"]
                    + loss_dict["chroma_stft_loss"]
                    * self.model.config.w_chroma_stft_loss
                )
            if self.model.config.add_vq:
                vq_states = output_dict["vq_states"]
                vq_ids = output_dict["vq_ids"]
                vq_loss = output_dict["vq_loss"]
                vq_quant_rate = self.get_quant_rate(
                    vq_ids, self.model.config.vq_codebook_size
                )
                vq_nuc = self.get_nuc(vq_ids)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["vq_loss"] = vq_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
                loss_dict["vq_entropy"] = self.model.vq.embedding.entropy()
                loss_dict["loss"] = (
                    loss_dict["loss"] + vq_loss.sum() * self.model.config.w_vq_loss
                )
                # if self.model.config.add_mulan:
                #     mulan_loss = output_dict["mulan_loss"]
                #     loss_dict["mulan_loss"] = mulan_loss
                #     loss_dict["loss"] = loss_dict["loss"] + mulan_loss * self.model.config.w_mulan_loss
            loss_dict["num_tokens"] = text_ids.size(1)
            loss_dict["num_frames"] = input_dict["feature"].size(1)
            loss_dict["feature_mean"] = input_dict["feature"].mean()
            loss_dict["feature_std"] = input_dict["feature"].std()
            return loss_dict

        return output_dict

    @torch.no_grad()
    def get_mel(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        feature = input_dict["feature"]
        recon_feature = output_dict["recon_feature"]
        return {
            "Reconstructed": recon_feature.transpose(1, 2),
            "Original": feature.transpose(1, 2),
        }

    @torch.no_grad()
    def get_audio(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        wav = input_dict["wav"]
        recon_wav = output_dict["recon_wav"]
        return {"Reconstructed": recon_wav, "Original": wav}


class Stage2Vocoder(BaseStage):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    def setup(self, stage: str) -> None:
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        state_dict = pretrained["init_fn"](
            pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"]
        )["state_dict"]
        print(f'Loading pretrained model from {pretrained["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        assert all(
            any(
                name.startswith(prefix_key)
                for prefix_key in ["model.lm_head", "model.unfolder", "model.quantizer"]
            )
            for name in unexpected_keys
        )
        assert all(
            any(
                name.startswith(prefix_key)
                for prefix_key in ["model.vocoder", "criterion"]
            )
            for name in missing_keys
        )
        if self.model.config.add_mulan:
            print(f'Loading mulan from {mulan["ckpt_path"]}')
            mulan = self.hparams.required_modules["mulan"]
            mulan = mulan["init_fn"](
                mulan["ckpt_path"], self.local_rank, mulan["cache_dir"]
            )
            self.requires.update(mulan)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        config = self.model.config
        if x.size(-1) % 960 > 0:
            return torch.nn.functional.pad(
                x, (0, 960 - (x.size(-1) % 960)), "constant", 0
            )
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.model.config.feature_cmvn is not None
        return self.model.model_input_transform(x, normalize=normalize)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        input_dict = {"feature": feature, "wav": audio}
        if self.model.config.add_mulan:
            mulan_embeds = self.get_mulan_embeds(audio, data_type="music")
            input_dict["mulan_embeds"] = mulan_embeds
        return input_dict

    def _shared_step(self, batch, return_loss: bool = True):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        if return_loss:
            feature = input_dict["feature"]
            recon_feature = output_dict["recon_feature"]
            if self.model.config.add_mulan:
                mulan_embeds = input_dict["mulan_embeds"]
                vq_embeds = output_dict["vq_embeds"]
            else:
                mulan_embeds = None
                vq_embeds = None
            loss_dict = self.criterion(
                recon_feature=recon_feature,
                feature=feature,
                logits=None,
                text_ids=None,
                recon_wav=output_dict["recon_wav"],
                wav=input_dict["wav"],
                vq_embeds=vq_embeds,
                mulan_embeds=mulan_embeds,
            )
            loss_dict["loss"] = (
                loss_dict["stft_loss"] * self.model.config.w_stft_loss
                + loss_dict["multi_stft_loss"] * self.model.config.w_multi_stft_loss
            )
            if self.model.config.add_vq:
                vq_states = output_dict["vq_states"]
                vq_ids = output_dict["vq_ids"]
                vq_loss = output_dict["vq_loss"]
                vq_quant_rate = self.get_quant_rate(
                    vq_ids, self.model.config.vq_codebook_size
                )
                vq_nuc = self.get_nuc(vq_ids)
                loss_dict["vq_nuc"] = vq_nuc
                loss_dict["vq_loss"] = vq_loss.sum()
                loss_dict["vq_quant_rate"] = vq_quant_rate
                loss_dict["loss"] = (
                    loss_dict["loss"] + vq_loss.sum() * self.model.config.w_vq_loss
                )
                if self.model.config.add_mulan:
                    mulan_loss = output_dict["mulan_loss"]
                    loss_dict["mulan_loss"] = mulan_loss
                    loss_dict["loss"] = (
                        loss_dict["loss"] + mulan_loss * self.model.config.w_mulan_loss
                    )
            loss_dict["num_frames"] = input_dict["feature"].size(1)
            loss_dict["feature_mean"] = input_dict["feature"].mean()
            loss_dict["feature_std"] = input_dict["feature"].std()
            return loss_dict

        return output_dict

    @torch.no_grad()
    def get_mel(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        feature = input_dict["feature"]
        recon_feature = output_dict["recon_feature"]
        return {
            "Reconstructed": recon_feature.transpose(1, 2),
            "Original": feature.transpose(1, 2),
        }

    @torch.no_grad()
    def get_audio(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        wav = input_dict["wav"]
        recon_wav = output_dict["recon_wav"]
        return {"Reconstructed": recon_wav, "Original": wav}


class Stage2VocoderOnly(pl.LightningModule):
    def __init__(
        self,
        generator_cls,
        discriminator_cls,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator, self.discriminator = generator_cls(), discriminator_cls()
        self.optimizer_g_cls, self.optimizer_d_cls = (
            self.hparams.optimizer_g_cls,
            self.hparams.optimizer_d_cls,
        )
        self.scheduler_g_cls, self.scheduler_d_cls = (
            self.hparams.scheduler_g_cls,
            self.hparams.scheduler_d_cls,
        )
        self.stft_loss = MultiResolutionSTFTLoss()

        self.extra_params = DotDict(extra_params)
        self.requires = {}

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # custom recorder for training step due to GAN training
        self.current_step = 0

    def setup(self, stage: str) -> None:
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        print(f"Loading bestrq_mel_ctc_vq")
        loaded = self.hparams.required_modules["bestrq_mel_ctc_vq"]
        bestrq_mel_ctc_vq = loaded["init_fn"](
            loaded["ckpt_path"], self.local_rank, loaded["cache_dir"]
        )
        self.requires.update(bestrq_mel_ctc_vq)

    def configure_optimizers(self):
        # generator
        optimizer_g = self.optimizer_g_cls(self.generator.parameters())
        scheduler_g = self.scheduler_g_cls(optimizer_g)
        # discriminator
        optimizer_d = self.optimizer_d_cls(self.discriminator.parameters())
        scheduler_d = self.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):
        # get optimizor and scheduler
        net_g, net_d = self.generator, self.discriminator
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        input_dict = self.prepare_feature(batch)
        wav = input_dict["wav"]

        mel = mel_spectrogram_torch(
            wav,
            self.extra_params.n_fft,
            self.extra_params.n_mels,
            self.extra_params.sample_rate,
            self.extra_params.hop_length,
            self.extra_params.win_length,
            self.extra_params.f_min,
            self.extra_params.f_max,
        )

        # train discriminator
        wav_hat = net_g(input_dict["feature"].transpose(1, 2)).squeeze(1)

        mel_hat = mel_spectrogram_torch(
            wav_hat,
            self.extra_params.n_fft,
            self.extra_params.n_mels,
            self.extra_params.sample_rate,
            self.extra_params.hop_length,
            self.extra_params.win_length,
            self.extra_params.f_min,
            self.extra_params.f_max,
        )

        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = net_d(
            wav.unsqueeze(1), wav_hat.detach().unsqueeze(1)
        )

        # discriminator loss
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

        # disciminator backward
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        grad_norm_d = clip_grad_value_(net_d.parameters(), 1.0)
        optim_d.step()
        scheduler_d.step()
        self.untoggle_optimizer(optim_d)

        # train generator
        self.toggle_optimizer(optim_g)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
            wav.unsqueeze(1), wav_hat.unsqueeze(1)
        )

        # generator loss
        loss_sc, loss_mag = self.stft_loss(wav, wav_hat)
        loss_mel = F.l1_loss(mel, mel_hat)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = (
            self.extra_params.w_gen * loss_gen
            + self.extra_params.w_fm * loss_fm
            + self.extra_params.w_mel * loss_mel
            + self.extra_params.w_stft * loss_sc
            + self.extra_params.w_stft * loss_mag
        )

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        grad_norm_g = clip_grad_value_(net_g.parameters(), 1.0)
        optim_g.step()
        scheduler_g.step()
        self.untoggle_optimizer(optim_g)

        # log
        self.log_dict(
            {
                "loss_disc": loss_disc,
                "loss_gen": loss_gen,
                "loss_fm": loss_fm,
                "loss_mel": loss_mel,
                "loss_sc": loss_sc,
                "loss_mag": loss_mag,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "opt_lr": optim_g.param_groups[0]["lr"],
                "sch_lr": scheduler_g.get_last_lr()[0],
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.current_step += 1

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        if x.size(-1) % 960 > 0:
            return torch.nn.functional.pad(
                x, (0, 960 - (x.size(-1) % 960)), "constant", 0
            )
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.generator.config.feature_cmvn is not None
        return self.requires["BestRQMelCTCVQ"].model.model_input_transform(
            x, normalize=normalize
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        encoded_feature = self.requires["BestRQMelCTCVQ"].model.frontend(feature)
        model_output = self.requires["BestRQMelCTCVQ"].model.encoder(
            encoded_feature, vq=self.requires["BestRQMelCTCVQ"].model.vq
        )
        hidden_state = model_output["last_hidden_state"]
        input_dict = {"feature": hidden_state, "wav": audio}
        return input_dict

    @torch.no_grad()
    def get_audio(self, batch):
        input_dict = self.prepare_feature(batch)
        feature, wav = input_dict["feature"], input_dict["wav"]
        recon_wav = self.generator(feature.transpose(1, 2)).squeeze(1)
        return {"Reconstructed": recon_wav, "Original": wav}


class Stage3(Stage2):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        encoded_feature = self.wav2embed(wav)
        shared_encoder_output = self.model.shared_encoder.forward_to_vq(
            encoded_feature, vq=self.model.vq
        )
        vq_ids = shared_encoder_output["vq_ids"]
        return vq_ids

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2embed(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        wav = self.pad_audio(wav.float())
        feature = self.preprocessing(wav)
        encoded_feature = self.model.audio_encoder(feature)
        return encoded_feature

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return


class MKII(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        checkpointing=False,
        extra_params=None,
        required_modules=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        state_dict = pretrained["init_fn"](
            pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"]
        )["state_dict"]
        print(f'Loading pretrained model from {pretrained["ckpt_path"]}')
        self.modify_state_dict(state_dict)
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return

    def modify_state_dict(self, state_dict):
        c = self.model.config
        for k in list(state_dict.keys()):
            if k.startswith("model.shared_encoder.layers"):
                k_i = int(k.split(".")[3])
                k_suffix = ".".join(k.split(".")[4:])
                if k_i < c.num_hidden_layers:
                    k_new = f"model.encoder_layers.{k_i}.{k_suffix}"
                    state_dict[k_new] = state_dict[k]
                    print(f"[MSD] {k} -> {k_new}")
                else:
                    if k_i < c.num_syllable_pre_layers + c.num_hidden_layers:
                        k_new = f"model.syllable_pre_layers.{k_i - c.num_hidden_layers}.{k_suffix}"
                    elif (
                        k_i
                        < c.num_syllable_pre_layers
                        + c.num_syllable_post_layers
                        + c.num_hidden_layers
                    ):
                        k_new = f"model.syllable_post_layers.{k_i - (c.num_syllable_pre_layers + c.num_hidden_layers)}.{k_suffix}"
                    state_dict[k_new] = state_dict[k]
                    print(f"[MSD] {k} -> {k_new}")
                    if k_i < c.num_chroma_pre_layers + c.num_hidden_layers:
                        k_new = f"model.chroma_pre_layers.{k_i - c.num_hidden_layers}.{k_suffix}"
                    elif (
                        k_i
                        < c.num_chroma_pre_layers
                        + c.num_chroma_post_layers
                        + c.num_hidden_layers
                    ):
                        k_new = f"model.chroma_post_layers.{k_i - (c.num_chroma_pre_layers + c.num_hidden_layers)}.{k_suffix}"
                    state_dict[k_new] = state_dict[k]
                    print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
            elif k.startswith("model.melrecon_head"):
                k_suffix = ".".join(k.split(".")[2:])
                k_new = f"model.mel_head.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
            elif k.startswith("model.ctc_head"):
                k_suffix = ".".join(k.split(".")[2:])
                k_new = f"model.syllable_head.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
            elif k.startswith("model.shared_encoder.embed_positions"):
                k_suffix = ".".join(k.split(".")[3:])
                k_new = f"model.embed_positions.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
        return

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        encoded_text = self.tokenizer(
            batch["text"],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        text_ids = encoded_text["input_ids"].to(audio.device)
        input_dict = {"text_ids": text_ids, "wav": audio}
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        loss_dict = self.criterion(
            logits=output_dict["syllable_out"],
            text_ids=input_dict["text_ids"],
            recon_chroma=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            recon_mel=output_dict["mel_out"],
            mel=input_dict["mel"],
        )

        loss_dict["loss"] = (
            loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
            + loss_dict["loss_chroma_stft"] * self.model.config.w_loss_chroma
            + loss_dict["loss_mel_stft"] * self.model.config.w_loss_mel
        )
        loss_dict["num_tokens"] = input_dict["text_ids"].size(1)
        loss_dict["num_frames"] = input_dict["mel"].size(1)
        loss_dict["feature_mean"] = input_dict["mel"].mean()
        loss_dict["feature_std"] = input_dict["mel"].std()

        if self.model.config.add_vq:
            loss_dict["loss_vq_syllable"] = output_dict["syllable_vq_loss"]
            loss_dict["loss_vq_chroma"] = output_dict["chroma_vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"]
                + output_dict["syllable_vq_loss"] * self.model.config.w_loss_vq
                + output_dict["chroma_vq_loss"] * self.model.config.w_loss_vq
            )
            nuc_chroma = self.get_nuc(output_dict["chroma_vq_indices"])
            nuc_syllable = self.get_nuc(output_dict["syllable_vq_indices"])
            qr_chroma = self.get_quant_rate(
                output_dict["chroma_vq_indices"],
                self.model.config.vq_chroma_codebook_size,
            )
            qr_syllable = self.get_quant_rate(
                output_dict["syllable_vq_indices"],
                self.model.config.vq_syllable_codebook_size,
            )
            loss_dict["nuc_chroma"] = nuc_chroma
            loss_dict["nuc_syllable"] = nuc_syllable
            loss_dict["qr_chroma"] = qr_chroma
            loss_dict["qr_syllable"] = qr_syllable
            loss_dict[
                "entropy_vq_syllable"
            ] = self.model.syllable_vq.embedding.entropy()
            loss_dict["entropy_vq_chroma"] = self.model.chroma_vq.embedding.entropy()
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     loss_dict = self._shared_step(batch)
    #     if dataloader_idx not in self.val_outputs:
    #         self.val_outputs[dataloader_idx] = []
    #     self.val_outputs[dataloader_idx].append(loss_dict)

    # def on_validation_epoch_end(self):
    #     for dataloader_idx, outputs in self.val_outputs.items():
    #         val_loss_dict = {}
    #         for loss in outputs:
    #             for k, v in loss.items():
    #                 k = f"val_{k}/{dataloader_idx}"
    #                 if k not in val_loss_dict:
    #                     val_loss_dict[k] = v
    #                 else:
    #                     val_loss_dict[k] = val_loss_dict[k] + v
    #         for k, v in val_loss_dict.items():
    #             val_loss_dict[k] = v / len(outputs)
    #         self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
    #         self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        # params = []
        # for name, p in self.model.named_parameters():
        #     if "vq" in name:
        #         print(f"Set lr={self.model.config.vq_lr}: {name}")
        #         params.append({"params": [p], "lr": self.model.config.vq_lr})
        #     else:
        #         params.append({"params": [p]})
        if self.model.config.vq_train_only:
            for name, param in self.model.named_parameters():
                if "vq" not in name:
                    print(f"Freezing {name}")
                    param.requires_grad = False

        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_nuc(self, target_tokens):
        nuc = (
            sum(
                [
                    len(target_tokens[i, :].unique())
                    for i in range(target_tokens.size(0))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
        )
        return nuc

    def get_quant_rate(self, quant_index, quant_token_num):
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = one_hot.sum() / quant_token_num
        return quant_rate

    @torch.no_grad()
    def get_spec(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        return {
            "mel": {
                "Reconstructed": output_dict["mel_out"].transpose(1, 2),
                "Original": input_dict["mel"].transpose(1, 2),
            },
            "chroma": {
                "Reconstructed": output_dict["chroma_out"].transpose(1, 2),
                "Original": input_dict["chroma"].transpose(1, 2),
            },
        }

    @torch.no_grad()
    def pad_audio(self, x):
        return self.model.pad_audio(x)

    @torch.no_grad()
    def preprocessing(self, x):
        return self.model.preprocessing(x)


class MKIIVQ(MKII):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        assert self.model.config.add_vq == True

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        state_dict = pretrained["init_fn"](
            pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"]
        )["state_dict"]
        print(f'Loading pretrained model from {pretrained["ckpt_path"]}')
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return

    @torch.no_grad()
    def tokenize(self, x):
        return self.model.tokenize(x)


class MKIIVocoder(pl.LightningModule):
    def __init__(
        self,
        generator_cls,
        discriminator_cls,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator, self.discriminator = generator_cls(), discriminator_cls()
        self.optimizer_g_cls, self.optimizer_d_cls = (
            self.hparams.optimizer_g_cls,
            self.hparams.optimizer_d_cls,
        )
        self.scheduler_g_cls, self.scheduler_d_cls = (
            self.hparams.scheduler_g_cls,
            self.hparams.scheduler_d_cls,
        )
        self.stft_loss = MultiResolutionSTFTLoss()
        self.ctc_loss = CTCLoss()

        self.extra_params = DotDict(extra_params)

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        # custom recorder for training step due to GAN training
        self.current_step = 0
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    def configure_optimizers(self):
        # generator
        optimizer_g = self.optimizer_g_cls(self.generator.parameters())
        scheduler_g = self.scheduler_g_cls(optimizer_g)
        # discriminator
        optimizer_d = self.optimizer_d_cls(self.discriminator.parameters())
        scheduler_d = self.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):
        # get optimizor and scheduler
        net_g, net_d = self.generator, self.discriminator
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()
        input_dict = self.prepare_feature(batch)

        # train discriminator
        net_g_out = net_g(input_dict["audio"])
        chroma_out = net_g_out["chroma_out"]
        chroma_vq_indices = net_g_out["chroma_vq_indices"]
        chroma_vq_loss = net_g_out["chroma_vq_loss"]
        syllable_out = net_g_out["syllable_out"]
        syllable_vq_indices = net_g_out["syllable_vq_indices"]
        syllable_vq_loss = net_g_out["syllable_vq_loss"]
        vocoder_out = net_g_out["vocoder_out"]
        vocoder_taget = net_g_out["vocoder_taget"]

        mel_hat = mel_spectrogram_torch(
            vocoder_out.squeeze(1),
            self.extra_params.n_fft,
            self.extra_params.n_mels,
            self.extra_params.sample_rate,
            self.extra_params.hop_length,
            self.extra_params.win_length,
            self.extra_params.f_min,
            self.extra_params.f_max,
        )
        mel = mel_spectrogram_torch(
            vocoder_taget.squeeze(1),
            self.extra_params.n_fft,
            self.extra_params.n_mels,
            self.extra_params.sample_rate,
            self.extra_params.hop_length,
            self.extra_params.win_length,
            self.extra_params.f_min,
            self.extra_params.f_max,
        )

        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = net_d(vocoder_taget, vocoder_out.detach())

        # discriminator loss
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc_all = loss_disc

        # disciminator backward
        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        grad_norm_d = clip_grad_value_(net_d.parameters(), 1.0)
        optim_d.step()
        scheduler_d.step()
        self.untoggle_optimizer(optim_d)

        # train generator
        self.toggle_optimizer(optim_g)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(vocoder_taget, vocoder_out)

        # generator loss
        loss_sc, loss_mag = self.stft_loss(
            vocoder_taget.suqeeze(1), vocoder_out.suqeeze(1)
        )
        loss_syllable = self.ctc_loss(syllable_out, input_dict["text_ids"])
        loss_mel = F.l1_loss(mel, mel_hat)
        loss_chroma = F.l1_loss(input_dict["chroma"], chroma_out)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = (
            self.extra_params.w_gen * loss_gen
            + self.extra_params.w_fm * loss_fm
            + self.extra_params.w_mel * loss_mel
            + self.extra_params.w_chroma * loss_chroma
            + self.extra_params.w_syllable * loss_syllable
            + self.extra_params.w_stft * loss_sc
            + self.extra_params.w_stft * loss_mag
            + self.extra_params.w_vq * chroma_vq_loss
            + self.extra_params.w_vq * syllable_vq_loss
        )

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        grad_norm_g = clip_grad_value_(net_g.parameters(), 1.0)
        optim_g.step()
        scheduler_g.step()
        self.untoggle_optimizer(optim_g)

        # log
        self.log_dict(
            {
                "loss_disc": loss_disc,
                "loss_gen": loss_gen,
                "loss_fm": loss_fm,
                "loss_mel": loss_mel,
                "loss_chroma": loss_chroma,
                "loss_chroma_vq": chroma_vq_loss,
                "loss_syllable": loss_syllable,
                "loss_syllable_vq": syllable_vq_loss,
                "loss_sc": loss_sc,
                "loss_mag": loss_mag,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "opt_lr": optim_g.param_groups[0]["lr"],
                "sch_lr": scheduler_g.get_last_lr()[0],
            },
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.current_step += 1

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        encoded_text = self.tokenizer(
            batch["text"],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        text_ids = encoded_text["input_ids"].to(batch["audio"].device)
        batch.update(text_ids=text_ids)
        return batch

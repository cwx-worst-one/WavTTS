import random
from typing import Optional, Union

import pytorch_lightning as pl
import time
import torch
import torch.distributed
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from recipes.umm_062.models.utils import clip_grad_value_, mel_spectrogram_torch
from recipes.umm_062.modules.criterion_vocoder import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from samantha.dataio.webdataset import ShardWriter
from samantha.utils.hparams import DotDict
from samantha.models.ctiga import gpt
from samantha.utils.ctiga.inference_params import InferenceParams


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: torch.Tensor, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thresh=0.95):
    num_logits = logits.shape[-1]
    k = max(int((1 - thresh) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def sample(predict_logits, temp, thresh=0.9, mode="naive", return_probs=False):
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample()
        if return_probs:
            sample_probs = torch.gather(probs, -1, samples.unsqueeze(1)).squeeze(1)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thresh=thresh)
        samples = gumbel_sample(predict_logits, temp)
        if return_probs:
            probs = (predict_logits / temp).softmax(dim=-1)
            sample_probs = torch.gather(probs, -1, samples.unsqueeze(1)).squeeze(1)
    else:
        raise NotImplementedError()

    if return_probs:
        return samples, sample_probs
    else:
        return samples


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
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

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


class Stage0(pl.LightningModule):
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
        self.flops = 0
        self.ts_before_forward = 0
        self.cached_log_dict = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        device_name = torch.cuda.get_device_name()
        if 'A100' in device_name or 'A800' in device_name:
            self.device_FLOPS = 312e12
        elif 'H100' in device_name or 'H800' in device_name:
            self.device_FLOPS = 989e12
        elif 'V100' in device_name:
            self.device_FLOPS = 125e12
        else:
            raise RuntimeError('unknow cuda device name: ', device_name)

    def setup(self, stage: str) -> None:
        if self.global_rank == 0:
            print(self.model)
        if (
            stage == "fit"
            and not self.requires
            and self.hparams.required_modules is not None
        ):
            self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        if pretrained["ckpt_path"].strip() != "":
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

        if "rmvpe" in self.hparams.required_modules:
            rmvpe = self.hparams.required_modules["rmvpe"]
            if rmvpe["ckpt_path"].strip() != "":
                state_dict = rmvpe["init_fn"](
                    rmvpe["ckpt_path"], self.local_rank, rmvpe["cache_dir"]
                )["state_dict"]
                print(f'Loading rmvpe model from {rmvpe["ckpt_path"]}')
                self.model.rmvpe.load_and_eval(state_dict)

    def modify_state_dict(self, state_dict):
        for k in list(state_dict.keys()):
            if k.startswith("model.encoder_pre_layers"):
                k_i = int(k.split(".")[2])
                k_suffix = ".".join(k.split(".")[3:])
                k_new = f"model.encoder_layers.{k_i}.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
            elif k.startswith("model.encoder_post_layers"):
                k_i = int(k.split(".")[2]) + 12
                k_suffix = ".".join(k.split(".")[3:])
                k_new = f"model.encoder_layers.{k_i}.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
        return

    def forward(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def _shared_step(self, batch):
        raise NotImplementedError()

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        # Not quite accurate, time used by optimizer is also counted.
        elapsed = time.time() - self.ts_before_forward
        mfu = self.flops / elapsed / self.device_FLOPS
        self.log_dict_cached({
            "training/mfu" : mfu,
            # FIXME: The below two should really be "max" or "use rank 0".
            "training/mem_gb" : torch.cuda.max_memory_allocated() / 2**30,
            "training/malloc_retries" : torch.cuda.memory_stats()["num_alloc_retries"],
        })

        # This makes sure we don't lose any log items added after forward pass.
        #
        # As a bonus, since by the time we reach here, the previous pass is
        # guaranteed to complete, so we won't need to wait on CUDA computation
        # synchronously.
        #
        # We defer actual logging operation to overlap it with CUDA computation.
        prev_log_dict, pending_deletion = self.flush_log_dict()

        self.ts_before_forward = time.time()
        loss_dict = self._shared_step(batch)

        # Overlap these operations with CUDA computation.
        for i in batch.keys():
            batch[i] = None
        self.log_dict(prev_log_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)
        del pending_deletion
        del prev_log_dict

        if "flops" in loss_dict:
            self.flops = loss_dict["flops"]
            del loss_dict["flops"]

        loss_dict["training/loss"] = loss_dict["loss"]
        self.log_dict_cached(loss_dict)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def pad_audio(self, x):
        return self.model.pad_audio(x)

    @torch.no_grad()
    def preprocessing(self, x):
        return self.model.preprocessing(x)

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
        """Deprecated in favor of `get_quant_rates`."""
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = one_hot.sum() / quant_token_num
        return quant_rate

    def get_quant_rates(self, quant_indices, quant_token_num):
        one_hots = []
        for index in quant_indices:
            one_hot = torch.nn.functional.one_hot(
                index.reshape(-1), quant_token_num
            ).sum(dim=0)
            one_hots.append(one_hot)
        one_hots = self.all_gather(torch.stack(one_hots))
        return one_hots.sum(dim=0).clamp(0, 1).sum() / quant_token_num

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

    def log_dict_cached(self, kvs):
        """Save `kvs` into cached log dict. The dict is flushed after each
        forward pass."""
        self.cached_log_dict.update({
            k : v if v is not torch.Tensor else v.detach()
            for k, v in kvs.items()
        })

    def flush_log_dict(self):
        orig_keys = []
        cpu_values = []
        cuda_values = []

        # Move all non-CUDA tensor in one go.
        for k, v in self.cached_log_dict.items():
            if v is not torch.Tensor:
                orig_keys.append(k)
                cpu_values.append(v)
            elif not v.is_cuda:
                orig_keys.append(k)
                cpu_values.append(v.item())
            else:
                assert v.is_cuda

        for k, v in self.cached_log_dict.items():
            if v is torch.Tensor and v.is_cuda:
                orig_keys.append(k)
                if len(v.size()) == 0:
                    v = torch.unsqueeze(v.float(), 0)
                cuda_values.append(v)

        values = torch.cat(
            [torch.tensor(cpu_values, dtype=torch.float, device='cuda')] + cuda_values
        )
        # Support different collectives is just a matter of gathering metrics
        # to rank 0 and reducing them locally.
        torch.distributed.reduce(values, 0, op=torch.distributed.ReduceOp.AVG)

        res_dict = {orig_keys[i] : values[i] for i in range(len(values))}
        self.cached_log_dict.clear()

        return res_dict, [orig_keys, cpu_values, cuda_values, values]


class Stage1(Stage0):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        seqlen_align=1,
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
        self.seqlen_align=seqlen_align

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
        if torch.all(start_indices == False):
            start_indices[
                random.randint(0, start_indices.size(0) - 1),
                random.randint(0, start_indices.size(1) - 1),
            ] = True
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
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        wav = batch["audio"].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        mel = self.preprocessing(wav)["mel"]
        masked_audio, masked_indices = self.masking(wav)
        masked_mel = self.preprocessing(masked_audio)["mel"]
        seqlen = mel.shape[1]
        if seqlen % self.seqlen_align != 0:
            pad_len = (seqlen + self.seqlen_align - 1) // self.seqlen_align * self.seqlen_align
            mel = torch.nn.functional.pad(mel, [0, 0, 0, pad_len - seqlen])
            masked_mel = torch.nn.functional.pad(masked_mel, [0, 0, 0, pad_len - seqlen])
        return {"masked_mel": masked_mel, "masked_indices": masked_indices, "mel": mel}

    def get_code_rate(self, target_tokens):
        code_rate = (
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
        return code_rate

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        rq_masked_logits = output_dict["rq_masked_logits"]
        rq_masked_target = output_dict["rq_masked_target"]
        loss_dict = self.criterion(rq_masked_logits, rq_masked_target)
        accu = (rq_masked_logits.argmax(1) == rq_masked_target).float().mean()
        loss_dict["accu"] = accu
        loss_dict["flops"] = output_dict["flops"]
        # target_tokens: [batch, time, codebook_idx]
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["rq_target"].transpose(1, 2))
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rates(
            [output_dict["rq_target"][:, :, i] for i in range(self.model.config.rq_codebook_num)],
            self.model.config.rq_codebook_size
        ) / self.model.config.rq_codebook_num
        loss_dict["aux/quant_rate"] = quant_rate

        mel = input_dict["mel"]
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        return loss_dict

    @torch.no_grad()
    def get_spec(self, batch):
        input_dict = self.prepare_feature(batch)
        mel = input_dict["mel"]
        maksed_mel = input_dict["masked_mel"]
        return {
            "Original Mel": mel.transpose(1, 2),
            "Masked Mel": maksed_mel.transpose(1, 2),
        }


class Stage2(Stage0):
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
        input_dict = {}
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        input_dict.update(text_ids=batch["token"])
        feature = self.preprocessing(audio)
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        if self.model.config.get("add_pitch", False):
            loss_dict = self.criterion(
                ctc_logits=output_dict["ctc_out"],
                text_ids=text_ids,
                recon_mel=output_dict["mel_out"],
                mel=mel,
                recon_f0=output_dict["f0_out"].squeeze(-1),
                f0=input_dict["f0"],
                recon_vuv=output_dict["vuv_out"].squeeze(-1),
                vuv=input_dict["vuv"],
            )
        else:
            loss_dict = self.criterion(
                ctc_logits=output_dict["ctc_out"],
                text_ids=text_ids,
                recon_chroma=output_dict["chroma_out"]
                if self.model.config.add_chroma
                else None,
                chroma=input_dict["chroma"] if self.model.config.add_chroma else None,
                recon_mel=output_dict["mel_out"],
                mel=mel,
            )
        loss_dict["loss"] = (
            loss_dict["loss_mel"] * self.model.config.w_loss_mel
            + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
        )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"]+loss_dict["vuv_loss"]) * self.model.config.w_loss_pitch
            )

        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if self.model.config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = self.model.config.w_loss_pitch

        loss_dict["flops"] = output_dict["flops"]
        return loss_dict

    @torch.no_grad()
    def get_spec(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)
        out = {
            "mel": {
                "Reconstructed": output_dict["mel_out"].transpose(1, 2),
                "Original": input_dict["mel"].transpose(1, 2),
            }
        }
        if self.model.config.add_chroma:
            out.update(
                {
                    "chroma": {
                        "Reconstructed": output_dict["chroma_out"].transpose(1, 2),
                        "Original": input_dict["chroma"].transpose(1, 2),
                    }
                }
            )
        return out


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

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        if self.model.config.get("add_pitch", False):
            loss_dict = self.criterion(
                ctc_logits=output_dict["ctc_out"],
                text_ids=text_ids,
                recon_mel=output_dict["mel_out"],
                mel=mel,
                recon_f0=output_dict["f0_out"].squeeze(-1),
                f0=input_dict["f0"],
                recon_vuv=output_dict["vuv_out"].squeeze(-1),
                vuv=input_dict["vuv"],
            )
        else:
            loss_dict = self.criterion(
                ctc_logits=output_dict["ctc_out"],
                text_ids=text_ids,
                recon_chroma=output_dict["chroma_out"]
                if self.model.config.add_chroma
                else None,
                chroma=input_dict["chroma"] if self.model.config.add_chroma else None,
                recon_mel=output_dict["mel_out"],
                mel=mel,
            )
        loss_dict["bs"] = text_ids.shape[0]
        loss_dict["loss"] = (
            loss_dict["loss_mel"] * self.model.config.w_loss_mel
            + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
        )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"]+loss_dict["vuv_loss"]) * self.model.config.w_loss_pitch
            )
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
        # logging
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rate(
            output_dict["vq_ids"].long(), self.model.config.vq_codebook_size
        )
        loss_dict["aux/quant_rate"] = quant_rate
        if getattr(self.model.vq, "entropy", None) is not None:
            loss_dict["aux/entropy"] = self.model.vq.entropy()
        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if self.model.config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = self.model.config.w_loss_pitch
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq

        loss_dict["flops"] = output_dict["flops"]
        return loss_dict

    def get_code_rate(self, target_tokens):
        code_rate = (
            sum(
                [
                    len(target_tokens[i, :].unique())
                    for i in range(target_tokens.size(0))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
        )
        return code_rate

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        return self.model.wav2token(wav)


class ASR(Stage0):
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
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        input_dict = {"wav": audio}
        encoded_text = self.tokenizer(
            batch["text"],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        text_ids = encoded_text["input_ids"].to(audio.device)
        input_dict.update(text_ids=text_ids)
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        loss_dict = self.criterion(ctc_logits=output_dict["ctc_out"], text_ids=text_ids)
        loss_dict["loss"] = loss_dict["loss_ctc"] * self.model.config.w_loss_ctc

        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        return loss_dict


class Stage3Improved(Stage3):
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

    def on_train_batch_start(self, batch, batch_idx):
        if self.trainer.global_step >= 20_000:
            if not hasattr(self.model.vq_proj_in, '__len__'):
                return
            if len(self.model.vq_proj_in) < 3:
                return
            module = self.model.vq_proj_in[2]
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                module.eval()
                if self.trainer.global_step % 1000 == 0:
                    print(module.running_mean)
        return

    def configure_optimizers(self):
        params_group = []
        normal_params = []
        special_params = []
        for name, params in self.model.named_parameters():
            if 'vq.embedding.weight' in name:
                print("Key {} use zero WD".format(name))
                special_params.append(params)
            else:
                normal_params.append(params)
        params_group.append({"params": special_params, "weight_decay": 0.0})
        params_group.append({"params": normal_params})
        optimizer = self.hparams.optimizer_cls(params_group)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class Stage2Vocoder(Stage0):
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


class UMMBase(pl.LightningModule):
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
        self.load_from_pretrained = False
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.hparams.required_modules is not None:
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

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        input_dict = {"text_ids": batch["token"], "wav": audio}
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=input_dict["text_ids"],
            recon_chroma=None
            if self.model.config.disable_chroma
            else output_dict["chroma_out"],
            chroma=None if self.model.config.disable_chroma else input_dict["chroma"],
            recon_mel=output_dict["mel_out"],
            mel=input_dict["mel"],
        )
        loss_dict["loss"] = (
            loss_dict["loss_mel"] * self.model.config.w_loss_mel
            + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
        )
        if not self.model.config.disable_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.add_vq:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
            if self.trainer.global_step % 100 == 0:
                code_rate = self.get_code_rate(output_dict["vq_ids"])
                loss_dict["aux/code_rate"] = code_rate
            quant_rate = self.get_quant_rate(
                output_dict["vq_ids"], self.model.config.vq_codebook_size
            )
            loss_dict["aux/quant_rate"] = quant_rate
            loss_dict["aux/entropy"] = self.model.vq.embedding.entropy()
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq

        loss_dict["aux/num_text_ids"] = input_dict["text_ids"].size(1) * input_dict[
            "text_ids"
        ].size(0)
        loss_dict["aux/num_mel_frames"] = input_dict["mel"].size(1) * input_dict[
            "mel"
        ].size(0)
        loss_dict["aux/mel_mean"] = input_dict["mel"].mean()
        loss_dict["aux/mel_std"] = input_dict["mel"].std()
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if not self.model.config.disable_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_code_rate(self, target_tokens):
        code_rate = (
            sum(
                [
                    len(target_tokens[i, :].unique())
                    for i in range(target_tokens.size(0))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
        )
        return code_rate

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
        if not self.model.config.disable_chroma:
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
        else:
            return {
                "mel": {
                    "Reconstructed": output_dict["mel_out"].transpose(1, 2),
                    "Original": input_dict["mel"].transpose(1, 2),
                }
            }

    @torch.no_grad()
    def pad_audio(self, x):
        return self.model.pad_audio(x)

    @torch.no_grad()
    def preprocessing(self, x):
        return self.model.preprocessing(x)

    @torch.no_grad()
    def wav2token(self, x):
        return self.model.wav2token(x)


class BaseUMM(UMMBase):
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


class Stage3AR(Stage3):
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
        self.requires = {}

    def load_required_modules(self):
        super().load_required_modules()
        soundstream = self.hparams.required_modules["soundstream"]
        print(f'Loading SoundStream from {soundstream["ckpt_path"]}')
        ss = soundstream["init_fn"](
            soundstream["ckpt_path"], self.local_rank, soundstream["cache_dir"]
        )
        self.requires.update(ss)
        print("SoundStream loaded")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        input_dict = {}
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        if hasattr(self, "tokenizer"):
            encoded_text = self.tokenizer(
                batch["text"],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            text_ids = encoded_text["input_ids"].to(audio.device)
            input_dict.update(text_ids=text_ids)
        else:
            input_dict.update(text_ids=batch["token"])
        feature = self.preprocessing(audio)
        input_dict.update(feature)

        soundstream_ids = self.requires["ss"](audio)[2]
        soundstream_ids = torch.stack(soundstream_ids, dim=2)
        soundstream_ids = soundstream_ids[:, :, 0:1]
        soundstream_ids = torch.reshape(soundstream_ids, [audio.size(0), -1])
        input_dict.update(ar_ids=soundstream_ids)

        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            ctc_ids=input_dict["text_ids"],
            chroma_out=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            mel_out=output_dict["mel_out"],
            mel=input_dict["mel"],
            ar_logits=output_dict["ar_out"],
            ar_ids=input_dict["ar_ids"],
        )
        loss_dict["loss_vq"] = output_dict["vq_loss"]
        loss_dict["loss"] = (
            loss_dict["loss_mel"] * self.model.config.w_loss_mel
            + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
            + loss_dict["loss_ar"] * self.model.config.w_loss_ar
            + output_dict["vq_loss"] * self.model.config.w_loss_vq
        )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rate(
            output_dict["vq_ids"], self.model.config.vq_codebook_size
        )
        loss_dict["aux/quant_rate"] = quant_rate
        loss_dict["aux/entropy"] = self.model.vq.embedding.entropy()
        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        loss_dict["aux/w_loss_ar"] = self.model.config.w_loss_ar
        return loss_dict


class UMMASR(UMMBase):
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
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(audio)
        # input_dict = {"text_ids": batch["token"], "wav": audio}
        input_dict = {"wav": audio}
        encoded_text = self.tokenizer(
            batch["text"],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        text_ids = encoded_text["input_ids"].to(audio.device)
        input_dict.update(text_ids=text_ids)
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"], text_ids=input_dict["text_ids"]
        )
        loss_dict["loss"] = loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
        if self.model.config.add_vq:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
            if self.trainer.global_step % 100 == 0:
                code_rate = self.get_code_rate(output_dict["vq_ids"])
                loss_dict["aux/code_rate"] = code_rate
            quant_rate = self.get_quant_rate(
                output_dict["vq_ids"], self.model.config.vq_codebook_size
            )
            loss_dict["aux/quant_rate"] = quant_rate
            loss_dict["aux/entropy"] = self.model.vq.embedding.entropy()
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq

        loss_dict["aux/num_text_ids"] = input_dict["text_ids"].size(1) * input_dict[
            "text_ids"
        ].size(0)
        loss_dict["aux/num_mel_frames"] = input_dict["mel"].size(1) * input_dict[
            "mel"
        ].size(0)
        loss_dict["aux/mel_mean"] = input_dict["mel"].mean()
        loss_dict["aux/mel_std"] = input_dict["mel"].std()
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        return loss_dict


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


class ARModule(pl.LightningModule):
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
        for name, item in self.hparams.required_modules.items():
            hpath = item["ckpt_path"]
            init_fn = item["init_fn"]
            cache_dir = item["cache_dir"]
            print(f"Loading {name} from {hpath}")
            self.requires.update(
                init_fn(hpath, local_rank=self.local_rank, cache_dir=cache_dir)
            )
        return

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_mkii_tokens(self, x):
        embeds, tokens = self.requires["mkii"].tokenize(x.float())
        return tokens

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_mkii_embeds(self, x):
        embeds, tokens = self.requires["mkii"].tokenize(x.float())
        return embeds

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        if x.size(-1) % (self.extra_params.hop_length * 4) > 0:
            return F.pad(
                x,
                (
                    0,
                    self.extra_params.hop_length * 4
                    - (x.size(-1) % (self.extra_params.hop_length * 4)),
                ),
                "constant",
                0,
            )
        else:
            return x

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class UnifiedDecoder(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}

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
        for name, item in self.hparams.required_modules.items():
            hpath = item["ckpt_path"]
            init_fn = item["init_fn"]
            cache_dir = item["cache_dir"]
            print(f"Loading {name} from {hpath}")
            self.requires.update(
                init_fn(hpath, local_rank=self.local_rank, cache_dir=cache_dir)
            )

    def _shared_step(self, batch):
        token_seq, token_length, target_length = self.prepare_feature(batch)
        input_seq = token_seq[:, :-1]
        target_seq = token_seq[:, 1:]
        if isinstance(self.model, gpt.GPTLMHeadModel):
            logits = self.model(input_ids=input_seq)[0]
        else:
            logits = self.model(input_ids=input_seq)["logits"]
        loss_mask = torch.zeros_like(target_seq)
        for i, (tok_len, tar_len) in enumerate(zip(token_length, target_length)):
            loss_mask[i, tok_len - tar_len - 1 : tok_len - 1] = 1
        loss_dict = self.criterion(logits, target_seq, mask=loss_mask)
        accu = ((logits.argmax(dim=-1) == target_seq).float() * loss_mask).sum() / sum(
            target_length
        )
        loss_dict.update(accu=accu)
        loss_dict["aux/num_target_tokens"] = sum(target_length)
        loss_dict["aux/num_input_tokens"] = sum(token_length)
        loss_dict["aux/num_tokens"] = token_seq.size(0) * token_seq.size(1)
        return loss_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def assemble_tokens(self, umm_tokens, text_tokens, text_length, token_map):
        text_sos_id = torch.LongTensor([self.extra_params.text_sos_id]).to(umm_tokens)
        audio_sos_id = torch.LongTensor([self.extra_params.audio_sos_id]).to(umm_tokens)
        break_id = torch.LongTensor([self.extra_params.break_id]).to(umm_tokens)
        eos_id = torch.LongTensor([self.extra_params.eos_id]).to(umm_tokens)
        token_seq = []
        token_length = []
        target_length = []
        for task_type, task_id, it_i, ia_i, tt_i, ta_i in token_map:
            tokens = []
            t_len = 0
            if it_i is not None:
                tokens.append(text_sos_id)
                tokens.append(text_tokens[it_i][: text_length[it_i]])
            if ia_i is not None:
                tokens.append(audio_sos_id)
                tokens.append(umm_tokens[ia_i])
            tokens.append(break_id)
            if tt_i is not None:
                tokens.append(text_sos_id)
                t_len += 1
                tokens.append(text_tokens[tt_i][: text_length[tt_i]])
                t_len += text_length[tt_i]
            if ta_i is not None:
                tokens.append(audio_sos_id)
                t_len += 1
                tokens.append(umm_tokens[ta_i])
                t_len += umm_tokens[ta_i].size(-1)
            tokens.append(eos_id)
            t_len += 1
            tokens = torch.cat(tokens, dim=-1)
            token_length.append(tokens.size(-1))
            token_seq.append(tokens)
            target_length.append(t_len)
        return token_seq, token_length, target_length

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        mono_map = batch["mono_map"]
        homo_map = batch["homo_map"]
        mono_audio = batch["mono_audio"]
        homo_audio = batch["homo_audio"]
        mono_text = batch["mono_text"]
        homo_text = batch["homo_text"]
        mono_text_length = batch["mono_text_length"]
        homo_text_length = batch["homo_text_length"]
        token_seq = []
        token_length = []
        target_length = []

        if len(mono_map) > 0:
            mono_umm_tokens = (
                self.get_umm_tokens(mono_audio) + self.extra_params.vocab_size_text
            )
            mono_assembled = self.assemble_tokens(
                mono_umm_tokens, mono_text, mono_text_length, mono_map
            )
            token_seq = token_seq + mono_assembled[0]
            token_length = token_length + mono_assembled[1]
            target_length = target_length + mono_assembled[2]
        if len(homo_map) > 0:
            homo_umm_tokens = (
                self.get_umm_tokens(homo_audio) + self.extra_params.vocab_size_text
            )
            homo_assembled = self.assemble_tokens(
                homo_umm_tokens, homo_text, homo_text_length, homo_map
            )
            token_seq = token_seq + homo_assembled[0]
            token_length = token_length + homo_assembled[1]
            target_length = target_length + homo_assembled[2]
        token_seq = pad_sequence(token_seq, batch_first=True, padding_value=0)
        if not self.extra_params.dynamic_batch:
            token_seq = F.pad(
                token_seq,
                [0, self.extra_params.max_length - token_seq.size(1)],
                mode="constant",
                value=0,
            )
        return token_seq, token_length, target_length

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_umm_tokens(self, x):
        tokens = self.requires["Stage3"].wav2token(x.float())
        return tokens

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        loss_dict = self._shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Dummy function for triggering callbacks
        return

    @torch.no_grad()
    def predict(self, input_ids, temperature=1.0, sample_mode="gumbel"):
        b, t = input_ids.size()
        assert b == 1, "Only support batch size 1."
        output_samples = None
        if isinstance(self.model, gpt.GPTLMHeadModel):
            inference_params = InferenceParams(
                max_sequence_len=self.extra_params.max_length, max_batch_size=b
            )
        else:
            past_key_values = None
        pbar = tqdm(range(self.extra_params.max_length - t))
        for _ in pbar:
            pbar.set_description(
                f"[UnifiedMirModel] {self.extra_params.max_length} - {t}"
            )
            if isinstance(self.model, gpt.GPTLMHeadModel):
                logits = self.model(
                    input_ids,
                    inference_params=inference_params,
                    position_ids=None,
                    last_token_only=False,
                ).logits
                inference_params.sequence_len_offset += input_ids.size(1)
            else:
                model_output = self.model(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = model_output["past_key_values"]
                logits = model_output["logits"]
            predict_logits = logits[:, -1:]
            samples = sample(predict_logits, temp=temperature, mode=sample_mode)
            input_ids = samples
            if input_ids.item() == self.extra_params.eos_id:
                print(
                    f"[UnifiedMirModel] EOS({self.extra_params.eos_id}) sampled, finish generation",
                    flush=True,
                )
                break
            if output_samples is None:
                output_samples = samples
            else:
                output_samples = torch.cat([output_samples, samples], dim=1)
        return output_samples

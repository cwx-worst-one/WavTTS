import os
import warnings
from turtle import back

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.profilers import PassThroughProfiler
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from core.models.pretrained.quantizer import RandomProjectionQuantizer
from recipes.best_rq.models.flash_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)
from samantha.utils.hparams import DotDict


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim, odim, kernel, conv_layers):
        super().__init__()
        assert len(conv_layers) in [1, 2]
        modules = []
        prev_c = 1
        conv_out_dim = conv_layers[-1] * idim
        for c in conv_layers:
            modules.append(nn.Conv2d(prev_c, c, kernel, 2, kernel // 2))
            modules.append(nn.ReLU())
            prev_c = c
            conv_out_dim //= 2
        self.conv = nn.Sequential(*modules)
        self.linear = nn.Linear(conv_out_dim, odim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x


class BestRQV1(pl.LightningModule):
    def __init__(
        self,
        optimizer_cls,
        scheduler_cls,
        criterion_cls,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        frontend = Conv2dSubsampling(
            idim=self.extra_params.n_mels,
            odim=self.extra_params.hidden_size,
            kernel=self.extra_params.kernel,
            conv_layers=self.extra_params.conv_dim,
        )
        backend = Wav2Vec2ConformerEncoder(
            Wav2Vec2ConformerConfig(
                hidden_size=self.extra_params.hidden_size,
                num_hidden_layers=self.extra_params.num_hidden_layers,
                position_embeddings_type="rotary",
                max_source_positions=self.extra_params.max_source_positions,
                num_attention_heads=self.extra_params.num_attention_heads,
                num_conv_pos_embeddings=self.extra_params.kernel,
            ),
            self.extra_params.is_causal,
        )
        head = nn.Linear(
            self.extra_params.hidden_size,
            self.extra_params.codebook_size * self.extra_params.n_softmax,
            bias=False,
        )
        self.model = nn.ModuleDict(
            {"frontend": frontend, "backend": backend, "head": head}
        )
        self.criterion = criterion_cls()
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.len_masking_raw = int(
            self.extra_params.sample_rate * self.extra_params.mask_hop
        )
        len_masking_token = (
            self.extra_params.sample_rate
            * self.extra_params.mask_hop
            / self.extra_params.hop_length
            / pow(2, len(self.extra_params.conv_dim))
        )
        self.len_masking_token = int(len_masking_token)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def setup(self, stage: str) -> None:
        self.load_required_modules()

    def load_required_modules(self):
        device = torch.device(f"cuda:{self.local_rank}")
        if self.extra_params.feature_mean_path is None:
            feature_mean = torch.zeros((1), device=device)
        else:
            feature_mean = torch.load(
                self.extra_params.feature_mean_path, map_location=device
            )
        if self.extra_params.feature_std_path is None:
            feature_std = torch.ones((1), device=device)
        else:
            feature_std = torch.load(
                self.extra_params.feature_std_path, map_location=device
            )

        melspec = MelSpectrogram(
            sample_rate=self.extra_params.sample_rate,
            n_fft=self.extra_params.n_fft,
            hop_length=self.extra_params.hop_length,
            n_mels=self.extra_params.n_mels,
        ).to(device)
        amp2db = AmplitudeToDB().to(device)

        unfolder = nn.Unfold(
            kernel_size=(self.extra_params.kernel, 1),
            dilation=1,
            padding=(self.extra_params.kernel // 2, 0),
            stride=(2, 1),
        ).to(device)
        quantizer = RandomProjectionQuantizer(
            input_dim=self.extra_params.n_mels
            * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
            codebook_dim=self.extra_params.codebook_dim,
            codebook_size=self.extra_params.codebook_size,
            quantizer_num=self.extra_params.n_softmax,
        ).to(device)
        self.requires["feature_mean"] = feature_mean.unsqueeze(0).unsqueeze(-1)
        self.requires["feature_std"] = feature_std.unsqueeze(0).unsqueeze(-1)
        self.requires["melspec"] = melspec
        self.requires["amp2db"] = amp2db
        self.requires["unfolder"] = unfolder
        self.requires["quantizer"] = quantizer

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.requires["unfolder"](feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        for _ in range(len(self.extra_params.conv_dim)):
            feature = self._unfold(feature)
        return feature

    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.len_masking_raw, device=device)
            < self.extra_params.mask_prob
        )
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), device=device) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x):
        x = self.requires["melspec"](x)
        x = self.requires["amp2db"](x)
        x = x - self.requires["feature_mean"]
        x = x / self.requires["feature_std"]
        return x[:, :, :-1].transpose(1, 2)

    @torch.no_grad()
    def prepare_feature(self, wav):
        if isinstance(wav, list):
            wav = wav[0]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        wav = wav.float()
        b, t = wav.size()

        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)

        # get target tokens
        target_tokens = self.requires["quantizer"](
            rearrange(self._subsample(feature), "b t d -> (b t) d")
        )
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=b)
        num_uni_codes = (
            sum(
                [
                    len(target_tokens[i, :, j].unique())
                    for i in range(b)
                    for j in range(self.extra_params.n_softmax)
                ]
            )
            / b
            / self.extra_params.n_softmax
        )
        return masked_feature, masked_indices, target_tokens, num_uni_codes

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        x = self.preprocessing(x)
        x = self.model["frontend"](x)
        emb = self.model["backend"](x, output_hidden_states=True)["hidden_states"]
        return emb[layer_idx]

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            (masked_feature, masked_indices, target_tokens, nuc) = self.prepare_feature(
                batch
            )
        with torch.cuda.amp.autocast(enabled=True):
            encoded_masked_feature = self.model["frontend"](masked_feature)
            hidden_state = self.model["backend"](encoded_masked_feature)[
                "last_hidden_state"
            ]
            logits = self.model["head"](hidden_state)
            logits = rearrange(
                logits, "b t (d c) -> b t d c", c=self.extra_params.n_softmax
            )
            # return logits and loss
            masked_logits = logits[tuple(masked_indices.t())]
            masked_logits = rearrange(
                masked_logits, "b d c -> (b c) d", c=self.extra_params.n_softmax
            )
            masked_tokens = target_tokens[tuple(masked_indices.t())]
            masked_tokens = rearrange(masked_tokens, "b c -> (b c)")
            loss = self.criterion(masked_logits, masked_tokens)
            accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return loss, accu, nuc

    def training_step(self, batch, batch_idx):
        loss, accu, nuc = self._shared_step(batch)
        self.log_dict(
            {"tr_loss": loss, "accu": accu, "nuc": nuc}, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu, nuc = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu, nuc))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            nuc = 0
            for l, a, n in outputs:
                loss += l
                accu += a
                nuc += n
            loss /= len(outputs)
            accu /= len(outputs)
            nuc /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                    f"val_nuc_{dataloader_idx}": nuc,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class BestRQ(pl.LightningModule):
    def __init__(
        self,
        optimizer_cls,
        scheduler_cls,
        criterion_cls,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        frontend = Conv2dSubsampling(
            idim=self.extra_params.n_mels,
            odim=self.extra_params.hidden_size,
            kernel=self.extra_params.kernel,
            conv_layers=self.extra_params.conv_dim,
        )
        backend = Wav2Vec2ConformerEncoder(
            Wav2Vec2ConformerConfig(
                hidden_size=self.extra_params.hidden_size,
                num_hidden_layers=self.extra_params.num_hidden_layers,
                position_embeddings_type="rotary",
                max_source_positions=self.extra_params.max_source_positions,
                num_attention_heads=self.extra_params.num_attention_heads,
                num_conv_pos_embeddings=self.extra_params.kernel,
            ),
            self.extra_params.is_causal,
        )
        head = nn.Linear(
            self.extra_params.hidden_size,
            self.extra_params.codebook_size * self.extra_params.n_softmax,
            bias=False,
        )
        self.model = nn.ModuleDict(
            {"frontend": frontend, "backend": backend, "head": head}
        )
        if self.extra_params.get("input_norm", True):
            self.model["ln"] = nn.LayerNorm(
                self.extra_params.n_mels
                * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
                # elementwise_affine=False,
            )

        if self.extra_params.feature_mean_path is None:
            feature_mean = torch.zeros((1))
        else:
            feature_mean = torch.load(self.extra_params.feature_mean_path)
        self.register_buffer("feature_mean", feature_mean)
        if self.extra_params.feature_std_path is None:
            feature_std = torch.ones((1))
        else:
            feature_std = torch.load(self.extra_params.feature_std_path)
        self.register_buffer("feature_std", feature_std)
        self.melspec = MelSpectrogram(
            sample_rate=self.extra_params.sample_rate,
            n_fft=self.extra_params.n_fft,
            hop_length=self.extra_params.hop_length,
            n_mels=self.extra_params.n_mels,
        )
        self.amp2db = AmplitudeToDB()

        self.unfolder = nn.Unfold(
            kernel_size=(self.extra_params.kernel, 1),
            dilation=1,
            padding=(self.extra_params.kernel // 2, 0),
            stride=(2, 1),
        )
        self.quantizer = RandomProjectionQuantizer(
            input_dim=self.extra_params.n_mels
            * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
            codebook_dim=self.extra_params.codebook_dim,
            codebook_size=self.extra_params.codebook_size,
            quantizer_num=self.extra_params.n_softmax,
        )

        self.criterion = criterion_cls()
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.len_masking_raw = int(
            self.extra_params.sample_rate * self.extra_params.mask_hop
        )
        len_masking_token = (
            self.extra_params.sample_rate
            * self.extra_params.mask_hop
            / self.extra_params.hop_length
            / pow(2, len(self.extra_params.conv_dim))
        )
        self.len_masking_token = int(len_masking_token)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        for _ in range(len(self.extra_params.conv_dim)):
            feature = self._unfold(feature)
        return feature

    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.len_masking_raw, device=device)
            < self.extra_params.mask_prob
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
        x = self.melspec.float()(x)
        x = self.amp2db.float()(x)
        x = x[:, :, :-1].transpose(1, 2)
        x = x - self.feature_mean.float()
        x = x / self.feature_std.float()
        return x

    @torch.no_grad()
    def prepare_feature(self, wav):
        if isinstance(wav, list):
            wav = wav[0]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        wav = wav.float()

        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)
        return masked_feature, masked_indices, feature

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        x = self.preprocessing(x)
        x = self.model["frontend"](x)
        emb = self.model["backend"](x, output_hidden_states=True)["hidden_states"]
        return emb[layer_idx]

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            masked_feature, masked_indices, feature = self.prepare_feature(batch)
        with torch.cuda.amp.autocast(enabled=True):
            encoded_masked_feature = self.model["frontend"](masked_feature)
            hidden_state = self.model["backend"](encoded_masked_feature)[
                "last_hidden_state"
            ]
            b = hidden_state.size(0)
            logits = self.model["head"](hidden_state)
            logits = rearrange(
                logits, "b t (d c) -> b t d c", c=self.extra_params.n_softmax
            )
            # return logits and loss
            masked_logits = logits[tuple(masked_indices.t())]
            masked_logits = rearrange(
                masked_logits, "b d c -> (b c) d", c=self.extra_params.n_softmax
            )

            quantizer_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
            if self.extra_params.get("input_norm", True):
                quantizer_input = self.model["ln"](quantizer_input)
            # get target tokens
            target_tokens = self.quantizer(quantizer_input)
            target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=b)
            nuc = (
                sum(
                    [
                        len(target_tokens[i, :, j].unique())
                        for i in range(b)
                        for j in range(self.extra_params.n_softmax)
                    ]
                )
                / b
                / self.extra_params.n_softmax
            )
            masked_tokens = target_tokens[tuple(masked_indices.t())]
            masked_tokens = rearrange(masked_tokens, "b c -> (b c)")
            loss = self.criterion(masked_logits, masked_tokens)
            accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return loss, accu, nuc

    def training_step(self, batch, batch_idx):
        loss, accu, nuc = self._shared_step(batch)
        self.log_dict(
            {"tr_loss": loss, "accu": accu, "nuc": nuc}, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu, nuc = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu, nuc))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            nuc = 0
            for l, a, n in outputs:
                loss += l
                accu += a
                nuc += n
            loss /= len(outputs)
            accu /= len(outputs)
            nuc /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                    f"val_nuc_{dataloader_idx}": nuc,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# Deprecated
class BestRq(pl.LightningModule):
    def __init__(
        self,
        optimizer_cls,
        scheduler_cls,
        criterion_cls,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        frontend = Conv2dSubsampling(
            idim=self.extra_params.n_mels,
            odim=self.extra_params.hidden_size,
            kernel=self.extra_params.kernel,
            conv_layers=self.extra_params.conv_dim,
        )
        backend = Wav2Vec2ConformerEncoder(
            Wav2Vec2ConformerConfig(
                hidden_size=self.extra_params.hidden_size,
                num_hidden_layers=self.extra_params.num_hidden_layers,
                position_embeddings_type="rotary",
                max_source_positions=self.extra_params.max_source_positions,
                num_attention_heads=self.extra_params.num_attention_heads,
                num_conv_pos_embeddings=self.extra_params.kernel,
            ),
            self.extra_params.is_causal,
        )
        head = nn.Linear(
            self.extra_params.hidden_size,
            self.extra_params.codebook_size * self.extra_params.n_softmax,
            bias=False,
        )
        self.model = nn.ModuleDict(
            {"frontend": frontend, "backend": backend, "head": head}
        )
        self.criterion = criterion_cls()
        self.requires = {}
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.len_masking_raw = int(
            self.extra_params.sample_rate * self.extra_params.mask_hop
        )
        len_masking_token = (
            self.extra_params.sample_rate
            * self.extra_params.mask_hop
            / self.extra_params.hop_length
            / pow(2, len(self.extra_params.conv_dim))
        )
        self.len_masking_token = int(len_masking_token)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def setup(self, stage: str) -> None:
        self.load_required_modules()

    def load_required_modules(self):
        device = torch.device(f"cuda:{self.local_rank}")
        if self.extra_params.feature_mean_path is None:
            feature_mean = torch.zeros((1), device=device)
        else:
            feature_mean = torch.load(
                self.extra_params.feature_mean_path, map_location=device
            )
        if self.extra_params.feature_std_path is None:
            feature_std = torch.ones((1), device=device)
        else:
            feature_std = torch.load(
                self.extra_params.feature_std_path, map_location=device
            )

        melspec = MelSpectrogram(
            sample_rate=self.extra_params.sample_rate,
            n_fft=self.extra_params.n_fft,
            hop_length=self.extra_params.hop_length,
            n_mels=self.extra_params.n_mels,
        ).to(device)
        amp2db = AmplitudeToDB().to(device)

        unfolder = nn.Unfold(
            kernel_size=(self.extra_params.kernel, 1),
            dilation=1,
            padding=(self.extra_params.kernel // 2, 0),
            stride=(2, 1),
        ).to(device)
        quantizer = RandomProjectionQuantizer(
            input_dim=self.extra_params.n_mels
            * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
            codebook_dim=self.extra_params.codebook_dim,
            codebook_size=self.extra_params.codebook_size,
            quantizer_num=self.extra_params.n_softmax,
        ).to(device)
        self.requires["feature_mean"] = feature_mean.unsqueeze(0).unsqueeze(-1)
        self.requires["feature_std"] = feature_std.unsqueeze(0).unsqueeze(-1)
        self.requires["melspec"] = melspec
        self.requires["amp2db"] = amp2db
        self.requires["unfolder"] = unfolder
        self.requires["quantizer"] = quantizer

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.requires["unfolder"](feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        for _ in range(len(self.extra_params.conv_dim)):
            feature = self._unfold(feature)
        return feature

    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.len_masking_raw, device=device)
            < self.extra_params.mask_prob
        )
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), device=device) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x):
        x = self.requires["melspec"](x)
        x = self.requires["amp2db"](x)
        x = x - self.requires["feature_mean"]
        x = x / self.requires["feature_std"]
        return x[:, :, :-1].transpose(1, 2)

    @torch.no_grad()
    def prepare_feature(self, wav):
        if isinstance(wav, list):
            wav = wav[0]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        wav = wav.float()
        b, t = wav.size()

        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)

        # get target tokens
        target_tokens = self.requires["quantizer"](
            rearrange(self._subsample(feature), "b t d -> (b t) d")
        )
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=b)
        num_uni_codes = (
            sum(
                [
                    len(target_tokens[i, :, j].unique())
                    for i in range(b)
                    for j in range(self.extra_params.n_softmax)
                ]
            )
            / b
            / self.extra_params.n_softmax
        )
        return masked_feature, masked_indices, target_tokens, num_uni_codes

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        x = self.preprocessing(x)
        x = self.model["frontend"](x)
        emb = self.model["backend"](x, output_hidden_states=True)["hidden_states"]
        return emb[layer_idx]

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            (masked_feature, masked_indices, target_tokens, nuc) = self.prepare_feature(
                batch
            )
        with torch.cuda.amp.autocast(enabled=True):
            encoded_masked_feature = self.model["frontend"](masked_feature)
            hidden_state = self.model["backend"](encoded_masked_feature)[
                "last_hidden_state"
            ]
            logits = self.model["head"](hidden_state)
            logits = rearrange(
                logits, "b t (d c) -> b t d c", c=self.extra_params.n_softmax
            )
            # return logits and loss
            masked_logits = logits[tuple(masked_indices.t())]
            masked_logits = rearrange(
                masked_logits, "b d c -> (b c) d", c=self.extra_params.n_softmax
            )
            masked_tokens = target_tokens[tuple(masked_indices.t())]
            masked_tokens = rearrange(masked_tokens, "b c -> (b c)")
            loss = self.criterion(masked_logits, masked_tokens)
            accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return loss, accu, nuc

    def training_step(self, batch, batch_idx):
        loss, accu, nuc = self._shared_step(batch)
        self.log_dict(
            {"tr_loss": loss, "accu": accu, "nuc": nuc}, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu, nuc = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu, nuc))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            nuc = 0
            for l, a, n in outputs:
                loss += l
                accu += a
                nuc += n
            loss /= len(outputs)
            accu /= len(outputs)
            nuc /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                    f"val_nuc_{dataloader_idx}": nuc,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class BestRQ(pl.LightningModule):
    def __init__(
        self,
        optimizer_cls,
        scheduler_cls,
        criterion_cls,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        frontend = Conv2dSubsampling(
            idim=self.extra_params.n_mels,
            odim=self.extra_params.hidden_size,
            kernel=self.extra_params.kernel,
            conv_layers=self.extra_params.conv_dim,
        )
        backend = Wav2Vec2ConformerEncoder(
            Wav2Vec2ConformerConfig(
                hidden_size=self.extra_params.hidden_size,
                num_hidden_layers=self.extra_params.num_hidden_layers,
                position_embeddings_type="rotary",
                max_source_positions=self.extra_params.max_source_positions,
                num_attention_heads=self.extra_params.num_attention_heads,
                num_conv_pos_embeddings=self.extra_params.kernel,
            ),
            self.extra_params.is_causal,
        )
        head = nn.Linear(
            self.extra_params.hidden_size,
            self.extra_params.codebook_size * self.extra_params.n_softmax,
            bias=False,
        )
        self.model = nn.ModuleDict(
            {"frontend": frontend, "backend": backend, "head": head}
        )
        if self.extra_params.get("input_norm", True):
            self.model["ln"] = nn.LayerNorm(
                self.extra_params.n_mels
                * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
                # elementwise_affine=False,
            )

        if self.extra_params.feature_mean_path is None:
            feature_mean = torch.zeros((1))
        else:
            feature_mean = torch.load(self.extra_params.feature_mean_path)
        self.register_buffer("feature_mean", feature_mean)
        if self.extra_params.feature_std_path is None:
            feature_std = torch.ones((1))
        else:
            feature_std = torch.load(self.extra_params.feature_std_path)
        self.register_buffer("feature_std", feature_std)
        self.melspec = MelSpectrogram(
            sample_rate=self.extra_params.sample_rate,
            n_fft=self.extra_params.n_fft,
            hop_length=self.extra_params.hop_length,
            n_mels=self.extra_params.n_mels,
        )
        self.amp2db = AmplitudeToDB()

        self.unfolder = nn.Unfold(
            kernel_size=(self.extra_params.kernel, 1),
            dilation=1,
            padding=(self.extra_params.kernel // 2, 0),
            stride=(2, 1),
        )
        self.quantizer = RandomProjectionQuantizer(
            input_dim=self.extra_params.n_mels
            * pow(self.extra_params.kernel, len(self.extra_params.conv_dim)),
            codebook_dim=self.extra_params.codebook_dim,
            codebook_size=self.extra_params.codebook_size,
            quantizer_num=self.extra_params.n_softmax,
        )

        self.criterion = criterion_cls()
        self.val_outputs = dict()

        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.len_masking_raw = int(
            self.extra_params.sample_rate * self.extra_params.mask_hop
        )
        len_masking_token = (
            self.extra_params.sample_rate
            * self.extra_params.mask_hop
            / self.extra_params.hop_length
            / pow(2, len(self.extra_params.conv_dim))
        )
        self.len_masking_token = int(len_masking_token)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        for _ in range(len(self.extra_params.conv_dim)):
            feature = self._unfold(feature)
        return feature

    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.len_masking_raw, device=device)
            < self.extra_params.mask_prob
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
        x = self.melspec.float()(x)
        x = self.amp2db.float()(x)
        x = x[:, :, :-1].transpose(1, 2)
        x = x - self.feature_mean.float()
        x = x / self.feature_std.float()
        return x

    @torch.no_grad()
    def prepare_feature(self, wav):
        if isinstance(wav, list):
            wav = wav[0]
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        wav = wav.float()

        feature = self.preprocessing(wav)
        masked_wav, masked_indices = self.masking(wav)
        masked_feature = self.preprocessing(masked_wav)
        return masked_feature, masked_indices, feature

    @torch.no_grad()
    def get_latent(self, x, layer_idx=12):
        x = self.preprocessing(x)
        x = self.model["frontend"](x)
        emb = self.model["backend"](x, output_hidden_states=True)["hidden_states"]
        return emb[layer_idx]

    def get_logits_from_layer(self, hidden_states, layer_idx=12):
        emb = self.model["backend"].forward_from_layer(hidden_states, layer_idx)[
            "last_hidden_state"
        ]
        logits = self.model["head"](emb)
        logits = rearrange(
            logits, "b t (d c) -> (b t c) d", c=self.extra_params.n_softmax
        )
        return logits

    @torch.no_grad()
    def get_logits(self, x):
        x = self.preprocessing(x)
        x = self.model["frontend"](x)
        emb = self.model["backend"](x)["last_hidden_state"]
        logits = self.model["head"](emb)
        logits = rearrange(
            logits, "b t (d c) -> (b t c) d", c=self.extra_params.n_softmax
        )
        return logits

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            masked_feature, masked_indices, feature = self.prepare_feature(batch)
        with torch.cuda.amp.autocast(enabled=True):
            encoded_masked_feature = self.model["frontend"](masked_feature)
            hidden_state = self.model["backend"](encoded_masked_feature)[
                "last_hidden_state"
            ]
            b = hidden_state.size(0)
            logits = self.model["head"](hidden_state)
            logits = rearrange(
                logits, "b t (d c) -> b t d c", c=self.extra_params.n_softmax
            )
            # return logits and loss
            masked_logits = logits[tuple(masked_indices.t())]
            masked_logits = rearrange(
                masked_logits, "b d c -> (b c) d", c=self.extra_params.n_softmax
            )

            quantizer_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
            if self.extra_params.get("input_norm", True):
                quantizer_input = self.model["ln"](quantizer_input)
            # get target tokens
            target_tokens = self.quantizer(quantizer_input)
            target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=b)
            nuc = (
                sum(
                    [
                        len(target_tokens[i, :, j].unique())
                        for i in range(b)
                        for j in range(self.extra_params.n_softmax)
                    ]
                )
                / b
                / self.extra_params.n_softmax
            )
            masked_tokens = target_tokens[tuple(masked_indices.t())]
            masked_tokens = rearrange(masked_tokens, "b c -> (b c)")
            loss = self.criterion(masked_logits, masked_tokens)
            accu = (masked_logits.argmax(1) == masked_tokens).float().mean() * 100
        return loss, accu, nuc

    def training_step(self, batch, batch_idx):
        loss, accu, nuc = self._shared_step(batch)
        self.log_dict(
            {"tr_loss": loss, "accu": accu, "nuc": nuc}, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu, nuc = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu, nuc))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            nuc = 0
            for l, a, n in outputs:
                loss += l
                accu += a
                nuc += n
            loss /= len(outputs)
            accu /= len(outputs)
            nuc /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                    f"val_nuc_{dataloader_idx}": nuc,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# Deprecated
class BestRq(pl.LightningModule):
    def __init__(self, model_cls, optimizer_cls, scheduler_cls, checkpointing=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.val_outputs = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        model_outputs = self.model(batch)
        return (
            model_outputs["loss"],
            model_outputs["accu"],
            model_outputs["num_uni_code"],
        )

    def training_step(self, batch, batch_idx):
        loss, accu, num_uni_code = self._shared_step(batch)
        self.log_dict(
            {"tr_loss": loss, "accu": accu, "num_uni_code": num_uni_code},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu, _ = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            for l, a in outputs:
                loss += l
                accu += a
            loss /= len(outputs)
            accu /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class Inference(pl.LightningModule):
    def __init__(self, extra_params, checkpointing=False):
        super().__init__()
        self.save_hyperparameters()
        self.extra_params = DotDict(extra_params)
        self.module = BestRQ.load_from_checkpoint(
            self.extra_params.state_dict_path
        ).eval()
        self.embeds_bucket = dict()
        self.bucket_idx = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        batch = batch.float()
        if self.extra_params.latent_type == "melspec":
            feature = self.module.preprocessing(batch)
            embeds = rearrange(
                self.module._subsample(feature), "b t d -> (b t) d"
            ).cpu()
        else:
            embeds = self.module.get_latent(
                batch, layer_idx=self.extra_params.layer_idx
            )
            embeds = embeds.reshape((-1, embeds.size(-1))).cpu()
        if self.global_rank not in self.embeds_bucket:
            self.embeds_bucket[self.global_rank] = embeds
        else:
            self.embeds_bucket[self.global_rank] = torch.cat(
                [self.embeds_bucket[self.global_rank], embeds], dim=0
            )
        if (
            self.embeds_bucket[self.global_rank].size(0)
            >= self.extra_params.bucket_size
        ):
            if self.global_rank not in self.bucket_idx:
                self.bucket_idx[self.global_rank] = 0
            np.save(
                os.path.join(
                    self.extra_params.output_dir,
                    f"rank{self.global_rank}_bucket{self.bucket_idx[self.global_rank]}.npy",
                ),
                self.embeds_bucket[self.global_rank].numpy(),
            )
            self.embeds_bucket.pop(self.global_rank)
            self.bucket_idx[self.global_rank] += 1


class WarmupCosine(_LRScheduler):
    def __init__(
        self,
        optimizer,
        init_lr,
        warmup_steps,
        cycle_steps,
        min_lr,
        last_epoch=-1,
        verbose=False,
    ):
        self.optimizer = optimizer

        if not isinstance(init_lr, list) and not isinstance(init_lr, tuple):
            self.init_lrs = [init_lr] * len(optimizer.param_groups)
        else:
            if len(init_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} init_lrs, but got {}".format(
                        len(optimizer.param_groups), len(init_lr)
                    )
                )
            self.init_lrs = list(init_lr)

        if not isinstance(min_lr, list) and not isinstance(min_lr, tuple):
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} min_lrs, but got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)

        # TODO: fix hyper bugs
        self.init_lrs = [eval(lr) if type(lr) == str else lr for lr in self.init_lrs]
        self.min_lrs = [eval(lr) if type(lr) == str else lr for lr in self.min_lrs]

        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]

    def cal_lr(self, init_lr, min_lr):
        if self.last_epoch <= self.warmup_steps:
            lr = max(0, self.last_epoch) / self.warmup_steps * init_lr
        else:
            lr = (init_lr - min_lr) * (
                1
                + np.cos(
                    np.pi
                    * min(self.last_epoch - self.warmup_steps, self.cycle_steps)
                    / self.cycle_steps
                )
            ) / 2 + min_lr
        return lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        lrs = [
            self.cal_lr(init_lr, min_lr)
            for init_lr, min_lr in zip(self.init_lrs, self.min_lrs)
        ]
        return lrs

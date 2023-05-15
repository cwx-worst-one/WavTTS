import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from samantha.utils.hparams import DotDict

from ..requires.w2v.w2v_infer import w2v_bert_tokenization
from ..utils.inference import sample


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.codec_token_num = self.extra_params.codec_token_num
        self.mulan_token_sep = mulan_token_sep
        self.mulan_token_num = self.extra_params.mulan_token_num
        self.requires = {}
        self.val_outputs = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch)
        self.log_dict({"tr_loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, accu = self._shared_step(batch)
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

            if self.local_rank == 0:
                print(f"[VAL/LOSS] {loss}")
                print(f"[VAL/ACCU] {accu}")

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
        params = []
        for name, p in self.model.named_parameters():
            if "ln_" in name or "bias" in name:
                print("Skip WD on {}".format(name))
                params.append({"params": [p], "weight_decay": 0.0})
            else:
                params.append({"params": [p]})
        optimizer = self.hparams.optimizer_cls(params)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def prepare_feature(self, wavs):
        raise NotImplementedError()

    @torch.no_grad()
    def get_mulan_embed(self, x):
        # [b, 1, d]
        return self.requires["mulan_infer_fn"](
            model=self.requires["mulan"], music=x.float(), device=x.device
        ).unsqueeze(1)

    @torch.no_grad()
    def get_quant(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_w2v_token(self, x):
        return w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x.float(),
            centers=self.requires["centroids"],
            device=x.device,
        )


class SemanticModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch.float())

        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        w2v_ids = self.get_w2v_token(wavs)
        # w2v_ids = torch.randint(0, 1024, (b, 750)).to(device)

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_ids, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )
        # mulan_ids = torch.randint(0, 1024, (b, 12)).to(device)
        mulan_ids = mulan_ids + 1024
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_ids.dtype, device=device)
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
        )

        input_ids = torch.cat(
            [mulan_ids, sos_ids, w2v_ids[:, : -1]], dim=1
        )
        return input_ids, w2v_ids


class SeerSemanticModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        n_priors,
        n_seers,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.n_priors = n_priors
        self.n_seers = n_seers
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch[0].squeeze(1).float())

        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        w2v_ids = (
            self.get_w2v_token(wavs)
            .reshape(b, self.n_seers, -1)
            .transpose(2, 1)
            .reshape(b, -1)
        )

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_ids, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )
        mulan_ids = mulan_ids + 1024
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        seer_ids = (
            torch.arange(self.n_seers).expand(b, -1).to(device)
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
        )

        if self.n_priors != mulan_ids.size(1):
            raise ValueError("Invalid prefix length.")
        if w2v_ids.size(1) % self.n_seers != 0:
            raise ValueError("Invalid number of seers.")

        input_ids = torch.cat([mulan_ids, seer_ids, w2v_ids[:, : -self.n_seers]], dim=1)
        return input_ids, w2v_ids


class CoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch[0].squeeze(1).float())

        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])

        w2v_ids = self.get_w2v_token(wavs)
        # w2v_ids = torch.randint(0, 1024, (b, 750)).to(device)
        w2v_ids = w2v_ids + num_coarse * self.codec_token_num

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_ids, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )
        # mulan_ids = torch.randint(0, 1024, (b, 12)).to(device)
        mulan_ids = mulan_ids + num_coarse * self.codec_token_num + 1024
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
        )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
            + 1
        )

        input_ids = torch.cat(
            [mulan_ids, sep_ids, w2v_ids, sos_ids, wav_ids[:, : -1]], dim=1
        )
        return input_ids, wav_ids


class SemanticFreeCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch.float())

        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_ids, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )
        mulan_ids = mulan_ids + num_coarse * self.codec_token_num
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + mulan_ids.size(1) * self.mulan_token_num
        )

        input_ids = torch.cat([mulan_ids, sos_ids, wav_ids[:, :-1]], dim=1)
        return input_ids, wav_ids

    @torch.no_grad()
    def predict(self, mulan_ids, sample_len=2000, temp=0.9, sample_mode="naive"):
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()

        mulan_ids = mulan_ids + num_coarse * self.codec_token_num
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + mulan_ids.size(1) * self.mulan_token_num
        )
        input_ids = torch.cat([mulan_ids, sos_ids], dim=1)
        self.model.transformer.init_cache()
        pbar = tqdm(range(sample_len))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerCoarse")
            logits = self.model(input_ids)
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1:, layer_idx * 1024 : (layer_idx + 1) * 1024]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        self.model.transformer.deinit_cache()
        return coarse_samples


class SeerCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        n_priors,
        n_seers,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.n_priors = n_priors
        self.n_seers = n_seers
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch.float())

        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    def seer_rearrange(self, seq, inv=False):
        b = seq.size(0)
        if inv:
            return seq.reshape(b, -1, self.n_seers).transpose(2, 1).reshape(b, -1)
        else:
            return seq.reshape(b, self.n_seers, -1).transpose(2, 1).reshape(b, -1)

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])
        wav_ids = self.seer_rearrange(wav_ids)

        w2v_ids = self.get_w2v_token(wavs) + num_coarse * self.codec_token_num

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_ids, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )
        mulan_ids = mulan_ids + num_coarse * self.codec_token_num + 1024
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        seer_ids = (
            torch.arange(self.n_seers).expand(b, -1).to(device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
            + self.n_seers
        )
        if self.n_priors != mulan_ids.size(1) + sep_ids.size(1) + w2v_ids.size(1):
            raise ValueError("Invalid prefix length.")
        if wav_ids.size(1) % self.n_seers != 0:
            raise ValueError("Invalid length.")

        input_ids = torch.cat(
            [mulan_ids, sep_ids, w2v_ids, seer_ids, wav_ids[:, : -self.n_seers]], dim=1
        )
        return input_ids, wav_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, w2v_ids, sample_len=2000, temp=0.9, sample_mode="naive"
    ):
        if sample_len % self.n_seers != 0:
            raise ValueError("Invalid sample length.")
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()

        mulan_ids = mulan_ids + num_coarse * self.codec_token_num + 1024
        if self.mulan_token_sep:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1)).to(device) * self.mulan_token_num
            )

        w2v_ids = w2v_ids + num_coarse * self.codec_token_num

        seer_ids = (
            torch.arange(self.n_seers).expand(b, -1).to(device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
        )
        sep_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
            + 1024
            + mulan_ids.size(1) * self.mulan_token_num
            + self.n_seers
        )
        if self.n_priors != mulan_ids.size(1) + sep_ids.size(1) + w2v_ids.size(1):
            raise ValueError("Invalid prefix length.")

        input_ids = torch.cat([mulan_ids, sep_ids, w2v_ids, seer_ids], dim=1)
        self.model.transformer.init_cache()
        pbar = tqdm(range(math.ceil(sample_len / self.n_seers)))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerCoarse")
            logits = self.model(input_ids)
            layer_idx = i % num_coarse
            predict_logits = logits[
                :, -self.n_seers :, layer_idx * 1024 : (layer_idx + 1) * 1024
            ]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        self.model.transformer.deinit_cache()
        return self.seer_rearrange(coarse_samples, inv=True)


class TextCoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with self.profiler.profile("[LightningModule]TextCoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                text_tokens, input_tokens = self.prepare_feature(batch)

        with self.profiler.profile("[LightningModule]TextCoarseModule.model_forward"):
            logits = self.model(input_tokens)

        loss_offset = text_tokens.size(1)
        x = logits[:, loss_offset : input_tokens.size(1) - 1, :]
        targets = input_tokens[:, loss_offset + 1 : input_tokens.size(1)]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, batch):
        wavs = batch["audio"].float()
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b = wavs.size(0)

        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        eos_ids = (
            torch.zeros(size=[b, 1], dtype=wav_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
        )

        text_ids = batch["input_ids"] + num_coarse * self.codec_token_num + 1

        input_tokens = torch.cat([text_ids, eos_ids, wav_ids], dim=1)

        return text_ids, input_tokens


class FineModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch):
        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                coarse_ids, input_tokens = self.prepare_feature(batch.float())
                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            logits = self.model(input_tokens)["logits"]

        loss_offset = coarse_ids.size(1)
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len] - (
            self.extra_params.num_coarse * self.codec_token_num
        )
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse, num_fine, num_res, codec_token_num = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
            self.extra_params.num_res,
            self.extra_params.codec_token_num,
        )
        wav_ids = self.get_quant(wavs)
        b = wav_ids.size(0)
        # get coarse ids
        coarse_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * codec_token_num
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            wav_ids[:, :, num_coarse : num_coarse + num_fine]
            + (torch.arange(num_fine).to(device) + num_coarse) * codec_token_num
        ).reshape((b, -1))
        # get eos id
        eos_id = num_res * self.codec_token_num
        eos_ids = torch.zeros([b, 1], dtype=wav_ids.dtype, device=device) + eos_id
        # final input tokens
        input_tokens = torch.cat([coarse_ids, eos_ids, fine_ids], dim=1)
        return coarse_ids, input_tokens


class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous().float()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return loss

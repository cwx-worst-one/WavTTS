import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ..requires.w2v.w2v_infer import w2v_bert_tokenization


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
        self.codec_token_num = self.extra_params.quant_token_num
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
        with self.profiler.profile("[LightningModule]SemanticModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, input_tokens = self.prepare_feature(batch.float())
                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]SemanticModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            logits = self.model(input_tokens)["logits"]

        loss_offset = mulan_tokens.size(1)
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100

        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        # prepare mulan tokens
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        cfg_rate = self.extra_params.get("cfg_rate", 0)
        if cfg_rate > 0:
            rand_mulan = torch.nn.functional.normalize(
                torch.rand_like(mulan_embeds), dim=-1, p=2
            )
            rand_rate = torch.rand(size=[b, 1, 1], device=device)
            mulan_embeds = torch.where(rand_rate > cfg_rate, mulan_embeds, rand_mulan)

        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, 12]
        mulan_tokens = mulan_tokens + 1024 + 1  # offset: semantic 1024 + EOS 1
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens
                + torch.arange(mulan_tokens.size(1)).to(device) * self.mulan_token_num
            )

        eos_ids = torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024
        input_tokens = torch.cat([mulan_tokens, eos_ids, w2v_tokens], dim=1)

        return mulan_tokens, input_tokens


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
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, w2v_tokens, input_tokens = self.prepare_feature(
                    batch.float()
                )
                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            logits = self.model(input_tokens)["logits"]

        loss_offset = mulan_tokens.size(1) + w2v_tokens.size(1) + 1
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len] - 1024  # remove w2v offset
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        # offset: semantic
        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
            + 1024
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # offset: semantic 1024, num_coarse
        eos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device)
            + 1024
            + num_coarse * self.codec_token_num
        )

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        cfg_rate = self.extra_params.get("cfg_rate", 0)
        if cfg_rate > 0:
            rand_mulan = torch.nn.functional.normalize(
                torch.rand_like(mulan_embeds), dim=-1, p=2
            )
            rand_rate = torch.rand(size=[b, 1, 1], device=device)
            mulan_embeds = torch.where(rand_rate > cfg_rate, mulan_embeds, rand_mulan)

        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        mulan_tokens = (
            mulan_tokens + 1024 + num_coarse * self.codec_token_num + 2
        )  # offset: semantic + coarse + EOS 2
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens
                + torch.arange(mulan_tokens.size(1)).to(device) * self.mulan_token_num
            )

        input_tokens = torch.cat(
            [mulan_tokens, eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1
        )

        return mulan_tokens, w2v_tokens, input_tokens


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
        with self.profiler.profile(
            "[LightningModule]SemanticFreeCoarseModule.prepare_feature"
        ):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, input_tokens = self.prepare_feature(batch.float())

        with self.profiler.profile(
            "[LightningModule]SemanticFreeCoarseModule.model_forward"
        ):
            logits = self.model(input_tokens)

        loss_offset = mulan_tokens.size(1)
        x = logits[:, loss_offset : input_tokens.size(1) - 1, :]
        targets = input_tokens[:, loss_offset + 1 : input_tokens.size(1)]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b = wavs.size(0)

        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * self.codec_token_num
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # offset: num_coarse
        eos_ids = (
            torch.zeros(size=[b, 1], dtype=wav_ids.dtype, device=device)
            + num_coarse * self.codec_token_num
        )

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        mulan_tokens = (
            mulan_tokens + num_coarse * self.codec_token_num + 1
        )  # offset: coarse + EOS
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens
                + torch.arange(mulan_tokens.size(1)).to(device) * self.mulan_token_num
            )

        input_tokens = torch.cat([mulan_tokens, eos_ids, wav_ids], dim=1)

        return mulan_tokens, input_tokens


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
        num_coarse, num_fine, num_res, quant_token_num = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
            self.extra_params.num_res,
            self.extra_params.quant_token_num,
        )
        wav_ids = self.get_quant(wavs)
        b = wav_ids.size(0)
        # get coarse ids
        coarse_ids = (
            wav_ids[:, :, 0:num_coarse]
            + torch.arange(num_coarse).to(device) * quant_token_num
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            wav_ids[:, :, num_coarse : num_coarse + num_fine]
            + (torch.arange(num_fine).to(device) + num_coarse) * quant_token_num
        ).reshape((b, -1))
        # get eos id
        eos_id = num_res * self.codec_token_num
        eos_ids = torch.zeros([b, 1], dtype=wav_ids.dtype, device=device) + eos_id
        # final input tokens
        input_tokens = torch.cat([coarse_ids, eos_ids, fine_ids], dim=1)
        return coarse_ids, input_tokens

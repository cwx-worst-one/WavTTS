import pytorch_lightning as pl
import torch
import torch.nn
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ...requires.mulan.mulan_infer import mulan_inference, mulan_rvq_indexs
from ...requires.w2v.w2v_infer import w2v_bert_tokenization


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        extra_hparams,
        checkpointing=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion_cls = criterion_cls()
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.required_modules = required_modules
        self.requires = {}
        self.extra_hparams = DotDict(extra_hparams)
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters())
        scheduler = self.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    @torch.no_grad()
    def get_mulan_embed(self, x):
        # [b, d]
        return mulan_inference(
            model=self.requires["mulan"], music=x.float(), device=x.device
        )

    @torch.no_grad()
    def get_quant(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_w2v_token(self, x):
        return w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x,
            centers=self.requires["centroids"],
            device=x.device,
        )

    @torch.no_grad()
    def prepare_feature(self, wavs):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()


class SemanticModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        extra_hparams,
        checkpointing=False,
    ):
        super().__init__(
            model_cls,
            criterion_cls,
            optimizer_cls,
            scheduler_cls,
            required_modules,
            extra_hparams,
            checkpointing,
        )

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]SemanticModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                input_tokens, target_tokens = self.prepare_feature(batch.float())
        with self.profiler.profile("[LightningModule]SemanticModule.model_forward"):
            logits = self.model(input_tokens[..., :-1])["logits"]
        if self.local_rank == 0:
            print("Semantic Training...")
        x = logits[:, -target_tokens.size(1) :, :]
        y = target_tokens
        loss = self.criterion_cls(x, y)
        acc = 100 * (x.argmax(dim=-1) == y).float().mean()
        self.log_dict({"ce": loss, "acc": acc}, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def prepare_feature(self, wavs):
        """
        mulan_tokens:   [0, 12288 (1024 * 12))
        w2v_tokens:     [12288, 13312 (12288 + 1024))
        eos_ids_0:      13312
        """
        device = wavs.device
        b = wavs.size(0)

        # preparation for mulan
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, d]
        mulan_tokens, ds = mulan_rvq_indexs(
            mulan_embeds, self.requires["mulan_centers"]
        )  # [b, 12]
        mulan_tokens = mulan_tokens + torch.arange(12).to(device) * 1024

        # preparation for w2v
        w2v_tokens = self.get_w2v_token(wavs)
        target_tokens = w2v_tokens
        # offset by mulan
        w2v_tokens = w2v_tokens + 1024 * 12

        # preparation for eos
        eos_id_0 = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024 * 13
        )

        input_tokens = torch.cat([mulan_tokens, eos_id_0, w2v_tokens], dim=1)

        return input_tokens, target_tokens


class CoarseModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        extra_hparams,
        checkpointing=False,
    ):
        super().__init__(
            model_cls,
            criterion_cls,
            optimizer_cls,
            scheduler_cls,
            required_modules,
            extra_hparams,
            checkpointing,
        )

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                input_tokens, target_tokens = self.prepare_feature(batch.float())
        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            logits = self.model(input_tokens[..., :-1])["logits"]

        if self.local_rank == 0:
            print("Coarse Training...")
        x = logits[:, -target_tokens.size(1) :, :]
        y = target_tokens
        loss = self.criterion_cls(x, y)
        acc = 100 * (x.argmax(dim=-1) == y).float().mean()
        self.log_dict({"ce": loss, "acc": acc}, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def prepare_feature(self, wavs):
        """
        mulan_tokens:   [0, 12288 (1024 * 12))
        w2v_tokens:     [12288, 13312 (12288 + 1024))
        coarse_tokens:  [13312, 17408 (13312 + 1024 * 4))
        eos_ids_0:      17408
        eos_ids_1:      17409
        """
        device = wavs.device
        b = wavs.size(0)
        num_coarse = self.extra_hparams.num_coarse

        # preparation for mulan
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, d]
        mulan_tokens, ds = mulan_rvq_indexs(
            mulan_embeds, self.requires["mulan_centers"]
        )  # [b, 12]
        mulan_tokens = mulan_tokens + torch.arange(12).to(device) * 1024

        # preparation for w2v
        w2v_tokens = self.get_w2v_token(wavs)
        # offset by mulan
        w2v_tokens = w2v_tokens + 1024 * 12

        # preparation for coarse
        wav_ids = self.get_quant(wavs)[:, :, 0:num_coarse]
        target_tokens = torch.reshape(wav_ids, [b, -1])
        wav_ids = wav_ids + torch.arange(num_coarse).to(device) * 1024
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, steps * quantizers]
        # offset by mulan and w2v
        wav_ids = wav_ids + 1024 * 13

        # preparation for eos
        eos_id_0 = (
            torch.zeros(size=[b, 1], dtype=wav_ids.dtype, device=device) + 1024 * 17
        )
        eos_id_1 = eos_id_0 + 1

        input_tokens = torch.cat(
            [mulan_tokens, eos_id_0, w2v_tokens, eos_id_1, wav_ids], dim=1
        )

        return input_tokens, target_tokens


class FineModule(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        extra_hparams,
        checkpointing=False,
    ):
        super().__init__(
            model_cls,
            criterion_cls,
            optimizer_cls,
            scheduler_cls,
            required_modules,
            extra_hparams,
            checkpointing,
        )

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                input_tokens, target_tokens = self.prepare_feature(batch.float())
        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            logits = self.model(input_tokens[..., :-1])["logits"]

        if self.local_rank == 0:
            print("Fine Training...")
        x = logits[:, -target_tokens.size(1) :, :]
        y = target_tokens
        loss = self.criterion_cls(x, y)
        acc = 100 * (x.argmax(dim=-1) == y).float().mean()
        self.log_dict({"ce": loss, "acc": acc}, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def prepare_feature(self, wavs):
        """
        coarse_tokens:  [0, 4096 (1024 * 4))
        fine_tokens:    [4096, 12288 (4096 + 1024 * 8))
        eos_ids_0:      12288
        """
        device = wavs.device
        b = wavs.size(0)
        num_res = self.extra_hparams.num_res
        num_coarse = self.extra_hparams.num_coarse
        num_fine = self.extra_hparams.num_fine

        # preparation for acoustic tokens
        wav_ids = self.get_quant(wavs)
        target_tokens = torch.reshape(
            wav_ids[:, :, num_coarse : num_coarse + num_fine], [b, -1]
        )
        wav_ids = wav_ids + torch.arange(num_res).to(device) * 1024
        coarse_tokens = wav_ids[:, :, 0:num_coarse]
        fine_tokens = wav_ids[:, :, num_coarse : num_coarse + num_fine]
        coarse_tokens = torch.reshape(coarse_tokens, [b, -1])
        fine_tokens = torch.reshape(fine_tokens, [b, -1])

        # preparation for eos
        eos_id_0 = (
            torch.zeros(size=[b, 1], dtype=wav_ids.dtype, device=device)
            + 1024 * num_res
        )

        input_tokens = torch.cat([coarse_tokens, eos_id_0, fine_tokens], dim=1)

        return input_tokens, target_tokens

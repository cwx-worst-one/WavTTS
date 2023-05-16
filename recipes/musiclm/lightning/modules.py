import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from samantha.utils.hparams import DotDict

from recipes.musiclm.models.compat.semantic_model import w2v_bert_tokenization
from ..inference.utils import sample


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
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def seer_rearrange(self, seq, inv=False):
        b = seq.size(0)
        if inv:
            return seq.reshape(b, -1, self.extra_params.n_seers).transpose(2, 1).reshape(b, -1)
        else:
            return seq.reshape(b, self.extra_params.n_seers, -1).transpose(2, 1).reshape(b, -1)

    def _shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            input_ids, target_ids = self.prepare_feature(batch[0].squeeze(1).float())
        logits = self.model(input_ids)

        x = logits[:, -target_ids.size(1) :, :]
        loss = self.criterion(x, target_ids)
        accu = (x.argmax(dim=-1) == target_ids).float().mean() * 100
        return loss, accu

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch)
        self.log_dict({"tr_loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
                print(f"Skip weight decay: {name}")
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
    def get_mulan_tokens(self, x):
        mulan_embeds = self.requires["mulan_infer_fn"](
            model=self.requires["mulan"], music=x.float(), device=x.device
        )
        mulan_tokens, _ = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )
        return mulan_tokens

    @torch.no_grad()
    def get_soundstream_tokens(self, x):
        output = self.requires["ss"](x.float())[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_wav2vec_tokens(self, x):
        wav2vec_tokens = w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x.float(),
            centers=self.requires["centroids"],
            device=x.device,
        )
        return wav2vec_tokens


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
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        wav2vec_ids = self.get_wav2vec_tokens(wavs)

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = mulan_ids + self.extra_params.wav2vec_codebook_size

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + self.extra_params.wav2vec_codebook_size
        )
        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )

        input_ids = torch.cat(
            [mulan_ids, sos_ids, wav2vec_ids[:, : -1]], dim=1
        )
        return input_ids, wav2vec_ids


class SeerSemanticModule(BaseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()

        wav2vec_ids = self.get_wav2vec_tokens(wavs)
        wav2vec_ids = self.seer_rearrange(wav2vec_ids)

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = mulan_ids + self.extra_params.wav2vec_codebook_size

        seer_ids = torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
        seer_ids = seer_ids + self.extra_params.wav2vec_codebook_size
    
        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")
        if wav2vec_ids.size(1) % self.extra_params.n_seers != 0:
            raise ValueError("Invalid number of seers.")

        input_ids = torch.cat([mulan_ids, seer_ids, wav2vec_ids[:, : -self.extra_params.n_seers]], dim=1)
        return input_ids, wav2vec_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, sample_len=250, temp=0.9, sample_mode="gumbel"
    ):
        if sample_len % self.model.config.n_seers != 0:
            raise ValueError("Invalid sample length.")
        device = mulan_ids.device
        b, _ = mulan_ids.size()
        mulan_ids = (
            mulan_ids
            + self.extra_params.wav2vec_codebook_size
        )
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + self.extra_params.wav2vec_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")

        input_ids = torch.cat([mulan_ids, seer_ids], dim=1)
        kv_cache = {}
        pbar = tqdm(range(math.ceil(sample_len / self.extra_params.n_seers)))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerSemantic")
            predict_logits = self.model(input_ids, kv_cache=kv_cache)
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            input_ids = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        return self.seer_rearrange(coarse_samples, inv=True)


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
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        wav2vec_ids = self.get_wav2vec_tokens(wavs)
        wav2vec_ids = wav2vec_ids + num_coarse * self.extra_params.soundstream_codebook_size

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + self.extra_params.wav2vec_codebook_size
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=soundstream_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + 1
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )

        input_ids = torch.cat(
            [mulan_ids, sep_ids, wav2vec_ids, sos_ids, soundstream_ids[:, : -1]], dim=1
        )
        return input_ids, soundstream_ids


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
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = mulan_ids + num_coarse * self.extra_params.soundstream_codebook_size

        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
            )

        input_ids = torch.cat([mulan_ids, sos_ids, soundstream_ids[:, :-1]], dim=1)
        return input_ids, soundstream_ids

    @torch.no_grad()
    def predict(self, mulan_ids, sample_len=2000, temp=0.9, sample_mode="naive"):
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()

        mulan_ids = mulan_ids + num_coarse * self.extra_params.soundstream_codebook_size
        sos_ids = (
            torch.zeros(size=[b, 1], dtype=mulan_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            sos_ids = (
                sos_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            sos_ids = (
                sos_ids
                + self.extra_params.mulan_codebook_size
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
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        b, _ = wavs.size()

        soundstream_ids = self.get_soundstream_tokens(wavs)
        soundstream_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + torch.arange(num_coarse, device=device) * self.extra_params.soundstream_codebook_size
        )
        soundstream_ids = torch.reshape(soundstream_ids, [b, -1])
        soundstream_ids = self.seer_rearrange(soundstream_ids)

        wav2vec_ids = (
            self.get_wav2vec_tokens(wavs)
            + num_coarse * self.extra_params.soundstream_codebook_size
        )

        mulan_ids = self.get_mulan_tokens(wavs)
        mulan_ids = (
            mulan_ids
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )

        sep_ids = (
            torch.zeros(size=[b, 1], dtype=wav2vec_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.n_seers
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) + sep_ids.size(1) + wav2vec_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")
        if soundstream_ids.size(1) % self.extra_params.n_seers != 0:
            raise ValueError("Invalid length.")

        input_ids = torch.cat(
            [mulan_ids, sep_ids, wav2vec_ids, seer_ids, soundstream_ids[:, : -self.extra_params.n_seers]], dim=1
        )
        return input_ids, soundstream_ids

    @torch.no_grad()
    def predict(
        self, mulan_ids, wav2vec_ids, sample_len=2000, temp=0.9, sample_mode="naive"
    ):
        if sample_len % self.extra_params.n_seers != 0:
            raise ValueError("Invalid sample length.")
        device = mulan_ids.device
        num_coarse = self.extra_params.num_coarse
        b, _ = mulan_ids.size()
        wav2vec_ids = wav2vec_ids + num_coarse * self.extra_params.soundstream_codebook_size
        mulan_ids = (
            mulan_ids
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
        )
        sep_ids = (
            torch.zeros(size=[b, 1], dtype=wav2vec_ids.dtype, device=device)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + self.extra_params.wav2vec_codebook_size
            + self.extra_params.n_seers
        )

        if not self.extra_params.mulan_codebook_shared:
            mulan_ids = (
                mulan_ids
                + torch.arange(mulan_ids.size(1), device=device) * self.extra_params.mulan_codebook_size
            )
            seer_ids = (
                seer_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + mulan_ids.size(1) * self.extra_params.mulan_codebook_size
            )
        else:
            seer_ids = (
                seer_ids
                + self.extra_params.mulan_codebook_size
            )
            sep_ids = (
                sep_ids
                + self.extra_params.mulan_codebook_size
            )

        if mulan_ids.size(1) + sep_ids.size(1) + wav2vec_ids.size(1) != self.model.config.n_priors:
            raise ValueError("Invalid prefix length.")

        input_ids = torch.cat([mulan_ids, sep_ids, wav2vec_ids, seer_ids], dim=1)
        self.model.transformer.init_cache()
        pbar = tqdm(range(math.ceil(sample_len / self.extra_params.n_seers)))
        coarse_samples = None
        for i in pbar:
            pbar.set_description("SeerCoarse")
            logits = self.model(input_ids)
            layer_idx = i % num_coarse
            predict_logits = logits[
                :, -self.extra_params.n_seers :, layer_idx * 1024 : (layer_idx + 1) * 1024
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
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()
        num_coarse, num_fine = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
        )
        soundstream_ids = self.get_soundstream_tokens(wavs)
        
        # get coarse ids
        coarse_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + (torch.arange(num_coarse, device=device) + num_fine) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            soundstream_ids[:, :, num_coarse : num_coarse + num_fine]
            + torch.arange(num_fine, device=device) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get sos id
        sos_ids = (
            torch.zeros([b, 1], dtype=soundstream_ids.dtype, device=device)
            + (num_coarse + num_fine) * self.extra_params.soundstream_codebook_size
        )
        # final input tokens
        input_tokens = torch.cat([coarse_ids, sos_ids, fine_ids[:, : -1]], dim=1)
        return input_tokens, fine_ids


class SeerFineModule(BaseModule):
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
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        b, _ = wavs.size()
        num_coarse, num_fine = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
        )
        soundstream_ids = self.get_soundstream_tokens(wavs)
        
        # get coarse ids
        coarse_ids = (
            soundstream_ids[:, :, 0 : num_coarse]
            + (torch.arange(num_coarse, device=device) + num_fine) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        # get fine ids
        fine_ids = (
            soundstream_ids[:, :, num_coarse : num_coarse + num_fine]
            + torch.arange(num_fine, device=device) * self.extra_params.soundstream_codebook_size
        ).reshape((b, -1))
        fine_ids = self.seer_rearrange(fine_ids)
        # get seer ids
        seer_ids = (
            torch.arange(self.extra_params.n_seers, device=device).expand(b, -1)
            + num_coarse * self.extra_params.soundstream_codebook_size
            + num_fine * self.extra_params.soundstream_codebook_size
        )
        # final input tokens
        input_tokens = torch.cat([coarse_ids, seer_ids, fine_ids[:, : -self.extra_params.n_seers]], dim=1)
        return input_tokens, fine_ids


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

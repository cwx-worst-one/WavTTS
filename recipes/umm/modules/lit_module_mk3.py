import random
from typing import Optional, Union

import pytorch_lightning as pl
import time
import torch
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from recipes.umm.models.utils import clip_grad_value_, mel_spectrogram_torch
from mariana.data.audio.transforms import KaldiFbank
from samantha.utils.hparams import DotDict


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

        if checkpointing:
            self.model.gradient_checkpointing_enable()

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
            del state_dict
            torch.cuda.empty_cache()

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
        mfu = self.flops / elapsed / 312e12
        self.ts_before_forward = time.time()

        loss_dict = self._shared_step(batch)
        if "flops" in loss_dict:
            self.flops = loss_dict["flops"]
            del loss_dict["flops"]
            loss_dict.update({"training/mfu" : mfu})
        loss_dict["training/loss"] = loss_dict["loss"]
        loss_dict["aux/bsz"] = batch["audio"].shape[0]
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
        one_hot = torch.nn.functional.one_hot(
            quant_index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hot = self.all_gather(one_hot)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = one_hot.sum() / quant_token_num
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
        if seqlen % 32 != 0:
            pad_len = (seqlen + 31) // 32 * 32
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
        quant_rate = 0
        for i in range(self.model.config.rq_codebook_num):
            quant_rate += self.get_quant_rate(
                output_dict["rq_target"][:, :, i], self.model.config.rq_codebook_size
            )
        quant_rate = quant_rate / self.model.config.rq_codebook_num
        loss_dict["aux/quant_rate"] = quant_rate

        mel = input_dict["mel"]
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        return loss_dict


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
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

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

        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        return loss_dict


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
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
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
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
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


class USMStage2(Stage0):
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
        # 16k audio -> fbank
        # 16k audio -> mel
        # 16k audio 计算 chroma -> chorma loss
        input_dict = {}
        audio = batch["audio"].squeeze(dim=1).float() # [b, t]
        # pad audio to multiple 32 frames
        pad_len = audio.shape[-1]  % (self.extra_params.hop_length * 32)
        if pad_len != 0:
            pad_len = self.extra_params.hop_length * 32 - pad_len
        audio = F.pad(audio, (0, pad_len))
        feature = self.preprocessing(audio)
        input_dict.update(feature)
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch) # fbank, mel, chroma
        input_dict.update(text_ids=batch["token"]) # text_ids
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]
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

        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        return loss_dict

    def extract_features(self, wav, dtype=torch.float32):
        # wav: 16k hz, shape=[b, t]
        # pad wav to multiple 32 frames
        pad_len = wav.shape[-1]  % (self.extra_params.hop_length * 32)
        if pad_len != 0:
            pad_len = self.extra_params.hop_length * 32 - pad_len
        wav = F.pad(wav, (0, pad_len))
        return self.model.extract_features(wav, dtype=dtype)


class USMStage3(USMStage2):
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
        input_dict = self.prepare_feature(batch) # fbank, mel, chroma
        input_dict.update(text_ids=batch["token"]) # text_ids
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

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
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
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
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        return loss_dict

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
    def wav2token(self, wav):
        return self.model.wav2token(wav)

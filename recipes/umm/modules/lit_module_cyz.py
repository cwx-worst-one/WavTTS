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
        f0 = batch["f0"].squeeze(dim=1).float() # [b, t]
        vuv = batch["vuv"].squeeze(dim=1).float() # [b, t]
        # pad audio to multiple 32 frames
        pad_len = audio.shape[-1]  % (self.extra_params.hop_length * 32)
        if pad_len != 0:
            pad_len = self.extra_params.hop_length * 32 - pad_len
        audio = F.pad(audio, (0, pad_len))
        feature = self.preprocessing(audio)
        input_dict.update(feature)
        f0_pad_len = feature['fbank'].shape[1] - f0.shape[1]
        f0 = F.pad(f0, (0, f0_pad_len))
        vuv = F.pad(vuv, (0, f0_pad_len))
        input_dict["f0"] = f0
        input_dict["vuv"] = vuv
        return input_dict

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch) # fbank, mel, chroma
        input_dict.update(text_ids=batch["token"]) # text_ids
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]
        f0 = input_dict["f0"]
        vuv = input_dict["vuv"]


        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=text_ids,
            recon_mel=output_dict["mel_out"],
            mel=mel,
            recon_f0=output_dict["f0_out"].squeeze(-1),
            f0=f0,
            recon_vuv=output_dict["vuv_out"].squeeze(-1),
            vuv=vuv,
        )
        loss_dict["loss"] = (
            loss_dict["loss_mel"] * self.model.config.w_loss_mel
            + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
        )
        loss_dict["loss"] = (
            loss_dict["loss"]
            + (loss_dict["f0_loss"]+loss_dict["vuv_loss"]) * self.model.config.w_loss_f0
        )

        loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        loss_dict["aux/w_loss_f0"] = self.model.config.w_loss_f0
        return loss_dict

    def extract_features(self, wav, dtype=torch.float32):
        # wav: 16k hz, shape=[b, t]
        # pad wav to multiple 32 frames
        pad_len = wav.shape[-1] % (self.extra_params.hop_length * 32)
        if pad_len != 0:
            pad_len = self.extra_params.hop_length * 32 - pad_len
        wav = F.pad(wav, (0, pad_len))
        return self.model.extract_features(wav, dtype=dtype)

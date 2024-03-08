import random
import time
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn.functional as F
from pytorch_lightning.profilers import PassThroughProfiler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer

from samantha.criterion.criterion_vocoder import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from samantha.dataio.webdataset import ShardWriter
from samantha.models.ctiga import gpt
from samantha.models.utils import clip_grad_value_, mel_spectrogram_torch
from samantha.utils.ctiga.inference_params import InferenceParams
from samantha.utils.hparams import DotDict


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
        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        device_name = torch.cuda.get_device_name()
        if "A100" in device_name or "A800" in device_name:
            self.device_FLOPS = 312e12
        elif "H100" in device_name or "H800" in device_name:
            self.device_FLOPS = 989e12
        elif "V100" in device_name:
            self.device_FLOPS = 125e12
        else:
            raise RuntimeError("unknow cuda device name: ", device_name)

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

        if "wvae_encoder" in self.hparams.required_modules:
            wvae_encoder = self.hparams.required_modules["wvae_encoder"]
            self.requires.update(
                wvae_encoder["init_fn"](
                    wvae_encoder["ckpt_path"],
                    self.local_rank,
                    wvae_encoder["cache_dir"],
                )
            )

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
        if self.flops is not None and self.flops > 0:
            mfu = self.flops / elapsed / self.device_FLOPS
            self.log_dict_cached({"training/mfu": mfu})
        self.log_dict_cached(
            {
                # FIXME: The below two should really be "max" or "use rank 0".
                "training/mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "training/malloc_retries": torch.cuda.memory_stats()[
                    "num_alloc_retries"
                ],
            }
        )

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
        self.log_dict(
            prev_log_dict, prog_bar=True, sync_dist=False, rank_zero_only=True
        )
        del pending_deletion
        del prev_log_dict

        if "flops" in loss_dict:
            self.flops = loss_dict["flops"]
            del loss_dict["flops"]

        loss_dict["training/loss"] = loss_dict["loss"]
        self.log_dict_cached(loss_dict)

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
                    k = f"val_{dataloader_idx}/{k}"
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
        self.cached_log_dict.update(
            {k: v if v is not torch.Tensor else v.detach() for k, v in kvs.items()}
        )

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
            [torch.tensor(cpu_values, dtype=torch.float, device="cuda")] + cuda_values
        )
        # Support different collectives is just a matter of gathering metrics
        # to rank 0 and reducing them locally.
        torch.distributed.reduce(values, 0, op=torch.distributed.ReduceOp.AVG)

        res_dict = {orig_keys[i]: values[i] for i in range(len(values))}
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
        mel_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(
                self.model.config.len_masking_token * 4, dim=1
            )
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
        return mx, token_domain_masked_indices, mel_domain_masked_indices

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        wav = batch["audio"].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        input_dict = self.preprocessing(wav)
        mel = input_dict["mel"]
        masked_audio, masked_indices, masked_mel_indices = self.masking(wav)
        if self.model.config.get("mask_mel", False):
            masked_mel = mel.clone()
            mel_noise = 0.1 * torch.randn(
                [len(masked_mel_indices), mel.shape[-1]],
                dtype=masked_mel.dtype,
                device=masked_mel.device,
            )
            masked_mel[tuple(masked_mel_indices.t())] = mel_noise
        else:
            masked_mel = self.preprocessing(masked_audio)["mel"]
        input_dict.update({"masked_mel": masked_mel, "masked_indices": masked_indices})
        return input_dict

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
        quant_rate = (
            self.get_quant_rates(
                [
                    output_dict["rq_target"][:, :, i]
                    for i in range(self.model.config.rq_codebook_num)
                ],
                self.model.config.rq_codebook_size,
            )
            / self.model.config.rq_codebook_num
        )
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
        config = self.model.config
        input_dict = {}
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        if config.get("add_ctc", True) or config.get("add_las", False):
            input_dict.update(text_ids=batch["token"])
        feature = self.preprocessing(audio)
        input_dict.update(feature)
        if config.get("add_kl", False):
            audio = F.pad(audio, (0, (600 - audio.shape[-1] % 600) % 600))
            mean, logs = self.requires["wvae_encoder"](audio)
            input_dict["mean"] = mean.transpose(1, 2)
            input_dict["logs"] = logs.transpose(1, 2)
        return input_dict

    def _shared_step(self, batch):
        config = self.model.config
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        if config.get("add_ctc", True) or config.get("add_las", False):
            text_ids = input_dict["text_ids"]
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"] if config.get("add_ctc", True) else None,
            text_ids=text_ids if config.get("add_ctc", True) else None,
            recon_mel=output_dict["mel_out"] if config.get("add_mel", True) else None,
            mel=mel if config.get("add_mel", True) else None,
            recon_chroma=output_dict["chroma_out"]
            if config.get("add_chroma", True)
            else None,
            chroma=input_dict["chroma"] if config.get("add_chroma", True) else None,
            recon_f0=output_dict["f0_out"].squeeze(-1)
            if config.get("add_pitch", False)
            else None,
            f0=input_dict["f0"] if config.get("add_pitch", False) else None,
            recon_vuv=output_dict["vuv_out"].squeeze(-1)
            if config.get("add_pitch", False)
            else None,
            vuv=input_dict["vuv"] if config.get("add_pitch", False) else None,
            las_logits=output_dict["las_out"] if config.get("add_las", False) else None,
            las_targets=output_dict["las_targets"]
            if config.get("add_las", False)
            else None,
        )
        loss_dict["loss"] = 0.0
        if config.get("add_mel", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_mel"] * config.w_loss_mel
            )
        if config.get("add_ctc", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_ctc"] * config.w_loss_ctc
            )
        if config.get("add_chroma", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_chroma"] * config.w_loss_chroma
            )
        if config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"]) * config.w_loss_pitch
            )
        if config.get("add_kl", False):
            loss_dict["kl_loss"] = output_dict["kl_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["kl_loss"] * config.w_loss_kl
            )
        if config.get("add_las", False):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_las"] * config.w_loss_las
            )
        # logging
        if config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        if config.get("add_mel", True):
            loss_dict["aux/w_loss_mel"] = config.w_loss_mel
        if config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = config.w_loss_ctc
        if config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = config.w_loss_chroma
        if config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = config.w_loss_pitch
        if config.get("add_kl", False):
            loss_dict["aux/w_loss_kl"] = config.w_loss_kl
        if config.get("add_las", False):
            loss_dict["aux/w_loss_las"] = config.w_loss_las
        if "flops" in output_dict:
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
        config = self.model.config
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        if config.get("add_ctc", True) or config.get("add_las", False):
            text_ids = input_dict["text_ids"]
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"] if config.get("add_ctc", True) else None,
            text_ids=text_ids if config.get("add_ctc", True) else None,
            recon_mel=output_dict["mel_out"] if config.get("add_mel", True) else None,
            mel=mel if config.get("add_mel", True) else None,
            recon_chroma=output_dict["chroma_out"]
            if config.get("add_chroma", True)
            else None,
            chroma=input_dict["chroma"] if config.get("add_chroma", True) else None,
            recon_f0=output_dict["f0_out"].squeeze(-1)
            if config.get("add_pitch", False)
            else None,
            f0=input_dict["f0"] if config.get("add_pitch", False) else None,
            recon_vuv=output_dict["vuv_out"].squeeze(-1)
            if config.get("add_pitch", False)
            else None,
            vuv=input_dict["vuv"] if config.get("add_pitch", False) else None,
            las_logits=output_dict["las_out"] if config.get("add_las", False) else None,
            las_targets=output_dict["las_targets"]
            if config.get("add_las", False)
            else None,
        )
        loss_dict["loss"] = 0.0
        if config.get("add_mel", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_mel"] * config.w_loss_mel
            )
        if config.get("add_ctc", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_ctc"] * config.w_loss_ctc
            )
        if config.get("add_chroma", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_chroma"] * config.w_loss_chroma
            )
        if config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"]) * config.w_loss_pitch
            )
        if config.get("add_kl", False):
            loss_dict["kl_loss"] = output_dict["kl_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["kl_loss"] * config.w_loss_kl
            )
        if config.get("add_las", False):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_las"] * config.w_loss_las
            )
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * config.w_loss_vq
            )
        # logging
        # code rate & quant rate & entropy
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rate(
            output_dict["vq_ids"].long(), config.vq_codebook_size
        )
        loss_dict["aux/quant_rate"] = quant_rate
        if getattr(self.model.vq, "entropy", None) is not None:
            loss_dict["aux/entropy"] = self.model.vq.entropy()
        # ctc, mel, chroma, pitch, kl, las, vq
        if config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        if config.get("add_mel", True):
            loss_dict["aux/w_loss_mel"] = config.w_loss_mel
        if config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = config.w_loss_ctc
        if config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = config.w_loss_chroma
        if config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = config.w_loss_pitch
        if config.get("add_kl", False):
            loss_dict["aux/w_loss_kl"] = config.w_loss_kl
        if config.get("add_las", False):
            loss_dict["aux/w_loss_las"] = config.w_loss_las
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        if "flops" in output_dict:
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
            if not hasattr(self.model.vq_proj_in, "__len__"):
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
            if "vq.embedding.weight" in name:
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

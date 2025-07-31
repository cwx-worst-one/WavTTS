import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import wandb
import logging
import soundfile as sf
from recipes.musiclm.utils.dist import local_zero_first, is_local_zero
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info
from samantha.utils.utils import download_checkpoint
from samantha.utils.hparams import DotDict
# from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric
from samantha.utils.flops_profiler import FlopsProfiler
from recipes.voicebox.modules.loss import sequence_mask
from recipes.voicebox.modules.loss import MaskedMAELoss, MaskedBCELoss, MaskedSSIMLoss, MaskedMSELoss
from recipes.voicebox.modules.loss import MaskedMSEWoReductionLoss
from recipes.voicebox.lit_modules.utils import log_audio, plot_mel
from recipes.sacodec.modules.sacodec_module_umm import SACodecModule as SACodecModuleUMM
from recipes.sacodec.modules.sacodec_module import SACodecModule
import random

from pathlib import Path
import torchaudio
import librosa

logger = logging.getLogger(__name__)

LOSS_DICT = {
        "l1": MaskedMAELoss,
        "l2": MaskedMSELoss,
        "l2_wo_reduction": MaskedMSEWoReductionLoss,
        "ssim": MaskedSSIMLoss
        }

def normalize_audio(t: torch.Tensor, eps=1e-8):
    return t / t.abs().amax(dim=(1, 2), keepdim=True).clamp(eps)

def get_max_scale_audio(t: torch.Tensor):
    t_max = t.abs().amax(dim=(1, 2), keepdim=True)
    t_scale = torch.ones_like(t_max)
    t_scale[t_max > 0.9] = 0.6 + random.random() * 0.3
    return t_scale
    
def init_latent2wav(*args, **kwargs): return init_sacodec(*args, **kwargs) # support for legacy renamed function
def init_sacodec(checkpoint_path, local_rank, cache_dir=None, version="umm"):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        if version == "umm":
            module_cls = SACodecModuleUMM
        elif version == "conv":
            module_cls = SACodecModule
        else:
            raise Exception("sacodec version not handled")
    
        # vocoder_model = vocoder_model_pl.generator.eval().to(device)
        sacodec_model = module_cls.load_from_checkpoint(
            checkpoint_path=local_path,
            strict=False,
            map_location="cpu"
        ).eval().to(device)
        sacodec_model.setup('predict')

        return {
            "sacodec": sacodec_model, 
        }


class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        required_modules,
        criterions,
        checkpointing=True,
        resume_ckpt_path = None,
        umm_dropout = 0.0,
        umm_perturb_p = 0.0,
        umm_pad=16384,
        umm_hz=25,
        codebook_tie_flag=False,
        bn_hz=50,
        diffusion_sample_rate=44100,
        val_output_samples_dir="",
        bn_config=None,
        optimizer_cls=None,
        scheduler_cls=None,
        normalize_audio=0,
        dpo_train=False,
        dpo_loss_type='sigmoid',
        dpo_beta=1.0,
        dpo_condition_sync=True,
        umm_slice_pct=0.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()

        self.criterion_dict = {}
        for criterion in criterions:
            self.criterion_dict[criterion] = LOSS_DICT[criterion]()

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        self.umm_dropout = umm_dropout
        self.umm_perturb_p = umm_perturb_p
        self.umm_pad = umm_pad
        self.umm_hz = umm_hz
        self.codebook_tie_flag = codebook_tie_flag
        self.bn_hz = bn_hz
        self.diffusion_sample_rate = diffusion_sample_rate
        self.bn_config = bn_config 

        self.normalize_audio = normalize_audio
        self.umm_slice_pct = umm_slice_pct
        self.dpo_train = dpo_train
        self.dpo_loss_type = dpo_loss_type
        self.dpo_beta = dpo_beta
        self.dpo_condition_sync = dpo_condition_sync

        self.resume_ckpt_path = resume_ckpt_path

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        if stage == "fit" and self.resume_ckpt_path is not None:
            self.load_from_pretrained(self.resume_ckpt_path)

        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )
        self.load_required_modules()
        if self.codebook_tie_flag:
            # replace token_embedding with codebook token_embedding, and freeze codebook
            Stage3 = self.requires["Stage3"].model
            assert isinstance(self.model.token_embedding, nn.ModuleList), "token_embedding must be ModuleList"

            tokenizer_codebook = [Stage3.stages[0].insert_modules[0].rvq.RVQ[i].embedding.weight
                                for i in range(self.model.hp.n_token_hierarchy)]
            for i in range(len(self.model.token_embedding)):
                tokenizer_codebook_vocab_size = tokenizer_codebook[i].shape[0]
                # [codebook_size, token_embed_dim]
                self.model.token_embedding[i].weight.data[:tokenizer_codebook_vocab_size].copy_(tokenizer_codebook[i])
                self.model.token_embedding[i].weight.data[tokenizer_codebook_vocab_size:].fill_(0)
                self.model.token_embedding[i].weight.requires_grad = False
            print("Initialized codebook by tokenizer codebook.")

    def load_required_modules(self, ignore=()):
        for name, item in self.hparams.required_modules.items():
            if name in ignore: 
                print("Skipping required module", name)
                continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            else: # to support setting required module to None in config
                print("Skipping required module", name)
                continue
            self.requires.update(initializer(hpath, local_rank=self.local_rank, cache_dir="./"))

    def load_from_pretrained(self, pretrained_path=None):
        rank_zero_info(f'Loading pre-trained model from checkpoint {pretrained_path}')
        with local_zero_first():
            local_path = download_checkpoint(pretrained_path, '.')

            ckpt_state_dict = torch.load(
                local_path, map_location=torch.device("cpu")
            )['state_dict']
            model_state_dict = self.model.state_dict()
            new_state_dict = {}

            for k in ckpt_state_dict:
                new_k = k.replace("model.", "") # saved model has prefix "model."
                if new_k in model_state_dict:
                    if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                        rank_zero_info(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[new_k].shape}, "
                                    f"loaded shape: {ckpt_state_dict[k].shape}")
                    else:
                        new_state_dict[new_k] = ckpt_state_dict[k]
                else:
                    rank_zero_info(f"Dropping parameter {k}")

            self.model.load_state_dict(new_state_dict, strict=False)

    @torch.no_grad()
    def get_umm_token(self, wav, audio_lengths=None, slice_pct=0.0):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            mode = 'even' if np.random.rand() < slice_pct else 'full'
            token = self.requires["Stage3"].wav2requires(wav, audio_length=audio_lengths, slice_method=mode, chunk_size=60, requires=['token'])
            if isinstance(token, dict):
                token = token["token"]
        return token

    @torch.no_grad()
    def get_sacodec_embedding(self, wav):
        # wav = B x CH x L
        model: SACodecModuleUMM = self.requires["sacodec"]
        latents = model.get_latents(wav) # B L D
        latents = model.normalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
        return latents
    
    @torch.no_grad()
    def sacodec_embs_to_wav(self, latents):
        model: SACodecModuleUMM = self.requires["sacodec"]
        latents = model.denormalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
        latents = latents.transpose(1, 2) # B D L -> B L D
        audio_hat = model.decode_latents(latents)
        return audio_hat # B x CH x L



    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)


    def align_umm_and_bn(self, batch):
        # align to mininmal length
        wav = batch["wav"]
        wav_24k = batch["wav_24k"]
        umm_token = batch["token"]
        bn = batch["bn"]
        umm_duration = umm_token.shape[1] / self.umm_hz
        wav_24k_duration = wav_24k.shape[1] / 24000
        wav_duration = wav.shape[-1] / self.diffusion_sample_rate
        bn_duration = bn.shape[1] / self.bn_hz
        
        min_duration = min(umm_duration, bn_duration, wav_24k_duration, wav_duration)
        umm_token = umm_token[:, :int(min_duration * self.umm_hz), ...]
        bn = bn[:, :int(min_duration * self.bn_hz), ...]
        batch["token"] = umm_token
        batch["bn"] = bn
        return batch

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        if self.normalize_audio != 0 and "token" in batch:
            raise Exception("normalize_audio is not supported when training on pre-extracted UMM features. Either use OTF training or normalize_audio=0")
        if self.normalize_audio == 1:
            raise DeprecationWarning("normalize_audio=1 has been deprecated")
        elif self.normalize_audio == 2:
            prob = random.random()
            if prob < 0.80:
                scale_val = get_max_scale_audio(batch["wav"])
                batch["wav_24k"] = batch["wav_24k"] * scale_val.squeeze(1)
                batch["wav"] = batch["wav"] * scale_val
        elif self.normalize_audio == 3:
            if random.random() < 0.75:
                batch["wav_24k"] = normalize_audio(batch["wav_24k"].unsqueeze(1)).squeeze(1)
                batch["wav"] = normalize_audio(batch["wav"])
                if random.random() < 0.8:
                    gain_db = random.randint(-6, -1)
                    batch["wav_24k"] = torchaudio.functional.gain(batch["wav_24k"], gain_db=gain_db)
                    batch["wav"] = torchaudio.functional.gain(batch["wav"], gain_db=gain_db)
        elif self.normalize_audio == 4: # increase umm volume. decrease wav volume
            batch["wav_24k"] = batch["wav_24k"] * 1.05
            batch["wav"] = batch["wav"]
        elif self.normalize_audio == 5: # random change and clip umm volume.
            gain_db = random.randint(-6, 6)
            random_clamp = random.uniform(0.99, 1)
            batch["wav_24k"] = torchaudio.functional.gain(batch["wav_24k"], gain_db=gain_db).clamp(-1*random_clamp, random_clamp)
            batch["wav"] = batch["wav"]
        elif self.normalize_audio == 6: # random change and clip umm volume. decrease wav volume -3db
            gain_db = random.randint(-6, 6)
            random_clamp = random.uniform(0.95, 1)
            batch["wav_24k"] = torchaudio.functional.gain(batch["wav_24k"], gain_db=gain_db).clamp(-1*random_clamp, random_clamp)
            batch["wav"] = torchaudio.functional.gain(batch["wav"], gain_db=-3)

        if "token" not in batch:
            wav_24k_lens = batch['wav_lens'] / self.diffusion_sample_rate * 24000
            batch["token"] = self.get_umm_token(batch["wav_24k"].unsqueeze(1), wav_24k_lens.int(), slice_pct=self.umm_slice_pct)
        if self.umm_dropout > 0:
            drop_idx = torch.rand([batch["token"].shape[0],batch["token"].shape[1]]) < self.umm_dropout
            if torch.sum(drop_idx) > 0:
                batch["token"][drop_idx] = self.umm_pad
                # print(f"umm_dropout drop_idx={torch.sum(drop_idx)}/{batch['token'].shape[0]}")
            if self.umm_perturb_p > 0 and self.model.hp.n_token_hierarchy > 1 and torch.sum(drop_idx) < batch["token"].shape[0]:
                for h in range(1, self.model.hp.n_token_hierarchy):
                    print(batch["token"][~drop_idx, ..., h].shape)
                    batch["token"][~drop_idx, ..., h] = self.model.perturb_tokens(batch["token"][~drop_idx, ..., h])

        if self.dpo_train and self.dpo_condition_sync:
            # for DPO training, make sure token condition of win and lose are the same.
            win_token, lose_token = batch["token"].chunk(2)
            batch["token"] = torch.cat([win_token, win_token], dim=0)

        batch["bn"] = self.get_sacodec_embedding(batch["wav"]) # B x L x D
        batch = self.align_umm_and_bn(batch)
        sacodec_lens = (batch["wav_lens"] / self.diffusion_sample_rate * self.bn_hz).round() # TODO: replace magic numbers with config values
        # print('Sacodec feat', batch["bn"].shape, sacodec_lens)
        batch["bn_lens"] = sacodec_lens
        batch["bn_mask"] = sequence_mask(sacodec_lens, max_len=batch["bn"].shape[1], device=sacodec_lens.device)
        # batch["bn_mask"] = sequence_mask(sacodec_lens, mask_len=sacodec_lens.max().item())
        # print('UMM', batch["token"].shape)

        ref = batch["bn"]
        loss_mask = batch["bn_mask"]
        feat_len = batch["bn_lens"]

        bsz, seqlen = ref.shape[0], ref.shape[1]
        batch_tokens = torch.sum(feat_len).item()

        self.model_metric.num_tokens += batch_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.model_metric.update(
                num_tokens=0,
                stage=self.trainer.state.stage,
                model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
            )
        
        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            pred, target = self.model(batch)

            loss_dict = {}
            loss = None
            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(pred, target, loss_mask)
                loss_dict[loss_type] = tmp_loss.sum().item()
                if loss is None:
                    loss = tmp_loss
                else:
                    loss += tmp_loss

            if self.dpo_train:
                loss = loss.sum(dim=list(range(1, len(loss.shape))))
                loss_w, loss_l = loss.chunk(2)
                bsz = loss_w.shape[0]

                with torch.no_grad():
                    ref_preds, ref_target = self.model(batch)
                    ref_loss = None
                    for loss_type, loss_func in self.criterion_dict.items():
                        tmp_loss = loss_func(ref_preds, ref_target, loss_mask)
                        loss_dict["ref_"+loss_type] = tmp_loss.sum().item()
                        if ref_loss is None:
                            ref_loss = tmp_loss
                        else:
                            ref_loss += tmp_loss
                    ref_loss = ref_loss.sum(dim=list(range(1, len(ref_loss.shape))))
                    ref_losses_w, ref_losses_l = ref_loss.detach().chunk(2)

                model_diff = loss_w - loss_l
                ref_diff = ref_losses_w - ref_losses_l
                # Final loss.
                logits = ref_diff - model_diff
                if self.dpo_loss_type == "sigmoid":
                    loss = -1 * F.logsigmoid(self.dpo_beta * logits).mean()
                elif self.dpo_loss_type == "hinge":
                    loss = torch.relu(1 - self.dpo_beta * logits).mean()
                elif self.dpo_loss_type == "ipo":
                    losses = (logits - 1 / (2 * self.dpo_beta)) ** 2
                    loss = losses.mean()
                elif self.dpo_loss_type == "dspo":
                    # https://openreview.net/pdf?id=xyfb9HHvMe
                    scale_term = -0.5 * self.dpo_beta
                    logits = scale_term * (model_diff - ref_diff) # [B]
                    pred2, _ = (pred - ref_preds).chunk(2) # [B, T, D]

                    expand_dims = (1,) * (pred2.ndim - 1)  # (1,1,1)
                    logits_expanded = F.sigmoid(logits).view(-1, *expand_dims)  # [B, 1, 1]
                    ref_losses_expanded = ref_losses_w.view(-1, *[1]*(pred2.ndim-1))  # [B, 1, 1]
                    loss = (ref_losses_expanded - self.dpo_beta * (1 - logits_expanded) * pred2).pow(2)
                    loss = loss.mean(dim=list(range(1, pred2.ndim))).mean()
                else:
                    raise ValueError(f"Unknown loss type {self.dpo_loss_type}")

                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)
                loss_dict["implicit_acc"] = implicit_acc.item()
            
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {
                "loss": loss.item(),
                "bsz": bsz,
                "seqlen": seqlen
                }
            log_dict.update(loss_dict)
            log_dict["training/loss"] = log_dict["loss"]
            metric = self.model_metric.compute(self.trainer.global_step)
            log_dict.update(metric)
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    @torch.no_grad()
    def inference(self, inputs, step):
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    predict = inference


    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # create folder based on current epoch
        if is_local_zero():
            if not hasattr(self, "wandb_logger"):
                wandb_logger = [logger for logger in self.loggers if isinstance(logger, pl.loggers.wandb.WandbLogger)]
                if len(wandb_logger) > 0:
                    wandb_logger = wandb_logger[0]
                else:
                    wandb_logger = None
                self.wandb_logger = wandb_logger
            
            if self.wandb_logger is None:
                os.makedirs(
                    f"{self.hparams.val_output_samples_dir}/{self.global_step}", exist_ok=True
                )
                os.makedirs(
                    f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}", exist_ok=True
                )
        return


    def validation_step(self, inputs, step):
        self.validation_from_dataset(inputs, step)
        # if step > 0: return
        # TODO: add validation loss

    def validation_from_dataset(self, batch, step):
        return
        if step > 1: return

        device_name = torch.cuda.get_device_name(0)
        if "H20" in device_name:
            return
        if self.model.target_type != "rectified-flow":
            return
        diffusion_nfe = 10
        diffusion_sampler = "ddim"
        text_cfg_w = 1.0
        max_val_output_samples_num = 10

        if "token" not in batch:
            wav_24k_lens = batch['wav_lens'] / self.diffusion_sample_rate * 24000
            batch["token"] = self.get_umm_token(batch["wav_24k"].unsqueeze(1), wav_24k_lens.int())
        # print('UMM', batch["token"].shape)

        batch["bn"] = self.get_sacodec_embedding(batch["wav"]) # B x L x D
        batch = self.align_umm_and_bn(batch)
        sacodec_lens = (batch["wav_lens"] / 44100 * 50).round() # TODO: replace magic numbers with config values
        batch["bn_lens"] = sacodec_lens
        batch["bn_mask"] = sequence_mask(sacodec_lens, max_len=batch["bn"].shape[1], device=sacodec_lens.device)
        # batch["bn_mask"] = sequence_mask(sacodec_lens, mask_len=sacodec_lens.max().item())

        def make_cfg_input(inputs):
            if text_cfg_w != 1:
                if inputs["token"].ndim == 2:
                    inputs["token"] = inputs["token"].repeat(2, 1)
                else:
                    inputs["token"] = inputs["token"].repeat(2, 1, 1)
                # inputs["bn"] = inputs["bn"].repeat(2, 1, 1)
                inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
                
            return inputs

        batch_num = batch["bn"].shape[0]
        
        if is_local_zero():
            all_gt_wavs=[]
            all_wavs=[]
            uttids=[]
            for b_idx in range(min(max_val_output_samples_num, batch_num)):
                single_batch_input = {}
                for key in batch:
                    single_batch_input[key] = batch[key][b_idx:b_idx+1]
                
                # print(f"single_batch_input")
                # for key in single_batch_input:
                #     if torch.is_tensor(single_batch_input[key]):
                #         print(f"\t {key} -> shape={single_batch_input[key].shape}")
                #     else:
                #         print(f"\t {key} -> {single_batch_input[key]}")


                single_batch_input["bn_ctx"] = torch.randn_like(single_batch_input["bn"])
                emb = self.model.inference(
                                inputs=make_cfg_input(single_batch_input),
                                timesteps=diffusion_nfe,
                                sampler=diffusion_sampler,
                                text_cfg_w=text_cfg_w)
                gt_emb = single_batch_input["bn"].transpose(1,2) # B L D -> B D L

                # print(f"gt_emb={gt_emb.shape} emb={emb.shape}")
                wavs = self.sacodec_embs_to_wav(emb).squeeze().cpu()
                gt_wavs = self.sacodec_embs_to_wav(gt_emb).squeeze().cpu()

                print(f"gt_wavs={gt_wavs.shape} wavs={wavs.shape}")

                utt_id = single_batch_input['utt_id'][0].replace("\\","")
                uttids.append(utt_id)
                if self.wandb_logger is None:
                    sf.write(
                        f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}/{utt_id}.wav",
                        wavs.T,
                        self.hparams.diffusion_sample_rate,
                    )
                    sf.write(
                        f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}/{utt_id}_gt.wav",
                        gt_wavs.T,
                        self.hparams.diffusion_sample_rate,
                    )
                else:
                    all_gt_wavs.append(gt_wavs.T)
                    all_wavs.append(wavs.T)
            if self.wandb_logger is not None:
                log_audio(
                self.wandb_logger,
                key="val/audio_gt",
                audios=all_gt_wavs, 
                uttids=uttids, 
                sample_rate=self.hparams.diffusion_sample_rate, 
                step=self.global_step, 
                rank=self.local_rank,
            )
            log_audio(
                self.wandb_logger,
                key="val/audio",
                audios=all_wavs, 
                uttids=uttids, 
                sample_rate=self.hparams.diffusion_sample_rate, 
                step=self.global_step, 
                rank=self.local_rank,
            )


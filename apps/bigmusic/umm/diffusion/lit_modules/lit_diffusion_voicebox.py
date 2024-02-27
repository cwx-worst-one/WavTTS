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

from recipes.musiclm.utils.dist import local_zero_first
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.diffusion.utils.utils import download_checkpoint
from samantha.utils.hparams import DotDict
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.criterion.masked_loss import MaskedMAELoss, MaskedMSELoss, MaskedSSIMLoss
from apps.bigtts.umm.diffusion.lit_modules.utils import plot_mel


logger = logging.getLogger(__name__)

LOSS_DICT = {
        "l1": MaskedMAELoss,
        "l2": MaskedMSELoss,
        "ssim": MaskedSSIMLoss
        }



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterions,
        checkpointing=True,
        resume_ckpt_path = None,
        umm_dropout = 0.0,
        umm_pad=16384,
        diffusion_sample_rate=24000,
        val_output_samples_dir="",
        bn_config=None,
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
        self.umm_pad = umm_pad
        self.bn_config = bn_config 

        if resume_ckpt_path is not None:
            self.load_from_pretrained(resume_ckpt_path)

    def setup(self, stage: str) -> None:
        setattr(self.model, "flops_fn", FlopsProfiler(self.model))

        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )
        self.load_required_modules()
        
    def load_required_modules(self, ignore=()):
        for name, item in self.hparams.required_modules.items():
            if name in ignore: continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
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
    def vocoder_embs_to_wav(self, x):
        # input x has shape (b, c, t)
        self.requires['vocoder'].eval()
        wav = self.requires['vocoder'].decode(x)
        return wav.detach().cpu().numpy()


    @torch.no_grad()
    def get_umm_token(self, wav):
        token = self.requires["umm"].wav2token(wav)
        return token

    def get_umm_embedding(self, token):
        embedding = F.embedding(token, 
                torch.cat([self.requires["umm_codebook"], self.model.umm_pad_vector], dim=0))
        return embedding

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        if "phone" in batch:
            batch["frontend"] = {
                "phone": batch["phone"],
                "tone": batch["tone"],
                "word_seg": batch["word_seg"]
                }
            if "lang" in batch:
                batch["frontend"]["lang"] = batch["lang"]

        if self.umm_dropout > 0:
            drop_idx = torch.rand([batch["token"].shape[0],batch["token"].shape[1]]) < self.umm_dropout
            if torch.sum(drop_idx) > 0:
                batch["token"][drop_idx] = self.umm_pad

        if "umm_codebook" in self.requires:
            batch["token"] = self.get_umm_embedding(batch["token"])

        if "mel" in batch:
            ref = batch["mel"]
            feat_len = batch["mel_lens"]
            loss_mask = batch["mel_ctx_mask"]
        else:
            ref = batch["bn"]
            feat_len = batch["bn_lens"]
            loss_mask = batch["bn_ctx_mask"]

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
            loss = 0
            for loss_type, loss_func in self.criterion_dict.items():
                tmp_loss = loss_func(pred, target, loss_mask)
                loss_dict[loss_type] = tmp_loss.item()
                loss += tmp_loss
            
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
    
    def log_mel(self, mels, t):
        for name, mel in mels.items():
            mel = mel.transpose(0,1).cpu().detach().numpy()
            self.loggers[1].log_image(name, [plot_mel(mel, t)])

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
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.global_step}", exist_ok=True
        )
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}", exist_ok=True
        )
        return

    def validation_step(self, inputs, step):
        diffusion_nfe = 10
        diffusion_sampler = "ddim"
        text_cfg_w = 0
        val_output_samples_num = 0
        max_val_output_samples_num = 20
        
        def make_cfg_input(inputs):
            if text_cfg_w != 1:
                # inputs["frontend"]["phone"] = inputs["frontend"]["phone"].repeat(2, 1)
                # inputs["frontend"]["phone"][1, :] = 1
                # inputs["frontend"]["tone"] = inputs["frontend"]["tone"].repeat(2, 1)
                # inputs["frontend"]["tone"][1, :] = 1
                # inputs["frontend"]["word_seg"] = inputs["frontend"]["word_seg"].repeat(2, 1)
                # inputs["frontend"]["word_seg"][1, :] = 1
                # if "lang" in inputs["frontend"]:
                #     inputs["frontend"]["lang"] = inputs["frontend"]["lang"].repeat(2, 1)
                #     inputs["frontend"]["lang"][1, :] = 1

                inputs["token"] = inputs["token"].repeat(2, 1)
                inputs["prompt_bn"] = inputs["prompt_bn"].repeat(2, 1, 1)
                inputs["bn_ctx"] = inputs["bn_ctx"].repeat(2, 1, 1)
                
            return inputs

        batch_num = inputs["bn"].shape[0]
        for b_idx in range(min(max_val_output_samples_num, batch_num)):
            single_batch_input = {}
            for key in inputs:
                single_batch_input[key] = inputs[key][b_idx:b_idx+1]
            
            # print(f"single_batch_input")
            # for key in single_batch_input:
            #     if torch.is_tensor(single_batch_input[key]):
            #         print(f"\t {key} -> shape={single_batch_input[key].shape}")
            #     else:
            #         print(f"\t {key} -> {single_batch_input[key]}")


            single_batch_input["bn_ctx"] = torch.ones_like(single_batch_input["bn_ctx"]) * self.bn_config['bn_padding']
            emb = self.model.inference(
                            inputs=make_cfg_input(single_batch_input),
                            timesteps=diffusion_nfe,
                            sampler=diffusion_sampler,
                            text_cfg_w=text_cfg_w)
            gt_emb = single_batch_input["bn"].transpose(1,2)

            #print(f"gt_emb={gt_emb.shape} emb={emb.shape}")

            emb = emb * self.bn_config['bn_norm_std'] + self.bn_config['bn_norm_mean']
            gt_emb = gt_emb * self.bn_config['bn_norm_std'] + self.bn_config['bn_norm_mean']

            with torch.autocast(device_type="cuda", enabled=False):
                wavs = self.vocoder_embs_to_wav(emb.float()).squeeze()
                gt_wavs = self.vocoder_embs_to_wav(gt_emb.float()).squeeze()

            #print(f"gt_wavs={gt_wavs.shape} wavs={wavs.shape}")

            utt_id = single_batch_input['utt_id'][0].replace("\\","")
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}/{utt_id}.wav",
                wavs,
                self.hparams.diffusion_sample_rate,
            )
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.global_step}/{self.local_rank}/{utt_id}_gt.wav",
                gt_wavs,
                self.hparams.diffusion_sample_rate,
            )




import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import wandb

from samantha.utils.hparams import DotDict
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric
from recipes.voicebox.modules.loss import sequence_mask
from recipes.voicebox.modules.speaker_encoder.speakers import SpeakerEncoder
import logging
from .utils import plot_mel

from .optimal_transport import OTPlanSampler
import inspect

logger = logging.getLogger(__name__)



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterion_cls,
        checkpointing=True,
        sigma_min = 0.1,
        ot_sampler = None,
        flow_matching_type = "cfm",
        use_len_mask = True,
        use_mask_loss = True,
        resume_ckpt_path = None,
        use_speaker_encoder=False,
        speaker_encoder_config=None,
        speaker_encoder_model_pth=None,
        local_speaker_attention=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()

        self.requires = {}

        self.use_speaker_encoder = use_speaker_encoder
        self.speaker_encoder_config = speaker_encoder_config
        if self.use_speaker_encoder:
            self.speaker_encoder = SpeakerEncoder(
                model_path=speaker_encoder_model_pth,
                config_path=speaker_encoder_config
            )
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        if ot_sampler is None:
            self.ot_sampler = None
        elif isinstance(ot_sampler, str):
            if ot_sampler == "exact":
                self.ot_sampler = OTPlanSampler(method="exact", normalize_cost=True)
            elif ot_sampler == "sinkhorn":
                # regularization (2*sigma^2) taken for optimal Schrodinger bridge relationship according to the last row of Table 1 in Ref [1]
                self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2*sigma_min**2)
            else:
                raise NotImplementedError(f"ot_sampler must be one of ['exact', 'sinkhorn'], but got {ot_sampler}")
        else:
            self.ot_sampler = ot_sampler
       
        self.criterion = criterion_cls()

        self.sigma_min = sigma_min
        self.flow_matching_type = flow_matching_type
        # check if flow_matching_type is not one of ['cfm', 'ot-cfm', 'sb-cfm', 'fm']
        assert self.flow_matching_type in ['cfm', 'ot-cfm', 'sb-cfm', 'fm'], \
            f"flow_matching_type must be one of ['cfm', 'ot-cfm', 'sb-cfm', 'fm'], but got {self.flow_matching_type}"
        self.use_len_mask = use_len_mask
        self.use_mask_loss = use_mask_loss
        self.local_speaker_attention = local_speaker_attention

    def load_required_modules(self):
        for name, initializer in self.hparams.required_modules.items():
            self.requires[name] = initializer(rank=self.local_rank)


    def setup(self, stage: str) -> None:
        self.model_metric = ModelMetric(
            precision=self.trainer.precision,
            model_obj_or_objs=self.model,
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                frontend_inputs = {
                        "phone": batch["phone"],
                        "tone": batch["tone"],
                        "word_seg": batch["word_seg"]
                        }
                text_len = batch["text_lens"]
                ref = batch["mel"]
                mel_ctx = batch["mel_ctx"]
                mel_len = batch["mel_lens"]

                if self.use_speaker_encoder:
                    spkenc_wav_16k = batch["spkenc_wav_16k"]

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):

            if self.flow_matching_type != "fm":
                if 'x0' not in batch:
                    # NOTE: x0 cannot necessarily be sampled from N(0, 1), it can be sampled from any distribution for ot-cfm, sb-cfm, and cfm. 
                    # For fm, it has to be sampled from N(0, 1)
                    x0 = torch.randn_like(ref)
                else:
                    x0 = batch['x0']
            else:
                if 'x0' in batch:
                    raise Warning("x0 is not used in fm, but got x0 in egs")
            x1 = ref


            if self.ot_sampler is not None:
                if not (self.flow_matching_type == "ot-cfm" or self.flow_matching_type == "sb-cfm"):
                    raise ValueError(f"ot_sampler can only be applied to ot-cfm or sb-cfm, but got {self.flow_matching_type}")
                x0, x1 = self.ot_sampler.sample_plan(x0, x1)

            # generate random t in [0, 1], B
            t = torch.rand((x1.shape[0], *([1] * (x1.dim() - 1))), device=x1.device)

            # get mu_t and sigma_t according to the Table 1 in Ref [1]
            if self.flow_matching_type == "cfm" or self.flow_matching_type == "ot-cfm":
                mu_t = t * x1 + (1 - t) * x0
                sigma_t = self.sigma_min
            elif self.flow_matching_type == "sb-cfm":
                mu_t = t * x1 + (1 - t) * x0
                sigma_t = torch.sqrt(t * (1 - t)) * self.sigma_min
            elif self.flow_matching_type == "fm":
                mu_t = t * x1
                sigma_t = 1 - (1 - self.sigma_min) * t
            else:
                raise NotImplementedError

            # get the samples at flow step t
            x = mu_t + sigma_t * torch.randn_like(x1)

            # get the vector field u_t(x|z) in eq (6) of Ref [1]
            if self.flow_matching_type == "fm":
                ut = (x1 - (1 - self.sigma_min)*x) / (1-(1-self.sigma_min)*t) # eq (13) of Ref [1]
            elif self.flow_matching_type == "cfm" or self.flow_matching_type == "ot-cfm":
                ut = x1 - x0 # eq (16) of Ref [1]
            elif self.flow_matching_type == "sb-cfm":
                ut = (1 - 2 * t)/(2 * t * (1 - t)) * (x - mu_t) + (x1 - x0) # eq (21) of Ref [1]
            else:
                raise NotImplementedError

            if self.use_speaker_encoder:
                spk_emb, bct_feat = self.speaker_encoder.encoder(spkenc_wav_16k, l2_norm=False)
                if self.local_speaker_attention:
                    vt = self.model(x, t, mel_ctx, frontend_inputs, mel_len, spk_emb, bct_feat, text_len)
                else:
                    vt = self.model(x, t, mel_ctx, frontend_inputs, mel_len, spk_emb)
            else:
                vt = self.model(x, t, mel_ctx, frontend_inputs, mel_len) # the nnet model should take x, t, and other conditional inputs

            # apply mask to vt if use_len_mask is True
            # if self.use_len_mask:
            #     len_mask = self.generate_len_mask(ut, egs["ref_len"])
            #     vt = vt * len_mask + ut * (1 - len_mask) # for the padded part, we use the ground truth ut

            # apply loss_mask to vt if use_mask_loss is True
            # TODO: ctx_mask is only used in the voicebox training, we need to find a more general way to apply mask loss

        if self.use_mask_loss and 'mel_ctx_mask' in batch:
            mask_loss = batch['mel_ctx_mask'].float() # [B, T]
            mask_loss = mask_loss.unsqueeze(-1) # [B, T, 1]
            vt = vt * mask_loss + ut * (1 - mask_loss)

        mel_len_mask = sequence_mask(mel_len, device=ref.device)
        loss = self.criterion(vt, ut, mel_len_mask) 

        if torch.isnan(loss).any():
            loss = torch.tensor([0.0], requires_grad=True).to(loss.device)
        
        bsz, seqlen = ref.shape[0], ref.shape[1]
        self.log_dict(
            {
                "mel_loss": loss.item(),
                "bsz": bsz,
                "seqlen": seqlen
            },
            prog_bar=True,
            sync_dist=True)
        
        batch_tokens = torch.sum(mel_len).item()
        self.model_metric.update(
            num_tokens=batch_tokens,
            stage=self.trainer.state.stage,
            model_kwargs=dict(batch_size=bsz, seqlen=seqlen),
        )
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            metric = self.model_metric.compute(self.trainer.global_step)
            self.log_dict(
                metric, sync_dist=True, prog_bar=True
            )
            self.log_mel(
                {
                    "mel_ctx": mel_ctx[0],
                    "x": x[0],
                    "ut": ut[0],
                    "vt": vt[0] 
                },
                t[0][0][0],
            )

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
    def inference(self, x, t, mel_ctx, frontend_inputs, mel_len=None, spkenc_wav_16k=None):
        with torch.autocast(device_type="cuda", enabled=True):
            if mel_len is None:
                mel_len = x.shape[1]
                mel_len = torch.tensor(mel_len).unsqueeze(0).to(x.device)
            if self.use_speaker_encoder:
                spk_emb, bct_feat = self.speaker_encoder.encoder(spkenc_wav_16k, l2_norm=False)
                if self.local_speaker_attention:
                    text_len = mel_len
                    x = self.model(x, t, mel_ctx, frontend_inputs, mel_len, spk_emb, bct_feat, text_len)
                else:
                    x = self.model(x, t, mel_ctx, frontend_inputs, mel_len, spk_emb)
            else:
                x = self.model(x, t, mel_ctx, frontend_inputs, mel_len)
            return x

    predict = inference

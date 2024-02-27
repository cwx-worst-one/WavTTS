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
from recipes.bark.lit_modules.sample import sample
from s3a.providers.ctiga.utils.generation import InferenceParams
from samantha.utils.model_metric import ModelMetric
import logging


logger = logging.getLogger(__name__)



class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=True,
        sigma_min: float = 0.1,
        flow_matching_type: str = "ot",
        loss_type: str = "l2",
        use_len_mask: bool = True,
        use_mask_loss: bool = False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.requires = {}

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

        if loss_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif loss_type == "l2":
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"loss_type must be one of ['l1', 'l2'], but got {loss_type}")

        self.sigma_min = sigma_min
        self.flow_matching_type = flow_matching_type
        # check if flow_matching_type is not one of ['cfm', 'ot-cfm', 'sb-cfm', 'fm']
        assert self.flow_matching_type in ['cfm', 'ot-cfm', 'sb-cfm', 'fm'], \
            f"flow_matching_type must be one of ['cfm', 'ot-cfm', 'sb-cfm', 'fm'], but got {self.flow_matching_type}"
        self.use_len_mask = use_len_mask
        self.use_mask_loss = use_mask_loss

    def generate_len_mask(self, 
                          ref: torch.Tensor, 
                          lens: torch.Tensor):
        """
        Generate the mask for the loss function according to the valid length of the example.
        Args:
            ref (Tensor): [B, T, D]
            lens (Tensor): [B]
        Returns:
            mask (Tensor): [B, T, D]
        """
        mask = torch.zeros_like(ref)
        for i, l in enumerate(lens):
            mask[i, :l] = 1
        return mask

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
                        }
                mel = batch["mel"]
                ref = batch["ref"]
                # bns, stop_tokens = batch["bn"], batch["stop_token"]
                # text_lens, bn_lens = batch["text_lens"], batch["bn_lens"]
                # max_text_len = max(text_lens)

                # seq_lens = text_lens + bn_lens
                # seq_len = max(seq_lens)

                # loss_mask = sequence_mask(seq_lens, device="cuda")
                # text_loss_mask = sequence_mask(text_lens, device="cuda")
                # pad_text_loss_mask = F.pad(text_loss_mask, (0, seq_len - text_loss_mask.shape[1]), "constant", 0)
                # z_loss_mask = loss_mask - pad_text_loss_mask
        if self.flow_matching_type != "fm":
            if 'x0' not in egs:
                # NOTE: x0 cannot necessarily be sampled from N(0, 1), it can be sampled from any distribution for ot-cfm, sb-cfm, and cfm. 
                # For fm, it has to be sampled from N(0, 1)
                x0 = torch.randn_like(ref)
            else:
                x0 = egs['x0']
        else:
            if 'x0' in egs:
                raise Warning("x0 is not used in fm, but got x0 in egs")
        x1 = ref

        if self.ot_sampler is not None:
            if not (self.flow_matching_type == "ot-cfm" or self.flow_matching_type == "sb-cfm"):
                raise ValueError(f"ot_sampler can only be applied to ot-cfm or sb-cfm, but got {self.flow_matching_type}")
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        
        # generate random t in [0, 1]
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
        

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
                    # get the vt in eq (10) of Ref [1]
            vt = self.model(x=x, t=t, **egs_input) # the nnet model should take x, t, and other conditional inputs
            # ret_dict = self.model(frontend_inputs, mel,  batch["seqlen"])
            if self.use_len_mask:
                len_mask = self.generate_len_mask(ut, egs["ref_len"])
                vt = vt * len_mask + ut * (1 - len_mask) # for the padded 

            if self.use_mask_loss and 'ctx_mask' in egs:
                mask_loss = egs['ctx_mask'].float() # [B, T]
                mask_loss = mask_loss.unsqueeze(-1) # [B, T, 1]
                vt = vt * mask_loss + ut * (1 - mask_loss)

            loss = self.criterion(vt, ut)


            # mel_loss = F.mse_loss(ret_dict['pred_mel'], mel)
            
        self.log_dict({
                "mel_loss": loss.item(),
            },
            prog_bar=True,
            sync_dist=True)
    
        return mel_loss

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
    def inference_from_text(self, batch, tokenizer):
        return None

    predict = inference_from_text
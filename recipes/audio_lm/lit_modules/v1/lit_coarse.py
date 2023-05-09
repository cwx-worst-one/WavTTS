import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ...requires.mulan.mulan_infer import mulan_inference
from ...requires.w2v.w2v_infer import w2v_bert_tokenization


class CoarseModule(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        device = batch.device

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                w2v_tokens, input_tokens, mulan_embeds = self.prepare_feature(
                    batch.float()
                )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            # add prompt condition
            inputs_embeds = self.model.wte(input_tokens)
            prompt_embeds = self.model.condition(mulan_embeds.to(batch.dtype))
            inputs_embeds = torch.cat(
                [prompt_embeds, inputs_embeds], dim=1
            ).contiguous()

            logits = self.model(inputs_embeds=inputs_embeds)["logits"]
        b, t = w2v_tokens.size()
        loss_mask = torch.arange(input_tokens.size(1)).to(device) >= (1 + t + 1)
        loss_mask = loss_mask.unsqueeze(0).repeat(b, 1)
        loss = self.criterion(logits[:, 0:-1, :], input_tokens, mask=loss_mask)
        self.log_dict({"tr_loss": loss.item()}, prog_bar=False, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        eos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device)
            + 1024
            + num_coarse * 1024
        )

        wav_ids = self.get_quant(wavs)

        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + (torch.arange(num_coarse).to(device) + 1) * 1024
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # add eos token
        # prompt 1 + eos 1 + w2v 748 + eos 1 + ss 4800
        input_tokens = torch.cat([eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1)

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]

        return w2v_tokens, input_tokens, mulan_embeds

    @torch.no_grad()
    def get_mulan_embed(self, x):
        # [b, 1, d]
        return mulan_inference(
            model=self.requires["mulan"], music=x.float(), device=x.device
        ).unsqueeze(1)

    @torch.no_grad()
    def get_quant(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_w2v_token(self, x):
        return w2v_bert_tokenization(
            frontend=None,
            w2v_model=self.requires["semantic"],
            wavs=x,
            centers=self.requires["centroids"],
            device=x.device,
        )

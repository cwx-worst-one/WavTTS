import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ...requires.mulan.mulan_infer import mulan_inference, mulan_rvq_indexs
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

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, w2v_tokens, input_tokens = self.prepare_feature(
                    batch.float()
                )
                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Coarse 3AR Training, V3...")
        loss_offset = mulan_tokens.size(1) + w2v_tokens.size(1) + 1
        loss = self.criterion(
            logits[:, loss_offset : org_len - 1, :],
            input_tokens[:, loss_offset + 1 : org_len],
        )
        accu = (
            100
            * (
                logits[:, loss_offset : org_len - 1, :].argmax(dim=-1)
                == input_tokens[:, loss_offset + 1 : org_len]
            ).sum()
            / torch.ones_like(input_tokens[:, loss_offset + 1 : org_len]).sum()
        )
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
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

        # offset: semantic
        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + (torch.arange(num_coarse).to(device) + 1) * 1024
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # offset: semantic 1024, num_coarse
        eos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device)
            + 1024
            + num_coarse * 1024
        )

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        cfg_rate = self.extra_params.get("cfg_rate", 0)
        if cfg_rate > 0:
            rand_mulan = torch.nn.functional.normalize(
                torch.rand_like(mulan_embeds), dim=-1, p=2
            )
            rand_rate = torch.rand(size=[b, 1, 1], device=device)
            mulan_embeds = torch.where(rand_rate > cfg_rate, mulan_embeds, rand_mulan)

        mulan_tokens, ds = mulan_rvq_indexs(
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        mulan_tokens = (
            mulan_tokens + 1024 + num_coarse * 1024 + 2
        )  # offset: semantic + coarse + EOS 2
        mulan_token_sep = self.extra_params.get("mulan_token_sep", False)
        if mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        # mulan_tokens 12 + eos 1 + w2v 747 + eos 1 + ss 4800
        input_tokens = torch.cat(
            [mulan_tokens, eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1
        )

        return mulan_tokens, w2v_tokens, input_tokens

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
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x,
            centers=self.requires["centroids"],
            device=x.device,
        )

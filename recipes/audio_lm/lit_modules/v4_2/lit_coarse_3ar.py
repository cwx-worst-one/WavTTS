import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ...requires.w2v.w2v_infer import w2v_bert_tokenization


class CoarseModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        base_model_path,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.base_model_path = base_model_path
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.mulan_token_sep = mulan_token_sep
        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        self.model.load_state_dict(torch.load(self.base_model_path))
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, w2v_tokens, input_tokens = self.prepare_feature(batch)
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
            print("Coarse 3AR Training, gpt_finetune...")
        loss_offset = mulan_tokens.size(1) + w2v_tokens.size(1) + 1
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if "ln_" in name or "bias" in name:
                print("Skip WD on {}".format(name))
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
    def prepare_feature(self, batch):
        wavs = batch["audio"].float()
        text_input = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        )

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
        mulan_embeds = self.requires["mulan"].text_encoder(*text_input)  # [b, d]
        cfg_rate = self.extra_params.get("cfg_rate", 0)
        if cfg_rate > 0:
            rand_mulan = torch.nn.functional.normalize(
                torch.rand_like(mulan_embeds), dim=-1, p=2
            )
            rand_rate = torch.rand(size=[b, 1, 1], device=device)
            mulan_embeds = torch.where(rand_rate > cfg_rate, mulan_embeds, rand_mulan)

        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds, self.requires["mulan_centers"]
        )  # [b, t]
        mulan_tokens = (
            mulan_tokens + 1024 + num_coarse * 1024 + 2
        )  # offset: semantic + coarse + EOS 2
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        # mulan_tokens 12 + eos 1 + w2v 747 + eos 1 + ss 4800
        input_tokens = torch.cat(
            [mulan_tokens, eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1
        )

        return mulan_tokens, w2v_tokens, input_tokens

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

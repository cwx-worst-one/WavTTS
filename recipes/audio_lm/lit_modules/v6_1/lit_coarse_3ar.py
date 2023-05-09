import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

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
        finetune=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.codec_token_num = self.extra_params.quant_token_num
        self.finetune = finetune
        self.requires = {}
        self.val_outputs = dict()
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

        loss, accu = self.shared_step(batch)
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, accu = self.shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            for l, a in outputs:
                loss += l
                accu += a
            loss /= len(outputs)
            accu /= len(outputs)

            if self.local_rank == 0:
                print(f"[VAL/LOSS] {loss}")
                print(f"[VAL/ACCU] {accu}")

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def shared_step(self, batch):

        with torch.autocast(device_type="cuda", enabled=False):
            mulan_embeds, w2v_tokens, input_tokens = self.prepare_feature(
                batch["audio"].float()
            )
            org_len = input_tokens.size(1) + mulan_embeds.size(1)
            if hasattr(self.model.config, "train_len"):
                train_len = self.model.config.train_len
            else:
                train_len = org_len
            assert train_len >= org_len
            input_tokens = torch.nn.functional.pad(
                input_tokens, [0, train_len - org_len]
            )

        logits = self.model(input_ids=input_tokens, inputs_embeds=mulan_embeds)[
            "logits"
        ]

        if self.local_rank == 0:
            print("Coarse 3AR Training, V6_1...")
        loss_offset = mulan_embeds.size(1) + 1 + w2v_tokens.size(1)
        x = logits[:, loss_offset : org_len - 1, :]
        targets = (
            input_tokens[
                :,
                loss_offset - mulan_embeds.size(1) + 1 : org_len - mulan_embeds.size(1),
            ]
            - 1024
        )  # remove w2v offset
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

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
            + torch.arange(num_coarse).to(device) * self.codec_token_num
            + 1024
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # offset: semantic 1024, num_coarse
        eos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device)
            + 1024
            + num_coarse * self.codec_token_num
        )

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]

        input_tokens = torch.cat([eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1)

        return mulan_embeds, w2v_tokens, input_tokens

    @torch.no_grad()
    def get_mulan_embed(self, x):
        # [b, 1, d]
        return self.requires["mulan_infer_fn"](
            model=self.requires["mulan"],
            music=x.float()[:, 0 : 24000 * 10],
            device=x.device,
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

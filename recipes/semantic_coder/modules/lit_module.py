import pytorch_lightning as pl
import torch

from recipes.best_rq.modules.lit_module import BestRQ
from samantha.utils.hparams import DotDict


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        semantic_module,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.semantic_module = BestRQ.load_from_checkpoint(semantic_module)
        self.semantic_module.freeze()
        self.val_outputs = dict()

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()


class VAE(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        semantic_module,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            semantic_module=semantic_module,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        elif isinstance(batch, dict):
            batch = batch["audio"]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            embeds = self.semantic_module.get_latent(
                batch.float(), self.extra_params.semantic_layer_idx
            )
            logits_target = self.semantic_module.get_logits(batch.float())

        embeds_rec, loss_kl, std_mean = self.model(embeds.transpose(1, 2))
        logits_rec = self.semantic_module.get_logits_from_layer(
            embeds_rec.transpose(1, 2), self.extra_params.semantic_layer_idx
        )
        loss_logits = self.criterion(logits_rec, logits_target.softmax(dim=1))
        loss = loss_logits + self.extra_params.loss_beta * loss_kl
        return loss, loss_kl, loss_logits, std_mean

    def training_step(self, batch, batch_idx):
        loss, loss_kl, loss_logits, std_mean = self._shared_step(batch, batch_idx)
        self.log_dict(
            {
                "tr_loss": loss,
                "loss_kl": loss_kl,
                "loss_logits": loss_logits,
                "std_mean": std_mean,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_kl, loss_logits, std_mean = self._shared_step(batch, batch_idx)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, loss_kl, loss_logits, std_mean))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            loss_kl = 0
            loss_logits = 0
            std_mean = 0
            for output in outputs:
                loss += output[0]
                loss_kl += output[1]
                loss_logits += output[2]
                std_mean += output[3]
            loss /= len(outputs)
            loss_kl /= len(outputs)
            loss_logits /= len(outputs)
            std_mean /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_loss_kl_{dataloader_idx}": loss_kl,
                    f"val_loss_logits_{dataloader_idx}": loss_logits,
                    f"val_std_mean_{dataloader_idx}": std_mean,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []


class VQVAE(BaseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        semantic_module,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            semantic_module=semantic_module,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def _shared_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        elif isinstance(batch, dict):
            batch = batch["audio"]
        if batch.dim() == 3:
            batch = batch.squeeze(1)
        with torch.autocast(device_type="cuda", enabled=False):
            embeds = self.semantic_module.get_latent(
                batch.float(), self.extra_params.semantic_layer_idx
            )
            logits_target = self.semantic_module.get_logits(batch.float())

        embeds_rec, loss_quant, quant_index = self.model(embeds.transpose(1, 2))

        logits_rec = self.semantic_module.get_logits_from_layer(
            embeds_rec.transpose(1, 2), self.extra_params.semantic_layer_idx
        )
        loss_logits = self.criterion(logits_rec, logits_target.softmax(dim=1))

        loss = loss_logits + self.extra_params.loss_beta * loss_quant

        quant_token_num = self.model.quant_token_num
        one_hot = torch.nn.functional.one_hot(quant_index.reshape(-1), quant_token_num)
        one_hot = self.all_gather(one_hot).sum(dim=0)
        one_hot = one_hot.sum(dim=0).clamp(0, 1)
        quant_rate = 100 * one_hot.sum() / quant_token_num
        return loss, loss_quant, loss_logits, quant_rate

    def training_step(self, batch, batch_idx):
        loss, loss_quant, loss_logits, quant_rate = self._shared_step(batch, batch_idx)
        self.log_dict(
            {
                "tr_loss": loss,
                "loss_quant": loss_quant,
                "loss_logits": loss_logits,
                "quant_rate": quant_rate,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_quant, loss_logits, quant_rate = self._shared_step(batch, batch_idx)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(
            (loss, loss_quant, loss_logits, quant_rate)
        )

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            loss_quant = 0
            loss_logits = 0
            quant_rate = 0
            for output in outputs:
                loss += output[0]
                loss_quant += output[1]
                loss_logits += output[2]
                quant_rate += output[3]
            loss /= len(outputs)
            loss_quant /= len(outputs)
            loss_logits /= len(outputs)
            quant_rate /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_loss_quant_{dataloader_idx}": loss_quant,
                    f"val_loss_logits_{dataloader_idx}": loss_logits,
                    f"val_quant_rate_{dataloader_idx}": quant_rate,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

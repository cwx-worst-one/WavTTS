import pytorch_lightning as pl
import torch


def print_model_params(model):
    total_params = 0
    trainable_params = 0
    for _, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
    print(f"Total Params: {total_params:,}, Trainable Params: {trainable_params:,}")


class MusicFMLitModule(pl.LightningModule):  # pragma: no cover
    def __init__(
        self, model, optimizer_class, scheduler_class, is_global=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        print_model_params(model)
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.is_global = is_global
        self.save_hyperparameters(ignore=["model"])

    def step(self, batch):
        audio = batch[0].squeeze(1)
        _, _, losses, accuracies = self.model(audio)
        overall_loss = []
        overall_accuracy = []
        for key in losses.keys():
            if not self.is_global and not key.startswith("global"):
                overall_loss.append(losses[key])
                overall_accuracy.append(accuracies[key])
            elif self.is_global and key.startswith("global"):
                overall_loss.append(losses[key])
                overall_accuracy.append(accuracies[key])
        losses["overall"] = torch.mean(torch.stack(overall_loss))
        accuracies["overall"] = torch.mean(torch.stack(overall_accuracy))
        return losses, accuracies

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        for key in loss.keys():
            self.log(
                "loss_%s/train" % key, loss[key], rank_zero_only=True, sync_dist=True
            )
            self.log(
                "accuracy_%s/train" % key, acc[key], rank_zero_only=True, sync_dist=True
            )
        return loss["overall"]

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        for key in loss.keys():
            self.log(
                "loss_%s/valid" % key, loss[key], rank_zero_only=True, sync_dist=True
            )
            self.log(
                "accuracy_%s/valid" % key, acc[key], rank_zero_only=True, sync_dist=True
            )
        return loss["overall"]

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters())
        scheduler = {
            "scheduler": self.scheduler_class(optimizer),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

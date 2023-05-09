import pytorch_lightning as pl


class TemplateModule(pl.LightningModule):
    def __init__(self, model_cls, optimizer_cls, scheduler_cls):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters()
        self.model = self.hparams.model_cls()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return [optimizer], [scheduler]

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.model.step = "train"
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        self.model.step = "validation"
        raise NotImplementedError

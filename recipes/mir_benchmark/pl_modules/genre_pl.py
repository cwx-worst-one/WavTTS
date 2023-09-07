import random
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from torch import optim

warnings.filterwarnings("ignore", category=UserWarning)


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self._lr = lr
        self._scheduler_patience = scheduler_patience
        self._scheduler_decay_factor = scheduler_decay_factor

    def configure_optimizers(self):
        # Config optimizer and scheduler
        optimizer = optim.Adam(
            list(self.model.parameters()), lr=self._lr, weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._scheduler_patience,
            factor=self._scheduler_decay_factor,
            verbose=True,
            mode="min",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


class LitGenre(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        hop_length,
        sample_rate,
        sample_len,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.CrossEntropyLoss()
        self.prds = []
        self.gts = []

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["genre_label"] = batch[1]
        inputs["aug_hop_size"] = self._hop_length

        # model prediction
        genre_pred = self.model(inputs)[0]
        loss = self.loss(genre_pred, inputs["genre_label"].squeeze(1))

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        sample = batch[0]
        sample = torch.nn.functional.pad(
            sample, (0, int(self._sample_rate * self._sample_len))
        )
        sample = sample.unfold(
            1,
            int(self._sample_rate * self._sample_len),
            int(self._sample_rate * self._sample_len),
        )

        inputs["audio"] = sample.squeeze(0)
        inputs["genre_label"] = batch[1]

        # model prediction
        genre_pred = self.model(inputs)[0].mean(0).unsqueeze(0)
        loss = self.loss(genre_pred, inputs["genre_label"].squeeze(0))

        self.log("valid_loss", loss, sync_dist=True)
        oup = {}
        self.prds.append(genre_pred.cpu().detach().numpy()[0].argmax())
        self.gts.append(inputs["genre_label"].cpu().detach().numpy()[0][0])
        return oup

    def on_validation_epoch_end(self):
        accuracy = accuracy_score(self.gts, self.prds)
        print("Accuracy: %.4f" % accuracy)
        self.log("accuracy", accuracy, sync_dist=True)

        self.prds = []
        self.gts = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

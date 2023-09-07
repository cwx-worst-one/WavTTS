import random
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn import metrics
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


class LitTaggingInstrument(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        sample_rate,
        sample_len,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.BCELoss()
        self.val_outputs = []
        self.instrument_names = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ensemble",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Synth Effects",
            "Synth Lead",
            "Synth Pad",
            "Vocal",
        ]
        self.genre_names = []

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["instrument_label"] = batch[1]

        # model prediction
        instrument_pred = self.model(inputs)[0]
        loss = self.loss(instrument_pred, inputs["instrument_label"])

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["instrument_label"] = batch[1]

        # model prediction
        instrument_pred = self.model(inputs)[0]
        loss = self.loss(instrument_pred, inputs["instrument_label"])

        self.log("valid_loss", loss, sync_dist=True)

        # detach predictions
        predictions = [prd.detach().cpu().numpy() for prd in instrument_pred]
        target_vectors = [
            trg.detach().cpu().numpy() for trg in inputs["instrument_label"]
        ]

        self.val_outputs.append(
            {"loss": loss, "predictions": predictions, "target_vectors": target_vectors}
        )
        oup = {}

        return oup

    def on_validation_epoch_end(self):
        predictions = []
        targets = []
        for outs in self.val_outputs:
            predictions.append(torch.tensor(outs["predictions"]))
            targets.append(torch.tensor(outs["target_vectors"]))
        prd = torch.cat(predictions)
        trg = torch.cat(targets)

        # get auc scores
        aucs = self.get_auc_scores(trg, prd)
        self.log("instrument_roc_auc", aucs["roc_auc"])
        self.log("instrument_pr_auc", aucs["pr_auc"])
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def get_auc_scores(self, targets, logits):
        roc_aucs = []
        pr_aucs = []
        for i, name in enumerate(self.instrument_names):
            try:
                roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
                pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
                roc_aucs.append(roc_auc)
                pr_aucs.append(pr_auc)
                print(name)
                print("roc_auc: %.4f" % roc_auc)
                print("pr_auc: %.4f" % pr_auc)
            except ValueError as e:
                print(name)
                print("not available")
        print("overall")
        print("roc_auc: %.4f" % np.mean(roc_aucs))
        print("pr_auc: %.4f" % np.mean(pr_aucs))
        return {"roc_auc": np.mean(roc_aucs), "pr_auc": np.mean(pr_aucs)}


class LitTaggingGenre(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        sample_rate,
        sample_len,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.BCELoss()
        self.val_outputs = []
        self.genre_names = [
            "pop",
            "rock",
            "electronic",
            "hiphop_rap",
            "reggae",
            "rnb_soul",
            "Metal",
            "Jazz",
            "Blues",
            "Country",
            "Folk",
            "Indie",
            "K_pop",
            "Indie_Pop",
            "Muslim",
            "Indo_Christian",
            "Bollywood",
            "Bollywood_Retro",
            "Urban_Punjabi_Pop",
            "Tamil_Film_Music",
            "Kannada_Film_Music",
            "Telugu_Film_Music",
            "Indian_Independent",
            "Malayalam_Film_Music",
            "Sertanejo",
            "Baile_Funk",
            "Gospel",
            "Samba",
            "Pagode",
            "MPB",
            "Forro",
            "Axe",
            "Reggaeton",
            "Brazilian_Punk",
        ]

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["genre_label"] = batch[1]

        # model prediction
        genre_pred = self.model(inputs)[0]
        loss = self.loss(genre_pred, inputs["genre_label"])

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["genre_label"] = batch[1]

        # model prediction
        genre_pred = self.model(inputs)[0]
        loss = self.loss(genre_pred, inputs["genre_label"])

        self.log("valid_loss", loss, sync_dist=True)

        # detach predictions
        predictions = [prd.detach().cpu().numpy() for prd in genre_pred]
        target_vectors = [trg.detach().cpu().numpy() for trg in inputs["genre_label"]]

        self.val_outputs.append(
            {"loss": loss, "predictions": predictions, "target_vectors": target_vectors}
        )
        oup = {}

        return oup

    def on_validation_epoch_end(self):
        predictions = []
        targets = []
        for outs in self.val_outputs:
            predictions.append(torch.tensor(outs["predictions"]))
            targets.append(torch.tensor(outs["target_vectors"]))
        prd = torch.cat(predictions)
        trg = torch.cat(targets)

        # get auc scores
        aucs = self.get_auc_scores(trg, prd)
        self.log("genre_roc_auc", aucs["roc_auc"])
        self.log("genre_pr_auc", aucs["pr_auc"])
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def get_auc_scores(self, targets, logits):
        roc_aucs = []
        pr_aucs = []
        for i, name in enumerate(self.genre_names):
            try:
                roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
                pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
                roc_aucs.append(roc_auc)
                pr_aucs.append(pr_auc)
                print(name)
                print("roc_auc: %.4f" % roc_auc)
                print("pr_auc: %.4f" % pr_auc)
            except ValueError as e:
                print(name)
                print("not available")
        print("overall")
        print("roc_auc: %.4f" % np.mean(roc_aucs))
        print("pr_auc: %.4f" % np.mean(pr_aucs))
        return {"roc_auc": np.mean(roc_aucs), "pr_auc": np.mean(pr_aucs)}


class LitTaggingVocal(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        sample_rate,
        sample_len,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.BCELoss()
        self.val_outputs = []
        self.vocal_tags = [
            "age_中老年",
            "age_中青年",
            "age_幼年",
            "age_青年",
            "gender_NO",
            "gender_女",
            "gender_男",
            "style_低沉和蔼",
            "style_厚实低沉",
            "style_嘹亮自信",
            "style_成熟明亮",
            "style_成熟磁性",
            "style_明亮细腻",
            "style_淘气萌娃",
            "style_甜美温柔",
            "style_磁性慵懒",
        ]

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["vocal_label"] = batch[1]

        # model prediction
        vocal_pred = self.model(inputs)[0]
        loss = self.loss(vocal_pred, inputs["vocal_label"])

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["vocal_label"] = batch[1]

        # model prediction
        vocal_pred = self.model(inputs)[0]
        loss = self.loss(vocal_pred, inputs["vocal_label"])

        self.log("valid_loss", loss, sync_dist=True)

        # detach predictions
        predictions = [prd.detach().cpu().numpy() for prd in vocal_pred]
        target_vectors = [trg.detach().cpu().numpy() for trg in inputs["vocal_label"]]

        self.val_outputs.append(
            {"loss": loss, "predictions": predictions, "target_vectors": target_vectors}
        )
        oup = {}

        return oup

    def on_validation_epoch_end(self):
        predictions = []
        targets = []
        for outs in self.val_outputs:
            predictions.append(torch.tensor(outs["predictions"]))
            targets.append(torch.tensor(outs["target_vectors"]))
        prd = torch.cat(predictions)
        trg = torch.cat(targets)

        # get auc scores
        aucs = self.get_auc_scores(trg, prd)
        self.log("vocal_roc_auc", aucs["roc_auc"])
        self.log("vocal_pr_auc", aucs["pr_auc"])
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def get_auc_scores(self, targets, logits):
        roc_aucs = []
        pr_aucs = []
        for i, name in enumerate(self.vocal_tags):
            try:
                roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
                pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
                roc_aucs.append(roc_auc)
                pr_aucs.append(pr_auc)
                print(name)
                print("roc_auc: %.4f" % roc_auc)
                print("pr_auc: %.4f" % pr_auc)
            except ValueError as e:
                print(name)
                print("not available")
        print("overall")
        print("roc_auc: %.4f" % np.mean(roc_aucs))
        print("pr_auc: %.4f" % np.mean(pr_aucs))
        return {"roc_auc": np.mean(roc_aucs), "pr_auc": np.mean(pr_aucs)}


class LitTaggingMulti(BaseLightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        sample_rate,
        sample_len,
    ):
        super().__init__(
            model=model,
            lr=lr,
            scheduler_patience=scheduler_patience,
            scheduler_decay_factor=scheduler_decay_factor,
        )
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.BCELoss()
        self.val_outputs = []
        self.instrument_tags = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ensemble",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Synth Effects",
            "Synth Lead",
            "Synth Pad",
            "Vocal",
        ]
        self.vocal_tags = [
            "age_中老年",
            "age_中青年",
            "age_幼年",
            "age_青年",
            "gender_NO",
            "gender_女",
            "gender_男",
            "style_低沉和蔼",
            "style_厚实低沉",
            "style_嘹亮自信",
            "style_成熟明亮",
            "style_成熟磁性",
            "style_明亮细腻",
            "style_淘气萌娃",
            "style_甜美温柔",
            "style_磁性慵懒",
        ]
        self.genre_tags = [
            "pop",
            "rock",
            "electronic",
            "hiphop_rap",
            "reggae",
            "rnb_soul",
            "Metal",
            "Jazz",
            "Blues",
            "Country",
            "Folk",
            "Indie",
            "K_pop",
            "Indie_Pop",
            "Muslim",
            "Indo_Christian",
            "Bollywood",
            "Bollywood_Retro",
            "Urban_Punjabi_Pop",
            "Tamil_Film_Music",
            "Kannada_Film_Music",
            "Telugu_Film_Music",
            "Indian_Independent",
            "Malayalam_Film_Music",
            "Sertanejo",
            "Baile_Funk",
            "Gospel",
            "Samba",
            "Pagode",
            "MPB",
            "Forro",
            "Axe",
            "Reggaeton",
            "Brazilian_Punk",
        ]

    def training_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["instrument_label"] = batch[1]
        inputs["vocal_label"] = batch[2]
        inputs["genre_label"] = batch[3]
        inputs["task"] = batch[4]

        # model prediction
        vocal_pred = self.model(inputs)[0]
        loss = self.loss(vocal_pred, inputs["vocal_label"])

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {}
        outputs = {}

        inputs["audio"] = batch[0]
        inputs["instrument_label"] = batch[1]
        inputs["vocal_label"] = batch[2]
        inputs["genre_label"] = batch[3]
        task = np.array(batch[4])

        # model prediction
        instrument_preds, vocal_preds, genre_preds = self.model(inputs)

        # get instrument loss
        overall_loss = torch.tensor(0.0).to(inputs["audio"].device)
        inst_labels = batch[1][task == "instrument"]
        inst_preds = instrument_preds[task == "instrument"]
        if len(inst_labels) > 0:
            inst_loss = self.loss(inst_preds, inst_labels)
            overall_loss += inst_loss * len(inst_labels)
            self.log("valid_instrument_loss", inst_loss, sync_dist=True)
            outputs["inst_predictions"] = [
                prd.detach().cpu().numpy() for prd in inst_preds
            ]
            outputs["inst_targets"] = [
                trg.detach().cpu().numpy() for trg in inst_labels
            ]

        # get vocal loss
        vocal_labels = batch[2][task == "vocal"]
        vocal_preds = vocal_preds[task == "vocal"]
        if len(vocal_labels) > 0:
            vocal_loss = self.loss(vocal_preds, vocal_labels)
            overall_loss += vocal_loss * len(vocal_labels)
            self.log("valid_vocal_loss", vocal_loss, sync_dist=True)
            outputs["vocal_predictions"] = [
                prd.detach().cpu().numpy() for prd in vocal_preds
            ]
            outputs["vocal_targets"] = [
                trg.detach().cpu().numpy() for trg in vocal_labels
            ]

        # get genre loss
        genre_labels = batch[3][task == "genre"]
        genre_preds = genre_preds[task == "genre"]
        if len(genre_labels) > 0:
            genre_loss = self.loss(genre_preds, genre_labels)
            overall_loss += genre_loss * len(genre_labels)
            self.log("valid_genre_loss", genre_loss, sync_dist=True)
            outputs["genre_predictions"] = [
                prd.detach().cpu().numpy() for prd in genre_preds
            ]
            outputs["genre_targets"] = [
                trg.detach().cpu().numpy() for trg in genre_labels
            ]

        overall_loss /= len(batch[0])
        self.log("valid_overall_loss", overall_loss, sync_dist=True)
        # outputs["loss"]: overall_loss

        self.val_outputs.append(outputs)
        oup = {}

        return oup

    def on_validation_epoch_end(self):
        inst_predictions, vocal_predictions, genre_predictions = [], [], []
        inst_targets, vocal_targets, genre_targets = [], [], []
        for outs in self.val_outputs:
            for key in outs.keys():
                if key == "inst_predictions":
                    inst_predictions.append(outs[key])
                elif key == "inst_targets":
                    inst_targets.append(outs[key])
                value = torch.tensor(outs[key])
                exec(f"{key}.append({value})")
        breakpoint()
        prd = torch.cat(predictions)
        trg = torch.cat(targets)

        # get auc scores
        aucs = self.get_auc_scores(trg, prd)
        self.log("vocal_roc_auc", aucs["roc_auc"])
        self.log("vocal_pr_auc", aucs["pr_auc"])
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def get_auc_scores(self, targets, logits):
        roc_aucs = []
        pr_aucs = []
        for i, name in enumerate(self.vocal_tags):
            try:
                roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
                pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
                roc_aucs.append(roc_auc)
                pr_aucs.append(pr_auc)
                print(name)
                print("roc_auc: %.4f" % roc_auc)
                print("pr_auc: %.4f" % pr_auc)
            except ValueError as e:
                print(name)
                print("not available")
        print("overall")
        print("roc_auc: %.4f" % np.mean(roc_aucs))
        print("pr_auc: %.4f" % np.mean(pr_aucs))
        return {"roc_auc": np.mean(roc_aucs), "pr_auc": np.mean(pr_aucs)}

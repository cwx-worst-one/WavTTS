import itertools
from collections import defaultdict

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from einops import rearrange
from matplotlib import pyplot as plt
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch import nn

from recipes.mae.models.mae import MAE
from recipes.mae.models.mut import MuT


class LitMutMAEModule(pl.LightningModule):
    def __init__(self, emb_dim, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()  # save hyperparameter in ckpt
        mut = MuT(
            spec_shape=(128, 1000),
            patch_shape=(128, 2),
            num_classes=1000,
            sample_rate=24000,
            dim=1280,
            depth=32,
            heads=16,
            dim_head=80,
            channels=1,
            mlp_dim=5120,
            checkpointing=True,
            use_flash_attn=False,
        )
        self.mae = MAE(
            encoder=mut,
            masking_ratio=0.75,  # the paper recommended 75% masked patches
            decoder_dim=512,  # paper showed good results with just 512
            decoder_depth=8,  # anywhere from 1 to 8
            decoder_checkpointing=False,
            decoder_use_flash_attn=False,
        )
        self.lr = lr
        self.weight_decay = weight_decay

        # Validation outputs
        self.val_outputs = dict()

    def on_fit_start(self):
        for k, v in self.mae.logmel_frontend["logmel"].feat_extract.items():
            self.mae.logmel_frontend["logmel"].feat_extract[k] = v.to(self.device)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            optimizer = FusedAdam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            base_params = []
            norm_params = []
            no_decay = ["bn", "bias", "norm.bias", "norm.weight", "rotary"]
            for name, param in self.named_parameters():  # self.parameters()
                _found = False
                for k in no_decay:
                    if k in name:
                        norm_params.append(param)
                        _found = True
                        break
                if not _found:
                    base_params.append(param)

            optimizer = torch.optim.AdamW(
                [{"params": base_params}, {"params": norm_params, "weight_decay": 0.0}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        return optimizer

    def training_step(self, batch, batch_idx):
        # Combine multiple dataloader batches into one batch

        results = self._shared_step(batch)

        return {"loss": results["loss"]}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        results = self._shared_step(batch)

        if self.local_rank == 0:
            mel_spec, masked_indices, pred_pixel_values = (
                results["mel_spec"],
                results["masked_indices"],
                results["pred_pixel_values"],
            )
            _id = batch["music_id"][0]
            plt.imsave(f"check/{_id}_mel.png", mel_spec[0][0].detach().cpu().numpy())
            # masked
            masked_mel = mel_spec.clone()
            for i in masked_indices[0]:
                masked_mel[0, 0, :, 2 * i : 2 * i + 2] = 0
            plt.imsave(
                f"check/{_id}_masked_mel.png", masked_mel[0][0].detach().cpu().numpy()
            )
            # pred
            pred_pixel_values = rearrange(
                pred_pixel_values, "b n (d w) -> b n d w", w=2
            )
            pred_mel = mel_spec.clone()
            for i, ii in enumerate(masked_indices[0]):
                pred_mel[0, 0, :, 2 * ii : 2 * ii + 2] = pred_pixel_values[0, i]
            plt.imsave(
                f"check/{_id}_pred_mel.png", pred_mel[0][0].detach().cpu().numpy()
            )

        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append(results)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = []
            for output in outputs:
                loss.append(output["loss"])
            loss = torch.stack(loss)
            loss = self.all_gather(loss)

            self.log_dict({f"loss_{dataloader_idx}": torch.mean(loss)}, prog_bar=True)
            self.val_outputs[dataloader_idx] = []

    def _shared_step(self, batch):
        loss, mel_spec, masked_indices, pred_pixel_values = self.mae(
            batch["audio"].unsqueeze(1)
        )

        return {
            "loss": loss,
            "mel_spec": mel_spec,
            "masked_indices": masked_indices,
            "pred_pixel_values": pred_pixel_values,
        }

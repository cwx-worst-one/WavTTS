import pytorch_lightning as pl
import torch


class ProjectVQ(pl.Callback):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            codebook_emb = pl_module.model.vq.codebook
            batch = batch["audio"].squeeze(1).float()
            batch_feature = pl_module.preprocessing(batch)
            batch_emb = pl_module.get_latent(
                batch_feature, layer_idx=pl_module.model.encoder_config.vq_layer_idx
            )
            batch_emb = pl_module.model.vq.project_in(batch_emb).flatten(end_dim=-2)

            pl_module.logger.experiment.add_embedding(
                torch.cat([codebook_emb, batch_emb], dim=0),
                metadata=(
                    ["codebook" for _ in range(codebook_emb.size(0))]
                    + ["batch" for _ in range(batch_emb.size(0))]
                ),
                global_step=pl_module.global_step,
            )

import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter

from samantha.utils.hdfs_helper import put


class MuLanInferWriter(BasePredictionWriter):
    r"""Predictions writer for MuLan inference.

    Args:
        save_path (str): Path to save the predictions.
        hdfs_save_path (str):
            Path to save the predictions on hdfs. Default: None.
        filename (str):
            Name of the file to save the predictions. Default: "music_emb".
            The filename will be appended with `_raw.npz` for intermediate
            results and `_avg.npz` for averaged results grouped by music_id.
        compress (bool):
            Whether to compress the predictions.
            Default: True.

    """

    def __init__(self, save_path, hdfs_save_path=None, filename="music_emb"):
        super().__init__(write_interval="batch_and_epoch")
        self.save_path = save_path
        self.hdfs_save_path = hdfs_save_path
        self.filename = filename
        self.music_vecs = []

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        # Group music_vec by music_id
        vecs = outputs["music_vec"].detach().cpu().numpy()
        self.music_vecs.extend(vecs)

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ) -> None:
        # Stack music_vecs
        music_vecs = np.stack(self.music_vecs)

        filename = f"{self.save_path}/{self.filename}_{trainer.global_rank}"
        np.save(f"{filename}.npy", music_vecs)

        if self.hdfs_save_path is not None:
            put(f"{filename}.npy", self.hdfs_save_path)

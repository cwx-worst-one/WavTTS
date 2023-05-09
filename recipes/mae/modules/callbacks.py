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

    def __init__(
        self, save_path, hdfs_save_path=None, filename="music_emb", compress=True
    ):
        super().__init__(write_interval="batch_and_epoch")
        self.save_path = save_path
        self.hdfs_save_path = hdfs_save_path
        self.filename = filename
        self.compress = compress
        self.music_vec_dict = {}

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
        ids = outputs["music_id"]
        for vec, id in zip(vecs, ids):
            if id not in self.music_vec_dict:
                self.music_vec_dict[id] = []
            self.music_vec_dict[id].append(vec)

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ) -> None:
        # Stack music_vecs and average by music_id
        music_ids = []
        music_vecs = []
        music_avg_vecs = []
        for id, vecs in self.music_vec_dict.items():
            vecs = np.stack(vecs)
            avg_vecs = np.mean(vecs, axis=0)
            music_ids.append(id)
            music_vecs.append(vecs)
            music_avg_vecs.append(avg_vecs)

        # Save results
        if self.compress:
            save_fn = np.savez_compressed
        else:
            save_fn = np.savez
        filename = f"{self.save_path}/{self.filename}_{trainer.global_rank}"
        save_fn(f"{filename}_raw.npz", music_ids=music_ids, music_vecs=music_vecs)
        save_fn(f"{filename}_avg.npz", music_ids=music_ids, music_vecs=music_avg_vecs)

        # Save music embed as a single matrix, without music is
        single_music_vecs = np.stack(music_avg_vecs)
        np.save(f"{filename}.npy", single_music_vecs)

        if self.hdfs_save_path is not None:
            put(f"{filename}_raw.npz", self.hdfs_save_path)
            put(f"{filename}_avg.npz", self.hdfs_save_path)
            put(f"{filename}.npy", self.hdfs_save_path)

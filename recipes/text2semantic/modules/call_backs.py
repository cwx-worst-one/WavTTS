import glob, os
from pytorch_lightning.callbacks import BasePredictionWriter


class PredMergeWriter(BasePredictionWriter):
    r"""Predictions writer for MuLan inference.

    Args:
        save_path (str): Path to save the predictions.

    """

    def __init__(self, save_path):
        super().__init__(write_interval="epoch")
        self.save_path = save_path

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ) -> None:
        pl_module.out_fid.close()
        if trainer.local_rank == 0:
            pred_files = glob.glob(os.path.dirname(self.save_path) + '/pred_hyp*.txt')
            with open(self.save_path, "w") as outfile:
                for f in pred_files:
                    with open(f, "r") as infile:
                        outfile.write(infile.read())

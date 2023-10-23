from pytorch_lightning.callbacks import BasePredictionWriter
import os


class BeatWriter(BasePredictionWriter):
    r"""Writes beat prediction to disk.
    """

    def __init__(self, output_dir=None) -> None:
        super().__init__()
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        output_path = os.path.join(self.output_dir, batch[-1][0]+'.txt')
        with open(output_path, 'w') as f:
            for line in prediction:
               f.write(f'{str(line[0])}\t{str(line[1])}\n')

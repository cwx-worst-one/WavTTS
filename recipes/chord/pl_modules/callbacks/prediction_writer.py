from pytorch_lightning.callbacks import BasePredictionWriter
import numpy as np
import os
import pickle


class ChordWriter(BasePredictionWriter):
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
        if self.output_dir is not None:
            output_path = os.path.join(self.output_dir, batch[1][0]+'.pickle')
            output = {}
            output['intervals'] = []
            output['labels'] = []
            for line in prediction:
                output['intervals'].append([line[0], line[1]])
                output['labels'].append(line[2])

            with open(output_path, 'wb') as handle:
                pickle.dump(output, handle)

            with open(os.path.join(self.output_dir, batch[1][0]+'.txt'), 'w') as output_file:
                for line in prediction:
                    output_file.write(f'{str(line[0])}\t{str(line[1])}\t{line[2]}\n')


class ChordSequenceWriter(BasePredictionWriter):
    r"""Writes chord sequence to disk.
    """

    def __init__(
        self,
        output_dir=None,
        write_genre_prefix=False,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.write_genre_prefix = write_genre_prefix

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
        if self.output_dir is None:
            return

        all_chords = [[x[-1] for x in y if x[-1] != "N"] for y in prediction]
        all_genres = [x.get("genres", []) for x in batch["style_metadata"]]
        output_path = os.path.join(self.output_dir, f"{dataloader_idx}.{batch_idx}.txt")
        with open(output_path, "w") as fw:
            for chords, genres in zip(all_chords, all_genres):
                if len(chords) == 0:
                    continue
                to_write = " ".join(chords)
                if self.write_genre_prefix:
                    if len(genres) == 0:
                        fw.write(f"n/a {to_write}\n")
                    else:
                        for genre in genres:
                            genre = "_".join(genre.lower().split())
                            fw.write(f"{genre} {to_write}\n")
                else:
                    fw.write(f"{to_write}\n")

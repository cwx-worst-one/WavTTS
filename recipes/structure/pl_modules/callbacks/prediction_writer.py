from pytorch_lightning.callbacks import BasePredictionWriter
import os
import pickle


def dedup(lst):
    lst_dedup = []
    for x in lst:
        if len(lst_dedup) == 0 or lst_dedup[-1] != x:
            lst_dedup.append(x)
    return lst_dedup


class StructureWriter(BasePredictionWriter):
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
            output_path = os.path.join(self.output_dir, batch[-1][0]+'.pickle')
            output = {}
            output['intervals'] = []
            output['labels'] = []
            for line in prediction:
                output['intervals'].append(line['interval'])
                output['labels'].append(line['funct_name'])

            with open(output_path, 'wb') as handle:
                pickle.dump(output, handle)


class StructureSequenceWriter(BasePredictionWriter):
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
        if self.output_dir is None:
            return

        all_structures = [
            dedup([x["funct_name"] for x in y if x["funct_name"] != "silence"])
            for y in prediction
        ]
        output_path = os.path.join(self.output_dir, f"{dataloader_idx}.{batch_idx}.txt")
        with open(output_path, "w") as fw:
            for structure in all_structures:
                if len(structure) == 0:
                    continue
                to_write = " ".join(structure)
                fw.write(f"{to_write}\n")
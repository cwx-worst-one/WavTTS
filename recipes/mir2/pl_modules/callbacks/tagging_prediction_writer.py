from pytorch_lightning.callbacks import BasePredictionWriter
import warnings
import numpy as np
import os
import pandas as pd
import json

from recipes.mir2.parquet_dataset.new_label_process import get_id_to_tag_maps

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TaggingWriter(BasePredictionWriter):

    def __init__(
        self,
        output_dir=None,
    ) -> None:
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
            warnings.warn("`output_dir` is not set, so `TaggingWriter` is not doing anything.")
            return 
        output_json = os.path.join(self.output_dir, batch[1][0]+'.json')
        json.dump(prediction, open(output_json, 'w'), indent=2,  cls=json_serialize)

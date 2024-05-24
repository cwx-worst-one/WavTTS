import pytorch_lightning as pl
import logging
import torch
from typing import Any, List, Union
import json
from pathlib import Path
import os
import textwrap
import shutil
import numpy as np
import glob
import tqdm
import librosa
import soundfile
from recipes.musiclm.inference.utils import (
    slugify,
    save_wav,
    generate_hash,
    format_name,
    load_wav,
)
from collections import defaultdict
from recipes.bigmusic.utils.format_utils import update_json
import numpy as np
from recipes.musiclm.utils.dist import local_zero_first
from recipes.bigmusic.utils.upload import (
    audio_tensor_to_bytes,
    upload_to_easycycle,
    upload_to_tos,
)
from recipes.bigmusic.datasets.mir_data_util import ID_TEMPO_LABEL_MAP, ID_KEY_MAP
from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes


from samantha.utils import groundtruth


class SaveGroundtruthCallback(pl.Callback):
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable

    def on_predict_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.enable:
            groundtruth.start()

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.enable:
            groundtruth.stop()

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.enable:
            return
        data = groundtruth.export()
        groundtruth.clear()
        if len(data) == 0:
            logging.warning(f"empty groundtruth.")
            return
        logging.info(f"{batch_idx=}, fetched groundtruth for {data.keys()}")

        output_path = Path(
            f"{pl_module.extra_params.output_dir}_groundtruth/groundtruth_{batch_idx+1}.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)

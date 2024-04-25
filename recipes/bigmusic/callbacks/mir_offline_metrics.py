"""
Call MIR models offline
"""
from pathlib import Path
from typing import Dict, Optional, List
import json

import pytorch_lightning as pl

from recipes.bigmusic.utils.metrics_mir import run_batch, key_metrics, tempo_metrics
from recipes.bigmusic.utils.format_utils import update_json
from recipes.bigmusic.datasets.mir_data_util import tempo_to_label


def _extract_tempo(mir_dict: Dict) -> int:
    return mir_dict["beat"]["tempo"]


def _extract_key(mir_dict: Dict) -> str:
    return mir_dict["key"]["song"]


CALLBACK_TASKS = {
    "tempo": {
        "mir_task": "beat",  # corresponding mir task
        "check_fn": tempo_metrics,  # calculate metrics
        "extractor": _extract_tempo,  # extract the mir value from the output json
        "cond": "tempo_label",   # condition name in the metadata.json 
    },
    "key": {
        "mir_task": "key",
        "check_fn": key_metrics,
        "extractor": _extract_key,
        "cond": "key",
    },
}


def _load_json(json_fp):
    with open(json_fp, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d


class MIROfflineCallback(pl.Callback):
    def __init__(self, tasks: Optional[List[str]] = None) -> None:
        """
        :param tasks: default (the keys in CALLBACK_TASKS): ["tempo", "key"]
        """
        if tasks is None:
            tasks = list(CALLBACK_TASKS.keys())
        if not set(tasks).issubset(CALLBACK_TASKS.keys()):
            raise ValueError(f"One or more tasks in {tasks} are not supported.")
        self.tasks = tasks
        super().__init__()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _process(pl_module.extra_params.output_dir, self.tasks)


def _process(output_dir, callback_tasks):
    audio_fps = list(Path(output_dir).glob('**/*.generated.wav'))
    rel_fps = [Path(fp).relative_to(output_dir) for fp in audio_fps]
    mir_dir = Path(output_dir) / '.mir_cache'
    run_batch(
        audio_dir=output_dir,
        output_dir=str(mir_dir),
        tasks=[CALLBACK_TASKS[task]["mir_task"] for task in callback_tasks],
    )

    for rel_fp, audio_fp in zip(rel_fps, audio_fps):
        mir_dict = _load_json((mir_dir / rel_fp).with_suffix(".json"))
        mir_result = {task: CALLBACK_TASKS[task]["extractor"](mir_dict) for task in callback_tasks}
        metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
        metadata = _load_json(metadata_fp)
        metrics = {}
        for task, value in mir_result.items():
            if task == "tempo":
                value = tempo_to_label(value)
            metrics[task] = CALLBACK_TASKS[task]["check_fn"](pred=value, gt=metadata[CALLBACK_TASKS[task]["cond"]])
        update_json(metadata_fp, {'mir': {"result": mir_result, "metrics": metrics}})


if __name__ == "__main__":
    _process("/opt/tiger/samantha/assets/phonetone_60k_4", CALLBACK_TASKS)
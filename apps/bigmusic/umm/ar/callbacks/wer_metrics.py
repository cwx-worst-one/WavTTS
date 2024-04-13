import json
from pathlib import Path

import pytorch_lightning as pl
import torch

from apps.bigmusic.umm.ar.utils.format_utils import (
    normalize_text,
    update_json,
)
from apps.bigmusic.umm.ar.utils.metrics_asr import (
    asr_transcribe_lyrics,
    edit_distance,
    init_asr,
    remove_punc_case,
)
from apps.bigmusic.umm.ar.utils.utils import load_wav


class WERMetricsCallback(pl.Callback):
    def __init__(self, asr_model_path="en_punc"):
        super().__init__()
        self.asr_model_path = asr_model_path

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if "output_paths" in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob("**/*.generated.wav"))
        run_wer_metrics(
            generated_output_fps,
            asr_model_path=self.asr_model_path,
            device=pl_module.device,
        )


def run_wer_metrics(generated_output_fps, asr_model_path="en_punc", device="cuda"):
    asr_requires = init_asr(asr_model_path, local_rank=torch.cuda.current_device())
    for idx, generated_output_fp in enumerate(generated_output_fps):
        wav = torch.tensor(load_wav(str(generated_output_fp))).to(device)
        wavs_batch = wav.unsqueeze(0)  # convert to batch format
        asr_lyrics = asr_transcribe_lyrics(asr_requires, wavs_batch, sample_rate=24000)
        metadata_fp = str(generated_output_fp).replace("generated.wav", "metadata.json")
        with open(metadata_fp, "r") as f:
            metadata = json.load(f)
        # actual transcript
        lyrics = metadata.get("lyrics")
        a = "" if lyrics is None else normalize_text(remove_punc_case(lyrics))
        g = normalize_text(remove_punc_case(asr_lyrics[0]))  # greedy transcript
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            "ins": ins,
            "subs": subs,
            "dels": dels,
            "wer": wer,
            "greedy_transcript": g,
            "actual_transcript": a,
        }
        update_json(metadata_fp, {"wer": wer_metadata})

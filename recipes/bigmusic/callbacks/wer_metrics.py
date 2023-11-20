import pytorch_lightning as pl
from typing import Any
from recipes.bigmusic.utils.format_utils import (
    concat_metadata_list,
    update_json,
    normalize_text,
)
from recipes.bigmusic.utils.metrics_asr import (
    asr_transcribe_lyrics,
    init_asr,
    edit_distance,
    remove_punc_case,
)
import torch
from recipes.musiclm.inference.utils import load_wav
import json
from pathlib import Path
import numpy as np
from string import punctuation
from recipes.bigmusic.utils.format_utils import normalize_text
from collections import defaultdict

class WERMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        run_wer_metrics(output_dir, device = pl_module.device)

def run_wer_metrics(output_dir, asr_model_path='en_punc', device='cuda'):
    output_dir = Path(output_dir)
    generated_output_fps = list(output_dir.glob('**/*.generated.wav'))
    if len(generated_output_fps) == 0:
        return
    category2wer = defaultdict(list)
    asr_requires = init_asr(asr_model_path, local_rank=torch.cuda.current_device())
    for idx, generated_output_fp in enumerate(generated_output_fps):
        wav = torch.tensor(load_wav(str(generated_output_fp))).to(device)
        wavs_batch = wav.unsqueeze(0) # convert to batch format
        asr_lyrics = asr_transcribe_lyrics(
            asr_requires,
            wavs_batch,
            sample_rate=24000,
        )
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        # actual transcript
        lyrics = metadata.get('lyrics')
        a = '' if lyrics is None else normalize_text(remove_punc_case(lyrics))
        g = normalize_text(remove_punc_case(asr_lyrics[0])) # greedy transcript
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'ins': ins,
            'subs': subs,
            'dels': dels,
            'wer': wer,
            'greedy_transcript': g,
            'actual_transcript': a
        }
        update_json(metadata_fp, { 'wer': wer_metadata })

        # update total metrics
        category_dir = generated_output_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2wer[str(category_dir)].append([wer, ins, subs, dels])
        category2wer[str(output_dir)].append([wer, ins, subs, dels]) # append to base directory to calculate total wer
        
    for dir_path, wers in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        wer, ins, subs, dels = np.array(wers).mean(axis=0)
        wer_metadata = {
            'wer': round(wer, 3),
            'ins': round(ins, 3),
            'subs': round(subs, 3),
            'dels': round(dels, 3),
        }
        update_json(metrics_fp, { 'wer': wer_metadata })

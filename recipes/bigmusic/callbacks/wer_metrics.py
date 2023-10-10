import pytorch_lightning as pl
from typing import Any
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json, normalize_text
from recipes.bigmusic.utils.metrics_asr import wav2lyrics, edit_distance
import torch
from recipes.musiclm.inference.utils import load_wav
import json
from pathlib import Path
import numpy as np

class WERMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        run_wer_metrics(output_dir, device = pl_module.device)

def run_wer_metrics(output_dir, device='cuda'):
    output_dir = Path(output_dir)
    generated_output_fps = list(output_dir.glob('**/*.generated.wav'))
    if len(generated_output_fps) == 0:
        return
    wer_totals = []
    for idx, generated_output_fp in enumerate(generated_output_fps):
        wav = torch.tensor(load_wav(str(generated_output_fp))).to(device)
        wavs_batch = wav.unsqueeze(0) # convert to batch format
        asr_lyrics, _ = wav2lyrics(wavs_batch, sr=24000, device_id=torch.cuda.current_device())
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        a = normalize_text(metadata['lyrics']) # actual transcript
        g = normalize_text(asr_lyrics[0]) # greedy transcript
        edits = edit_distance(a, g)
        ins = round(edits.ins / len(a), 3)
        subs = round(edits.subs / len(a), 3)
        dels = round(edits.dels / len(a), 3)
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
        wer_totals.append([wer, ins, subs, dels])
        
    metrics_fp = output_dir/'metrics.json'
    wer, ins, subs, dels = np.array(wer_totals).mean(axis=0)
    wer_metadata = {
        'wer': round(wer, 3),
        'ins': round(ins, 3),
        'subs': round(subs, 3),
        'dels': round(dels, 3),
    }
    update_json(metrics_fp, { 'wer': wer_metadata })





# WARNING: DEPRECATED. 
# Use WERMetricsCallback. This callback is unstable and may crash during inference due to OOM
class WERMetricsBatchCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.batched_wer_results = []

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        wavs = outputs['generated_audio_tensor']
        lyrics = batch['lyrics']
        wer_results, wer_metadatas = WERMetricsBatchCallback.calculate_batch_wer(wavs, lyrics)
        self.batched_wer_results.extend(wer_results)
        outputs['metadata'] = concat_metadata_list(outputs.get('metadata'), wer_metadatas)

    @staticmethod
    def calculate_batch_wer(wavs, lyrics):
        wer_results = []
        wer_metadatas = []
        asr_lyrics, _ = wav2lyrics(wavs)
        actual_transcript = [normalize_text(l) for l in lyrics]
        greedy_transcript = [normalize_text(l) for l in asr_lyrics]
        for j, (a, g) in enumerate(zip(actual_transcript, greedy_transcript)):
            edits = edit_distance(a, g)
            ins = round(edits.ins / len(a), 3)
            subs = round(edits.subs / len(a), 3)
            dels = round(edits.dels / len(a), 3)
            wer = sum([ins, subs, dels])
            wer_metadata = {
                'ins': ins,
                'subs': subs,
                'dels': dels,
                'wer': wer,
                'greedy_transcript': g,
                'actual_transcript': a
            }
            wer_results.append([wer, ins, dels, subs])
            wer_metadatas.append({ 'wer': wer_metadata })
        return wer_results, wer_metadatas
    
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        metrics_fp = Path(output_dir)/'metrics.json'
        wer, ins, subs, dels = np.array(self.batched_wer_results).mean(axis=0)
        wer_metadata = {
            'wer': round(wer, 3),
            'ins': round(ins, 3),
            'subs': round(subs, 3),
            'dels': round(dels, 3),
        }
        update_json(metrics_fp, { 'wer': wer_metadata })
import pytorch_lightning as pl
from typing import Any
from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json
import torch
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length
from recipes.musiclm.inference.utils import load_wav
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

class MCSMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        requires = pl_module.semantic_module.requires
        sample_rate = pl_module.extra_params.sample_rate
        output_dir = pl_module.extra_params.output_dir
        run_mcs_metrics(requires, output_dir, device=pl_module.device, sample_rate=sample_rate)

def run_mcs_metrics(requires, output_dir, device='cuda', sample_rate=24000):
    output_dir = Path(output_dir)
    generated_output_fps = list(output_dir.glob('**/*.generated.wav'))
    category2mcs = defaultdict(list)
    mulan_min_duration = 10 * sample_rate
    def _load_audio_tensor(audio_path):
        wav_tensor = torch.tensor(load_wav(str(audio_path), sr=sample_rate)).to(device)
        if wav_tensor.shape[-1] < mulan_min_duration:
            wav_tensor = crop_pad_to_seq_length(wav_tensor, mulan_min_duration)
        return wav_tensor.unsqueeze(0)
    for idx, generated_output_fp in enumerate(generated_output_fps):
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        conditions = metadata['conditions']
        # Compute MCS based on wavs
        if 'style_text' in conditions:
            gt_emb = get_mulan_embeds(
                requires, metadata["style_text"], data_type='text'
            ).to(device)
        if 'style_audio' in conditions:
            style_audio_fp = str(generated_output_fp).replace('.generated.wav', '.style_audio.wav')
            wav_style = _load_audio_tensor(style_audio_fp)
            gt_emb = get_mulan_embeds(
                requires, wav_style, data_type='music'
            ).to(device)
        wav_gen = _load_audio_tensor(generated_output_fp)
        audio_emb = get_mulan_embeds(
            requires, wav_gen, data_type='music'
        ).to(device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().item()
        update_json(metadata_fp, { 'mcs': round(mcs, 3)})
        
        # update total metrics
        category_dir = generated_output_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2mcs[category_dir].append(mcs)
        category2mcs[output_dir].append(mcs) # append to base directory to calculate total wer
        
    for dir_path, mcs_totals in category2mcs.items():
        metrics_fp = dir_path/'metrics.json'
        avg_mcs = round(float(np.array(mcs_totals).mean(axis=0)), 3)
        update_json(metrics_fp, { 'mcs': avg_mcs })




# WARNING: DEPRECATED. Use MCSMetricsCallback.
class MCSMetricsBatchCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.batched_mcs_results = []

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
        semantic_module = pl_module.semantic_module
        sample_rate = pl_module.extra_params.sample_rate
        mcs = MCSMetricsBatchCallback.calculate_mcs(semantic_module, wavs, batch, sample_rate)
        mcs_metadata = [ { 'mcs': round(x.item(), 3) } for x in mcs ]
        outputs['metadata'] = concat_metadata_list(outputs.get('metadata'), mcs_metadata)
        self.batched_mcs_results.extend(mcs)

    @staticmethod
    def calculate_mcs(semantic_module, wavs, batch, sample_rate):
        device = semantic_module.device
        mulan_max_duration = 10 * sample_rate
        if len(wavs.shape) == 3:
            wavs = wavs.squeeze(1)
        wavs = wavs.to(device)
        conditions = batch['conditions']
        mulan_emb_names = [key for key in semantic_module.input_embedders.keys() if key.startswith("mulan")]
        if not mulan_emb_names:
            return torch.zeros((wavs.shape[0]))
        mulan_emb_name = mulan_emb_names[0]
        # Compute MCS based on wavs
        if 'style_text' in conditions:
            gt_emb = semantic_module.input_embedders[mulan_emb_name].get_embeds(
                semantic_module.requires, batch["style_text"], data_type='text'
            ).squeeze(1).to(device)
        if 'style_audio' in conditions:
            gt_emb = semantic_module.input_embedders[mulan_emb_name].get_embeds(
                semantic_module.requires, batch["style_audio"][:,0:mulan_max_duration], data_type='music'
            ).squeeze(1).to(device)
        audio_emb = semantic_module.input_embedders[mulan_emb_name].get_embeds(
            semantic_module.requires, wavs[:,0:mulan_max_duration], data_type='music'
        ).squeeze(1).to(device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().numpy()
        return mcs
    
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mcs = np.array(self.batched_mcs_results).mean()
        output_dir = pl_module.extra_params.output_dir
        metrics_fp = Path(output_dir)/'metrics.json'
        update_json(metrics_fp, { 'mcs': float(round(mcs, 3)) })

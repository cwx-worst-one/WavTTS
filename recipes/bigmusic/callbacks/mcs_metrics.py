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
from torchaudio.functional import resample

class MCSMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        requires = pl_module.semantic_module.requires
        sample_rate = pl_module.extra_params.sample_rate
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        generated_output_fps.sort()
        run_mcs_metrics(requires, generated_output_fps, device=pl_module.device, sample_rate=sample_rate)
        if 'output_paths' in pl_module.extra_params:
            gather_all_result(str(pl_module.extra_params.output_dir))

def run_mcs_metrics(requires, generated_output_fps, device='cuda', sample_rate=24000):
    mulan_min_duration = 10 * sample_rate
    def _load_audio_tensor(audio_path):
        wav_tensor = torch.tensor(load_wav(str(audio_path), sr=sample_rate)).to(device)
        if wav_tensor.shape[-1] < mulan_min_duration:
            wav_tensor = crop_pad_to_seq_length(wav_tensor, mulan_min_duration)
        if sample_rate != 24000:    # bochen: note that mulan only works on 24k audio
            wav_tensor = resample(
                wav_tensor,
                orig_freq=sample_rate,
                new_freq=24000,
            )
        return wav_tensor.unsqueeze(0)
    for idx, generated_output_fp in enumerate(generated_output_fps):
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        conditions = metadata['conditions']
        # Compute MCS based on wavs
        if 'freeform_text' in conditions and metadata["freeform_text"]:
            gt_emb = get_mulan_embeds(
                requires, metadata["freeform_text"], data_type='text'
            ).to(device)
        # elif 'style_text' in conditions or 'style_category' in conditions:
        #     gt_emb = get_mulan_embeds(
        #         requires, metadata["style_text"], data_type='text'
        #     ).to(device)
        elif 'style_audio' in conditions:
            style_audio_fp = str(generated_output_fp).replace('.generated.wav', '.style_audio.wav')
            wav_style = _load_audio_tensor(style_audio_fp)
            gt_emb = get_mulan_embeds(
                requires, wav_style, data_type='music'
            ).to(device)
        else:
            print('Could not calculate MCS metrics. Could not find freeform_text or style_text or style_audio in conditions', conditions)
            return
        wav_gen = _load_audio_tensor(generated_output_fp)
        audio_emb = get_mulan_embeds(
            requires, wav_gen, data_type='music'
        ).to(device)
        mcs = torch.nn.functional.cosine_similarity(gt_emb, audio_emb).cpu().item()
        update_json(metadata_fp, { 'mcs': round(mcs, 3)})


def gather_all_result(path_result):
    import os
    import glob
    import pandas as pd
    from tabulate import tabulate
    # metadata_fps = list(Path(output_dir).glob('**/*.metadata.json'))
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']

    title = ['index', 'mcs']
    df = pd.DataFrame(columns=title)
    mcs_all = {}
    mcs_all['all'] = []

    for category in categories:
        mcs_all[category] = []
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        filenames.sort()
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            mcs = metadata.get('mcs', None)
            data = {}
            data['index'] = index
            data['mcs'] = mcs if mcs else '/'
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            mcs_all[category].append(mcs)
            mcs_all['all'].append(mcs)
    for category in categories:
        data['index'] = 'category: ' + category
        data['mcs'] = round(np.array( list(filter(None, mcs_all[category])) ).mean(), 3)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    data['index'] = 'all'
    data['mcs'] = round(np.array( list(filter(None, mcs_all['all'])) ).mean(), 3)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    filename_out = os.path.join(path_result, 'mcs_for_all_samples.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print ('--- MCS metrics:')
    print(table)
    with open(filename_out, 'w') as f:
        f.write(table)

if __name__ == '__main__':

    # help to run mcs metrics offline
    from recipes.musiclm.requires.model_initializer import init_mulan
    from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()

    generated_output_fps = list(Path(args.input_dir).glob('**/*.generated.wav'))
    generated_output_fps.sort()

    hpath = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v9/mulan/mulan-step=005000-median_rank_1=61-kaggle-minimal.ckpt'
    cache_dir = '/opt/tiger/samantha/.module_cache/bigmusic'
    version = 'sstkmae_v3'
    local_rank = 0
    requires = init_mulan(hpath, local_rank, cache_dir=cache_dir, version=version)

    run_mcs_metrics(requires, generated_output_fps, sample_rate=44100)
    gather_all_result(args.input_dir)


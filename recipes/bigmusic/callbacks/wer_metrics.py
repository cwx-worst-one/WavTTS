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
    remove_space,
)
import torch
from recipes.musiclm.inference.utils import load_wav
import json
from pathlib import Path
import numpy as np
import tqdm
from string import punctuation
from recipes.bigmusic.utils.format_utils import normalize_text
from collections import defaultdict

class WERMetricsCallback(pl.Callback):
    def __init__(self, asr_model_path='en_punc'):
        super().__init__()
        self.asr_model_path = asr_model_path

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        run_wer_metrics(generated_output_fps, asr_model_path=self.asr_model_path, device=pl_module.device)

def run_wer_metrics(generated_output_fps, asr_model_path='en_punc', device='cuda'):
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
        with open(metadata_fp, 'r', encoding='utf-8') as f:
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

class WERMetricsCallbackV2(pl.Callback):
    def __init__(self, asr_model_path='en_punc'):
        super().__init__()
        self.asr_model_path = asr_model_path

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

        language = pl_module.extra_params.lyrics_lang
        if 'en' in language:
            asr_model_path = 'en_punc'
        elif 'zh' in language:
            asr_model_path = 'zh'
        else:
            raise NotImplementedError
        run_wer_metrics_svs(generated_output_fps, asr_model_path=asr_model_path, device=pl_module.device)


def run_wer_metrics_svs(generated_output_fps, asr_model_path='en_punc', device='cuda'):
    asr_requires = init_asr(asr_model_path, local_rank=torch.cuda.current_device())
    category2wer = defaultdict(list)

    for idx, generated_output_fp in enumerate(generated_output_fps):
        wav = torch.tensor(load_wav(str(generated_output_fp))).to(device)
        wavs_batch = wav.unsqueeze(0) # convert to batch format
        asr_lyrics = asr_transcribe_lyrics(
            asr_requires,
            wavs_batch,
            sample_rate=24000,
        )
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # actual transcript
        lyrics = metadata.get('lyrics')

        if asr_model_path == 'zh':
            lyrics = remove_space(lyrics)
        
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
        if dels > 0.2:
            print("gt trans: ", a, "asr result: ", g, "path: ",str(generated_output_fp), "might be asr model error")
        update_json(metadata_fp, { 'wer': wer_metadata })
        if isinstance(generated_output_fp, str):
            import os
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append([wer, ins, subs, dels]) # append to base directory to calculate total wer
        
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
        print(f"output_dir={str(category_dir)}, WER={wer_metadata}")

class RelativeWERMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        language = pl_module.extra_params.lyrics_lang
        if 'en' in language:
            asr_model_path = 'en_punc'
        elif 'zh' in language:
            asr_model_path = 'zh'
        else:
            raise NotImplementedError
        run_relative_wer_metrics(output_dir, device = pl_module.device, asr_model_path=asr_model_path)


def relative_wer_on_single_file(inputs):
    generated_output_fp, asr_requires = inputs
    wav = torch.tensor(load_wav(str(generated_output_fp))).to(torch.cuda.current_device())
    wavs_batch = wav.unsqueeze(0) # convert to batch format
    asr_lyrics = asr_transcribe_lyrics(
        asr_requires,
        wavs_batch,
        sample_rate=24000,
    )
    metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
    target_audio_fp = str(generated_output_fp).replace('generated.wav', 'target_audio.wav')
    wav = torch.tensor(load_wav(str(target_audio_fp))).to(torch.cuda.current_device())
    wavs_batch = wav.unsqueeze(0) # convert to batch format
    lyrics = asr_transcribe_lyrics(
        asr_requires,
        wavs_batch,
        sample_rate=24000,
    )
    return (asr_lyrics, lyrics, metadata_fp, generated_output_fp)


def run_relative_wer_metrics(output_dir, asr_model_path='zh', device='cuda'):
    asr_requires = init_asr(asr_model_path, local_rank=torch.cuda.current_device())
    output_dir = Path(output_dir)
    generated_output_fps = list(output_dir.glob('**/*.generated.wav'))
    if len(generated_output_fps) == 0:
        return
    category2wer = defaultdict(list)
    all_pred_text = ''
    all_label_text = ''

    for fp in tqdm.tqdm(generated_output_fps,
                        desc="Running relative_wer_metrics",
                        total=len(generated_output_fps)):
        asr_lyrics, lyrics, metadata_fp, generated_output_fp = relative_wer_on_single_file((fp, asr_requires))
        a = normalize_text(remove_punc_case(lyrics[0])) # greedy transcript
        g = normalize_text(remove_punc_case(asr_lyrics[0])) # greedy transcript

        all_label_text += a
        all_pred_text += g
        
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
        update_json(metadata_fp, { 'target_audio_asr': lyrics })
        update_json(metadata_fp, { 'generated_audio_asr': asr_lyrics})

        # update total metrics
        category_dir = generated_output_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2wer[str(category_dir)].append([wer, ins, subs, dels])
        category2wer[str(output_dir)].append([wer, ins, subs, dels]) # append to base directory to calculate total wer
        
    for dir_path, wers in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        edits = edit_distance(all_label_text, all_pred_text)
        denom = 1.0 if len(all_label_text) == 0 else len(all_label_text)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'wer': round(wer, 3),
            'ins': round(ins, 3),
            'subs': round(subs, 3),
            'dels': round(dels, 3),
        }
        update_json(metrics_fp, { 'rWER': wer_metadata })
        print(f"output_dir={output_dir}, rWER={wer_metadata}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    # run_relative_wer_metrics(args.input_dir, asr_model_path='zh')
    run_wer_metrics_svs(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh')
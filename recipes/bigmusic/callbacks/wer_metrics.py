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
    normalize_lyrics,
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
import requests
import time

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

class WERMetricsSAOnlineCallback(pl.Callback):
    def __init__(self, asr_model_path='en_punc'):
        super().__init__()
        self.asr_model_path = asr_model_path

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        run_wer_metrics_sa_online(generated_output_fps, asr_model_path=self.asr_model_path)

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
        a = normalize_text(remove_punc_case(lyrics))
        normalize_a = a.lower().replace("[verse]", "").replace("[chorus]", "").replace("[intro]", "").replace("[outro]", "").replace("[inst]", "").replace("[bridge]", "")
        g = normalize_text(remove_punc_case(asr_lyrics[0])) # greedy transcript
        edits = edit_distance(normalize_a.replace(" ", ""), g.replace(" ", ""))
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


def run_asr_lyrics_sa_online(filepath, language):
    #base_url = 'http://speech-test.byted.org/api/v1/vc'
    #appid = "api_dev"
    #token = "lv_token"
    base_url = 'http://speech.byted.org/api/v1/vc'
    appid = 'bkseig7309i0'
    token = 'lv_token'
    access_token = 'YLp0eHdvLZH_IvgNSEHQrHsBeTdliqe0'
    caption_type = "singing"
    max_retry_num = 5
    try_num = 0

    text = ''
    while try_num < max_retry_num:
        try:
            with open(filepath, 'rb') as fp:
                data = fp.read()
                response = requests.post(
                             '{base_url}/submit'.format(base_url=base_url),
                             params=dict(
                                 appid=appid,
                                 token=token,
                                 language=language,
                                 caption_type=caption_type,
                                 use_itn='False',
                                 dirt_filter='False',
                                 use_capitalize='False',
                                 use_spell_correct='False',
                                 with_gender_info='False',
                                 with_speaker_info='False',
                                 max_lines=1,
                                 words_per_line=15,
                                 with_confidence='True',
                                 verbose='True'
                             ),
                             data=data,
                             headers={
                                'content-type': 'audio/m4a',
                                "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                             }
                         )
                job_id = response.json()['id']
                response = requests.get(
                        '{base_url}/query'.format(base_url=base_url),
                        params=dict(
                            appid=appid,
                            token=token,
                            id=job_id,
                        ),
                        headers={
                            "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                        }
                )
                for item in response.json()['utterances']:
                    text += item['text'] + ' '
                try_num = max_retry_num + 1
        except Exception as e:
            try_num += 1
            text = ''
            time.sleep(30)
    return text.strip()


def run_wer_metrics_sa_online(generated_output_fps, asr_model_path='zh-CN'):
    category2wer = defaultdict(list)

    for idx, generated_output_fp in enumerate(generated_output_fps):
        asr_lyrics = run_asr_lyrics_sa_online(generated_output_fp, asr_model_path)
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # actual transcript
        lyrics = metadata.get('lyrics')
        lyrics = normalize_lyrics(lyrics)

        if asr_model_path == 'zh-CN':
            lyrics = lyrics.replace(' ', '')
            asr_lyrics = asr_lyrics.replace(' ', '')
        
        a = '' if lyrics is None else normalize_text(remove_punc_case(lyrics))
        g = normalize_text(remove_punc_case(asr_lyrics)) # greedy transcript
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'ins error': edits.ins,
            'subs error': edits.subs,
            'dels error': edits.dels,
            'ref length': denom,
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
        category2wer[str(category_dir)].append([denom, edits.ins, edits.subs, edits.dels]) # append to base directory to calculate total wer
    
    all_ref_length = 0
    all_ins_err = 0
    all_subs_err = 0
    all_dels_err = 0
    for dir_path, wers in category2wer.items():
        metrics_fp = Path(dir_path)/'metrics.json'
        #wer, ins, subs, dels = np.array(wers).mean(axis=0)
        ref_length, ins_err, subs_err, dels_err = np.array(wers).sum(axis=0)
        all_ref_length += ref_length
        all_ins_err += ins_err
        all_subs_err += subs_err
        all_dels_err += dels_err
        wer_metadata = {
            'wer': round((ins_err+subs_err+dels_err)/ref_length, 3),
            'ins': round(ins_err/ref_length, 3),
            'subs': round(subs_err/ref_length, 3),
            'dels': round(dels_err/ref_length, 3),
        }
        update_json(metrics_fp, { 'wer': wer_metadata })
        print(f"output_dir={metrics_fp}, WER={wer_metadata}")
    wer_metadata = {
        'wer': round((all_ins_err+all_subs_err+all_dels_err)/all_ref_length, 3),
        'ins': round(all_ins_err/all_ref_length, 3),
        'subs': round(all_subs_err/all_ref_length, 3),
        'dels': round(all_dels_err/all_ref_length, 3),
    }
    all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
    update_json(all_metrics_fp, { 'wer': wer_metadata })
    print(f"output_dir={all_metrics_fp}, WER={wer_metadata}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    # run_relative_wer_metrics(args.input_dir, asr_model_path='zh')
    #run_wer_metrics_svs(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh')
    run_wer_metrics_sa_online(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh-CN')

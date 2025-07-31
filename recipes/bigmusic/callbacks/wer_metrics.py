import pytorch_lightning as pl
from recipes.bigmusic.utils.format_utils import (
    update_json,
    normalize_text,
    remove_space_in_zh,
    load_json_locked
)
from prettytable import PrettyTable
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
import json, os
from pathlib import Path
import numpy as np
import tqdm
from recipes.bigmusic.utils.format_utils import normalize_text
from collections import defaultdict
import requests
import time
from uuid import uuid4
import base64
from recipes.bigmusic.callbacks.mir_metrics import upload_audio_file_to_tos
from recipes.bigmusic.callbacks.common_callbacks import metadata_check_decorator, ForceAlignCallback
from recipes.bigmusic.callbacks.plot_metrics import plot_wer
from recipes.musiclm.utils.dist import local_zero_first
from multiprocess.pool import ThreadPool
import re
from bytedance.easycycle import get_current_region, Region
from recipes.bigmusic.callbacks.common_callbacks import ASRCallback


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
    def __init__(self, asr_model_path='en_punc', transliteration=False, parallel=1, no_gt_lyrics=False):
        super().__init__()
        self.asr_model_path = asr_model_path
        self.transliteration = transliteration
        self.parallel = parallel
        self.no_gt_lyrics = no_gt_lyrics

    @metadata_check_decorator
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None) -> None:
        # process current chunk
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()
        if trainer.is_global_zero:
            ts = time.time()
            while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                time.sleep(10)
                print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")              
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            if self.no_gt_lyrics:
                run_wer_metrics_sa_online_wo_gtlyrics(generated_output_fps, asr_model_path=self.asr_model_path, transliteration=self.transliteration)
            else:
                run_wer_metrics_sa_online(generated_output_fps, asr_model_path=self.asr_model_path, transliteration=self.transliteration, parallel=self.parallel)
            try:
                output_wer_for_all_samples_into_one_file(output_dir)
                plot_wer(output_dir)
            except:
                print ('failed to plot or generate wer for all samples, for some unknown reason...')    # 只要文件夹结构没变，就不应该有问题，这里兜一下以防万一

class LinebreakMetricsOnlineCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    @metadata_check_decorator
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()
        if trainer.is_global_zero:
            ts = time.time()
            while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                time.sleep(10)
                print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")          
            # Upload the audio files to TOS
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

            # first retrieve audio url from metadata
            audio_list = []
            for generated_output_fp in generated_output_fps:
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                metadata = load_json_locked(metadata_fp)
                audio_url = metadata.get("easycycle_url", None)
                audio_list.append([str(generated_output_fp), audio_url])

            # back-up plan
            if len(audio_list) == 0:
                raise ValueError(f"No audio found in {output_dir}")  
                audio_list = upload_audio_file_to_tos(generated_output_fps)

            # Read the lyrics from the metadata files
            lyrics_list = []
            for idx, generated_output_fp in enumerate(generated_output_fps):
                generated_output_fp = Path(generated_output_fp)
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = load_json_locked(metadata_fp)
                lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))
                lyrics_list.append(normalize_lyrics(lyrics))

            print(audio_list)
            print(lyrics_list)

            durations = run_line_break_metrics_online(audio_list, lyrics_list)                
            num_bad_linebreaks, word_dur, linebreak_neighbor_dur, linebreak_dur = zip(*durations)
            id = list(range(len(word_dur)))
            avg_num_bad_linebreaks = sum(num_bad_linebreaks) / len(num_bad_linebreaks)
            avg_word_dur = sum(word_dur) / len(word_dur)
            avg_linebreak_neighbor_dur = sum(linebreak_neighbor_dur) / len(linebreak_neighbor_dur)
            avg_linebreak_dur = sum(linebreak_dur) / len(linebreak_dur)
            print("average num bad linebreaks per song: ", avg_num_bad_linebreaks)
            print("average word duration ms: ", avg_word_dur)
            print("average linebreak neighbor duration ms: ", avg_linebreak_neighbor_dur)
            print("average linebreak duration ms: ", avg_linebreak_dur)
            if 'output_dir' in pl_module.extra_params:
                with open(os.path.join(pl_module.extra_params.output_dir, f'linebreak_report.txt'), 'w') as fw:
                    report_tab = PrettyTable()
                    report_tab.add_column("index", id)
                    report_tab.add_column("num_bad_linebreaks", num_bad_linebreaks)
                    report_tab.add_column("word_dur", word_dur)
                    report_tab.add_column("linebreak_neighbor_dur", linebreak_neighbor_dur)
                    report_tab.add_column("linebreak_dur", linebreak_dur)
                    report_tab.add_row(["average", avg_num_bad_linebreaks, avg_word_dur, avg_linebreak_neighbor_dur, avg_linebreak_dur])
                    fw.write(report_tab.get_string())

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
        lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))
        a = normalize_text(remove_punc_case(lyrics))
        normalize_a = re.sub(r'\[.+?\]\n', '', a.lower())
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
        lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))

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

def run_lyric_transliteration_sa_online(origin_text, language):
    def sa_tts(content, voice, voice_type, codec="wav", timeout=20):
        """
        Example
        {
        "reqid": "a3273f8ee3db11e7bf2ff3223ff33638",
        "code": 3000,
        "message": "Success",
        "operation": "query",
        "sequence": -1,
        "data": "audio data encoded in base64"
        }
        """
        appid="xuling.9427"
        token="access_token"
        cluster="demo_test"
        region = get_current_region()
        if region == Region.CN:
            host = "speech.byted.org"
        elif region == Region.MALIVA:
            host = "speech-maliva.byted.org"
        else:
            raise ValueError("Unsupported region: {}".format(region))
        try:
            srv_url = f"http://{host}/api/v1/tts"
            req = {
                "app": {
                    "appid": appid,
                    "token": token,
                    "cluster": cluster,
                },
                "user": {
                    "uid": "388808087185088"
                },
                "audio": {
                    "voice": voice,
                    "voice_type": voice_type,
                    "encoding": codec,
                    "speed": 10,
                    "volume": 10,
                    "pitch": 10
                },
                "request": {
                    "reqid": str(uuid4()),
                    "text": content,
                    "text_type": "plain",
                    "operation": "query",
                    "with_frontend": 1,
                    "split_sentence": 0,
                    "return_mel": 0
                }
            }

            resp = requests.post(srv_url, json=req, timeout=timeout)
            if resp.status_code != 200:
                return None, None
            resp_json = resp.json()
            frontend_res = None
            audio = None
            if "description" in resp_json["addition"]:
                frontend_res = resp_json["addition"]["description"]
            if "data" in resp_json:
                b64_audio = resp_json["data"]
                audio = base64.b64decode(b64_audio)
            return frontend_res, audio
        except Exception as e:
            print(str(e))
        return None, None

    # support zh-CN: character to pinyin
    def _request_sa_tts_zh_cn(origin_text):
        voice = "CN_EN_MULTITASK"
        voice_type = "multitask_frontend"
        valid_list = [
          ["\u4E00", "\u9FA5"],
          ["\u9FA6", "\u9FFF"],
          ["\u3400", "\u4DBF"]
        ]
        text = ""
        output_text = ""
        for char in origin_text:
            valid_flag = 0
            if char == " " and text.strip() != "":
                valid_flag = 1
            for (v1, v2) in valid_list:
                if char >= v1 and char <= v2:
                    valid_flag = 1
            if valid_flag == 1:
                text += char
            else:
                if text.strip() != "":
                    frontend_res, _ = sa_tts(text, voice, voice_type)
                    sy_text = ""
                    for item in json.loads(frontend_res):
                        for key in item["json"]:
                            tk_list = item["json"][key]
                            ph_text = ""
                            for tk in tk_list:
                                orth = tk["orth"]
                                if tk["unitType"] == "text":
                                    if tk["isEnglish"] == 1:
                                        sy_text += orth
                                        sy_text += " "
                                    else:
                                        sy_text += tk["pinYin"].lower()
                                        sy_text += " "
                    #sy_text = re.sub("[0-9]", "", sy_text)
                    output_text += " %s" % sy_text.strip()
                    output_text += " %s" % char
                    text = ""
                else:
                    output_text += "%s" % char
                    text = ""
        if text.strip() != "":
            frontend_res, _ = sa_tts(text, voice, voice_type)
            sy_text = ""
            for item in json.loads(frontend_res):
                for key in item["json"]:
                    tk_list = item["json"][key]
                    ph_text = ""
                    for tk in tk_list:
                        orth = tk["orth"]
                        if tk["unitType"] == "text":
                            if tk["isEnglish"] == 1:
                                sy_text += orth
                                sy_text += " "
                            else:
                                sy_text += tk["pinYin"].lower()
                                sy_text += " "
            #sy_text = re.sub("[0-9]", "", sy_text)
            output_text += " %s" % sy_text.strip()
            text = ""
        return output_text

    if language == 'zh-CN':
        return _request_sa_tts_zh_cn(origin_text)

@DeprecationWarning
def run_asr_lyrics_sa_online(filepath, language='zh-CN'):
    print('Please use recipes.bigmusic.callbacks.common_callbacks.ASRCallback.run_asr_lyrics_sa_online() instead.')

    #base_url = 'http://speech-test.byted.org/api/v1/vc'
    #appid = "api_dev"
    #token = "lv_token"
    region = get_current_region()
    if region == Region.CN:
        host = 'speech.byted.org'
    elif region == Region.I18n:
        host = 'speech-maliva.byted.org'
    else:
        raise ValueError(f"Unsupported region: {region}")
    
    base_url = f'http://{host}/api/v1/vc'
    appid = 'bkseig7309i0'
    token = 'lv_token'
    access_token = 'YLp0eHdvLZH_IvgNSEHQrHsBeTdliqe0'
    #caption_type = "singing"
    caption_type = "auto"
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
    return text.strip(), None


def run_wer_metrics_sa_online(generated_output_fps, asr_model_path='zh-CN', transliteration=False, parallel=1):
    def compute_wer(ref, res):
        if not ref:
            return {key: np.nan for key in ['wer', 'ins error', 'subs error', 'dels error', 'ref length', 'ins', 'subs', 'dels', 'greedy_transcript', 'actual_transcript', 'badcase']}

        a = '' if ref is None else ref
        g = '' if res is None else res
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'wer': wer,
            'ins error': edits.ins,
            'subs error': edits.subs,
            'dels error': edits.dels,
            'ref length': denom,
            'ins': ins,
            'subs': subs,
            'dels': dels,
            'greedy_transcript': g,
            'actual_transcript': a,
            'badcase': 0,
        }
        if dels > 0.8:
            print("gt trans: ", a, "asr result: ", g, "path: ",str(generated_output_fp), "might be asr model error")
            wer_metadata['badcase'] = 1
        return wer_metadata

    def merge_all_wer(category2wer):
        all_ref_length = 0
        all_ins_err = 0
        all_subs_err = 0
        all_dels_err = 0
        all_badcase = 0
        all_support = 0
        wer_metadata_list = []
        for dir_path, wers in category2wer.items():
            metrics_fp = Path(dir_path)/'metrics.json'
            ref_length, ins_err, subs_err, dels_err, badcase, support = np.array(wers).sum(axis=0)
            if not np.isnan(ref_length):
                all_ref_length += ref_length
                all_ins_err += ins_err
                all_subs_err += subs_err
                all_dels_err += dels_err
                all_badcase += badcase
                all_support += support
            wer_metadata = {
                'wer': round((ins_err+subs_err+dels_err)/ref_length, 3),
                'ins': round(ins_err/ref_length, 3),
                'subs': round(subs_err/ref_length, 3),
                'dels': round(dels_err/ref_length, 3),
                'badcases': int(badcase) if not np.isnan(badcase) else np.nan,
                'badcase_rate': round(badcase/support, 3),
                'support': int(support) if not np.isnan(badcase) else np.nan,
            }
            wer_metadata_list.append([metrics_fp, wer_metadata])
        all_ref_length = all_ref_length if all_ref_length else np.nan
        all_wer_metadata = {
            'wer': round((all_ins_err+all_subs_err+all_dels_err)/all_ref_length, 3),
            'ins': round(all_ins_err/all_ref_length, 3),
            'subs': round(all_subs_err/all_ref_length, 3),
            'dels': round(all_dels_err/all_ref_length, 3),
            'badcases': int(all_badcase) if not np.isnan(all_badcase) else np.nan,
            'badcase_rate': round(all_badcase/all_support, 3) if all_support else np.nan,
            'support': int(all_support) if not np.isnan(all_support) else np.nan,
        }
        all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
        wer_metadata_list.append([all_metrics_fp, all_wer_metadata])
        return wer_metadata_list

    category2wer = defaultdict(list)
    category2pinyinwer = defaultdict(list)
    # all_asr_lyrics, all_generated_output_fps = [], []
    # pool = ThreadPool(parallel)
    # generated_output_fps_groups = np.array_split(generated_output_fps, parallel)
    # def run_asr_lyrics_sa_online_group(fps):
    #     group_lyrics = []
    #     for idx, fp in enumerate(fps):
    #         asr_lyrics = run_asr_lyrics_sa_online(fp, asr_model_path)
    #         group_lyrics.append(asr_lyrics)
    #     return group_lyrics
    
    # rets = []
    # for _fps in generated_output_fps_groups:
    #     ret = pool.apply_async(run_asr_lyrics_sa_online_group, args=(_fps,))
    #     rets.append(ret)
    # pool.close()

    # for _fps, ret in zip(generated_output_fps_groups, rets):
    #     all_asr_lyrics += ret.get()
    #     all_generated_output_fps += list(_fps)
    # pool.join()

    for idx, generated_output_fp in enumerate(generated_output_fps):
        # asr_lyrics = run_asr_lyrics_sa_online(generated_output_fp, asr_model_path)
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # actual transcript
        lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))
        lyrics = normalize_lyrics(lyrics)

        wer_metadata = {}
        if "wer" in metadata and "pinyin_wer" in metadata:
            wer_metadata["wer"] = metadata["wer"]
            wer_metadata["pinyin_wer"] = metadata["pinyin_wer"]
        else:
            timestamps = None
            if lyrics:      # only run asr when reference lyrics exists
                asr_lyrics = metadata.get('asr_lyrics', None)
                timestamps = metadata.get('asr_timestamps', None)
                if asr_lyrics is None:
                    asr_lyrics, timestamps = run_asr_lyrics_sa_online(generated_output_fp, asr_model_path)
                if asr_model_path in ['zh-CN', 'ja-JP']:
                    lyrics = normalize_text(remove_punc_case(lyrics))
                    lyrics = remove_space_in_zh(lyrics)
                    asr_lyrics = normalize_text(remove_punc_case(asr_lyrics))
                    asr_lyrics = remove_space_in_zh(asr_lyrics)
                    wer_metadata['wer'] = compute_wer(lyrics, asr_lyrics)
                    if transliteration:
                        trans_lyrics = run_lyric_transliteration_sa_online(lyrics, asr_model_path)
                        trans_asr_lyrics = run_lyric_transliteration_sa_online(asr_lyrics, asr_model_path)
                        wer_metadata['pinyin_wer'] = compute_wer(trans_lyrics, trans_asr_lyrics)
            else:
                wer_metadata['wer'] = {key: np.nan for key in ['wer', 'ins error', 'subs error', 'dels error', 'ref length', 'ins', 'subs', 'dels', 'greedy_transcript', 'actual_transcript', 'badcase']}
                wer_metadata['pinyin_wer'] = {key: np.nan for key in ['wer', 'ins error', 'subs error', 'dels error', 'ref length', 'ins', 'subs', 'dels', 'greedy_transcript', 'actual_transcript', 'badcase']}

            update_json(metadata_fp, replace_nan_with_none(wer_metadata))
            if timestamps:
                update_json(metadata_fp, {'asr_timestamps': timestamps})

        if isinstance(generated_output_fp, str):
            import os
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append([
                wer_metadata['wer']['ref length'],
                wer_metadata['wer']['ins error'],
                wer_metadata['wer']['subs error'],
                wer_metadata['wer']['dels error'],
                wer_metadata['wer']['badcase'],
                1
        ]) # append to base directory to calculate total wer
        if asr_model_path == 'zh-CN' and transliteration:
            category2pinyinwer[str(category_dir)].append([
                wer_metadata['pinyin_wer']['ref length'],
                wer_metadata['pinyin_wer']['ins error'],
                wer_metadata['pinyin_wer']['subs error'],
                wer_metadata['pinyin_wer']['dels error'],
                wer_metadata['pinyin_wer']['badcase'],
                1
            ])
    
    for metrics_fp, wer_metadata in merge_all_wer(category2wer):
        update_json(metrics_fp, {'wer': wer_metadata})
        print(f"output_dir={metrics_fp}, WER={wer_metadata}")
    if asr_model_path == 'zh-CN' and transliteration:
        for metrics_fp, wer_metadata in merge_all_wer(category2pinyinwer):
            update_json(metrics_fp, {'pinyin_wer': wer_metadata})
            print(f"output_dir={metrics_fp}, PINYIN_WER={wer_metadata}")

def run_wer_metrics_sa_online_wo_gtlyrics(generated_output_fps, asr_model_path='zh-CN', transliteration=False):
    def compute_wer(ref, res):
        if not ref:
            return {key: np.nan for key in ['wer', 'ins error', 'subs error', 'dels error', 'ref length', 'ins', 'subs', 'dels', 'greedy_transcript', 'actual_transcript', 'badcase']}
        a = '' if ref is None else ref
        g = '' if res is None else res
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'wer': wer,
            'ins error': edits.ins,
            'subs error': edits.subs,
            'dels error': edits.dels,
            'ref length': denom,
            'ins': ins,
            'subs': subs,
            'dels': dels,
            'greedy_transcript': g,
            'actual_transcript': a,
            'badcase': 0,
        }
        if dels > 0.8:
            print("gt trans: ", a, "asr result: ", g, "path: ",str(generated_output_fp), "might be asr model error")
            wer_metadata['badcase'] = 1
        return wer_metadata

    def merge_all_wer(category2wer):
        all_ref_length = 0
        all_ins_err = 0
        all_subs_err = 0
        all_dels_err = 0
        all_badcase = 0
        all_support = 0
        wer_metadata_list = []
        for dir_path, wers in category2wer.items():
            metrics_fp = Path(dir_path)/'metrics.json'
            ref_length, ins_err, subs_err, dels_err, badcase, support = np.array(wers).sum(axis=0)
            if not np.isnan(ref_length):
                all_ref_length += ref_length
                all_ins_err += ins_err
                all_subs_err += subs_err
                all_dels_err += dels_err
                all_badcase += badcase
                all_support += support
            wer_metadata = {
                'wer': round((ins_err+subs_err+dels_err)/ref_length, 3),
                'ins': round(ins_err/ref_length, 3),
                'subs': round(subs_err/ref_length, 3),
                'dels': round(dels_err/ref_length, 3),
                'badcases': int(badcase) if not np.isnan(badcase) else np.nan,
                'badcase_rate': round(badcase/support, 3),
                'support': int(support) if not np.isnan(badcase) else np.nan,
            }
            wer_metadata_list.append([metrics_fp, wer_metadata])
        all_ref_length = all_ref_length if all_ref_length else np.nan
        all_wer_metadata = {
            'wer': round((all_ins_err+all_subs_err+all_dels_err)/all_ref_length, 3),
            'ins': round(all_ins_err/all_ref_length, 3),
            'subs': round(all_subs_err/all_ref_length, 3),
            'dels': round(all_dels_err/all_ref_length, 3),
            'badcases': int(all_badcase) if not np.isnan(all_badcase) else np.nan,
            'badcase_rate': round(all_badcase/all_support, 3) if all_support else np.nan,
            'support': int(all_support) if not np.isnan(all_support) else np.nan,
        }
        all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
        wer_metadata_list.append([all_metrics_fp, all_wer_metadata])
        return wer_metadata_list

    category2wer = defaultdict(list)
    category2pinyinwer = defaultdict(list)
    for idx, generated_output_fp in tqdm.tqdm(enumerate(generated_output_fps)):
        asr_lyrics, _ = ASRCallback.run_asr_lyrics_sa_online(generated_output_fp, asr_model_path)
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        targetaudio_fp = str(generated_output_fp).replace('generated.wav', 'target_audio.wav')
        # actual transcript
        lyrics, _ = ASRCallback.run_asr_lyrics_sa_online(targetaudio_fp, asr_model_path)
        if lyrics == '': 
            asr_lyrics = lyrics = 'THIS CASE IS INST'
        wer_metadata = {}
        if asr_model_path in ['zh-CN', 'ja-JP']:
            lyrics = normalize_text(remove_punc_case(lyrics))
            lyrics = remove_space_in_zh(lyrics)
            asr_lyrics = normalize_text(remove_punc_case(asr_lyrics))
            asr_lyrics = remove_space_in_zh(asr_lyrics)
            wer_metadata['wer'] = compute_wer(lyrics, asr_lyrics)
            if transliteration:
                trans_lyrics = run_lyric_transliteration_sa_online(lyrics, asr_model_path)
                trans_asr_lyrics = run_lyric_transliteration_sa_online(asr_lyrics, asr_model_path)
                wer_metadata['pinyin_wer'] = compute_wer(trans_lyrics, trans_asr_lyrics)
        

        update_json(metadata_fp, wer_metadata)
        if isinstance(generated_output_fp, str):
            import os
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()

        category2wer[str(category_dir)].append([
                wer_metadata['wer']['ref length'],
                wer_metadata['wer']['ins error'],
                wer_metadata['wer']['subs error'],
                wer_metadata['wer']['dels error'],
                wer_metadata['wer']['badcase'],
                1
        ]) # append to base directory to calculate total wer
        if asr_model_path == 'zh-CN' and transliteration:
            category2pinyinwer[str(category_dir)].append([
                    wer_metadata['pinyin_wer']['ref length'],
                    wer_metadata['pinyin_wer']['ins error'],
                    wer_metadata['pinyin_wer']['subs error'],
                    wer_metadata['pinyin_wer']['dels error'],
                    wer_metadata['pinyin_wer']['badcase'],
                    1
            ])
    
    for metrics_fp, wer_metadata in merge_all_wer(category2wer):
        update_json(metrics_fp, {'wer': wer_metadata})
        print(f"output_dir={metrics_fp}, WER={wer_metadata}")
    if asr_model_path == 'zh-CN' and transliteration:
        for metrics_fp, wer_metadata in merge_all_wer(category2pinyinwer):
            update_json(metrics_fp, {'pinyin_wer': wer_metadata})
            print(f"output_dir={metrics_fp}, PINYIN_WER={wer_metadata}")


def run_wer_metrics_sa_online_parallel(generated_output_fps, asr_model_path='zh-CN', transliteration=False, parallel=10):
    def compute_wer(ref, res):
        if not ref:
            return {key: np.nan for key in ['wer', 'ins error', 'subs error', 'dels error', 'ref length', 'ins', 'subs', 'dels', 'greedy_transcript', 'actual_transcript', 'badcase']}

        a = '' if ref is None else ref
        g = '' if res is None else res
        edits = edit_distance(a, g)
        denom = 1.0 if len(a) == 0 else len(a)
        ins = round(edits.ins / denom, 3)
        subs = round(edits.subs / denom, 3)
        dels = round(edits.dels / denom, 3)
        wer = sum([ins, subs, dels])
        wer_metadata = {
            'wer': wer,
            'ins error': edits.ins,
            'subs error': edits.subs,
            'dels error': edits.dels,
            'ref length': denom,
            'ins': ins,
            'subs': subs,
            'dels': dels,
            'greedy_transcript': g,
            'actual_transcript': a,
            'badcase': 0,
        }
        if dels > 0.8:
            print("gt trans: ", a, "asr result: ", g, "might be asr model error")
            wer_metadata['badcase'] = 1
        return wer_metadata

    def merge_all_wer(category2wer):
        all_ref_length = 0
        all_ins_err = 0
        all_subs_err = 0
        all_dels_err = 0
        all_badcase = 0
        all_support = 0
        wer_metadata_list = []
        for dir_path, wers in category2wer.items():
            metrics_fp = Path(dir_path)/'metrics.json'
            ref_length, ins_err, subs_err, dels_err, badcase, support = np.array(wers).sum(axis=0)
            if not np.isnan(ref_length):
                all_ref_length += ref_length
                all_ins_err += ins_err
                all_subs_err += subs_err
                all_dels_err += dels_err
                all_badcase += badcase
                all_support += support
            wer_metadata = {
                'wer': round((ins_err+subs_err+dels_err)/ref_length, 3),
                'ins': round(ins_err/ref_length, 3),
                'subs': round(subs_err/ref_length, 3),
                'dels': round(dels_err/ref_length, 3),
                'badcases': int(badcase) if not np.isnan(badcase) else np.nan,
                'badcase_rate': round(badcase/support, 3),
                'support': int(support) if not np.isnan(badcase) else np.nan,
            }
            wer_metadata_list.append([metrics_fp, wer_metadata])
        all_ref_length = all_ref_length if all_ref_length else np.nan
        all_wer_metadata = {
            'wer': round((all_ins_err+all_subs_err+all_dels_err)/all_ref_length, 3),
            'ins': round(all_ins_err/all_ref_length, 3),
            'subs': round(all_subs_err/all_ref_length, 3),
            'dels': round(all_dels_err/all_ref_length, 3),
            'badcases': int(all_badcase) if not np.isnan(all_badcase) else np.nan,
            'badcase_rate': round(all_badcase/all_support, 3) if all_support else np.nan,
            'support': int(all_support) if not np.isnan(all_support) else np.nan,
        }
        all_metrics_fp = Path(dir_path.rsplit('/', 1)[0])/'all_metrics.json'
        wer_metadata_list.append([all_metrics_fp, all_wer_metadata])
        return wer_metadata_list

    def process_single_file(generated_output_fp):
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        metadata = load_json_locked(metadata_fp)

        # actual transcript
        lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))
        lyrics = normalize_lyrics(lyrics)

        if lyrics:
            asr_lyrics = metadata.get('asr_lyrics', None)
            if asr_model_path in ['zh-CN', 'ja-JP']:
                if "wer" in metadata and "pinyin_wer" in metadata:
                    wer_data = metadata['wer']
                    pinyin_wer_data = metadata['pinyin_wer']
                else:
                    lyrics_norm = remove_space_in_zh(normalize_text(remove_punc_case(lyrics)))
                    asr_lyrics_norm = remove_space_in_zh(normalize_text(remove_punc_case(asr_lyrics)))
                    wer_data = compute_wer(lyrics_norm, asr_lyrics_norm)
                    if transliteration:
                        trans_lyrics = run_lyric_transliteration_sa_online(lyrics_norm, asr_model_path)
                        trans_asr_lyrics = run_lyric_transliteration_sa_online(asr_lyrics_norm, asr_model_path)
                        pinyin_wer_data = compute_wer(trans_lyrics, trans_asr_lyrics)
        else:
            wer_data = {key: np.nan for key in ['wer','ins error','subs error','dels error','ref length','ins','subs','dels','greedy_transcript','actual_transcript','badcase']}
            pinyin_wer_data = {key: np.nan for key in ['wer','ins error','subs error','dels error','ref length','ins','subs','dels','greedy_transcript','actual_transcript','badcase']}

        update_json(metadata_fp, replace_nan_with_none({"wer": wer_data, "pinyin_wer": pinyin_wer_data}))

        if isinstance(generated_output_fp, str):
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()

        wer_vec = [
            wer_data['ref length'],
            wer_data['ins error'],
            wer_data['subs error'],
            wer_data['dels error'],
            wer_data['badcase'],
            1
        ]
        pinyin_wer_vec = None
        if asr_model_path=='zh-CN' and transliteration:
            pinyin_wer_vec = [
                pinyin_wer_data['ref length'],
                pinyin_wer_data['ins error'],
                pinyin_wer_data['subs error'],
                pinyin_wer_data['dels error'],
                pinyin_wer_data['badcase'],
                1
            ]
        print(f"output_dir={category_dir}, WER={wer_data}")
        return str(category_dir), wer_vec, pinyin_wer_vec

    def process_group(file_list):
        local_cat2wer = defaultdict(list)
        local_cat2pinyinwer = defaultdict(list)

        for fp in file_list:
            cat_dir, wer_vec, pinyin_wer_vec = process_single_file(fp)
            local_cat2wer[cat_dir].append(wer_vec)
            if pinyin_wer_vec is not None:
                local_cat2pinyinwer[cat_dir].append(pinyin_wer_vec)

        return (local_cat2wer, local_cat2pinyinwer)

    pool = ThreadPool(parallel)
    file_groups = np.array_split(generated_output_fps, parallel)

    rets = []
    for group in file_groups:
        ret = pool.apply_async(process_group, args=(group,))
        rets.append(ret)

    pool.close()

    category2wer_dict = defaultdict(list)
    category2pinyinwer_dict = defaultdict(list)

    for ret in rets:
        local_c2wer, local_c2pinyin = ret.get()
        for cat_dir, wer_list in local_c2wer.items():
            category2wer_dict[cat_dir].extend(wer_list)
        for cat_dir, pwer_list in local_c2pinyin.items():
            category2pinyinwer_dict[cat_dir].extend(pwer_list)

    pool.join()

    for metrics_fp, wer_metadata in merge_all_wer(category2wer_dict):
        update_json(metrics_fp, {'wer': wer_metadata})
        print(f"[WER] output_dir={metrics_fp}, WER={wer_metadata}")

    if asr_model_path=='zh-CN' and transliteration:
        for metrics_fp, pwer_metadata in merge_all_wer(category2pinyinwer_dict):
            update_json(metrics_fp, {'pinyin_wer': pwer_metadata})
            print(f"[PINYIN_WER] output_dir={metrics_fp}, PINYIN_WER={pwer_metadata}")

@DeprecationWarning
def run_force_align_online(url, gt_lyrics, verbose=True):
    print('Please use recipes.bigmusic.callbacks.common_callbacks.ForceAlignCallback.run_force_align_online() instead.')
    base_url = 'http://speech.byted.org/api/v1/vc'
    appid = 'ks9x6edfo8x1h0ts'
    token = 'lQpuXhcIeG_khGJRPdKK-5XpsR7Z6pDm'
    caption_type = "singing"
    max_retry_num = 5
    try_num = 0
    if verbose:
        print(url)
    gt_lyrics = re.sub(r'\[.+?\]\n', '', gt_lyrics)
    gt_lyrics = gt_lyrics.replace('\n', ',')
    if verbose:
        print(gt_lyrics)
    while try_num < max_retry_num:
        try:
            data = {
                # "url": "https://tosv.byted.org/obj/tostest/mingfei_zh_music_sample_30.wav",
                # "audio_text": "曾梦想仗剑走天涯,看一看世界的繁华,年少的心总有些轻狂,如今你四海为家"
                "url": url,
                "audio_text": gt_lyrics
            }
            response = requests.post(
                            '{base_url}/ata/submit'.format(base_url=base_url),
                            params=dict(
                                appid=appid,
                                token=token,
                                caption_type=caption_type,                                 
                                with_confidence='True',
                                verbose='True'
                            ),
                            json=data,
                            headers={
                            'content-type': 'application/json',
                            "Authorization": 'Bearer; {access_token}'.format(access_token=token)
                            }
                        )
            job_id = response.json()['id']
            print(response)
            time.sleep(3)
            response = requests.get(
                    '{base_url}/query'.format(base_url=base_url),
                    params=dict(
                        appid=appid,
                        token=token,
                        id=job_id,
                    ),
                    headers={
                        "Authorization": 'Bearer; {access_token}'.format(access_token=token)
                    }
            )
            print(response)
            aligned_utts = response.json()['utterances'][0]
            try_num = max_retry_num + 1
        except Exception as e:
            try_num += 1
            time.sleep(10)
    return aligned_utts 


def run_line_break_metrics_online(url_list, lyrics_list):
    durations = []
    for url, gt_lyrics in zip(url_list, lyrics_list):

        metadata_fp = str(url[0]).replace('generated.wav','metadata.json')
        metadata = load_json_locked(metadata_fp)

        if "force_align" in metadata:
            words = metadata["force_align"]["words"]
        else:
            words = ForceAlignCallback.run_force_align_online(url[1], gt_lyrics)["words"]
        word_ind = [i for i, word in enumerate(words) if (
            word["text"] != "," and word["text"] != "\n" and word["text"] != " ")]
        linebreak_neighbor_ind = []
        for i in word_ind:
            if (i+2 not in word_ind) or (i+3 not in word_ind) or (i-1 not in word_ind) or (i-2 not in word_ind):
                linebreak_neighbor_ind.append(i)
        

        word_list = []
        for i, j in zip(word_ind, word_ind[1:]):
            if i in word_ind:
                word_idx_a, word_idx_b = i, j        
                word_a = words[word_idx_a]
                word_b = words[word_idx_b]
                dur_a = word_b["start_time"] - word_a["start_time"]
                word_list.append({
                    "text": word_a["text"],
                    "start_time": word_a["start_time"],
                    "end_time": word_a["end_time"],
                    "duration": dur_a,
                    "is_linebreak": False if (i+1) in word_ind else True,
                    "is_linebreak_neighbor": True if i in linebreak_neighbor_ind else False
                })
        last_word = words[word_ind[-1]]
        word_list.append({
            "text": last_word["text"],
            "start_time": last_word["start_time"],
            "end_time": last_word["end_time"],
            "duration": last_word["end_time"] - last_word["start_time"],
            "is_linebreak": True,
            "is_linebreak_neighbor": False
        })

        # Compute average word duration.
        word_dur, word_ct = 0.0, 0
        linebreak_neighbor_dur, linebreak_neighbor_ct = 0.0, 0
        linebreak_dur, linebreak_ct = 0.0, 0
        for word in word_list:
            # print(word["text"], word['is_linebreak'], word['is_linebreak_neighbor'])                
            if word['is_linebreak'] == True:
                linebreak_dur += word['duration']
                linebreak_ct += 1
            if word['is_linebreak_neighbor'] == True:
                linebreak_neighbor_dur += word['duration']
                linebreak_neighbor_ct += 1
                word_dur += word['duration']
                word_ct += 1        
            else:
                word_dur += word['duration']
                word_ct += 1

        # Compute numbers of linebreak bad cases.
        num_bad_break = 0        
        for i, w in enumerate(word_list):     
            if (i - 2 < 0) or (i + 2 >= len(word_list)):
                continue   # hotfix by Yixiao Zhang，避免越界bug. 我们假设句子永远多于4个words。   
            
            if w['is_linebreak'] == True:
                long_dur = w['duration']
                short_dur = (
                    word_list[i-1]['duration'] + word_list[i-2]['duration']
                    + word_list[i+1]['duration'] + word_list[i+2]['duration']) / 4.0
                if long_dur < short_dur:
                    num_bad_break += 1                    
                    w = word_list[i-2]
                    print(w['text'], w['duration'], w['is_linebreak'])
                    w = word_list[i-1]
                    print(w['text'], w['duration'], w['is_linebreak'])
                    w = word_list[i]
                    print(w['text'], w['duration'], w['is_linebreak'])
                    w = word_list[i+1]
                    print(w['text'], w['duration'], w['is_linebreak'])
                    w = word_list[i+2]
                    print(w['text'], w['duration'], w['is_linebreak'])
                    print("")

        durations.append([
            num_bad_break, word_dur/word_ct, linebreak_neighbor_dur/linebreak_neighbor_ct, linebreak_dur/linebreak_ct])

    return durations

def replace_nan_with_none(obj):     # only for output json, convert NaN to null
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    return obj


def output_wer_for_all_samples_into_one_file(path_result):

    import os
    import glob
    import pandas as pd
    from tabulate import tabulate
    path_result = str(path_result)
    wer = {}
    pinyin_wer = {}
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    wer['all'] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
    pinyin_wer['all'] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
    keys = ['wer', 'ins', 'subs', 'dels']

    title = ['index', 'wer', 'ins', 'subs', 'dels', 'badcase',  'pinyin_wer', 'pinyin_ins', 'pinyin_subs', 'pinyin_dels', 'pinyin_badcase']
    df = pd.DataFrame(columns=title)

    for category in categories:
        wer[category] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
        pinyin_wer[category] = {'wer': [], 'ins': [], 'subs': [], 'dels': [], 'badcase': 0}
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            data = {}
            data['index'] = index
            data['wer'] = metadata['wer']['wer']
            data['ins'] = metadata['wer']['ins']
            data['subs'] = metadata['wer']['subs']
            data['dels'] = metadata['wer']['dels']
            data['badcase'] = metadata['wer']['badcase']
            data['pinyin_wer'] = metadata['pinyin_wer']['wer']
            data['pinyin_ins'] = metadata['pinyin_wer']['ins']
            data['pinyin_subs'] = metadata['pinyin_wer']['subs']
            data['pinyin_dels'] = metadata['pinyin_wer']['dels']
            data['pinyin_badcase'] = metadata['pinyin_wer']['badcase']
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    filename_out = os.path.join(path_result, 'wer_for_all_samples.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print(table)
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(table)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    parser.add_argument("--metrics_type", type=str, help="wer metrics type to calculate, wer/linebreak/all")
    args = parser.parse_args()
    # # run_relative_wer_metrics(args.input_dir, asr_model_path='zh')
    # #run_wer_metrics_svs(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh')
    # #run_wer_metrics_sa_online(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh-CN')
    # run_wer_metrics_sa_online(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh-CN', transliteration=True)
    # #run_wer_metrics_sa_online(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh-CN', transliteration=False)
    # output_wer_for_all_samples_into_one_file(args.input_dir)
    # plot_wer(args.input_dir)
    # output_dir = "assets/linebreak_sft_2287_10k_line_cfg_cfg3"

    generated_output_fps = list(Path(args.input_dir).glob('**/*.generated.wav'))
    # print(generated_output_fps)
    # first retrieve audio url from metadata
    audio_list = []
    for generated_output_fp in generated_output_fps:
        metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        audio_url = metadata.get("easycycle_url", None)
        audio_list.append([str(generated_output_fp), audio_url])

    # back-up plan
    if len(audio_list) == 0:
        raise ValueError(f"No audio found in {output_dir}")  
        audio_list = upload_audio_file_to_tos(generated_output_fps)
    if args.metrics_type in ['wer', 'all']:
        run_wer_metrics_sa_online(list(Path(args.input_dir).glob('**/*.generated.wav')), asr_model_path='zh-CN', transliteration=True)

    if args.metrics_type in ['linebreak', 'all']:
        # Read the lyrics from the metadata files
        lyrics_list = []
        for idx, generated_output_fp in enumerate(generated_output_fps):
            generated_output_fp = Path(generated_output_fp)
            metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
            with open(metadata_fp, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            lyrics = metadata.get('lyrics_eval', metadata.get('lyrics'))
            lyrics_list.append(lyrics)

        durations = run_line_break_metrics_online(audio_list, lyrics_list)
        # breakpoint()
        num_bad_linebreaks, word_dur, linebreak_neighbor_dur, linebreak_dur = zip(*durations)
        id = list(range(len(word_dur)))
        avg_num_bad_linebreaks = sum(num_bad_linebreaks) / len(num_bad_linebreaks)
        avg_word_dur = sum(word_dur) / len(word_dur)
        avg_linebreak_neighbor_dur = sum(linebreak_neighbor_dur) / len(linebreak_neighbor_dur)
        avg_linebreak_dur = sum(linebreak_dur) / len(linebreak_dur)
        print(avg_num_bad_linebreaks, avg_word_dur, avg_linebreak_neighbor_dur, avg_linebreak_dur)

        with open(os.path.join(args.input_dir, f'linebreak_report.txt'), 'w') as fw:
            report_tab = PrettyTable()
            report_tab.add_column("index", id)
            report_tab.add_column("num_bad_linebreaks", num_bad_linebreaks)
            report_tab.add_column("word_dur", word_dur)
            report_tab.add_column("linebreak_neighbor_dur", linebreak_neighbor_dur)
            report_tab.add_column("linebreak_dur", linebreak_dur)
            report_tab.add_row(["average", avg_num_bad_linebreaks, avg_word_dur, avg_linebreak_neighbor_dur, avg_linebreak_dur])
            fw.write(report_tab.get_string())

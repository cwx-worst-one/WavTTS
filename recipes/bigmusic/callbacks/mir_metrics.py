from operator import gt
import os
import time
import json
import euler
import thriftpy2
import logging
import numpy as np
import tempfile
import torch
from torch.cuda.amp import autocast
from typing import Any, Union
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
import pytorch_lightning as pl
from recipes.bigmusic.datasets.zh_inference import format_style_text
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json, load_json_locked
from recipes.bigmusic.utils.upload import upload_to_tos_v2
from recipes.bigmusic.datasets.mir_data_util import SA_GENRE20, MAP_SUB_GENRE_2_SA_GENRE20, MACRO_STYLE_MAP_MOOD2_SA_MOOD19, SA_MOOD19
from recipes.bigmusic.datasets.utils.zh_vocab import AUDIO_V4_TO_SA_TAG_MAP
from recipes.bigmusic.scripts.tos import upload_to_easycycle
from recipes.mir_benchmark.tagging_inference.genre import GenreTagging
from recipes.mir_benchmark.tagging_inference.instrument import InstrumentTagging
from recipes.mir_benchmark.tagging_inference.vocal import VocalTagging
from recipes.mir2.pl_modules.pl_datamodule import convert_audio_ffmpeg_file_to_file
import librosa
import soundfile as sf
import pandas as pd
from tabulate import tabulate
import re
from recipes.musiclm.utils.dist import local_zero_first
from multiprocessing.pool import ThreadPool
import requests
import random
from copy import deepcopy
from hyperpyyaml import load_hyperpyyaml
from samantha.utils.parser import parse_arguments
from tqdm import tqdm
from samantha.utils.hdfs_helper import get as hdfs_get
from recipes.bigmusic.callbacks.common_callbacks import metadata_check_decorator

thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "../utils/services/idl/music_tagging.thrift"), "music_tagging_thrift"
)


from music_tagging_thrift import MusicTagging, TaggingRequest
from urllib.request import Request, urlopen
import base64
import uuid
import ast

try:
    from recipes.bigmusic.callbacks.api_call import api_call
except Exception as e:
    print(f"[WARNING] Failed to import recipes.bigmusic.callbacks.api_call due to {e}")
    print(f"It's a must for MIRPitchMetricsCallback/MIRSectionTransitionCallback/MIRSectionLabelCallback")
    api_call = None
    

class MIRTagMetricsCallback(pl.Callback):
    def __init__(self, tags="genre,instrument,vocal"):
        super().__init__()
        tagging_models = {}
        if "genre" in tags:
            tagging_models['genre'] = GenreTagging()
        if "instrument" in tags:
            tagging_models['instrument'] = InstrumentTagging()
        if "vocal" in tags:
            tagging_models['vocal'] = VocalTagging()
        self.tagging_models = tagging_models
        self.batch_accuracies = { tag: [] for tag in tagging_models.keys() }

    @staticmethod
    def gt_model_predict(model, target_audio, generated_audio, device):
        model.to(device) # to fix bug in pytorch lightning where "setup" does not have correct device yet
        if target_audio.shape == 2:
            target_audio = target_audio.unsqueeze(1)
        if generated_audio.shape == 2:
            generated_audio = generated_audio.unsqueeze(1)
        gt_tags, gt_preds = model.predict(target_audio.to(device))
        gen_tags, gen_preds = model.predict(generated_audio.to(device))
        accuracies = MIRTagMetricsCallback.multilabel_accuracy(gt_tags, gen_tags)
        return accuracies

    @staticmethod
    def semantic_model_predict(model, target_tags, generated_audio, device):
        model.to(device) # to fix bug in pytorch lightning where "setup" does not have correct device yet
        target_tags = [tags.split(',') if isinstance(tags, str) else tags for tags in target_tags]
        if generated_audio.shape == 2:
            generated_audio = generated_audio.unsqueeze(1)
        gen_tags, gen_preds = model.predict(generated_audio.to(device))
        accuracies = MIRTagMetricsCallback.multilabel_accuracy(target_tags, gen_tags)
        return accuracies

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # for now, calculate using target audio and generate audio
        device = pl_module.device
        generated_audio = outputs['generated_audio_tensor']
        batch_size = generated_audio.shape[0]
        if 'target_audio' in batch: # Ground Truth use case
            target_audio = batch['target_audio']
            tag_metadata = {}
            for tag, model in self.tagging_models.items():
                accuracies = MIRTagMetricsCallback.gt_model_predict(model, target_audio, generated_audio, device)
                accuracy_metadata = [ { tag: round(x, 3) } for x in accuracies ]
                tag_metadata = concat_metadata_list(tag_metadata, accuracy_metadata)
            self.batch_accuracies[tag].extend(accuracies)
            outputs['metadata'] = concat_metadata_list(outputs.get('metadata'), tag_metadata)
        elif 'style_text' in batch: # style_text use case
            pass # TODO (AS) finish tagging for MIR

    @staticmethod
    def multilabel_accuracy(pred_labels, target_labels):
        accuracies = []
        for pred, target in zip(pred_labels, target_labels):
            pred_set = set(pred)
            target_set = set(target)
            target_set - pred_set
            correct = len(pred_set & target_set)
            if len(target_set) == 0:
                accuracy = 1
            else:
                accuracy = correct / len(target_set)
            accuracies.append(accuracy)
        return accuracies

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tag_metadata = { tag: np.mean(tag_accuracies) for tag, tag_accuracies in self.batch_accuracies.items() }
        output_dir = pl_module.extra_params.output_dir
        metrics_fp = Path(output_dir)/'metrics.json'
        update_json(metrics_fp, { 'MIR tag acc': tag_metadata })


def upload_to_tos(audio_fp, prefix, tos_cli="/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli", bucket="sa-music-model-zoo", ak="9NC3OBANMDH4TTPTO52E"):
    os.system(f'{tos_cli} -bucket {bucket} -accessKey {ak} put -prefix {prefix} "{audio_fp}"')
    file_name = os.path.basename(audio_fp)
    url = f'https://tosv.byted.org/obj/{bucket}/{prefix}/{file_name}'
    return url


def map_gt_categorical_genre(genre_text):
    if isinstance(genre_text, str):
        genres = genre_text.split(',')
    else:
        genres = genre_text
    gt_genres = []
    for g in genres:
        if not g:
            continue
        if g in SA_GENRE20:
            gt_genres.append(g)
        elif g in AUDIO_V4_TO_SA_TAG_MAP:
            gt_genres.append(AUDIO_V4_TO_SA_TAG_MAP[g])
        else:
            print(f"unrecognized genre {g}")
    return list(set(gt_genres))

def map_gt_categorical_mood(mood_text):
    moods = mood_text.split(',')
    gt_moods = []
    for m in moods:
        if m in SA_MOOD19:
            gt_moods.append(m)
        elif m in MACRO_STYLE_MAP_MOOD2_SA_MOOD19:
            gt_moods.append(MACRO_STYLE_MAP_MOOD2_SA_MOOD19[m])
        else:
            print(f"unrecognized mood {m}")
    return list(set(gt_moods))

def map_gt_categorical_gender(gender_text):
    genders = gender_text.split(',')
    gt_genders = []
    for g in genders:
        g = g.lower().strip()
        gt_genders.append(g)
    return list(set(gt_genders))

def map_gt_categorical_lang(lang_text):
    if isinstance(lang_text, str):
        langs = lang_text.split(',')
    else:
        langs = lang_text
    gt_langs = []
    for l in langs:
        gt_langs.append(l.capitalize())
    return list(set(gt_langs))

def TaggingGender(audio_url):
    cluster = "gender_detect"
    audio_type = 'wav'
    app_id = "bigmusic_data_test"
    threshold_config= {
        "male": 0.6,
        "female": 0.65,
        #"adult": 0.5,
        #"child": 0.85
    }

    #with open(audio_path, 'rb') as f:
    #    content = f.read()
    #    content = base64.b64encode(bytes(content))
    req = Request(audio_url)
    response = urlopen(req, timeout=30)
    content = response.read()
    content = base64.b64encode(bytes(content))
    headers = {'Content-Type': "application/json;"}
    uuid_str = str(uuid.uuid4())
    url = "https://speech-test.byted.org/api/v1/aed?reqid=%s" % uuid_str
    post_data = {
        "app":{
            "appid": app_id,
            "token": "access_token",
            "cluster": cluster,
        },
        "user": {
            "uid": "388808087185088"
        },
        "audio":{
            "rate": 16000,
            "format": audio_type,
            "data": content,
        },
        "request": {
            "reqid": uuid_str,
            "sequence": -1,
            "nbest": 5,
            "workflow": "audio_in,resample,partition,vad",
        }
    }

    max_retry_num = 5
    retry_num = 0
    while retry_num < max_retry_num:
        res = requests.post(url, json=post_data, headers=headers)
        try:
            res_json = res.json()
        except Exception as e:
            res_json = {"code": -1}
            retry_num += 1
        else:
            break
    #print(json.dumps(res_json))
    res = []
    if res_json['code'] == 1000:
        for item in res_json['event_items']:
            event = item['event']
            utt_prob = item['utt_prob']
            if event not in threshold_config:
                continue
            thres = threshold_config[event]
            if utt_prob > thres:
                res.append(event)
    return res, res_json['event_items']


def SA_online_tagging_predict(client, audio_url, tag="genre"):
    if tag == "genre":
        rlts = client.TaggingGenre20(TaggingRequest(track_id="test", url=audio_url))
        try:
            result = eval(rlts.result_json)['Genre20']['result']
            result_json = eval(rlts.result_json)['Genre20']
        except:
            print(f"failed: {audio_url} check silence")
            result = []
            result_json = {}
    if tag == 'mood':
        rlts = client.TaggingMood(TaggingRequest(track_id="test", url=audio_url))
        result = eval(rlts.result_json)['Mood']['result']
        result_json = eval(rlts.result_json)['Mood']
    if tag == 'theme':
        rlts = client.TaggingTheme(TaggingRequest(track_id="test", url=audio_url))
        result = eval(rlts.result_json)['Theme']['result']
        result_json = eval(rlts.result_json)['Theme']
    if tag == 'gender':
        result, result_json = TaggingGender(audio_url)
    if tag == 'lang':
        rlts = client.TaggingLanguage(TaggingRequest(track_id="test", url=audio_url))
        try:
            result = eval(rlts.result_json)['Language']['result']
            result_json = eval(rlts.result_json)['Langauage']
        except:
            print(f"failed: {audio_url} check silence")
            result = []
            result_json = {}
    return result, result_json


def parse_gt_tag_from_metadata(audio_file_path, tag="genre"):
    metadata_fp = str(audio_file_path).replace('generated.wav', 'metadata.json')
    with open(metadata_fp, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    gt_style_text = metadata.get('style_text')
    #print('gt_style_text: ', gt_style_text)
    #gt_style_text = format_style_text(gt_style_text)  # insert a reformat logic to make it compatible with the previous code
    if '|' in gt_style_text:
        gt_style_text = gt_style_text.split('|')
    if len(gt_style_text) == 3:
        genre_idx = 0
        genre_extra_idx = None
        mood_idx = 1
        gender_idx = 2
        lang_idx = None
    elif len(gt_style_text) == 9:
        # [genre,genre_extra,extra,mood,scene,gender,vocal,language,is_sinking]
        genre_idx = 0
        genre_extra_idx = 1
        mood_idx = 3
        gender_idx = 5
        lang_idx = 7
    elif len(gt_style_text) == 13:
        genre_idx = 0
        genre_extra_idx = 1
        mood_idx = 3
        gender_idx = 5
        lang_idx = 7
    else:
        genre_idx = 0
        genre_extra_idx = None
        mood_idx = None
        gender_idx = None
        lang_idx = None

    if tag == "genre":
        genres = gt_style_text[genre_idx]
        if genre_extra_idx is not None:
            genres_extra = gt_style_text[genre_extra_idx]
        else:
            genres_extra = []
        if genre_idx is not None:
            return map_gt_categorical_genre(genres + genres_extra)
        else:
            return []
    if tag == 'mood':
        if mood_idx is not None:
            moods = gt_style_text[mood_idx]
            return map_gt_categorical_mood(moods)
        else:
            return []
    if tag == 'gender':
        if gender_idx is not None:
            genders = gt_style_text[gender_idx]
            return map_gt_categorical_gender(genders)
        else:
            return []
    if tag == 'lang':
        if lang_idx is not None:
            langs = gt_style_text[lang_idx]
            return map_gt_categorical_lang(langs)
        else:
            return []


def upload_audio_file_to_tos(audio_file_paths):
    tos_prefix = 'tmp/infer/wavs_test/' + datetime.now().strftime("%m%d%Y%H%M%S")

    audio_list = []
    for audio_fp in audio_file_paths:
        url = upload_to_tos_v2(audio_fp, tos_prefix)
        audio_list.append([str(audio_fp), url])
    return audio_list


def run_tagging_acc(audio_list, tag="genre", parallel=4):
    target = 'sd://lab.speech.music_tagging?cluster=default'
    client = euler.Client(MusicTagging,target=f'{target}&idc=lf', timeout=700)
    report_tab = PrettyTable([tag, "accuracy", "avg probs", "support"])
    rlts = {}
    total_correct = 0
    all_total = 0
    # parallel
    def predict_tagging(url, audio_fp):
        gt_tags = parse_gt_tag_from_metadata(audio_fp, tag)
        predict_tags, result_json = SA_online_tagging_predict(client, url, tag)
        return (gt_tags, predict_tags, result_json)

    all_results = []

    rets = []
    pool = ThreadPool(parallel)
    audio_lst_groups = np.array_split(audio_list, parallel)

    def predict_tagging_group(audio_lst):
        group_results = []
        for (audio_fp, url) in audio_lst:
            group_results.append(predict_tagging(url, audio_fp))
        return group_results

    for audio_lst in audio_lst_groups:
        ret = pool.apply_async(
            predict_tagging_group,
            args=(audio_lst,),
        )
        rets.append(ret)
    pool.close()

    for ret in rets:
        result = ret.get()
        all_results+=result
    pool.join()

    for (gt_tags, predict_tags, result_json) in all_results:
        # audio_fp, url = item[0], item[1]
        # gt_tags = parse_gt_tag_from_metadata(audio_fp, tag)
        # predict_tags, result_json = SA_online_tagging_predict(client, url, tag)
        print(gt_tags, predict_tags)
        if len(gt_tags) == 1 and gt_tags[0] == '':      # skip tagging acc calculation, if no gt style parsed (e.g., freeform text was used)
            continue
        all_total += 1
        flag = False
        for pt_tag in predict_tags:
            if pt_tag in gt_tags:
                flag = True
                total_correct += 1
                if pt_tag not in rlts:
                    rlts[pt_tag] = {"correct": 1, "total": 1, "probs": result_json[pt_tag]}
                else:
                    rlts[pt_tag]['correct'] += 1
                    rlts[pt_tag]['total'] += 1
                    rlts[pt_tag]['probs'] += result_json[pt_tag]
                break

        if not flag:
            for gt_tag in gt_tags:
                if gt_tag not in rlts:
                    rlts[gt_tag] = {"correct": 0, "total": 1, "probs": result_json.get(gt_tag, 0.0)}
                else:
                    rlts[gt_tag]['total'] += 1
                    rlts[gt_tag]['probs'] += result_json.get(gt_tag, 0.0)

    total_probs = 0.0
    for k, v in rlts.items():
        k_acc = round(v["correct"] / v["total"], 4)
        avg_prob = round(v["probs"] / v["total"], 4)
        total_probs += v["probs"]
        report_tab.add_row([k, k_acc, avg_prob, v["total"]])

    total_acc = round(total_correct / all_total, 4) if all_total else np.nan
    total_avg_prob = round(total_probs / all_total, 4) if all_total else np.nan
    report_tab.add_row(['total', total_acc, total_avg_prob, all_total])

    return report_tab


class MIRTagMetricsSAOnlineCallback(pl.Callback):
    #def __init__(self, tags=["genre", "mood", "gender"]) -> None:
    #def __init__(self, tags=["genre", "gender"]) -> None:
    def __init__(self, tags="['genre']", parallel=4) -> None:
        super().__init__()
        self.tags = ast.literal_eval(tags)
        self.parallel = parallel

    @metadata_check_decorator
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()

        if trainer.is_global_zero:
            ts = time.time()
            while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                time.sleep(10)
                print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)")
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))

            # first retrieve audio url from metadata
            audio_list = []
            for generated_output_fp in generated_output_fps:
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                with open(metadata_fp, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                audio_url = metadata.get("easycycle_url", None)
                if audio_url is None:
                    # warning, please run easycycle first
                    print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] No audio url found in {metadata_fp}, please run easycycle first")
                    
                audio_list.append([str(generated_output_fp), audio_url])

            # back-up plan
            if len(audio_list) == 0:
                raise ValueError(f"No audio found in {output_dir}")  
                audio_list = upload_audio_file_to_tos(generated_output_fps)

            for tag in self.tags:
                report = run_tagging_acc(audio_list, tag, self.parallel)
                print(report)
            if 'output_dir' in pl_module.extra_params:
                with open(os.path.join(pl_module.extra_params.output_dir, f'{tag}_report.txt'), 'w') as fw:
                    fw.write(report.get_string())


class MIRSectionTransitionCallback(pl.Callback):

    def __init__(self, online=True) -> None:
        super().__init__()
        self.online = online

    def calculate_prf(self, predicted_indices, ground_truth_indices):
        """
        Calculate Precision, Recall, and F1-score for evaluating prediction performance.

        Args:
            predicted_indices:  List of predicted lyric line indices (list).
            ground_truth_indices: List of ground truth lyric line indices (list).

        Returns:
            A dictionary containing Precision, Recall, and F1-score.
        """

        predicted_set = set(predicted_indices)  # Convert predicted indices list to a set for efficient operations
        ground_truth_set = set(ground_truth_indices) # Convert ground truth indices list to a set

        # Calculate True Positives (TP): Number of correctly predicted and actually correct indices (intersection)
        tp_set = predicted_set.intersection(ground_truth_set)
        tp = len(tp_set)

        # Calculate False Positives (FP): Number of incorrectly predicted as correct indices (predicted set - intersection)
        fp_set = predicted_set.difference(ground_truth_set)
        fp = len(fp_set)

        # Calculate False Negatives (FN): Number of incorrectly predicted as incorrect indices (ground truth set - intersection)
        fn_set = ground_truth_set.difference(predicted_set)
        fn = len(fn_set)

        # Calculate Precision - avoid division by zero
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        # Calculate Recall - avoid division by zero
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        # Calculate F1-score - avoid division by zero (F1 is 0 if Precision or Recall is 0)
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score

    def merge_timestamps(self, asr_timestamp_reformat):
        lyrics_line_timestamp = []
        current_line_words = []

        for item in asr_timestamp_reformat:
            if item[0] == ',':
                if current_line_words:  # 确保当前行有词语
                    line_start_time = current_line_words[0][1][0]
                    line_end_time = current_line_words[-1][1][1]
                    line_text = "".join([word[0] for word in current_line_words])

                    lyrics_line_timestamp.append({
                        "text": line_text.strip(),
                        "start": line_start_time / 1000,
                        "end": line_end_time / 1000
                    })
                    current_line_words = []  # 重置当前行词语列表
            else:
                current_line_words.append(item)

        # 处理最后一行，如果最后一行没有逗号分隔
        if current_line_words:
            line_start_time = current_line_words[0][1][0]
            line_end_time = current_line_words[-1][1][1]
            line_text = "".join([word[0] for word in current_line_words])
            lyrics_line_timestamp.append({
                "text": line_text.strip(),
                "start": line_start_time / 1000,
                "end": line_end_time / 1000
            })

        return lyrics_line_timestamp

    def align_deepchorus_to_lines(self, deepchorus_transition_timestamps, lyrics_line_timestamp):
        deepchorus_line_indices = []
        for target_time in deepchorus_transition_timestamps:
            min_time_diff = float('inf') 
            closest_index = -1          

            for index, lyric_item in enumerate(lyrics_line_timestamp):
                lyric_start_time = lyric_item['start']
                time_diff = abs(target_time - lyric_start_time) 

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_index = index

            deepchorus_line_indices.append(closest_index)

        return deepchorus_line_indices

    def get_timestamps_for_lyrics(self, generated_output_fp, lyrics):
        metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            asr_timestamp = metadata.get("force_align", {})
        if len(asr_timestamp) == 0:
            return None
        asr_timestamp = asr_timestamp['words']
        asr_timestamp_reformat = []
        for item in asr_timestamp:
            text = item['text']
            start_time = item['start_time']
            end_time = item['end_time']
            asr_timestamp_reformat.append([text, [start_time, end_time]])
        
        return asr_timestamp_reformat

    def get_ground_truth_indices(self, lyrics):
        lines = lyrics.strip().split('\n')
        transition_indices_gt = []
        is_new_section = True
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                is_new_section = True
                continue
            if line:
                if is_new_section:
                    transition_indices_gt.append(1)
                    is_new_section = False
                else:
                    transition_indices_gt.append(0)
        return transition_indices_gt
    
    @metadata_check_decorator
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None) -> None:
        """ Capture the section transition accuracy at lyrics line level. Yixiao Zhang, 2025.03.06.

        1. We use DeepChorus (MusicFM) to estimate the structure of the AR generated audio.
        2. We align the audio timestamp to lyrics indexes.
        3. We extract the lyrics text to get line-level transition points.
        4. We calculate P, R, F scores and return.
        """
        # get estimated structure for generated audio, write into metadata.json

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if trainer is None or trainer.is_global_zero:

            
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()  

            # check if deepchorus_segment exist. If all exist, do not run deepchorus
            counter = 0
            for generated_output_fp in generated_output_fps:
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                metadata = json.load(open(metadata_fp, 'r', encoding='utf-8'))
                deepchorus_segment = metadata.get('deepchorus_segment')
                if deepchorus_segment is not None:
                    counter += 1
            if counter == len(generated_output_fps):
                print(f"[{self.__class__.__name__}] All {len(generated_output_fps)} audio have deepchorus_segment, skip running deepchorus.")
            else:
                run_deepchorus(trainer, output_dir, online=self.online) 
            
            # calculate scores per audio
            precision_list = []
            recall_list = []
            f1_list = []


            output_dir = output_dir
            metrics_fp = Path(output_dir)/'MIRTransition.txt'

            # pretty table qol improvement
            results_table = PrettyTable(['index', 'precision', 'recall', 'f1', 'gt', 'pred'])
            results_table.align['gt'] = 'l'
            results_table.align['pred'] = 'l'

            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):
                # read estimated structure from metadata.json
                index = generated_output_fp.stem
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = json.load(open(metadata_fp, 'r', encoding='utf-8'))
                deepchorus_segment = metadata.get('deepchorus_segment') 
                lyrics = metadata['lyrics']

                # get transition timestamps
                deepchorus_transition_timestamps = [section[0][0] for section in deepchorus_segment if section[1] not in ["intro", "outro", "inst", "interlude", "end", "silence"]]

                # get ground truth indices from lyrics
                transition_indices_gt_onehot = self.get_ground_truth_indices(lyrics)
                transition_indices_gt = [index for index, value in enumerate(transition_indices_gt_onehot) if value == 1]

                # get timestamp for lyrics. 
                asr_timestamp_reformat = self.get_timestamps_for_lyrics(generated_output_fp, lyrics)

                if asr_timestamp_reformat is None:
                    continue

                # merge timestamp to get line-level timestamp
                lyrics_line_timestamp = self.merge_timestamps(asr_timestamp_reformat)

                # align deepchorus timestamp to line indices
                deepchorus_line_indices = self.align_deepchorus_to_lines(deepchorus_transition_timestamps, lyrics_line_timestamp)

                # print(f"Lyrics GT : {transition_indices_gt}")
                # print(f"DeepChorus: {deepchorus_line_indices}")

                # calculate P R F scores between aligned lines and gt lines
                p, r, f1 = self.calculate_prf(deepchorus_line_indices, transition_indices_gt)
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f1)

                # restrict to 4 decimal places
                p = round(p, 4)
                r = round(r, 4)
                f1 = round(f1, 4)

                results_table.add_row([index, p, r, f1, str(transition_indices_gt), str(deepchorus_line_indices)])

            # calculate average scores
            avg_precision = sum(precision_list) / len(precision_list)
            avg_recall = sum(recall_list) / len(recall_list)
            avg_f1 = sum(f1_list) / len(f1_list)

            # restrict to 4 decimal places
            avg_precision = round(avg_precision, 4)
            avg_recall = round(avg_recall, 4)
            avg_f1 = round(avg_f1, 4)

            results_table.add_row(['AVERAGE', avg_precision, avg_recall, avg_f1, '', ''])

            print(results_table.get_string())

            # write to file
            with open(metrics_fp, 'w', encoding='utf-8') as f:
                f.write(results_table.get_string())

            return {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    def run_parallel(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None, parallel=3) -> None:
        """ Capture the section transition accuracy at lyrics line level. Yixiao Zhang, 2025.03.06.

        1. We use DeepChorus (MusicFM) to estimate the structure of the AR generated audio.
        2. We align the audio timestamp to lyrics indexes.
        3. We extract the lyrics text to get line-level transition points.
        4. We calculate P, R, F scores and return.
        """
        # get estimated structure for generated audio, write into metadata.json

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if trainer is None or trainer.is_global_zero:

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()  
            # check if deepchorus_segment exist. If all exist, do not run deepchorus
            counter = 0
            for generated_output_fp in generated_output_fps:
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                metadata = load_json_locked(metadata_fp)
                deepchorus_segment = metadata.get('deepchorus_segment')
                if deepchorus_segment is not None and isinstance(deepchorus_segment, list):
                    counter += 1
            if counter == len(generated_output_fps):
                print(f"[{self.__class__.__name__}] All {len(generated_output_fps)} audio have deepchorus_segment, skip running deepchorus.")
            else:
                run_deepchorus_parallel(trainer, output_dir, online=self.online, parallel=parallel) 
            
            # calculate scores per audio
            precision_list = []
            recall_list = []
            f1_list = []

            output_dir = output_dir
            metrics_fp = Path(output_dir)/'MIRTransition.txt'

            # pretty table qol improvement
            results_table = PrettyTable(['index', 'precision', 'recall', 'f1', 'gt', 'pred'])
            results_table.align['gt'] = 'l'
            results_table.align['pred'] = 'l'

            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):
                # read estimated structure from metadata.json
                index = generated_output_fp.stem
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = load_json_locked(metadata_fp)
                deepchorus_segment = metadata.get('deepchorus_segment')  # str to list
                lyrics = metadata['lyrics']

                # get transition timestamps
                deepchorus_transition_timestamps = [section[0][0] for section in deepchorus_segment if section[1] not in ["intro", "outro", "inst", "interlude", "end", "silence"]]

                # get ground truth indices from lyrics
                transition_indices_gt_onehot = self.get_ground_truth_indices(lyrics)
                transition_indices_gt = [index for index, value in enumerate(transition_indices_gt_onehot) if value == 1]

                # get timestamp for lyrics. 
                asr_timestamp_reformat = self.get_timestamps_for_lyrics(generated_output_fp, lyrics)

                if asr_timestamp_reformat is None:
                    continue

                # merge timestamp to get line-level timestamp
                lyrics_line_timestamp = self.merge_timestamps(asr_timestamp_reformat)

                # align deepchorus timestamp to line indices
                deepchorus_line_indices = self.align_deepchorus_to_lines(deepchorus_transition_timestamps, lyrics_line_timestamp)

                # print(f"Lyrics GT : {transition_indices_gt}")
                # print(f"DeepChorus: {deepchorus_line_indices}")

                # calculate P R F scores between aligned lines and gt lines
                p, r, f1 = self.calculate_prf(deepchorus_line_indices, transition_indices_gt)
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f1)

                # restrict to 4 decimal places
                p = round(p, 4)
                r = round(r, 4)
                f1 = round(f1, 4)
                results_table.add_row([index, p, r, f1, str(transition_indices_gt), str(deepchorus_line_indices)])
            
                with open(metrics_fp, 'a') as f:
                    f.write(f"{index}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{str(transition_indices_gt)}\t{str(deepchorus_line_indices)}\n")

            # calculate average scores
            avg_precision = sum(precision_list) / len(precision_list)
            avg_recall = sum(recall_list) / len(recall_list)
            avg_f1 = sum(f1_list) / len(f1_list)

            avg_precision = round(avg_precision, 4)
            avg_recall = round(avg_recall, 4)
            avg_f1 = round(avg_f1, 4)

            results_table.add_row(['AVERAGE', avg_precision, avg_recall, avg_f1, '', ''])
            print(results_table.get_string())

            with open(metrics_fp, 'w', encoding='utf-8') as f:
                f.write(results_table.get_string())

            return {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
        

class StructureConfidenceCallback(pl.Callback):
    """计算结构一些额外的指标的callback。

    1. Average function score
    2. Chorus onset boundary score
    [TODO] 3. 段落之间的起承转合 (intensity change) chorus-verse / pairwise contrast
    """

    def __init__(self, online=True) -> None:
        super().__init__()
        self.online = online

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None) -> None:

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir
        
        if trainer is None or trainer.is_global_zero:

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()

            output_dir = output_dir
            metrics_fp = Path(output_dir)/'MIRStructureConfidence.txt'

            result_table = PrettyTable(['index', 'chorus_onset','function_score'])
            result_table.align["function_score"] = "l"

            function_confidence_aggregate = {}

            chorus_boundary_score_list = []

            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):
                index = generated_output_fp.stem
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = load_json_locked(metadata_fp)

                function_confidence_dict = {}

                chorus_boundary_score = []

                # 首先检测dir的metadata.json里面有没有deepchorus_segment_raw字段
                deepchorus_segment_raw = metadata.get('deepchorus_segment_raw', None)
                
                if deepchorus_segment_raw is None:
                    print(f"[MIRStructureConfidenceCallback: Warning] {generated_output_fp} does not have deepchorus_segment_raw in metadata.json")
                    
                    run_deepchorus(trainer, output_dir, online=self.online)

                for j, segment in enumerate(deepchorus_segment_raw):
                    # example: [{'interval': [0.0, 9.6], 'label': 'intro', 'funct_prob': 0.9689, 'start_prob': 1.0}, {'interval': [9.6, 28.8], 'label': 'verse', 'funct_prob': 0.9841, 'start_prob': 0.8365}, {'interval': [28.8, 43.0], 'label': 'pre-chorus', 'funct_prob': 0.9088, 'start_prob': 0.804}, {'interval': [43.0, 52.4], 'label': 'chorus', 'funct_prob': 0.9861, 'start_prob': 0.7905}, {'interval': [52.4, 62.2], 'label': 'chorus', 'funct_prob': 0.9902, 'start_prob': 0.0174}, {'interval': [62.2, 71.8], 'label': 'inst', 'funct_prob': 0.7146, 'start_prob': 0.7422}, {'interval': [71.8, 91.2], 'label': 'verse', 'funct_prob': 0.9783, 'start_prob': 0.6887}, {'interval': [91.2, 110.4], 'label': 'chorus', 'funct_prob': 0.8879, 'start_prob': 0.773}, {'interval': [110.4, 119.8], 'label': 'chorus', 'funct_prob': 0.4131, 'start_prob': 0.5649}, {'interval': [119.8, 130.4], 'label': 'intro', 'funct_prob': 0.382, 'start_prob': 0.1995}]

                    # 计算每次transit到chorus的边界分数，以及每个function的平均confident
                    # 计算chorus的边界分数
                    if segment['label'] == 'chorus' and ((j == 0) or (deepchorus_segment_raw[j-1]['label'] != 'chorus')):
                        chorus_boundary_score.append(segment['start_prob'])

                    # 计算每个function的平均confident
                    if segment['label'] not in function_confidence_dict:
                        function_confidence_dict[segment['label']] = []
                    function_confidence_dict[segment['label']].append(segment['funct_prob'])

                    # 乐段间对比。需要instrument标注和vocals的标注。
                    # TODO: Yixiao

                # 计算每个function的平均confident
                for function, confidence_list in function_confidence_dict.items():
                    function_confidence_dict[function] = sum(confidence_list) / len(confidence_list)
                
                    if function not in function_confidence_aggregate:
                        function_confidence_aggregate[function] = []
                    function_confidence_aggregate[function].append(function_confidence_dict[function])

                # round to 4
                for function, confidence in function_confidence_dict.items():
                    function_confidence_dict[function] = round(confidence, 4)

                function_confidence_dict_string = json.dumps(function_confidence_dict).replace('"', '').replace('{', '').replace('}', '')

                # 计算chorus的边界分数
                if len(chorus_boundary_score) == 0:
                    chorus_boundary_score = 0
                else:
                    chorus_boundary_score = sum(chorus_boundary_score) / len(chorus_boundary_score)

                chorus_boundary_score_list.append(chorus_boundary_score)

                # round to 4
                chorus_boundary_score = round(chorus_boundary_score, 4)

                result_table.add_row([index, chorus_boundary_score, function_confidence_dict_string])

            # 计算每个function的平均confident
            for function, confidence_list in function_confidence_aggregate.items():
                function_confidence_aggregate[function] = sum(confidence_list) / (len(confidence_list) + 1e-6)
            
            # 计算chorus的边界分数 (filter掉分数为0的）
            chorus_boundary_score_list = [elem for elem in chorus_boundary_score_list if elem !=0]
            avg_chorus_boundary_score = sum(chorus_boundary_score_list) / (len(chorus_boundary_score_list) + 1e-6)

            # round to 4
            avg_chorus_boundary_score = round(avg_chorus_boundary_score, 4)
            # round to 4
            for function, confidence in function_confidence_aggregate.items():
                function_confidence_aggregate[function] = round(confidence, 4)

            result_table.add_row(['AVERAGE', avg_chorus_boundary_score, json.dumps(function_confidence_aggregate).replace('"', '').replace('{', '').replace('}', '')])
            print(result_table.get_string())
            with open(metrics_fp, 'w', encoding='utf-8') as f:
                f.write(result_table.get_string())
            return {
                "chorus_boundary_score": avg_chorus_boundary_score,
                "function_score": function_confidence_aggregate
            }

class MIRSectionDurationCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()
    
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if trainer.is_global_zero:
            if 'output_dir' in pl_module.extra_params:
                run_deepchorus(trainer, pl_module.extra_params.output_dir)
                output_section_duration_metric(pl_module.extra_params.output_dir)
                output_dir = pl_module.extra_params.output_dir
                (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()

                if trainer.is_global_zero:
                    ts = time.time()
                    while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                        time.sleep(10)
                        print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)")

                    output_total_duration_metric(pl_module.extra_params.output_dir)


def output_total_duration_metric(output_dir):
    generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
    title = ['index', 'reference_total_duration', 'detected_total_duration']
    df = pd.DataFrame(columns=title)
    for generated_output_fp in generated_output_fps:
        index = generated_output_fp.stem
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        metadata = load_json_locked(metadata_fp)
        reference_total_duration = metadata.get('total_duration')
        if reference_total_duration:
            data = {}
            data['index'] = index
            wav, sr = librosa.load(str(generated_output_fp))
            detected_total_duration = wav.shape[-1] / sr
            data['reference_total_duration'] = reference_total_duration
            data['detected_total_duration'] = detected_total_duration
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    if not df.empty:
        filename_out = os.path.join(output_dir, 'total_duration_metrics.txt')
        table = tabulate(df, headers='keys', tablefmt='grid')
        print(table)
        with open(filename_out, 'w') as f:
            f.write(table)
            f.write('\n')
        duration_errors = abs( df['reference_total_duration'].to_numpy() - df['detected_total_duration'].to_numpy() )
        with open(filename_out, 'a') as f:
            line = 'duration_error_median: %.2f sec' % np.median(duration_errors)
            print (line)
            f.write(line + '\n')
            line = 'duration_error_mean: %.2f sec' % np.mean(duration_errors)
            print (line)
            f.write(line + '\n')


class MIRSectionLabelCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_dir' in pl_module.extra_params:
            run_deepchorus(trainer, pl_module.extra_params.output_dir)
            output_section_label_metric(pl_module.extra_params.output_dir)
            output_section_duration_metric(pl_module.extra_params.output_dir)


class MIRPitchMetricsCallback(pl.Callback):
    def __init__(self, online=True) -> None:
        super().__init__()
        self.online = online

    def on_predict_end(self, 
            trainer: "pl.Trainer", 
            pl_module: "pl.LightningModule",
            output_dir=None) -> None:
        
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir
       

        if trainer is None or trainer.is_global_zero:
            print('\n\n\n\n\n\n\n===================')
            print(' MIR Evaluation...')
            print('output_dir:', output_dir )
            import glob
            # filelist = glob.glob(
                # os.path.join(output_dir, '*.mp3')) + glob.glob(os.path.join(output_dir, '*.wav'))

            filelist = list(Path(f'{output_dir}').glob('**/*.mp3')) + list(Path(f'{output_dir}').glob('**/*.wav'))
            print(' [i] num files:', len(filelist))

            for file in tqdm(filelist):
                print(' [i] file:', file)
                os.system('ffmpeg -i "{}" -y -ar 24000 -ac 1 tmp_24k_mono.wav'.format(file))
                if self.online:
                    data = api_call('tmp_24k_mono.wav', model="bigmir-vocal2midi", verbose=False)[0][0]
                else:
                    raise NotImplementedError
                print(data)
                of = str(file) + '_v2m.json'
                print(' >>> out:', of)
                with open(of, 'w') as f:
                    json.dump(data, f)
                    
            ### run pitch metrics ### 
            def pitch_range(notes):
                '''the difference of highest and lowest ptiches'''
                pitches = [note['pitch'] for note in notes]
                return max(pitches) - min(pitches) if pitches else 0

            def pitch_interval_diversity(notes):
                '''the diversity of the intervals between 2 successive notes'''
                intervals = [abs(notes[i]['pitch'] - notes[i-1]['pitch']) for i in range(1, len(notes))]
                return len(set(intervals))
        
            def compute_melody_metrics(melody):

                PR = pitch_range(melody)
                PID = pitch_interval_diversity(melody)

                return {
                    'pitch_range': PR,
                    'pitch_interval_diversity': PID,
                }

            melody_files = list(Path(f'{output_dir}').glob('**/*_v2m.json'))

            # compute stats
            data = []
            for fn in melody_files:
                # load notes
                with open(fn, 'r') as f:
                    notes = json.load(f)

                # update metadata of each song
                metadata_fp = str(fn).replace(
                    '.generated.wav_v2m.json', '.metadata.json')
                song_result = {
                    "melody": compute_melody_metrics(notes)}
                update_json(metadata_fp, {
                    'leadsheet_metrics': song_result})  
                
                # append to dataset stats
                data.append(song_result)
            
            # compute dataset stats
            score_keys = data[0]["melody"].keys()
            result = {"melody": {}}
            for key in score_keys:
                values = [song["melody"][key] for song in data]
                result["melody"][key] = {
                    "mean": round(np.mean(values), 3), 
                    "std": round(np.std(values, ddof=1), 3)}
            
            path_score = os.path.join(str(output_dir), 'MIR_score.json')
            with open(path_score, 'w') as f:
                json.dump(result, f, indent=4)
            print('/n/n==============/n/n')
            print('result:', result)
            print('[o] PASS')

    def run_parallel(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None, parallel=3) -> None:
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if trainer is None or trainer.is_global_zero:
            print('\n\n\n\n\n\n\n===================')
            print(' MIR Evaluation...')
            print('output_dir:', output_dir)

            filelist = list(Path(output_dir).glob('**/*.mp3')) + list(Path(output_dir).glob('**/*.wav'))
            print(f' [i] num files: {len(filelist)}')

            tmp_dir = Path(output_dir) / "tmp_ffmpeg"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            def process_single_file(file_path: Path):
                out_json = str(file_path).replace('generated.wav', 'v2m.json')
                if os.path.exists(out_json):
                    print(f"{out_json} already exists, skip")
                    return

                unique_name = f"{uuid.uuid4().hex}_24k_mono.wav"
                tmp_file = tmp_dir / unique_name

                cmd = f'ffmpeg -i "{file_path}" -y -ar 24000 -ac 1 "{tmp_file}"'
                os.system(cmd)
                data = api_call(str(tmp_file), model="bigmir-vocal2midi", verbose=False)[0][0]
                
                with open(out_json, 'w') as f:
                    json.dump(data, f)

                try:
                    tmp_file.unlink() 
                except OSError as e:
                    print(f"[WARN] Failed to remove tmp file {tmp_file}: {e}")

            def process_group(file_list):
                for fp in file_list:
                    process_single_file(fp)

            pool = ThreadPool(parallel)
            file_groups = np.array_split(filelist, parallel)
            print("Begin calling Vocal2midi (online) in parallel...")
            rets = []
            for group in file_groups:
                ret = pool.apply_async(process_group, args=(group,))
                rets.append(ret)
            pool.close()
            for ret in rets:
                ret.get()
            pool.join()
            print("Vocal2midi online done")

            def pitch_range(notes):
                '''the difference of highest and lowest ptiches'''
                pitches = [note['pitch'] for note in notes]
                return max(pitches) - min(pitches) if pitches else 0

            def pitch_interval_diversity(notes):
                '''the diversity of the intervals between 2 successive notes'''
                intervals = [abs(notes[i]['pitch'] - notes[i-1]['pitch']) for i in range(1, len(notes))]
                return len(set(intervals))
        
            def compute_melody_metrics(melody):

                PR = pitch_range(melody)
                PID = pitch_interval_diversity(melody)

                return {
                    'pitch_range': PR,
                    'pitch_interval_diversity': PID,
                }

            melody_files = list(Path(f'{output_dir}').glob('**/*.v2m.json'))

            data = []
            for fn in melody_files:
                with open(fn, 'r') as f:
                    notes = json.load(f)

                metadata_fp = str(fn).replace('v2m.json', 'metadata.json')
                song_result = {"melody": compute_melody_metrics(notes)}
                update_json(metadata_fp, {'leadsheet_metrics': song_result})  
                data.append(song_result)
            
            score_keys = data[0]["melody"].keys()
            result = {"melody": {}}
            for key in score_keys:
                values = [song["melody"][key] for song in data]
                result["melody"][key] = {
                    "mean": round(np.mean(values), 3), 
                    "std": round(np.std(values, ddof=1), 3)}
            
            path_score = os.path.join(str(output_dir), 'MIR_score.json')
            with open(path_score, 'w') as f:
                json.dump(result, f, indent=4)
            print('\n\n==============\n\n')
            print('result:', result)
        
        return result


def run_deepchorus(trainer, output_dir, musicfm_version="20250210", online=True):

    def merge_labels(raws):
        # merging consecutive raw segments with the same labels
        segments = []
        ss = 0
        for i in range(len(raws)):
            if i == len(raws) - 1 or raws[i]["label"] != raws[i + 1]["label"]:
                ee = i
                funct = raws[i]["label"]
                start = raws[ss]["interval"][0]
                end = raws[ee]["interval"][1]
                segment = {
                    "interval": [start, end],
                    "label": funct,
                }
                segments.append(segment)
                ss = i + 1
        return segments

    if trainer is None or trainer.global_rank == 0:

        if musicfm_version == "20250210":

            if online:
                """ DeepChorus 2, using online service. Supported by @Yixiao Zhang and @Yakun Sun.
                
                """

                output_dir = os.path.abspath(output_dir)
                musicfm_dir = os.path.join(output_dir, 'musicfm_result')

                os.makedirs(musicfm_dir, exist_ok=True)

                generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
                generated_output_fps.sort()

                generated_output_fps_str = [str(fp) for fp in generated_output_fps]

                for generated_output_fp in tqdm(generated_output_fps):

                    resp = api_call(audio_path=generated_output_fp, model="bigmir-deepchorus2")

                    musicfm_segment_raw = resp[0]
                    musicfm_segment = merge_labels(musicfm_segment_raw)
                    musicfm_segment = [[item['interval'], item['label']] for item in musicfm_segment]
                    # print(musicfm_segment)
                    metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                    update_json(metadata_fp, { 'deepchorus_segment': musicfm_segment})  
                    update_json(metadata_fp, { 'deepchorus_segment_raw': musicfm_segment_raw})

                # command = f'rm -r {musicfm_dir}'
                # os.system(command)

            else:
                """ MusicFM new structure model ver. 20250210. @Yixiao Zhang

                New feature:
                - New keyword: "pre-chorus"
                - Improved performance on HR5F and ACC.
                """

                # additional dependency
                # os.system("pip3 install jiwer mir_eval colorednoise simplejson pretty_midi madmom music21 mido confusables")

                # os.system("pip3 install pytorch-lightning --upgrade")  # bug when loading deepspeed ckpt when lightning version <=2.1.0. 
                #                                                     # Best solution: upgrade lightning.

                output_dir = os.path.abspath(output_dir)
                musicfm_dir = os.path.join(output_dir, 'musicfm_result')

                os.makedirs(musicfm_dir, exist_ok=True)

                original_world_size = os.environ.get("WORLD_SIZE")
                os.environ["WORLD_SIZE"] = "1"
                
                # check pretrained ckpt exists
                if not os.path.exists('pretrained_musicfm/'):
                    print('The pretrained ckpt of MusicFM is not found.')
                    os.system(f"hdfs dfs -get hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/pretrained_musicfm/")
                if not os.path.exists("epoch=349-step=140000.ckpt"):
                    print('The pretrained ckpt of structure head is not found.')
                    os.system(f"hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/yixiao/epoch=349-step=140000.ckpt")

                generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
                generated_output_fps.sort()

                generated_output_fps_str = [str(fp) for fp in generated_output_fps]

                # inference script
                command = f"python3 -m samantha.main predict \
                            -c recipes/mir_benchmark/conf/structure/structure_inference_50s_prechorus.yaml \
                            --ckpt_path epoch=349-step=140000.ckpt \
                            --pl_datamodule.audio_files {generated_output_fps_str} \
                            --prediction_writer.output_dir {musicfm_dir}"
                os.system(command)

                if original_world_size is not None:
                    os.environ["WORLD_SIZE"] = original_world_size

                # write the musicfm results into metadata.json
                for generated_output_fp in generated_output_fps:
                    index = generated_output_fp.stem
                    musicfm_output_fp = musicfm_dir + f"/{index}.json"
                    if not os.path.exists(musicfm_output_fp):
                        continue
                    metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                    musicfm = json.load(open(musicfm_output_fp, 'r'))
                    musicfm_segment = musicfm['results']
                    update_json(metadata_fp, { 'deepchorus_segment': musicfm_segment })  
                
                # command = f'rm -r {musicfm_dir}'
                # os.system(command)
            
        else:
            """ Legacy verion.
            """
            if not os.path.exists('bigmusic_sami_models'):  # a folder of deepchorus repo, pulled from entry script (will replace with service call in the future)
                print ('The git submodule of [bigmusic_sami_models] is not found. Skip running section_duration accuracy metric.')
                return

            output_dir = os.path.abspath(output_dir)
            deepchorus_dir = os.path.join(output_dir, 'deepchorus_result')
            os.chdir('bigmusic_sami_models')
            command = f'python3 batch.py -i {output_dir}/ -o {deepchorus_dir} -t structure'
            os.system(command)
            os.chdir('..')

            # write the deepchorus results into metadata.json
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()
            for generated_output_fp in generated_output_fps:
                index = generated_output_fp.stem
                deepchorus_output_fp = deepchorus_dir / generated_output_fp.relative_to(output_dir).with_suffix('.json')
                if not deepchorus_output_fp.exists():
                    continue
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                deepchorus = json.load(open(deepchorus_output_fp, 'r'))
                deepchorus_segment = deepchorus['structure']['segment']
                update_json(metadata_fp, { 'deepchorus_segment': deepchorus_segment })  
            
            command = f'rm -r {deepchorus_dir}'
            os.system(command)

def run_deepchorus_parallel(trainer, output_dir, musicfm_version="20250210", online=True, parallel=10):
    
    def merge_labels(raws):
        # merging consecutive raw segments with the same labels
        segments = []
        ss = 0
        for i in range(len(raws)):
            if i == len(raws) - 1 or raws[i]["label"] != raws[i + 1]["label"]:
                ee = i
                funct = raws[i]["label"]
                start = raws[ss]["interval"][0]
                end = raws[ee]["interval"][1]
                segment = {
                    "interval": [start, end],
                    "label": funct,
                }
                segments.append(segment)
                ss = i + 1
        return segments
    
    if trainer is None or trainer.global_rank == 0:
        
        if musicfm_version == "20250210":

            if online:
                """ DeepChorus 2, using online service. Supported by @Yixiao Zhang and @Yakun Sun.
                
                """

                output_dir = os.path.abspath(output_dir)
                musicfm_dir = os.path.join(output_dir, 'musicfm_result')

                os.makedirs(musicfm_dir, exist_ok=True)

                generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
                generated_output_fps.sort()

                def process_single_file(generated_output_fp):

                    resp = api_call(audio_path=generated_output_fp, model="bigmir-deepchorus2")

                    musicfm_segment_raw = resp[0]
                    musicfm_segment = merge_labels(musicfm_segment_raw)
                    musicfm_segment = [[item['interval'], item['label']] for item in musicfm_segment]

                    metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                    update_json(metadata_fp, { 'deepchorus_segment': musicfm_segment})
                    update_json(metadata_fp, { 'deepchorus_segment_raw': musicfm_segment_raw})
                    print(f"Successfully write DeepChorus result into {metadata_fp}")
                
                def process_group(file_list):
                    for fp in file_list:
                        process_single_file(fp)

                # 创建线程池
                pool = ThreadPool(parallel)
                file_groups = np.array_split(generated_output_fps, parallel)
                print("Begin calling DeepChorus (online) in parallel...")
                rets = []
                for group in file_groups:
                    ret = pool.apply_async(process_group, args=(group,))
                    rets.append(ret)
                pool.close()
                for ret in rets:
                    ret.get()
                pool.join()
                print("DeepChorus online done")



def output_section_duration_metric(output_dir):
    generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
    generated_output_fps.sort()

    deepchorus_dir = os.path.join(output_dir, 'deepchorus_result')
    
    n_boundaries = 0
    n_correct_boundaries = 0
    title = ['index', 'reference_total_duration', 'detected_total_duration', 'total_duration_error', 'reference_boundaries', 'detected_boundaries', 'section_boundary_recall']
    df = pd.DataFrame(columns=title)
    for generated_output_fp in generated_output_fps:
        index = generated_output_fp.stem
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        metadata = json.load(open(metadata_fp, 'r'))

        data = {}
        data['index'] = index
        data['detected_total_duration'] = None
        data['reference_total_duration'] = None
        data['total_duration_error'] = None
        data['detected_boundaries'] = None
        data['reference_boundaries'] = None
        data['section_boundary_recall'] = None
        valid_sample = False

        try:
            deepchorus_segment = metadata.get('deepchorus_segment')
            detected_total_duration = deepchorus_segment[-1][0][1]
        except:
            deepchorus_segment = None
            wav, sr = librosa.load(str(generated_output_fp))
            detected_total_duration = wav.shape[-1] / sr

        section_durations = metadata.get('section_durations')
        reference_total_duration = metadata.get('total_duration')
        if not reference_total_duration:
            if section_durations:
                reference_total_duration = sum(section_durations)
        # reference total_duration can be either from prompt total_duration, or sum of section_durations
        data['reference_total_duration'] = reference_total_duration
        if reference_total_duration:        # in case sometime there is no reference_total_duration
            # detected total_duration can be either from deepchorus (preferred), or generated audio duration
            valid_sample = True
            data['detected_total_duration'] = detected_total_duration
            data['total_duration_error'] = abs(reference_total_duration - detected_total_duration)

        # calculate section duration metric
        # when section_durations exsits in prompt AND deepchorus result is available
        if section_durations and deepchorus_segment:
            reference_boundaries = [sum(section_durations[:i+1]) for i in range(len(section_durations)-1)]
            detected_boundaries = [x[0][1] for x in deepchorus_segment[:-1]]
            n_boundary = len(reference_boundaries)
            n_correct_boundary = sum(1 for x in reference_boundaries for y in detected_boundaries if abs(x-y) < 1 )       # threshold is 1sec
            data['reference_boundaries'] = ', '.join(map(str, reference_boundaries))
            data['detected_boundaries'] = ', '.join(map(str, detected_boundaries))
            data['section_boundary_recall'] = n_correct_boundary / n_boundary
            # used for calculate global metric across all pieces
            n_boundaries += n_boundary
            n_correct_boundaries += n_correct_boundary   
        
        if valid_sample:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        section_duration_metrics = data.copy()
        del section_duration_metrics['index']
        update_json(metadata_fp, {'section_duration_metrics': section_duration_metrics})

    if not df.empty:
        filename_out = os.path.join(output_dir, 'section_duration_metrics.txt')
        table = tabulate(df, headers='keys', tablefmt='grid')
        print(table)
        with open(filename_out, 'w') as f:
            f.write(table)
            f.write('\n')

        section_boundary_recall = n_correct_boundaries / n_boundaries if n_boundaries else np.nan
        duration_error_median = np.nanmedian(abs(df['reference_total_duration'].to_numpy() - df['detected_total_duration'].to_numpy()))
        duration_error_mean = np.nanmean(abs(df['reference_total_duration'].to_numpy() - df['detected_total_duration'].to_numpy()))

        with open(filename_out, 'a') as f:
            line = 'duration_error_mean: %.2f sec' % duration_error_mean
            print (line)
            f.write(line + '\n')
            line = 'duration_error_median: %.2f sec' % duration_error_median
            print (line)
            f.write(line + '\n')
            line = 'section_boundary_recall: %.1f%%' % (section_boundary_recall*100)
            print (line)
            f.write(line+ '\n')

def output_section_label_metric(output_dir):

    generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
    generated_output_fps.sort()

    title = ['index', 'reference_section_labels', 'detected_section_labels', 'section_label_accuracy']
    df = pd.DataFrame(columns=title)

    reference_section_labels_all = []   # used to calculate P/R for all pieces
    detected_section_labels_all = []

    for generated_output_fp in generated_output_fps:

        # print ('run section label accuracy for: ', generated_output_fp)
        index = generated_output_fp.stem
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        deepchorus_segment = metadata.get('deepchorus_segment')
        asr_timestamps = metadata.get('asr_timestamps')
        wer = metadata['wer']['wer']
        if not deepchorus_segment:
            print (f'The deepchorus result is not found. Skip running section_label accuracy metric for {generated_output_fp.stem}.')
            continue
        if not asr_timestamps:
            print (f'The ASR timestamps result is not found. Skip running section_label accuracy metric for {generated_output_fp.stem}.')
            continue
        if not wer or wer > 0.5:
            print (f'The wer is not valid. Skip running section_label accuracy metric for {generated_output_fp.stem}.')
            continue
        
        lyrics = metadata.get('lyrics', None)
        asr_lyrics = asr_timestamps
        deepchorus_segment = deepchorus_segment
        section_timestamps, alignment_messages = align_lyrics(lyrics, asr_lyrics)
        
        reference_section_labels, detected_section_labels = align_section_labels(section_timestamps, deepchorus_segment)
        if len(reference_section_labels):
            section_label_accuracy = sum(x==y for x, y in zip(reference_section_labels, detected_section_labels)) / len(reference_section_labels)
        else:
            section_label_accuracy = np.nan
        reference_section_labels_all.extend(reference_section_labels)
        detected_section_labels_all.extend(detected_section_labels)
        data = {}
        data['index'] = index
        data['reference_section_labels'] = ', '.join(reference_section_labels)
        data['detected_section_labels'] = ', '.join(detected_section_labels)
        data['section_label_accuracy'] = '%.1f%%' % (section_label_accuracy*100) if section_label_accuracy else section_label_accuracy
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        section_label_metrics = data.copy()
        del section_label_metrics['index']
        section_label_metrics['alignment_messages'] = alignment_messages

        update_json(metadata_fp, {'section_label_metrics': section_label_metrics})
    
    metrics = calculate_section_label_p_r(reference_section_labels_all, detected_section_labels_all)
    
    title = ['section_label', 'TP', 'n_reference', 'n_detection', 'precision', 'recall']
    df_all = pd.DataFrame(columns=title)
    for section_label in metrics:
        data = {}
        data['section_label'] = section_label
        for k in ['TP', 'n_detection', 'n_reference']:
            data[k] = metrics[section_label][k]
        for k in ['precision', 'recall']:
            data[k] = '%.1f%%' % (metrics[section_label][k]*100) if (metrics[section_label][k] and not np.isnan(metrics[section_label][k])) else metrics[section_label][k]
        df_all = pd.concat([df_all, pd.DataFrame([data])], ignore_index=True)
    
    filename_out = os.path.join(output_dir, 'section_label_metrics.txt')
    if not df_all.empty:

        overall_accuracy = sum(a==b for a, b in zip(reference_section_labels_all, detected_section_labels_all)) / len(reference_section_labels_all)
        with open(filename_out, 'w') as f:
            f.write('====================================================================\n')
            f.write('Overall accuracy (hit rate) for all section labels across all pieces\n')
            f.write('====================================================================\n')
            f.write('%.1f%%' % (overall_accuracy*100))
            f.write('\n\n')

            f.write('==========================================================\n')
            f.write('Precision/Recall for each section label, across all pieces\n')
            f.write('==========================================================\n')
            table_all = tabulate(df_all, headers='keys', tablefmt='grid')
            f.write(table_all)
            f.write('\n\n')
    
    if not df.empty:
        with open(filename_out, 'a') as f:
            f.write('==========================================================\n')
            f.write('Section label accuracy for each individual piece\n')
            f.write('==========================================================\n')
            table = tabulate(df, headers='keys', tablefmt='grid')
            f.write(table)
            f.write('\n\n')
    
    if os.path.exists(filename_out):
        with open(filename_out, 'r', encoding='utf-8') as file: print(file.read())


def calculate_section_label_p_r(reference_section_labels_all, detected_section_labels_all):
    '''
    - input:
        - reference_section_labels_all:     ['verse', 'chorus', 'verse', 'chorus', 'verse', 'bridge']
        - detected_section_labels_all:      ['verse', 'verse', 'verse', 'chorus', 'chorus', 'verse']
    - output:
        - metrics:      {
                            'verse':    {'TP': 2, 'FP': 2, 'FN': 1, 'n_detection': 4, 'n_reference': 3, 'precision': 0.5, 'recall': 0.66},
                            'chorus':   {'TP': 1, 'FP': 1, 'FN': 1, 'n_detection': 2, 'n_reference': 2, 'precision': 0.5, 'recall': 0.5},
                            'bridge':   {'TP': 0, 'FP': 0, 'FN': 1, 'n_detection': 0, 'n_reference': 1, 'precision': nan, 'recall': 0.0}}
                        }
    '''
    assert len(reference_section_labels_all) == len(detected_section_labels_all), "Lists must have the same length"
    unique_labels = set(reference_section_labels_all + detected_section_labels_all)
    # unique_labels = [x for x in unique_labels if x.strip()]

    metrics = {label: {"TP": 0, "FP": 0, "FN": 0} for label in unique_labels}

    for ref_label, det_label in zip(reference_section_labels_all, detected_section_labels_all):
        # if not det_label:   # sometimes asr lyrics is empty for a specific section, and it cannot align to any deepchorus tag, so det_label is empty
        #     continue
        if ref_label == det_label:
            metrics[ref_label]["TP"] += 1
        else:
            metrics[ref_label]["FN"] += 1
            metrics[det_label]["FP"] += 1

    for label, metric in metrics.items():
        TP = metric["TP"]
        FP = metric["FP"]
        FN = metric["FN"]
        n_detection = (TP + FP)
        n_reference = (TP + FN)
        precision = TP / n_detection if n_detection > 0 else np.nan
        recall = TP / n_reference if n_reference > 0 else np.nan

        metric["n_detection"] = n_detection
        metric["n_reference"] = n_reference
        metric["precision"] = round(precision, 3)
        metric["recall"] = round(recall, 3)

    metrics = dict(sorted(metrics.items(), key=lambda x: x[1]['TP'], reverse=True))

    return metrics

def align_section_labels(section_timestamps, deepchorus_segments):
    '''
    - input:
        - section_timestamps:   [
                                    {'section_label': '[verse]', 'start_time': 0.08, 'end_time': 30.92}, 
                                    {'section_label': '[chorus]', 'start_time': 30.92, 'end_time': 55.0}]
                                ]
        - deepchorus_segments:  [
                                    [ [0,      31.68 ], "verse" ],
                                    [ [31.68,  46.848], "chorus" ]
                                    [ [46.848, 55.872], "chorus" ]
                                ]
    - output:
        - reference_section_labels:     ['verse', 'chorus']
        - detected_section_labels:      ['verse', 'chorus']
    '''
    reference_section_labels = []
    detected_section_labels = []
    for section in section_timestamps:
        section_start = section['start_time']
        section_end = section['end_time']
        max_overlap = 0
        aligned_label = 'null'
        for segment in deepchorus_segments:
            segment_start, segment_end = segment[0]
            segment_label = segment[1]
            overlap_start = max(section_start, segment_start)
            overlap_end = min(section_end, segment_end)
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    aligned_label = segment_label
        reference_section_labels.append( section['section_label'][1:-1] )
        detected_section_labels.append( aligned_label )
    return reference_section_labels, detected_section_labels


def align_lyrics(lyrics, asr_lyrics):
    '''
    input:
        - lyrics:       [verse]阳光撒在街头每个角落 快乐的旋律随风融进我心窝[inst][chorus]看那蓝天白云轻轻飘过
        - asr_lyrics:   [
                            ["阳", [80, 240]], ["光", [240, 440]], ["洒", [440, 800]], ["在", [800, 1320]], ...
                        ]
    output
        - section_timestamps:   [
                                    {'section_label': '[verse]', 'start_time': 0.08, 'end_time': 30.92}, 
                                    {'section_label': '[chorus]', 'start_time': 30.92, 'end_time': 55.0}]
                                ]
    '''
    try:
        from dtw import dtw
    except ImportError:
        os.system('pip3 install dtw==1.4.0')
        from dtw import dtw
    text, section_index = extract_lyrics_sections_and_text(lyrics)
    asr_text = [item[0] for item in asr_lyrics]
    asr_timestamps = [(item[1][0], item[1][1]) for item in asr_lyrics]

    dist, d, D, path = dtw(text, asr_text, dist=lambda x, y: 0 if x == y else 1)
    gt_text_indices, asr_text_indices = path
    alignment_messages = []
    section_timestamps = []
    for section in section_index:
        section_label = section['section']
        start, end = section['index_range']
        idx_start, idx_end = np.where(gt_text_indices==start)[0][0], np.where(gt_text_indices==end)[0][0]
        aligned_lyrics_text = asr_text[asr_text_indices[idx_start] : asr_text_indices[idx_end]+1]
        aligned_lyrics_text_str = ''.join(aligned_lyrics_text)
        if len(aligned_lyrics_text_str) > 8:
            aligned_lyrics_text_str = aligned_lyrics_text_str[:4] + '...' + aligned_lyrics_text_str[-4:]
        section_time_span = asr_timestamps[asr_text_indices[idx_start]][0]/1000,  asr_timestamps[asr_text_indices[idx_end]][1]/1000
        message = f'{section_label} ({section_time_span[0]}s-{section_time_span[1]}s), aligned lyrics: {aligned_lyrics_text_str}'
        alignment_messages.append(message)
        section_timestamps.append({'section_label': section_label, 'start_time': section_time_span[0], 'end_time': section_time_span[1]})

    return section_timestamps, alignment_messages


def extract_lyrics_sections_and_text(lyrics):
    '''
    input: 
        lyrics:             [verse]阳光撒在街头每个角落 ha ha 快乐的旋律随风融进我心窝[inst][chorus]看那蓝天白云轻轻飘过
    output:
        text:               ['阳', '光', '撒', '在', '街', '头', '每', '个', '角', '落', 'ha', 'ha', '快', '乐', '的', '旋', '律', '随', '风', '融', '进', '我', '心', '窝', '看', '那', '蓝', '天', '白', '云', '轻', '轻', '飘', '过']
        section_index:      {'section': '[verse]', 'index_range': (0, 23)}
                            {'section': '[chorus]', 'index_range': (24, 33)}
    '''
    def split_text(text_str):
        '''
        - input: '再需要 ha ha 刺激我大脑'
        - output: ['再', '需', '要', 'ha', 'ha', '刺', '激', '我', '大', '脑']
        '''

        input_text = text_str.split()
        output_text = []
        for item in input_text:
            temp = []           # list of English char
            for char in item:
                if '\u4e00' <= char <= '\u9fff':  # if Chinese char
                    output_text.append(char)
                else:
                    temp.append(char)
            if temp:            # if find English word, add into output
                output_text.append(''.join(temp))
        return output_text

    section_index = []
    text = []
    current_position = 0
    parts = re.split(r'(\[[^\]]+\])', lyrics)
    for part in parts:
        if re.match(r'\[[^\]]+\]', part):
            current_section = part
        else:
            part = part.replace('\n', ' ')
            part = split_text(part)
            if part:
                text.extend(part)
                if current_section:
                    section_index.append({
                        'section': current_section,
                        'index_range': (current_position, current_position + len(part) - 1)
                    })
                    current_section = None
                current_position += len(part)
    return text, section_index

def create_parent_directories(file_path):
    path = Path(file_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent


def run_finegrained_tagging_offline_multigpu(audio_files):
    
    CONFIG_LPATH = "recipes/mir2/conf/bigmusic/bigmusic_tag_umm2_inference.yaml"
    TAGGING_CKPT = "hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/samantha_tagging/bigmusic/umm_stage2_8203_7905/checkpoints/step=0030000.ckpt"
    CKPT_LOCAL_PATH = "./tmp/umm_stage2_8203_7905.step=0030000.ckpt"
    
    create_parent_directories(CKPT_LOCAL_PATH)
    
    if not os.path.exists(CONFIG_LPATH):
        print(
            f"Finegrained tagging config file not found at {CONFIG_LPATH}\n"
            f"Please double check the branch and commit id.\n"
        )
        return
    if not os.path.exists(CKPT_LOCAL_PATH):
        # To prevent collision with main training checkpoint
        print(
            f"Finegrained tagging checkpoint not found at {CKPT_LOCAL_PATH}\n"
            f"Downloading from {TAGGING_CKPT}..."
        )
        os.system(f"hdfs dfs -get {TAGGING_CKPT} {CKPT_LOCAL_PATH}")
        print(f"Finegrained tagging checkpoint downloaded to {CKPT_LOCAL_PATH}")

    audio_files = [Path(fp) for fp in audio_files]
    out_dict = {}
    with tempfile.TemporaryDirectory() as prediction_dir:
        # HACK (vibertthio): Had to add ARNOLD_TRIAL_ID to prevent reusing rdzv endpoint
        command = f"RDMAV_FORK_SAFE=1 ARNOLD_TRIAL_ID='' ARNOLD_WORKSPACE_ID='' \
                    /bin/bash /opt/tiger/samantha/recipes/bigmusic/bootstrap.sh predict \
                    -c {CONFIG_LPATH} \
                    --ckpt_path {CKPT_LOCAL_PATH} \
                    --pl_datamodule.audio_files {[str(fp) for fp in audio_files]} \
                    --prediction_writer.output_dir {prediction_dir}"
        os.system(command)
        
        for audio_fp in audio_files:
            index = audio_fp.stem
            output_json_fp = prediction_dir + f"/{index}.json"
            if not os.path.exists(output_json_fp):
                continue
            predictions = json.load(open(output_json_fp, 'r'))
            metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
            update_json(metadata_fp, {
                "predicted_finegrained_tags": {**predictions}
            })

            out_dict[index] = predictions  # TODO: modify the key to something better

    return out_dict if out_dict else None


def run_finegrained_tagging(audio_files, device: Union[str, torch.device] = "cuda:0", online=True):
    
    def load_yaml(yaml_path: str):
        args = [
            "predict", "-c", yaml_path,
            "--run_opts.num_workers", "0",
            "--trainer", "null",
        ]
        hparams_file, run_opts, overrides = parse_arguments(args)
        with open(hparams_file, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        return hparams

    def load_pl_module(yaml_path: str, ckpt_path: str) -> pl.LightningModule:
        hparams = load_yaml(yaml_path)
        pl_module = hparams["pl_module"]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        pl_module.load_state_dict(state_dict)
        return pl_module
    
    def make_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        if isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}  # Recursively process dict
        if isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]  # Recursively process lists
        
        return obj  # Return as is if not ndarray
    
    def resample_and_monofy(audio_fp: Path, sr: int = 24000, mono: bool = True) -> np.ndarray:
        """
        Use convert_audio_ffmpeg_file_to_file from the prediction pipeline to align the resample method
        """
        _converted_temp_dir = tempfile.TemporaryDirectory()
        converted_audio_fp = os.path.join(
            _converted_temp_dir.name, "converted.wav"
        )
        convert_audio_ffmpeg_file_to_file(
            audio_fp,
            converted_audio_fp,
            sampling_rate=sr,
            mono=True,
        )
        audio, sr_out = sf.read(converted_audio_fp)
        _converted_temp_dir.cleanup()
        return audio, sr_out

    CONFIG_LPATH = "recipes/mir2/conf/bigmusic/bigmusic_tag_umm2_inference.yaml"
    TAGGING_CKPT = "hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/samantha_tagging/bigmusic/umm_stage2_8203_7905/checkpoints/step=0030000.ckpt"
    CKPT_LOCAL_PATH = "./tmp/umm_stage2_8203_7905.step=0030000.ckpt"

    create_parent_directories(CKPT_LOCAL_PATH)
    
    if not os.path.exists(CONFIG_LPATH):
        print(
            f"Finegrained tagging config file not found at {CONFIG_LPATH}\n"
            f"Please double check the branch and commit id.\n"
        )
        return
    if not os.path.exists(CKPT_LOCAL_PATH):
        # To prevent collision with main training checkpoint
        print(
            f"Finegrained tagging checkpoint not found at {CKPT_LOCAL_PATH}\n"
            f"Downloading from {TAGGING_CKPT}..."
        )
        os.system(f"hdfs dfs -get {TAGGING_CKPT} {CKPT_LOCAL_PATH}")
        print(f"Finegrained tagging checkpoint downloaded to {CKPT_LOCAL_PATH}")

    audio_files = [Path(fp) for fp in audio_files]
    results = []
    pl_module = load_pl_module(CONFIG_LPATH, CKPT_LOCAL_PATH)
    pl_module.eval()
    pl_module.to(device)
    sr = pl_module._sample_rate

    for audio_fp in tqdm(audio_files, desc=f"Finegrained Tagging Model on {device}"):

        if online:
            predictions = api_call(audio_fp, model='bigmir-bigmusic-tag', verbose=True)[0]
        else:
            audio, _ = resample_and_monofy(str(audio_fp))
            audio = torch.from_numpy(audio) 
            batch = (audio.unsqueeze(0).to(device), "filename_placeholder")
            
            with torch.no_grad():
                with autocast(dtype=torch.float16):  # to align with train/val/test dtype of 16-mixed
                    predictions = pl_module.predict_step(batch, batch_idx=0)
            
            # convert from float16 to float32 for more precise logits to be used as rewards
            predictions['logits'] = {k: v.astype(np.float32) for k, v in predictions['logits'].items()}
            predictions = make_json_serializable(predictions)

        metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
        update_json(metadata_fp, {
            "predicted_finegrained_tags": {**predictions}
        })

        results.append({
            "audio_fp": audio_fp,
            "predictions": predictions,
        })

    return results if results else None

def run_finegrained_tagging_online_parallel(audio_files, parallel=3):
    audio_files = [Path(fp) for fp in audio_files]
    def process_single_file(audio_fp):
        metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
        metadata = load_json_locked(metadata_fp)
        if 'predicted_finegrained_tags' in metadata:
            return
        predictions = api_call(audio_fp, model='bigmir-bigmusic-tag', verbose=True)[0]
        update_json(metadata_fp, {
            "predicted_finegrained_tags": {**predictions}
        })
    def process_group(file_list):
        for fp in file_list:
            process_single_file(fp)
    
    pool = ThreadPool(parallel)
    file_groups = np.array_split(audio_files, parallel)
    rets = []
    for group in file_groups:
        ret = pool.apply_async(process_group, args=(group,))
        rets.append(ret)
    pool.close()
    for ret in rets:
        ret.get()
    pool.join()
    print("Fine grained tagging call done.")


def run_finegrained_tagging_metrics(generated_out_dir, device: Union[str, torch.device] = "cuda:0", online=True, parallel=1):
    
    # TODO: check on prerequisie files in AR generated directory

    def get_vocab2id():
        # NOTE: another way to get vocab2id is from pl_module.semantic_module.input_embedders.multitags_categorical.vocab2id
        # but semantic module is not available when metrics used as a standalone script, so we exported a json file from the model
        VOCAB2ID_HPATH = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/vibertthio/bigmusic/assets/semantic_module_vocab2id.json"
        VOCAB2ID_JSON_FP = "./tmp/semantic_module_vocab2id.json"
        create_parent_directories(VOCAB2ID_JSON_FP)
        hdfs_get(VOCAB2ID_HPATH, VOCAB2ID_JSON_FP, overwrite=True)
        with open (VOCAB2ID_JSON_FP, "r") as f:
            vocab2id = json.load(f)
        return vocab2id
    
    output_dir = generated_out_dir
    audio_files = list(Path(output_dir).glob('**/*.generated.wav'))
    if parallel > 1:
        run_finegrained_tagging_online_parallel(audio_files, parallel)
    else:
        _ = run_finegrained_tagging(audio_files, device, online=online)
    
    metrics = {"all": [], "stats": {}}
    INDICES_KEYS_PAIRS = [{"style_text_idx": i, "finegrained_tags_key": k} for i, k in [(0, "GENRE"), (4, "THEME"), (3, "MOOD"), (5, "GENDER"), (6, "TIMBRE")]]
    KEY_TO_IDX = {p["finegrained_tags_key"]: p["style_text_idx"] for p in INDICES_KEYS_PAIRS}
    IDX_TO_KEY = {p["style_text_idx"]: p["finegrained_tags_key"] for p in INDICES_KEYS_PAIRS}
    
    vocab2id = get_vocab2id()

    for audio_fp in audio_files:
        # TODO: Cannot get condition tags using parse_gt_tag_from_metadata.
        # It was for SA20 tags, but we want to parse fine-grained tags.
        metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
        with open(metadata_fp, 'r', encoding='utf-8') as f: metadata = json.load(f)

        acc_per_tag = {}
        condition_tags = {}
        predicted_tags = metadata["predicted_finegrained_tags"]["finegrained_tags"]
        for tag_type in ['GENRE', 'THEME', 'MOOD', 'GENDER', 'TIMBRE']:
            condition_tags[tag_type] = metadata["style_text"][KEY_TO_IDX[tag_type]]
            condition_tag_ids = [vocab2id[tag] for tag in condition_tags[tag_type] if tag in vocab2id]
            predicted_tag_ids = [vocab2id[tag] for tag in predicted_tags[tag_type] if tag in vocab2id]

            # TODO: calculate metrics using "predicted_tags" vs. "gt_tags"
            if len(condition_tag_ids) == 0:
                acc_per_tag[tag_type] = None
            else:                    
                acc_per_tag[tag_type] = len(set(predicted_tag_ids) & set(condition_tag_ids)) / len(set(condition_tag_ids))
            
        
        metrics["all"].append({
            "audio_fp": str(audio_fp),
            "condition_tags": condition_tags,
            "predicted_tags": predicted_tags,
            "accuracy": acc_per_tag,
        })
    
    # from all metrics to calculate metrics stats
    for tag_type in ['GENRE', 'THEME', 'MOOD', 'GENDER', 'TIMBRE']:
        accuracy = np.mean([m["accuracy"][tag_type] for m in metrics["all"] if m["accuracy"][tag_type] is not None])
        metrics["stats"][tag_type] = {
            "accuracy": accuracy,
        }
        print(f"[Finegrained Tagging Metrics]: {tag_type} accuracy: {accuracy}")
    
    # write metrics to json
    metrics_fp = os.path.join(output_dir, 'finegrained_tagging_metrics.json')
    with open(metrics_fp, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[Finegrained Tagging Metrics] computed and saved in {metrics_fp}")
    
    # TODO: write a txt file output
    

class MIRFineGrainedTaggingCallback(pl.Callback):
    def __init__(self, online=True):
        super().__init__()
        self.online=online

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        
        output_dir = pl_module.extra_params["output_dir"]
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()
        
        if not trainer.is_global_zero:
            return
        
        ts = time.time()
        while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
            time.sleep(10)
            print(f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)")
        
        device = pl_module.device
        run_finegrained_tagging_metrics(output_dir, device, online=self.online)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()

    generated_output_fps = list(Path(args.input_dir).glob('**/*.generated.wav'))
    # audio_list = upload_audio_file_to_tos(generated_output_fps)
    # report = run_tagging_acc(audio_list, tag='genre')
    # print(report)

    # for tag in ['genre', 'mood', 'gender', 'lang']:
    #     report = run_tagging_acc(audio_list, tag=tag)
    #     print(report)
    #     with open(os.path.join(args.input_dir, f'{tag}_report.txt'), 'w') as fw:
    #         fw.write(report.get_string())

    # run_deepchorus(None, Path(args.input_dir))
    # output_section_label_metric(Path(args.input_dir))
    
    audio_files = list(Path(args.input_dir).glob('**/*.generated.wav'))
    audio_files.sort()
    results = run_finegrained_tagging(audio_files)

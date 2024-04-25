from operator import gt
import os
import json
import euler
import thriftpy2
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime
from prettytable import PrettyTable
import pytorch_lightning as pl
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json
from recipes.bigmusic.datasets.mir_data_util import SA_GENRE20, MAP_SUB_GENRE_2_SA_GENRE20
from recipes.mir_benchmark.tagging_inference.genre import GenreTagging
from recipes.mir_benchmark.tagging_inference.instrument import InstrumentTagging
from recipes.mir_benchmark.tagging_inference.vocal import VocalTagging


thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "../utils/services/idl/music_tagging.thrift"), "music_tagging_thrift"
)


from music_tagging_thrift import MusicTagging, TaggingRequest


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
    os.system(f'{tos_cli} -bucket {bucket} -accessKey {ak} put -prefix {prefix} {audio_fp}')
    file_name = os.path.basename(audio_fp)
    url = f'https://tosv.byted.org/obj/{bucket}/{prefix}{file_name}'
    return url


def map_gt_categorical_genre(genre_text):
    genres = genre_text.split(',')
    gt_genres = []
    for g in genres:
        if g in SA_GENRE20:
            gt_genres.append(g)
        elif g in MAP_SUB_GENRE_2_SA_GENRE20:
            gt_genres += MAP_SUB_GENRE_2_SA_GENRE20[g]
        else:
            print(f"unrecognized genre {g}")
    return list(set(gt_genres))


def SA_online_tagging_predict(client, audio_url, tag="genre"):
    if tag == "genre":
        rlts = client.TaggingGenre20(TaggingRequest(track_id="test", url=audio_url))
        try:
            result = eval(rlts.result_json)['Genre20']['result']
        except:
            print(f"failed: {audio_url} check silence")
            result = []
    if tag == 'mood':
        rlts = client.TaggingMood(TaggingRequest(track_id="test", url=audio_url))
        result = eval(rlts.result_json)['Mood']['result']
    if tag == 'theme':
        rlts = client.TaggingTheme(TaggingRequest(track_id="test", url=audio_url))
        result = eval(rlts.result_json)['Theme']['result']
    return result


def parse_gt_tag_from_metadata(audio_file_path, tag="genre"):
    metadata_fp = str(audio_file_path).replace('generated.wav', 'metadata.json')
    with open(metadata_fp, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    gt_style_text = metadata.get('style_text')
    if tag == "genre":
        if '|' in gt_style_text:
            genres = gt_style_text.split('|')[0]
            if ',' in genres:
                return map_gt_categorical_genre(genres)
            else:
                return [genres.strip()]
        else:
            return [gt_style_text.split(',')[0].strip()]


def run_tagging_acc(audio_file_paths, tag="genre"):
    if os.path.exists("/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli"):
        tos_cli = "/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli"
    elif os.path.exists("/opt/tiger/1.0.0.20/toscli"):
        tos_cli = "/opt/tiger/1.0.0.20/toscli"
    else:
        os.system("hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhaohang.ai/scripts/1.0.0.20 /opt/tiger/")
        os.system("chmod +x /opt/tiger/1.0.0.20/toscli")
        tos_cli = "/opt/tiger/1.0.0.20/toscli"

    target = 'sd://lab.speech.music_tagging?cluster=default'
    client = euler.Client(MusicTagging,target=f'{target}&idc=lf', timeout=1200)

    tos_prefix = 'tmp/infer/wavs_test/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + '/'

    report_tab = PrettyTable([tag, "accuracy", "support"])

    rlts = {}
    total_correct = 0
    for audio_fp in audio_file_paths:
        url = upload_to_tos(audio_fp, tos_prefix, tos_cli=tos_cli)
        gt_tags = parse_gt_tag_from_metadata(audio_fp, tag)
        predict_tags = SA_online_tagging_predict(client, url, tag)
        
        flag = False
        for pt_tag in predict_tags:
            if pt_tag in gt_tags:
                flag = True
                total_correct += 1
                if pt_tag not in rlts:
                    rlts[pt_tag] = {"correct": 1, "total": 1}
                else:
                    rlts[pt_tag]['correct'] += 1
                    rlts[pt_tag]['total'] += 1
                break
        
        if not flag:
            gt_tag = ','.join(gt_tags)
            if gt_tag not in rlts:
                rlts[gt_tag] = {"correct": 0, "total": 1}
            else:
                rlts[gt_tag]['total'] += 1
    
    for k, v in rlts.items():
        k_acc = round(v["correct"] / v["total"], 4)
        report_tab.add_row([k, k_acc, v["total"]])

    total_acc = round(total_correct / len(audio_file_paths), 4)
    report_tab.add_row(['total', total_acc, len(audio_file_paths)])
    
    return report_tab
        

class MIRTagMetricsSAOnlineCallback(pl.Callback):
    def __init__(self, tags=["genre"]) -> None:
        self.tags = tags
        super().__init__()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        
        for tag in self.tags:
            report = run_tagging_acc(generated_output_fps, tag)
            print(report)
            if 'output_dir' in pl_module.extra_params:
                with open(os.path.join(pl_module.extra_params.output_dir, f'{tag}_report.txt'), 'w') as fw:
                    fw.write(report.get_string())
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    )
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    generated_output_fps = list(Path(args.input_dir).glob('**/*.generated.wav'))
    report = run_tagging_acc(generated_output_fps)
    print(report)
    with open(os.path.join(args.input_dir, 'genre_report.txt'), 'w') as fw:
        fw.write(report.get_string())

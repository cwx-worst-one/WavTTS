"""
Call Musicfm+ models offline
"""
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json, os, subprocess
import pytorch_lightning as pl
from collections import Counter
from recipes.bigmusic.datasets.mir_data_util import tempo_to_label, tempo_to_coarse_label
import numpy as np
from recipes.bigmusic.datasets.transforms.inst import tagging_inst_to_38, get_inst_family

tempo_range = [[0, 48], [49, 76], [77, 120], [121, 176], [177, 200]]
tempo_range_tolerance = list(range(1, len(tempo_range) + 1))
    

def _extract_song_tempo(wav_dir: Path) -> float:
    beat_txt_path = wav_dir / "beat.txt"
    if not os.path.exists(beat_txt_path):
        return -1
    f = open(beat_txt_path, "r").readlines()
    beat_list = []
    for line in f:
        beat_time, beat_idx = line.strip().split()
        beat_list.append([float(beat_time), int(beat_idx)])
    tempo = 60 / np.mean(np.diff(np.array(beat_list), n=1, axis=0)[:, 0])
    return tempo


def _extract_song_key(wav_dir: Path) -> str:
    key_txt_path = wav_dir / "key.txt"
    
    f = open(key_txt_path, "r").readlines()
    key_list = []
    for line in f:
        key_time, key_name = line.strip().split()
        key_list.append([float(key_time), key_name])
    
    key_list = [key[1] for key in key_list]
    # key_root_list = [key.split(":")[0] if key != "X" else "X" for key in key_list]
    # key_mode_list = [key.split(":")[-1] if key != "X" else "X" for key in key_list]
    key_counter = Counter(key_list)
    key = key_counter.most_common(1)[0][0]
    if key == 'X':
        root, mode = "X", "X"
    else:
        root, mode = key.split(":")
    return key, root, mode

def _extract_song_inst(wav_dir: Path) -> str:
    inst_json_path = wav_dir / "inst.json"
    f = _load_json(inst_json_path)
    insts = f["instruments"]
    inst_list = [inst.replace(" ", "_") for inst in insts.keys()]
    return inst_list


CHROMA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
EMPTY_KEY = "X"


def is_key_relative_key_error(pred: str, gt: str) -> bool:
    if EMPTY_KEY in [pred, gt]:
        return False
    pred_root, pred_type = pred.split(":")
    gt_root, gt_type = gt.split(":")
    pred_idx = CHROMA.index(pred_root)
    gt_idx = CHROMA.index(gt_root)
    if pred_type == "Maj" and gt_type == "Min":
        ind = [pred_idx, gt_idx]
    elif pred_type == "Min" and gt_type == "Maj":
        ind = [gt_idx, pred_idx]
    # elif pred_type == "Min" and gt_type == "Min":   # root_convert_to_maj, gt->Maj
    #     ind = [gt_idx, pred_idx]
    else:
        return False
    return (ind[0] + 9) % 12 == ind[1]  # maj->min
        

def is_key_fifth_error(pred: str, gt: str) -> bool:
    if EMPTY_KEY in [pred, gt]:
        return False

    pred_root, pred_type = pred.split(":")
    gt_root, gt_type = gt.split(":")
    pred_idx = CHROMA.index(pred_root)
    gt_idx = CHROMA.index(gt_root)
    return (
        (gt_type == pred_type) and 
        (
            ((pred_idx - 7) % 12 == gt_idx) or
            ((pred_idx + 7) % 12 == gt_idx)
        )
    )


def key_metrics(pred: List[str], gt: Tuple[str], gt_transform=None) -> Dict[str, bool]:
    if isinstance(gt, tuple):
        gt, gt_root, gt_mode = gt
    else:
        gt_root, gt_mode = gt.split(":") if gt != "X" else ("X", "X")
    if gt_transform == 'root_convert_to_maj' and gt_mode in ["Min", "Minor"]:
        gt_root = CHROMA[(CHROMA.index(gt_root) + 9) % 12]  # convert back to original user input
        gt = f"{gt_root}:{gt_mode}"

    if isinstance(pred, tuple) or isinstance(pred, list):
        pred, pred_root, pred_mode = pred
    else:
        pred_root, pred_mode = pred.split(":") if pred != "X" else ("X", "X")

    return {
        "key_match": pred == gt or gt == "X",
        "root_match": pred_root == gt_root or gt_root == "X",
        "mode_match": pred_mode == gt_mode or gt_mode == "X",
        # "fifth_error": is_key_fifth_error(pred, gt),
        "rel_key_match": is_key_relative_key_error(pred, gt)
    }

def inst_metrics(pred: List[str], gt: List[str]) -> Dict[str, bool]:
    gt = gt[9]

    pred_cls1, pred_cls2, pred38 = [], [], []
    for _pred in pred:
        if _pred:
            _pred = tagging_inst_to_38(_pred)
            _pred_cls1, _pred_cls2, _ = get_inst_family(_pred)
            pred_cls1.append(_pred_cls1) if _pred_cls1 is not None else None
            pred_cls2.append(_pred_cls2) if _pred_cls2 is not None else None
            pred38.append(_pred)
    # print(f"pred: {pred}, pred_cls1: {pred_cls1}, pred_cls2: {pred_cls2}, pred38: {pred38}")

    gt_cls1, gt_cls2, gt38 = [], [], []
    for _gt in gt:
        _gt = tagging_inst_to_38(_gt)
        _gt_cls1, _gt_cls2, _ = get_inst_family(_gt)
        gt_cls1.append(_gt_cls1) if _gt_cls1 is not None else None
        gt_cls2.append(_gt_cls2) if _gt_cls2 is not None else None
        gt38.append(_gt)
    # print(f"gt: {gt}, gt_cls1: {gt_cls1}, gt_cls2: {gt_cls2}, gt38: {gt38}")

    return {
        "inst_acc": len(set(pred38) & set(gt38)) / len(set(gt38)),
        "inst_cls1_acc": len(set(pred_cls1) & set(gt_cls1)) / len(set(gt_cls1)) if len(set(gt_cls1)) > 0 else np.nan,
        "inst_cls2_acc": len(set(pred_cls2) & set(gt_cls2)) / len(set(gt_cls2)) if len(set(gt_cls2)) > 0 else np.nan,
    }

def tempo_metrics(pred: str, pred_coarse: str, pred_raw: str, gt: str, gt_coarse: str, gt_raw: str) -> Dict[str, bool]:
    for idx in range(len(tempo_range)):
        l, h = tempo_range[idx]
        if l <= float(gt_raw) <= h:
            tolerance = tempo_range_tolerance[idx]
            break

    return {
        "tempo_8cls": pred == gt or gt == "",
        "tempo_4cls": pred_coarse == gt_coarse or gt_coarse == "",
        "tempo_rg": abs(float(pred_raw) - float(gt_raw)) <= tolerance,
    }

def _load_json(json_fp):
    with open(json_fp, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d


CALLBACK_TASKS = {
    "tempo": {
        "mir_task": "beat",  # corresponding mir task
        "check_fn": tempo_metrics,  # calculate metrics
        "extractor": _extract_song_tempo,  # extract the mir value from the output json
        "cond": "tempo_label",   # condition name in the metadata.json
        "raw": "tempo", # raw name in the metadata.json 
    },
    "key": {
        "mir_task": "key",
        "check_fn": key_metrics,
        "extractor": _extract_song_key,
        "cond": "key",
        "raw": "key", # raw name in the metadata.json 
    },
    "instrument": {
        "mir_task": "inst_tagging",
        "check_fn": inst_metrics,
        "extractor": _extract_song_inst,
        "cond": "style_text",
        "raw": "style_text" 
    }
}

class MusicFMOfflineCallback(pl.Callback):
    def __init__(self, tasks: Optional[List[str]] = None) -> None:
        """
        :param tasks: default (the keys in CALLBACK_TASKS): ["tempo", "key", "instrument"]
        """
        if tasks is None:
            tasks = list(CALLBACK_TASKS.keys())
        if not set(tasks).issubset(CALLBACK_TASKS.keys()):
            raise ValueError(f"One or more tasks in {tasks} are not supported.")
        self.tasks = tasks
        super().__init__()

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        audio_fps, rel_fps, mir_dir = _run_result(pl_module.extra_params.output_dir)
        _process(audio_fps, rel_fps, mir_dir, self.tasks)
        output_mir_metric_for_all_samples(pl_module.extra_params.output_dir)

        # _process(pl_module.extra_params.output_dir, self.tasks)

def _init_musicfm():
    """
    Return: stat_file, local_config_path, ckpt_path
    """
    # ckpt_path = "hdfs://haruna/home/byte_speech_sv/ju-chiang.wang/melrof_ckpt/epoch=64-step=6500.ckpt"
    ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/mir/epoch=64-step=6500.ckpt"    # more stable for CN
    
    stat_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/mir/playlist_classic_stats.json"
    local_stat_path = [
        "/tmp/playlist_classic_stats.json",
        "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/playlist_classic_stats.json",
        "/mnt/bn/qinxin/mir/playlist_classic_stats.json",
    ]
    local_stat_path_exist_flag = [os.path.exists(p) for p in local_stat_path]
    if sum(local_stat_path_exist_flag) == 0:    # none of local_stat exists
        local_stat_path = "/tmp/playlist_classic_stats.json"
        get_cmd = f"hdfs dfs -get {stat_path} {local_stat_path}"
        subprocess.call(get_cmd, shell=True)
    else:
        local_stat_path = local_stat_path[local_stat_path_exist_flag.index(True)]

    return local_stat_path, ckpt_path

def _init_musicfm_inst_tagging():
    """
    Return: local_stat_path, local_ckpt_dir, local_model_path
    """
    # deepspeed zero-stage2 ckpt must be local (not hdfs)
    stat_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/mir/playlist_classic_stats.json"
    local_stat_path = [
        "/tmp/playlist_classic_stats.json",
        "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/playlist_classic_stats.json",
        "/mnt/bn/qinxin/mir/playlist_classic_stats.json",
    ]
    local_stat_path_exist_flag = [os.path.exists(p) for p in local_stat_path]
    if sum(local_stat_path_exist_flag) == 0:    # none of local_stat exists
        local_stat_path = "/tmp/playlist_classic_stats.json"
        get_cmd = f"hdfs dfs -get {stat_path} {local_stat_path}"
        subprocess.call(get_cmd, shell=True)
    else:
        local_stat_path = local_stat_path[local_stat_path_exist_flag.index(True)]

    ckpt_dir="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/mir/random_mix_inst_109400.ckpt"
    local_ckpt_dir = [
        "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/random_mix_inst_109400.ckpt",
        "/mnt/bn/qinxin/mir/random_mix_inst_109400.ckpt",
        "/tmp/random_mix_inst_109400.ckpt",
        ]
    local_ckpt_dir_exist_flag = [os.path.exists(p) for p in local_ckpt_dir]
    if sum(local_ckpt_dir_exist_flag) == 0:    # none of local_ckpt exists
        local_ckpt_dir = "/tmp/random_mix_inst_109400.ckpt"
        get_cmd = f"hdfs dfs -get {ckpt_dir} {local_ckpt_dir}"
        subprocess.call(get_cmd, shell=True)
    else:
        local_ckpt_dir = local_ckpt_dir[local_ckpt_dir_exist_flag.index(True)]
    
    # model_path="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/qinxin.025/mir/musicfm_25hz_playlist_330m_520k.pt"
    # local_model_path = [
    #     "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/musicfm_25hz_playlist_330m_520k.pt",
    #     "/mnt/bn/qinxin/mir/musicfm_25hz_playlist_330m_520k.pt",
    #     "/tmp/musicfm_25hz_playlist_330m_520k.pt",
    # ]
    # local_model_path_exist_flag = [os.path.exists(p) for p in local_model_path]
    # if sum(local_model_path_exist_flag) == 0:    # none of local_model exists
    #     local_model_path = "/tmp/musicfm_25hz_playlist_330m_520k.pt"
    #     get_cmd = f"hdfs dfs -get {model_path} {local_model_path}"
    #     subprocess.call(get_cmd, shell=True)
    # else:
    #     local_model_path = local_model_path[local_model_path_exist_flag.index(True)]
    git_cmd = f"pip3 install music21; git clone -b qx/mir_inst --single-branch git@code.byted.org:seed/samantha.git /opt/tiger/samantha_mir; cd /opt/tiger/samantha_mir; git checkout f2e26ec419fb07d4b66f6cad02cd43c74cda5d3b; cd -"
    subprocess.call(git_cmd, shell=True)
    return local_stat_path, local_ckpt_dir

def run_musicfm(audio_dirs, output_dirs, stat_path, ckpt_path):
    # ensure running in the venv
    config_path = "recipes/mir_benchmark/conf/multitask/multi_melrof_inference.yaml"
    assert len(audio_dirs) == len(output_dirs)
    for i in range(len(audio_dirs)):
        cmd1 = f"cd /opt/tiger/bigmusic_sami_ai_models/ && . ./venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32; \
                sh launch.sh predict --config {config_path} --ckpt_path {ckpt_path} --pl_datamodule.audio_dir {audio_dirs[i]} \
                --prediction_writer.output_dir {output_dirs[i]} --frontend.model.stat_path {stat_path} --train_params.roformer_path ; deactivate; cd -"
        print(cmd1)
        subprocess.call(cmd1, shell=True)

def run_musicfm_inst(audio_dirs, output_dirs, stat_path, ckpt_dir):
    config_path = "recipes/mir_benchmark/conf/tagging/instrument_random_mix_infererce.yaml"
    assert len(audio_dirs) == len(output_dirs)
    for i in range(len(audio_dirs)):
        cmd2 = f"cd /opt/tiger/samantha_mir/; bash launch.sh predict --config {config_path} --ckpt_path {ckpt_dir} --extra_params.input_audio {audio_dirs[i]} \
            --extra_params.output_path {output_dirs[i]} --frontend.model.stat_path {stat_path}  --extra_params.model_path ; cd -"
        print(cmd2)
        subprocess.call(cmd2, shell=True)

    
def get_wavs_from_dir(output_dir):
    wav_subfolders = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_subfolders.append(root)
                break
    audio_fps = list(Path(output_dir).glob('**/*.generated.wav'))
    rel_fps = [Path(fp).relative_to(output_dir) for fp in audio_fps]
    mir_dir = Path(output_dir) / '.mir_cache'
    return audio_fps, rel_fps, mir_dir, wav_subfolders
    
def _run_result(wav_subfolders, mir_dir):
    stat_path, ckpt_path = _init_musicfm()

    audio_dirs, output_dirs = [], []
    for subfolder in wav_subfolders:
        audio_dirs.append(subfolder)
        output_dirs.append(str(mir_dir / subfolder.split("/")[-1]))

    run_musicfm(
        audio_dirs=audio_dirs, output_dirs=output_dirs,
        stat_path=stat_path, ckpt_path=ckpt_path,
    )

def _run_inst_result(wav_subfolders, mir_dir):
    stat_path, ckpt_dir = _init_musicfm_inst_tagging()

    audio_dirs, output_dirs = [], []
    for subfolder in wav_subfolders:
        audio_dirs.append(subfolder)
        output_dirs.append(str(mir_dir / subfolder.split("/")[-1]))

    run_musicfm_inst(
        audio_dirs=audio_dirs, output_dirs=output_dirs,
        stat_path=stat_path, ckpt_dir=ckpt_dir
    )

def _process(audio_fps, rel_fps, mir_dir, callback_tasks):
    """
    # musicfm+ processed data tree
    output_dir
    ├── cnlh001.generated
    │   ├── beat.txt
    │   ├── chord.mid
    │   ├── chord.txt
    │   ├── key.txt
    │   ├── inst.json
    │   └── structure.txt
    """
    for rel_fp, audio_fp in zip(rel_fps, audio_fps):
        # mir_dict = _load_json((mir_dir / rel_fp).with_suffix(".json"))
        wav_dir = mir_dir / str(rel_fp).split(".wav")[0]
        # import pdb; pdb.set_trace()

        metadata_fp = str(audio_fp).replace('generated.wav', 'metadata.json')
        metadata = _load_json(metadata_fp)
        # mir_result = {task: CALLBACK_TASKS[task]["extractor"](wav_dir) for task in callback_tasks}
        gt_raw, mir_result = {}, {}
        for task in callback_tasks:
            if task == 'instrument':
                gt_raw[task] = list(set([tagging_inst_to_38(_inst) for _inst in metadata[callback_tasks[task]["raw"]][9]]))
                mir_result[task] = list(set([tagging_inst_to_38(_inst) for _inst in CALLBACK_TASKS[task]["extractor"](wav_dir)]))
            else:
                gt_raw[task] = metadata[callback_tasks[task]["raw"]] 
                mir_result[task] = callback_tasks[task]["extractor"](wav_dir)

        metrics = {}
        for task, value in mir_result.items():
            if task == "tempo":
                tempo_label, tempo_coarse_label = tempo_to_label(value), tempo_to_coarse_label(value)
                metrics[task] = CALLBACK_TASKS[task]["check_fn"](pred=tempo_label, pred_coarse=tempo_coarse_label, pred_raw=value,
                                                                 gt=metadata[CALLBACK_TASKS[task]["cond"]],
                                                                 gt_coarse=tempo_to_coarse_label(metadata[CALLBACK_TASKS[task]["raw"]]),
                                                                 gt_raw=metadata[CALLBACK_TASKS[task]["raw"]])
            elif task == 'key':
                metrics[task] = CALLBACK_TASKS[task]["check_fn"](pred=value, 
                                                                 gt=metadata[CALLBACK_TASKS[task]["cond"]])
            elif task == 'instrument':
                metrics[task] = CALLBACK_TASKS[task]["check_fn"](pred=value, 
                                                                 gt=metadata[CALLBACK_TASKS[task]["cond"]])

        from recipes.bigmusic.utils.format_utils import update_json
        update_json(metadata_fp, {'musicfm': {"result": mir_result, "metrics": metrics, "gt": gt_raw}})


def output_mir_metric_for_all_samples(path_result, callback_tasks):
    import os
    import glob
    import pandas as pd
    from tabulate import tabulate
    
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    
    task_dict = {}
    for task in callback_tasks.keys():
        task_dict[task] = {}
        if task == 'tempo':
            task_dict[task]['all'] = {'tempo_4cls': [], 'tempo_8cls': [], 'tempo_rg': [], }
        elif task == 'key':
            task_dict[task]['all'] = {'key_match': [], 'root_match': [],'mode_match': [],'rel_key_match': []}
        elif task == 'instrument':
            task_dict[task]['all'] = {'inst_acc': [], 'inst_cls1_acc': [], 'inst_cls2_acc': []}

    title = ['index']
    detail_title = ['index']
    for task in task_dict.keys():
        title = title + list(task_dict[task]['all'].keys())
        detail_title = detail_title + ['gt_'+task, 'gen_'+task]

    df = pd.DataFrame(columns=title)
    detail_df = pd.DataFrame(columns=detail_title)
    for category in categories:
        path_category = os.path.join(path_result, category)
        filenames = sorted(glob.glob(os.path.join(path_category, '*.metadata.json')))
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            mir_metrics = metadata['musicfm']["metrics"]
            data = {}
            data['index'] = index
            for task in task_dict.keys():
                for tkey in task_dict[task]['all'].keys():
                    if task in ['key', 'tempo']:
                        data[tkey] = int(mir_metrics[task][tkey])
                    elif task == 'instrument':
                        data[tkey] = float(mir_metrics[task][tkey])
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

            detail_data = {}
            detail_data['index'] = index
            for task in task_dict.keys():
                detail_data['gt_'+task] = metadata['musicfm']["gt"][task]
                detail_data['gen_'+task] = metadata['musicfm']["result"][task]

            detail_df = pd.concat([detail_df, pd.DataFrame([detail_data])], ignore_index=True)

            for task in task_dict.keys():
                if task == 'instrument' and detail_data["gt_instrument"] in [None, [], '', ['']]:
                    continue
                for tkey in task_dict[task]['all'].keys():
                    task_dict[task]['all'][tkey].append(data[tkey])

    total_data = {}
    total_data['index'] = 'total'
    for task in task_dict.keys():
        for tkey in task_dict[task]['all'].keys():
            total_data[tkey] = round(sum(task_dict[task]['all'][tkey]) / len(task_dict[task]['all'][tkey]), 2)
    df = pd.concat([df, pd.DataFrame([total_data])], ignore_index=True)

    # metrics
    filename_out = os.path.join(path_result, 'musicfm_report.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print(table)
    with open(filename_out, 'w') as f:
        f.write(table)
    print()
    
    # details
    detail_filename_out = os.path.join(path_result, 'musicfm_for_all_samples.txt')
    detail_table = tabulate(detail_df, headers='keys', tablefmt='grid')
    print(detail_table)
    with open(detail_filename_out, 'w') as f:
        f.write(detail_table)

if __name__ == "__main__":
    import argparse
    # Create a parser object
    parser = argparse.ArgumentParser()
    # Add an argument, -o ossr --output_dir, to the parser
    parser.add_argument('-o', '--output_dir', help='Output directory')
    parser.add_argument('-t', '--tasks', help='callback tasks', default='key,tempo,instrument')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Output directory
    output_dir = args.output_dir
    tasks = args.tasks.split(",")
    
    audio_fps, rel_fps, mir_dir, wav_subfolders = get_wavs_from_dir(output_dir)
    if any([x in ['key', 'tempo'] for x in tasks]):
        _run_result(wav_subfolders, mir_dir)
    if 'instrument' in tasks:
        _run_inst_result(wav_subfolders, mir_dir)

    _process(audio_fps, rel_fps, mir_dir, {k: CALLBACK_TASKS[k] for k in tasks})
    output_mir_metric_for_all_samples(output_dir, {k: CALLBACK_TASKS[k] for k in tasks})

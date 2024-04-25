import os
import concurrent.futures
import glob
from typing import Dict, List, Union, Optional

import pandas as pd
from tqdm import tqdm

import logging

from sami_models.serve.MIRService import PoolMIRService


logger = logging.getLogger(__file__)


default_tasks = ["beat", "chord", "key", "structure", "trans_5stem"]


def check_output_existence(filename, output_path, write_json=True):
    if isinstance(filename, tuple):
        filename = filename[1]
    else:
        filename = os.path.basename(filename)
    fn, _ = os.path.splitext(filename)
    output_path = os.path.join(output_path, fn)
    if write_json:
        return os.path.exists(output_path+'.json')
    else:
        return os.path.exists(output_path)


def run_batch(
    audio_dir: str = '',
    input_csv: str = '',
    output_dir: str = 'mir_output',
    tasks: List[str] = None,
    extension: Optional[List[str]] = None,
    gpu_ids: Optional[List[Union[str, int]]] = None,
    reverse_list: bool = False,
    start_perc: float = 0.0,
):
    """
    :param path_audio: root directory for audio files (will include all subdirectories)
    :param input_csv: csv file that contains 'song_id' and 'url' columns
    :param output_path: output root path
    :param tasks: root directory for audio files (will include all subdirectories)
    :param extension: audio file extensions
    :param gpu_ids: audio file extensions
    :param reverse_list: reverse the input list
    :param start_perc: starting percentage in the url list
    """
    if not audio_dir and not input_csv:
        raise ValueError('No audio path')

    if gpu_ids is None:
        NUM_GPUS = os.environ.get('ARNOLD_WORKER_GPU')
        if NUM_GPUS and int(NUM_GPUS)>0:
            cuda_ids = list(range(int(NUM_GPUS)))
        else:
            cuda_ids = ["cpu"] * 4
    else:
        cuda_ids = [int(i) for i in gpu_ids]
    logger.info(f"cuda IDs applied: {cuda_ids}")

    if tasks is None:
        run_tasks = default_tasks
    else:
        run_tasks = [t for t in tasks if t in default_tasks]
    logger.info(run_tasks)

    if extension is None:
        exts = ['mp3', 'wav', 'm4a']
    else:
        exts = [str(i) for i in extension]

    mir_workers = PoolMIRService(
        cuda_ids=cuda_ids,
        output_path=output_dir,
        base_batch_size=1,
        tasks=run_tasks,
    )

    if audio_dir:
        if audio_dir[-1] != "/":
            audio_dir = audio_dir + "/"
        urls = []
        for ext in exts:
            urls += glob.glob(f"{audio_dir}/**/*.{ext}", recursive=True)
        urls = sorted(urls, reverse=reverse_list)
        urls = sorted(urls, reverse=reverse_list)
        urls = [(audio_dir, url.replace(audio_dir, '')) for url in urls]
        urls = [url for url in urls if not check_output_existence(url, output_dir)]
        urls = urls[int(round(len(urls)*start_perc)):]
    else:
        df = pd.read_csv(input_csv)
        urls = list(zip(df['url'].tolist(), df['song_id'].tolist()))
        urls = [('', url) for url in urls]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cuda_ids)) as executor:
        results = list(tqdm(executor.map(mir_workers.predict, urls), total=len(urls))) 

    results = [res for res in results if res is not None]
    return results


CHROMA = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
EMPTY_KEY = "N"


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
    else:
        return False
    return (ind[0] + 9) % 12 == ind[1]
        

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


def key_metrics(pred: str, gt: str) -> Dict[str, bool]:
    return {
        "key_match": pred == gt or gt == "N",
        "fifth_error": is_key_fifth_error(pred, gt),
        "relative_key_error": is_key_relative_key_error(pred, gt)
    }


def tempo_metrics(pred: str, gt: str) -> Dict[str, bool]:
    return {
        "tempo_match": pred == gt or gt == "",
    }


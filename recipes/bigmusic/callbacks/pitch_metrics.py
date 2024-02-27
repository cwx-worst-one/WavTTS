from functools import partial
import pytorch_lightning as pl
from typing import Any
import torch
from recipes.bigmusic.utils.format_utils import (
    concat_metadata_list,
    update_json,
    normalize_text,
)
import sklearn
from recipes.musiclm.inference.utils import load_wav
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map
import pyworld as pw
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error


F0_FLOOR = 65 # C2
F0_CEIL = 2093 # C7

def get_pitch_world(wav_array, sr):
    x = wav_array.astype(np.double)
    _f0, t = pw.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL)    # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, sr)  # pitch refinement

    aperiodicity = pw.d4c(x, _f0, t, sr)
    # https://github.com/mmorise/World/issues/35#issuecomment-306521887
    vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)
    f0 *= vuv
    f0[f0<F0_FLOOR] = 0.0 # C2
    f0[f0>F0_CEIL] = 0.0 # C7
    return f0, vuv.astype(np.bool_).flatten()

def get_vuv_acc(pred, label):
    # Reshape the lists to be 2D as required by fastdtw
    list1 = np.reshape(pred, (len(pred), 1))
    list2 = np.reshape(label, (len(label), 1))

    # Align the sequences based on the DTW path
    aligned_list1 = []
    aligned_list2 = []
   
    list1 = list(list1.flatten())
    list2 = list(list2.flatten())
    if len(list1) > len(list2):
        list2.extend([0.0] * (len(list1) - len(list2)))
    else:
        list1.extend([0.0] * (len(list2) - len(list1)))

    aligned_list1 = np.array(list1)
    aligned_list2 = np.array(list2)

    acc = sklearn.metrics.accuracy_score(aligned_list1, aligned_list2)
    return acc


def dtw_mae(pred, label, plot_f0_path="", alignment="dtw"):
    # Reshape the lists to be 2D as required by fastdtw
    list1 = np.reshape(pred, (len(pred), 1))
    list2 = np.reshape(label, (len(label), 1))

    # Align the sequences based on the DTW path
    aligned_list1 = []
    aligned_list2 = []

    if alignment=="dtw":
        # Compute the DTW distance and path
        distance, path = fastdtw(list1, list2, dist=euclidean)
        for node in path:
            aligned_list1.append(list1[node[0]])
            aligned_list2.append(list2[node[1]])
    elif alignment=="interpolate":

        list1 = list(list1)
        list2 = list(list2)
        target_size = max(len(list1), len(list2))

        list1 = torch.tensor(list1).view(1,1,-1,1)
        list2 = torch.tensor(list2).view(1,1,-1,1)

        if list1.shape[2] > list2.shape[2]:
            list2 = torch.nn.functional.interpolate(list2, size=(target_size, 1), mode='bilinear', align_corners=True)
        else:
            list1 = torch.nn.functional.interpolate(list1, size=(target_size, 1), mode='bilinear', align_corners=True)
        list1 = list1.numpy().flatten()
        list2 = list2.numpy().flatten()
        aligned_list1 = list1
        aligned_list2 = list2

    else:
        list1 = list(list1.flatten())
        list2 = list(list2.flatten())
        if len(list1) > len(list2):
            list2.extend([0.0] * (len(list1) - len(list2)))
        else:
            list1.extend([0.0] * (len(list2) - len(list1)))

        aligned_list1 = np.array(list1)
        aligned_list2 = np.array(list2)


    # Compute the MSE between the aligned sequences
    mae = np.mean(np.abs(np.array(aligned_list1) - np.array(aligned_list2)))

    if plot_f0_path != "":
        max_len = max(len(aligned_list1), len(aligned_list2))
        x_axis = np.arange(max_len)
        plt.plot(x_axis, aligned_list1, label="generated")
        plt.plot(x_axis, aligned_list2, label="target_audio")
        plt.xlabel("Time(frame index)")
        plt.ylabel("Freqency(Hz)")
        plt.title(f"Pitch Frequency Diff of {os.path.basename(plot_f0_path)} ({alignment}), MAE={mae:.2f}")
        plt.legend(loc="upper right")
        # save the plot as a PNG image
        plt.savefig(os.path.join(plot_f0_path+f"_f0.{alignment}_bias.png"))
        plt.cla()
    return mae

class PitchMetricsCallback(pl.Callback):
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        output_dir = pl_module.extra_params.output_dir
        run_pitch_metrics(output_dir, device = pl_module.device)

def run_single_pitch_metrics(input):
    generated_output_fp, pred_tag = input
    style_audio_fp = str(generated_output_fp).replace(f'.{pred_tag}.wav', '.target_audio.wav')
    metadata_fp = str(generated_output_fp).replace(f'{pred_tag}.wav', 'metadata.json')
    plot_f0_path = str(generated_output_fp).replace(f'{pred_tag}.wav', '')

    vuv_acc = 0.0
    pitch_mae = 99
    duration = 0
    if os.path.exists(style_audio_fp):
        style_audio = load_wav(str(style_audio_fp)).reshape(-1,)
        duration = len(style_audio) / 24000
        wav = load_wav(str(generated_output_fp)).reshape(-1,)
        if len(wav) != 0:
            predict_f0, predict_vuv = get_pitch_world(wav, sr=24000)
            target_f0, target_vuv = get_pitch_world(style_audio, sr=24000)

            # actual transcript
            if duration > 0.1:
                # vuv_acc = np.sum(predict_vuv == target_vuv) / len(predict_vuv)
                vuv_acc = get_vuv_acc(predict_vuv, target_vuv)
                pitch_mae = dtw_mae(predict_f0, target_f0, plot_f0_path, alignment='dtw')
        else:
            print(generated_output_fp, "wavshape=0")

    pitch_metadata = {
        'duration': round(duration, 3),
        'vuv_acc': round(vuv_acc, 3),
        'pitch_mae(Hz)': round(pitch_mae, 3),
    }
    update_json(metadata_fp, {'pitch': pitch_metadata })
    return pitch_metadata, generated_output_fp

def run_pitch_metrics(output_dir, device='cuda', pred_tag="generated", save_json='metrics.json'):
    output_dir = Path(output_dir)
    generated_output_fps = list(output_dir.glob(f'**/*.{pred_tag}.wav'))
    if len(generated_output_fps) == 0:
        return
    category2pitch = defaultdict(list)

    data = []
    for generated_output_fp in generated_output_fps:
        data.append((generated_output_fp, pred_tag))

    for pitch_metadata, generated_output_fp in process_map(run_single_pitch_metrics, data,
                                                desc="Running pitch_metrics", chunksize=1):
        duration = pitch_metadata['duration']
        vuv_acc = pitch_metadata['vuv_acc']
        pitch_mae = pitch_metadata['pitch_mae(Hz)']
        # update total metrics
        vuv_acc *= duration
        pitch_mae *= duration
        category_dir = generated_output_fp.parent.resolve()
        if category_dir != output_dir.resolve(): # ignore category if there are none
            category2pitch[str(category_dir)].append([duration, vuv_acc, pitch_mae])
        category2pitch[str(output_dir)].append([duration, vuv_acc, pitch_mae]) # append to base directory to calculate total wer
        
    for dir_path, pitch_metrics in category2pitch.items():
        metrics_fp = f'{Path(dir_path)}/{save_json}'
        total_duration, vuv_acc, pitch_mae = np.array(pitch_metrics).sum(axis=0)
        vuv_acc /= total_duration
        pitch_mae /= total_duration
        pitch_metadata = {
            'total_duration': round(total_duration, 3),
            'vuv_acc': round(vuv_acc, 3),
            'pitch_mae(Hz)': round(pitch_mae, 3),
        }
        update_json(metrics_fp, {'pitch': pitch_metadata })
        print(f"output_dir={output_dir}, pitch metrics={pitch_metadata}")

def test():
    run_pitch_metrics("./generated_output/0208_debug_SFT_master_v3_20k_exclude_eos")


def run():
    run_pitch_metrics("./generated_output/0126_Ssvs_Lzh_Cleadsheet_bound01_20k", pred_tag="generated", save_json='metrics.json')
    run_pitch_metrics("./generated_output/0126_Ssvs_Lzh_Cleadsheet_bound01_20k", pred_tag="rerender_iter1", save_json='metrics_rerender_iter1.json')
    run_pitch_metrics("./generated_output/0126_Ssvs_Lzh_Cleadsheet_bound01_20k", pred_tag="rerender_iter2", save_json='metrics_rerender_iter2.json')
    run_pitch_metrics("./generated_output/0126_Ssvs_Lzh_Cleadsheet_bound01_20k", pred_tag="rerender_iter3", save_json='metrics_rerender_iter3.json')

if __name__ == "__main__":
    test()
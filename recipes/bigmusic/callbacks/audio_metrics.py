from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchaudio
import tqdm
import librosa

import json
import glob
import pandas as pd
import numpy as np
import pyloudnorm as pyln
from tabulate import tabulate
from recipes.musiclm.utils.dist import local_zero_first
from recipes.bigmusic.utils.format_utils import update_json
from recipes.bigmusic.utils.rewards import get_audio_metrics, get_audio_metrics_score
from recipes.musiclm.inference.utils import dump_wav
from tqdm.contrib.concurrent import thread_map
from recipes.musiclm.utils.dist import local_zero_first

import os
import time
from pydub import AudioSegment
from pydub.silence import detect_silence
from prettytable import PrettyTable
from multiprocess.pool import ThreadPool


class AudioLoundessDiffCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        (
            Path(output_dir)
            / f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS"
        ).touch()

        if trainer.is_global_zero:
            ts = time.time()
            while not all(
                [
                    (
                        Path(output_dir) / f"{self.__class__.__name__}.{rank}.SUCCESS"
                    ).exists()
                    for rank in range(trainer.world_size)
                ]
            ):
                time.sleep(10)
                print(
                    f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)"
                )

            generated_output_fps = list(
                Path(output_dir).glob("**/*.generated.wav")
            )

            loudness_diff_mean, loudness_diff_std = run_audio_loudness_diff(generated_output_fps)
            metrics_fp = Path(output_dir) / "metrics.json"
            print(f"AudioLoundessDiff: {loudness_diff_mean=}, {loudness_diff_std=}")
            update_json(
                metrics_fp,
                {
                    "loudness_diff_mean": loudness_diff_mean,
                    "loudness_diff_std": loudness_diff_std,
                },
            )

class AudioLoundessDiffCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        (
            Path(output_dir)
            / f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS"
        ).touch()

        if trainer.is_global_zero:
            ts = time.time()
            while not all(
                [
                    (
                        Path(output_dir) / f"{self.__class__.__name__}.{rank}.SUCCESS"
                    ).exists()
                    for rank in range(trainer.world_size)
                ]
            ):
                time.sleep(10)
                print(
                    f"[{self.__class__.__name__}(rank={trainer.global_rank})] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)"
                )

            generated_output_fps = list(
                Path(output_dir).glob("**/*.generated.wav")
            )

            loudness_diff_mean, loudness_diff_std = run_audio_loudness_diff(generated_output_fps)
            metrics_fp = Path(output_dir) / "metrics.json"
            print(f"AudioLoundessDiff: {loudness_diff_mean=}, {loudness_diff_std=}")
            update_json(
                metrics_fp,
                {
                    "loudness_diff_mean": loudness_diff_mean,
                    "loudness_diff_std": loudness_diff_std,
                },
            )


def run_audio_loudness_diff(generated_output_fps):
    loudness_diffs = []
    for fp in generated_output_fps:
        fp = str(fp)
        target_song = fp.replace(".generated.", ".target_audio.")
        target_loudness = get_audio_loudness(target_song)
        pred_loudness = get_audio_loudness(fp)
        audio_loudness_diff = pred_loudness-target_loudness

        metadata_fp = str(fp).replace("generated.wav", "metadata.json")
        update_json(metadata_fp, {
            "audio_loudness_diff": audio_loudness_diff,
            })
        loudness_diffs.append(audio_loudness_diff)
    return np.mean(loudness_diffs), np.std(loudness_diffs)

def get_audio_loudness(wavfile):
    wav, sr = librosa.load(wavfile, sr = None, mono=False)
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(wav.T)

def run_audio_loudness_diff(generated_output_fps):
    loudness_diffs = []
    for fp in generated_output_fps:
        fp = str(fp)
        target_song = fp.replace(".generated.", ".target_audio.")
        target_loudness = get_audio_loudness(target_song)
        pred_loudness = get_audio_loudness(fp)
        audio_loudness_diff = pred_loudness-target_loudness

        metadata_fp = str(fp).replace("generated.wav", "metadata.json")
        update_json(metadata_fp, {
            "audio_loudness_diff": audio_loudness_diff,
            })
        loudness_diffs.append(audio_loudness_diff)
    return np.mean(loudness_diffs), np.std(loudness_diffs)

def get_audio_loudness(wavfile):
    wav, sr = librosa.load(wavfile, sr = None, mono=False)
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(wav.T)

class AudioMetricsCallback(pl.Callback):
    def __init__(
        self,
        window_dur: float = 0.2,
        stride_dur: Optional[float] = None,
        max_db: float = -0.5,
        clip_ratio_threshold: float = 0.05,
        drop_last: bool = True,
    ) -> None:
        super().__init__()

        self.window_dur = window_dur
        if stride_dur is None:
            stride_dur = self.window_dur / 2
        self.stride_dur = stride_dur
        self.max_db = max_db
        self.clip_ratio_threshold = clip_ratio_threshold
        self.drop_last = drop_last

    def on_predict_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()

        with local_zero_first():
            if trainer.is_global_zero:
                ts = time.time()
                while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                    time.sleep(10)
                    print(f"[{self.__class__.__name__}] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)")

                generated_output_fps = list(Path(output_dir).glob("**/*.generated.wav"))
                run_audio_metrics(
                    output_dir,
                    generated_output_fps,
                    self.window_dur,
                    self.stride_dur,
                    self.max_db,
                    self.clip_ratio_threshold,
                    self.drop_last,
                )
                try:
                    output_for_all_samples_into_one_file(
                        output_dir
                    )
                except:
                    print(output_dir)
                    print(
                        "failed to plot or generate audio_metrics for all samples, for some unknown reason..."
                    )  # 只要文件夹结构没变，就不应该有问题，这里兜一下以防万一


        with open(
            Path(output_dir) / f"inference_params.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pl_module.extra_params, f, indent=2)


class AudioSilenceCallback(pl.Callback):
    def __init__(
        self,
        silence_threshold=-50,
        min_silence_len=50,
        silence_dur_thresh=1,
        parallel=4,
    ) -> None:
        super().__init__()
        self.silence_threshold = silence_threshold
        self.min_silence_len = min_silence_len
        self.silence_dur_thresh = silence_dur_thresh
        self.parallel = parallel

    def on_predict_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        output_dir = pl_module.extra_params.output_dir
        (Path(output_dir)/f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS").touch()

        with local_zero_first():
            if trainer.is_global_zero:
                ts = time.time()
                while not all([(Path(output_dir)/f"{self.__class__.__name__}.{rank}.SUCCESS").exists() for rank in range(trainer.world_size)]):
                    time.sleep(10)
                    print(f"[{self.__class__.__name__}] waiting for all ranks done ... (cost {round(time.time() - ts, 3)} s)")

                run_audio_silence_metrics(
                    list(Path(output_dir).glob("**/*.generated.wav")),
                    self.silence_threshold,
                    self.min_silence_len,
                    self.silence_dur_thresh,
                    self.parallel,
                    )


def get_audio_amplitude(wavfile):
    wav, sr = torchaudio.load(wavfile, normalize=True)
    return (torchaudio.transforms.AmplitudeToDB()(wav), sr)


def detect_audio_clipping(
    amplitude, window_size, stride, max_db, clip_ratio_threshold, drop_last=True
):
    if stride is None:
        stride = window_size // 2
    remain_samples = amplitude.shape[-1] % stride
    drop_last = drop_last and (remain_samples == window_size - stride)
    amplitude_slices = torch.nn.functional.unfold(
        amplitude.unsqueeze(1).unsqueeze(-1),
        kernel_size=(window_size, 1),
        stride=(stride, 1),
    )
    if drop_last:
        amplitude_slices = amplitude_slices[:, :, :-1]
    audio_clippings = (amplitude_slices > max_db).sum(
        dim=(0, 1)
    ) > int(clip_ratio_threshold * window_size)
    return audio_clippings


def run_audio_silence_metrics(audio_list, silence_threshold=-50, min_silence_len=50, silence_dur_thresh=1, parallel=4):
    def detect_audio_silence(fp):
        audio = AudioSegment.from_file(fp)
        silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)
        first_silence_dur = (silences[0][1] - silences[0][0]) / 1000 if len(silences) > 0 else 0
        max_silence_dur = max(end - start for start, end in silences) / 1000
        # total_silence_duration = sum(end - start for start, end in silences) / 1000
        # return total_silence_duration
        return first_silence_dur, max_silence_dur

    rlts = []
    pool = ThreadPool(parallel)
    for audio_fp in audio_list:
        rlt = pool.apply_async(detect_audio_silence, (audio_fp,))
        rlts.append(rlt)
    pool.close()
    final_rlts = [rlt.get() for rlt in rlts]
    pool.join()

    def plot_silence_table(silence_durations):
        table = PrettyTable(["idx", "1st silence duration (s)", "max silence duration (s)"])
        audio_sils = []
        for fp, sil_dur in zip(audio_list, silence_durations):
            if sil_dur[0] > silence_dur_thresh or sil_dur[1] > silence_dur_thresh:
                table.add_row([f"{os.path.basename(fp).split('.')[0]}", sil_dur[0], sil_dur[1]])
            audio_sils.append((fp, sil_dur))
        if len(table.rows) > 0:
            print(table)
        else:
            print("no silence longer than {silence_dur_thresh}s detected")
        return audio_sils

    return plot_silence_table(final_rlts)


def run_audio_clipping_metrics(
    wavfiles, window_dur, stride_dur, max_db, clip_ratio_threshold, drop_last
):
    if len(wavfiles) == 0:
        return
    score_list = []
    for wavfile in tqdm.tqdm(wavfiles):
        amplitude, sr = get_audio_amplitude(wavfile)
        window_size = int(window_dur * sr)
        stride_size = int(stride_dur * sr)
        metadata_fp = str(wavfile).replace("generated.wav", "metadata.json")
        audio_clippings = detect_audio_clipping(
            amplitude, window_size, stride_size, max_db, clip_ratio_threshold, drop_last
        )

        audio_clipping_rate = audio_clippings.float().mean().item()
        audio_clipping_metadata = {
            "audio_clippings": audio_clippings.detach().cpu().numpy().tolist(),
            "audio_clipping_rate": audio_clipping_rate,
        }
        update_json(metadata_fp, {"audio_clipping": audio_clipping_metadata})

        wav, sr = torchaudio.load(wavfile, normalize=True)
        wav = wav.detach().cpu().numpy()
        audio_bytes = dump_wav(wav.T, sr)
        metrics = get_audio_metrics(audio_bytes)
        if len(metrics) == 0:
            continue
        score = get_audio_metrics_score(metrics)
        score_list.append(score)
        if score < 1.:
            print(f"AudioMetrics {wavfile} score={score} {metrics['score']['worst_type']}")
        update_json(metadata_fp, metrics)

    print(f"AudioMetrics avg_score={torch.FloatTensor(score_list).mean().item()}")


def single_audio_metrics(task):
    wavfile, window_dur, stride_dur, max_db, clip_ratio_threshold, drop_last = task
    amplitude, sr = get_audio_amplitude(wavfile)
    window_size = int(window_dur * sr)
    stride_size = int(stride_dur * sr)
    metadata_fp = str(wavfile).replace("generated.wav", "metadata.json")
    audio_clippings = detect_audio_clipping(
        amplitude, window_size, stride_size, max_db, clip_ratio_threshold, drop_last
    )

    audio_clipping_rate = audio_clippings.float().mean().item()
    audio_clipping_metadata = {
        "audio_clippings": audio_clippings.detach().cpu().numpy().tolist(),
        "audio_clipping_rate": audio_clipping_rate,
    }

    wav, sr = torchaudio.load(wavfile, normalize=True)
    wav = wav.detach().cpu().numpy()
    audio_bytes = dump_wav(wav.T, sr)
    metrics = get_audio_metrics(audio_bytes)
    return metrics, audio_clipping_metadata, metadata_fp

def run_audio_metrics(
    output_dir, wavfiles, window_dur, stride_dur, max_db, clip_ratio_threshold, drop_last
):
    if len(wavfiles) == 0:
        return
    score_list = []
    downmixdiff_list = []
    leftrightdiff_list = []
    cutofffreq_list = []
    integrated_loudness_list = []
    max_short_term_loud_list = []
    clipping_rate_list = []
    audio_clipping_rate_list = []

    task_list = []
    for wavfile in wavfiles:
        task_list.append((wavfile, window_dur, stride_dur, max_db, clip_ratio_threshold, drop_last))
    for metrics, audio_clipping_metadata, metadata_fp in thread_map(single_audio_metrics, task_list, desc="Evaluting AudioMetrics"):
        update_json(metadata_fp, {"audio_clipping": audio_clipping_metadata})
        if len(metrics) == 0:
            continue
        score = get_audio_metrics_score(metrics)
        score_list.append(score)
        if score < 1.:
            filename = Path(metadata_fp).name
            print(f"AudioMetrics {filename} score={score} {metrics['score']['worst_type']}")

        if 'rms_downmix_diff' in metrics['phase_check']:
            downmixdiff_list.append(metrics['phase_check']["rms_downmix_diff"])
        if 'left_right_diff' in metrics['rms_stats']:
            leftrightdiff_list.append(metrics['rms_stats']["left_right_diff"])
        if 'cutoff_frequency' in metrics and 'rel_left' in metrics['cutoff_frequency']:
            cutofffreq_list.append(metrics['cutoff_frequency']["rel_left"])
        if 'integrated_loudness' in metrics["loudness"]:
            integrated_loudness_list.append(metrics["loudness"]["integrated_loudness"])
        if 'max_short_term_loud' in metrics["loudness"]:
            max_short_term_loud_list.append(metrics["loudness"]["max_short_term_loud"])
        if 'rate' in metrics["clipping"]:
            clipping_rate_list.append(metrics["clipping"]["rate"])
        # TODO: if rate and audio_clipping_rate are similar. Remove audio_clipping_rate
        if 'audio_clipping_rate' in audio_clipping_metadata:
            audio_clipping_rate_list.append(audio_clipping_metadata['audio_clipping_rate'])

        update_json(metadata_fp, metrics)

    all_metrics_fp = os.path.join(output_dir, 'metrics.json')
    avg_metrics = {
        "AudioMetrics":{
            "score": torch.FloatTensor(score_list).mean().item(),
            "cutofffreq": torch.FloatTensor(cutofffreq_list).mean().item(),
            "downmixdiff": torch.FloatTensor(downmixdiff_list).mean().item(),
            "leftrightdiff": torch.FloatTensor(leftrightdiff_list).mean().item(),
            "integrated_loudness": torch.FloatTensor(integrated_loudness_list).mean().item(),
            "max_short_term_loud": torch.FloatTensor(max_short_term_loud_list).mean().item(),
            "clipping_rate": torch.FloatTensor(clipping_rate_list).mean().item(),
            "audio_clipping_rate": torch.FloatTensor(audio_clipping_rate_list).mean().item(),
        }
    }
    update_json(all_metrics_fp, avg_metrics)
    print(f"AudioMetrics avg_cutofffreq={torch.FloatTensor(cutofffreq_list).mean().item()}")
    print(f"AudioMetrics avg_downmixdiff={torch.FloatTensor(downmixdiff_list).mean().item()}")
    print(f"AudioMetrics avg_leftrightdiff={torch.FloatTensor(leftrightdiff_list).mean().item()}")
    print(f"AudioMetrics avg_score={torch.FloatTensor(score_list).mean().item()}")
    print(f"AudioMetrics avg_integrated_loudnesss={torch.FloatTensor(integrated_loudness_list).mean().item()}")
    print(f"AudioMetrics avg_max_short_term_loud={torch.FloatTensor(max_short_term_loud_list).mean().item()}")
    print(f"AudioMetrics avg_clipping_rate={torch.FloatTensor(clipping_rate_list).mean().item()}")
    print(f"AudioMetrics avg_audio_clipping_rate={torch.FloatTensor(audio_clipping_rate_list).mean().item()}")

def output_for_all_samples_into_one_file(path_result):
    path_result = str(path_result)
    asv = {}
    categories = [name for name in os.listdir(path_result) if os.path.isdir(os.path.join(path_result, name))]
    categories = [x for x in categories if x[0]!='.']
    asv['all'] = {'cutoff_frequency': [], 'rms_downmix_diff': [], 'left_right_diff': [], "clipping_rate": []}

    title = ['index', 'cutoff_frequency', 'rms_downmix_diff', 'left_right_diff', "clipping_rate"]
    df = pd.DataFrame(columns=title)

    for category in categories:
        asv[category] = {'cutoff_frequency': [], 'rms_downmix_diff': [], 'left_right_diff': [], "clipping_rate": []}
        path_category = os.path.join(path_result, category)
        filenames = glob.glob(os.path.join(path_category, '*.metadata.json'))
        for filename in filenames:
            index = os.path.basename(filename).split('.')[0]
            metadata = json.load(open(filename, 'r', encoding='utf-8'))
            data = {}
            data['index'] = index
            data['cutoff_frequency'] = metadata['cutoff_frequency']['rel_left']
            data['rms_downmix_diff'] = metadata['phase_check']['rms_downmix_diff']
            data['left_right_diff'] = metadata['rms_stats']['left_right_diff']
            data['clipping_rate'] = metadata['clipping']['rate']
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    filename_out = os.path.join(path_result, 'all_metrics.txt')
    table = tabulate(df, headers='keys', tablefmt='grid')
    print(table)
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(table)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()

    import sys
    # rootdir=Path("/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/diffusion/eval/20240618-1910542965")
    rootdir = Path(args.input_dir)
    # exp = "20240607-1705272156"
    exp = sys.argv[1]
    print(f"analyse {exp}")
    rootdir=Path(f"{rootdir}/{exp}")
    run_audio_clipping_metrics(
        list(rootdir.glob('**/*.generated.wav')),
        window_dur= 0.2,
        stride_dur=0.2,
        max_db=-0.5,
        clip_ratio_threshold=0.05,
        drop_last=True,
    )

    # silence metrics
    run_audio_silence_metrics(list(rootdir.glob('**/*.generated.wav')),  silence_dur_thresh=1, parallel=4)

    run_audio_metrics(
        rootdir,
        list(rootdir.glob('**/*.generated.wav')),
        window_dur= 0.2,
        stride_dur=0.2,
        max_db=-0.5,
        clip_ratio_threshold=0.05,
        drop_last=True,
    )
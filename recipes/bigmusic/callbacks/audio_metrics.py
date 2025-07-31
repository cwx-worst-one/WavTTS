from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchaudio
import tqdm
import time

import json
import os
import glob
import pandas as pd
from recipes.musiclm.utils.dist import local_zero_first
from recipes.bigmusic.utils.format_utils import update_json
from recipes.bigmusic.utils.rewards import get_audio_metrics, get_audio_metrics_score
from recipes.musiclm.inference.utils import dump_wav
from tqdm.contrib.concurrent import thread_map


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
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # process current chunk
        output_dir = pl_module.extra_params.output_dir
        (
            Path(output_dir)
            / f"{self.__class__.__name__}.{trainer.global_rank}.SUCCESS"
        ).touch()
        with local_zero_first():
            if trainer.is_global_zero:
                ts = time.time()
                while not all(
                    [
                        (
                            Path(output_dir)
                            / f"{self.__class__.__name__}.{rank}.SUCCESS"
                        ).exists()
                        for rank in range(trainer.world_size)
                    ]
                ):
                    time.sleep(10)
                    print(
                        f"[{self.__class__.__name__}] waiting for all ranks done ... (cost {round(time.time() - ts, 3)}s)"
                    )
                output_dir = pl_module.extra_params.output_dir
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


        with open(
            Path(output_dir) / f"inference_params.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pl_module.extra_params, f, indent=2)


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
            print(f"AudioMetrics {wavfile} score={score} {metrics['score']['worst_type']}")

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

if __name__=="__main__":
    import sys
    # rootdir=Path("/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/diffusion/eval/20240618-1910542965")
    rootdir = Path("./h1")
    # exp = "20240607-1705272156"
    # exp = sys.argv[1]
    # print(f"analyse {exp}")
    # rootdir=Path(f"{rootdir}/{exp}")
    run_audio_metrics(
        rootdir,
        list(rootdir.glob('**/*.generated.wav')),
        window_dur= 0.2,
        stride_dur=0.2,
        max_db=-0.5,
        clip_ratio_threshold=0.05,
        drop_last=True,
    )

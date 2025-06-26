from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchaudio
import tqdm

from recipes.bigmusic.utils.format_utils import update_json
from recipes.bigmusic.utils.rewards import get_audio_metrics, get_audio_metrics_score
from recipes.musiclm.inference.utils import dump_wav
from recipes.musiclm.utils.dist import local_zero_first

import os
import time
from pydub import AudioSegment
from pydub.silence import detect_silence
from prettytable import PrettyTable
from multiprocess.pool import ThreadPool


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

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "output_paths" in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob("**/*.generated.wav"))
        run_audio_clipping_metrics(
            generated_output_fps,
            self.window_dur,
            self.stride_dur,
            self.max_db,
            self.clip_ratio_threshold,
            self.drop_last,
        )


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

    def on_predict_batch_end(
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


if __name__=="__main__":
    import sys
    # rootdir=Path("/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/diffusion/eval/20240618-1910542965")
    rootdir = "/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/diffusion/eval/"
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

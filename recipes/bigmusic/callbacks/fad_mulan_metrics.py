import librosa
import pytorch_lightning as pl
import torch
import tqdm
from torch.nn.functional import cosine_similarity
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import time
import os
from recipes.musiclm.utils.dist import local_zero_first
from recipes.bigmusic.utils.format_utils import update_json
from recipes.musiclm.requires.model_initializer import init_mulan
# from apps.bigmusic.umm.ar.lightning.embedding_modules import get_mulan_embeds
from functools import partial

MAX_DURATION = 5 * 60

CONFIG = {
    'mulan61': {
        'version': 'sstkmae_v3',
        'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/lixingxing.cs/models/bigmusic/instrumental/sstk_v9/mulan/mulan-step=005000-median_rank_1=61-kaggle-minimal.ckpt',
        'description': 'trained on sstk only'
    },
    'mulan63': {
        'version': 'sstkmae_v3',
        'hpath': "hdfs://harunawl/home/byte_data_seed_wl/user/bochenli/lq_backup/logs/mix_mulan/20250922_mulan_v3_0_31/checkpoints/mulan-step=004000-median_rank_1=63-kaggle.ckpt",
        # 'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/logs/mix_mulan/20250922_mulan_v3_0_31/checkpoints/mulan-step=004000-median_rank_1=63-kaggle.ckpt',
        'description': 'MuLan V3.0.31'
    },
    'mulan89': {
        'version': 'sstkmae_v3',
        'hpath': 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/mulan-step=002500-median_rank_1=89-kaggle.ckpt',
        # 'hpath': '/mnt/bn/music-llm-nas-lq/bochen/logs/mix_mulan/20250113_mix_mulan_4/checkpoints/mulan-step=002500-median_rank_1=89-kaggle.ckpt',
        'description': 'trained on inst, en vocal, cn vocal, more data, better training tricks (Bochen 20251113)'
    },
}

@torch.no_grad()
def get_mulan_embeds(requires, x, data_type="music", average=True, return_sequence=False):
    if data_type == "music":
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"], music=x.float(), device=x.device, avg=average, return_sequence=return_sequence
        )
    elif data_type == "text":
        # x should be a list of strings
        mulan_embeds = requires["mulan_infer_fn"](
            model=requires["mulan"], text=x, device=requires["mulan"].device
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return mulan_embeds

class FADMulanDistanceCallback(pl.Callback):
    def __init__(self, 
                sample_rate=24000, 
                cache_dir = "./.module_cache/musiclm",
                mulan_name = "mulan63",
                ):
        super().__init__()
        self.mulan_requires = None
        self.mulan_name = mulan_name
        mulan_config = CONFIG[mulan_name]
        mulan_ckpt = mulan_config["hpath"]
        mulan_version = mulan_config["version"]
        self.mulan_init_func = partial(init_mulan, hpath=mulan_ckpt, cache_dir=cache_dir, version=mulan_version, load_text_tower=False)
        self.sample_rate = sample_rate

    def load_mulan(self, local_rank):
        if self.mulan_requires is not None: return
        self.mulan_requires = self.mulan_init_func(local_rank=local_rank)

    def get_fad_mulan_distance(self, audio_path1, audio_path2):
        # 加载音频文件
        waveform1, sample_rate1 = librosa.load(
            str(audio_path1), mono=True, sr=self.sample_rate, duration=MAX_DURATION
        )
        waveform2, sample_rate2 = librosa.load(
            str(audio_path2), mono=True, sr=self.sample_rate, duration=MAX_DURATION
        )

        device = self.mulan_requires["mulan"].device
        waveform1 = torch.FloatTensor(waveform1).unsqueeze(0).to(device)
        waveform2 = torch.FloatTensor(waveform2).unsqueeze(0).to(device)

        # 提取音频特征
        features1 = get_mulan_embeds(self.mulan_requires, waveform1, data_type="music", average=False).squeeze(0)
        features2 = get_mulan_embeds(self.mulan_requires, waveform2, data_type="music", average=False).squeeze(0)

        # 裁剪到最短长度
        min_length = min(features1.shape[0], features2.shape[0])
        features1 = features1[:min_length]
        features2 = features2[:min_length]

        # 计算余弦相似度
        similarity = cosine_similarity(features1, features2)
        # 计算平均相似度
        average_similarity = similarity.mean().item()
        return average_similarity

    def get_fad_mulan_distance_sequence(self, audio_path1, audio_path2):
        # 加载音频文件
        waveform1, sample_rate1 = librosa.load(
            str(audio_path1), mono=True, sr=self.sample_rate, duration=MAX_DURATION
        )
        waveform2, sample_rate2 = librosa.load(
            str(audio_path2), mono=True, sr=self.sample_rate, duration=MAX_DURATION
        )

        device = self.mulan_requires["mulan"].device
        waveform1 = torch.FloatTensor(waveform1).unsqueeze(0).to(device)
        waveform2 = torch.FloatTensor(waveform2).unsqueeze(0).to(device)

        # 提取音频特征
        features1 = get_mulan_embeds(self.mulan_requires, waveform1, data_type="music", average=False, return_sequence=2)
        features2 = get_mulan_embeds(self.mulan_requires, waveform2, data_type="music", average=False, return_sequence=2)

        # 裁剪到最短长度
        min_length = min(features1.shape[0], features2.shape[0])
        features1 = features1[:min_length]
        features2 = features2[:min_length]

        # 计算余弦相似度
        similarity = cosine_similarity(features1, features2)
        # 计算平均相似度
        average_similarity = similarity.mean().item()
        return average_similarity

    def get_fad_mulan_distance_stereo(self, audio_path1, audio_path2):
        # 加载音频文件
        waveform1, sample_rate1 = librosa.load(
            str(audio_path1), mono=False, sr=self.sample_rate, duration=MAX_DURATION
        )
        waveform2, sample_rate2 = librosa.load(
            str(audio_path2), mono=False, sr=self.sample_rate, duration=MAX_DURATION
        )

        device = self.mulan_requires["mulan"].device
        waveform1 = torch.FloatTensor(waveform1).to(device)
        waveform2 = torch.FloatTensor(waveform2).to(device)

        # original waveform in mono. convert to stereo
        if len(waveform1.shape) == 1: waveform1 = torch.stack([waveform1, waveform1], dim=0)
        if len(waveform2.shape) == 1: waveform2 = torch.stack([waveform2, waveform2], dim=0)

        # 提取音频特征
        features1 = get_mulan_embeds(self.mulan_requires, waveform1, data_type="music", average=False)
        features2 = get_mulan_embeds(self.mulan_requires, waveform2, data_type="music", average=False)

        # 裁剪到最短长度
        min_length = min(features1.shape[1], features2.shape[1])
        features1 = features1[:, :min_length].reshape(-1, features1.shape[-1])
        features2 = features2[:, :min_length].reshape(-1, features2.shape[-1])

        # 计算余弦相似度
        similarity = cosine_similarity(features1, features2, dim=1)
        # 计算平均相似度
        average_similarity = similarity.mean().item()
        return average_similarity

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
            self.load_mulan(trainer.local_rank)
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

                run_fad_mulan_distance_metrics(
                    output_dir,
                    fad_mulan_distance_func=self.get_fad_mulan_distance,
                    metric_name=f"{self.mulan_name}_distance_pool"
                )

                run_fad_mulan_distance_metrics(
                    output_dir,
                    fad_mulan_distance_func=self.get_fad_mulan_distance_sequence,
                    metric_name=f"{self.mulan_name}_distance_seq"
                )

                # run_fad_mulan_distance_metrics(
                #     output_dir,
                #     fad_mulan_distance_func=self.get_fad_mulan_distance_stereo,
                #     metric_name=f"{self.mulan_name}_distance_stereo"
                # )

        with open(
            Path(output_dir) / f"inference_params.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pl_module.extra_params, f, indent=2)


def run_fad_mulan_distance_metrics(input_dir, fad_mulan_distance_func, metric_name="fad_mulan_distance"):
    def compute_fad_mulan_distance(gen_path, gt_path, fad_mulan_distance_func):
        fad_mulan_distance = fad_mulan_distance_func(gen_path, gt_path)
        metadata = {metric_name: fad_mulan_distance}
        return metadata

    def merge_all_fad_mulan_distance(category2wer):
        all_fad_mulan_distance = []
        count = 0
        fad_mulan_distance_metadata_list = []
        for dir_path, fad_mulan_distances in category2wer.items():
            metrics_fp = Path(dir_path) / "metrics.json"
            values = np.array(
                [d[metric_name] for d in fad_mulan_distances]
            )
            mean = values.mean()
            metadata = {f"{metric_name}_avg": mean, f"{metric_name}_avg_old": values[0], f"{metric_name}_count": len(values) }
            fad_mulan_distance_metadata_list.append([metrics_fp, metadata])
            all_fad_mulan_distance.append(mean)
            count += len(values)

        all_metadata = {f"{metric_name}_avg": np.mean(all_fad_mulan_distance), f"{metric_name}_avg_old": values[0], f"{metric_name}_count": count }
        all_metrics_fp = Path(dir_path.rsplit("/", 1)[0]) / "metrics.json"
        fad_mulan_distance_metadata_list.append([all_metrics_fp, all_metadata])
        return fad_mulan_distance_metadata_list

    all_metrics_fp = Path(input_dir) / "metrics.json"
    with open(all_metrics_fp, 'r', encoding='utf-8') as f:
        all_metrics_json = json.load(f)
    if metric_name in all_metrics_json:
        print(f"Metric {metric_name} exists in dir path {input_dir}. Skipping...")
        return

    generated_output_fps = list(Path(input_dir).glob("**/*.generated.wav"))
    category2wer = defaultdict(list)

    for idx, generated_output_fp in tqdm.tqdm(enumerate(generated_output_fps), desc="Evaluating FAD Mulan Distance", total=len(generated_output_fps)):
        targetaudio_fp = str(generated_output_fp).replace(
            "generated.wav", "target_audio.wav"
        )
        metadata_fp = str(generated_output_fp).replace("generated.wav", "metadata.json")

        # skip calculations if exists
        if os.path.exists(metadata_fp):
            with open(metadata_fp, 'r', encoding='utf-8') as f:
                fad_mulan_distance_metadata = json.load(f)
        else:
            fad_mulan_distance_metadata = {}

        if metric_name not in fad_mulan_distance_metadata:
            fad_mulan_distance_metadata[metric_name] = compute_fad_mulan_distance(
                generated_output_fp, targetaudio_fp, fad_mulan_distance_func
            )
            update_json(metadata_fp, fad_mulan_distance_metadata)

        if isinstance(generated_output_fp, str):
            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append(
            fad_mulan_distance_metadata[metric_name]
        )  # append to base directory to calculate total wer

    for metrics_fp, wer_metadata in merge_all_fad_mulan_distance(category2wer):
        update_json(metrics_fp, {metric_name: wer_metadata})
        print(f"output_dir={metrics_fp}, fad_mulan_distance={wer_metadata}")

# Run on single directory
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory. Comma separate for multiple directories")
    parser.add_argument("--mulan_name", type=str, default="mulan63", help="Mulan version - mulan63 (latest), mulan89 (previous)")
    parser.add_argument('--is_nested_results_dir', action='store_true', help='If true, will run metrics on each nested directory')
    args = parser.parse_args()

    mulan_name = args.mulan_name
    callback = FADMulanDistanceCallback(mulan_name=mulan_name)
    callback.load_mulan(0)

    if args.is_nested_results_dir:
        input_dirs = list(Path(args.input_dir).iterdir())
    elif "," in args.input_dir:
        input_dirs = args.input_dir.split(",")
    else:
        input_dirs = [args.input_dir]

    for input_dir in input_dirs:
        if not Path(input_dir).is_dir(): 
            print("Warning: Path is not directory... skipping", input_dir)
            continue
        print("Running distance for input_dir", input_dir)
        run_fad_mulan_distance_metrics(
            input_dir,
            callback.get_fad_mulan_distance,
            metric_name=f"{mulan_name}_distance_pool"
        )
        run_fad_mulan_distance_metrics(
            input_dir,
            callback.get_fad_mulan_distance_sequence,
            metric_name=f"{mulan_name}_distance_seq"
        )
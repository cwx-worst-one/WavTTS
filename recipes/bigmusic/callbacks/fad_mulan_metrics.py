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
                mulan_version = "sstkmae_v3",
                mulan_ckpt = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bochen/mulan/mulan-step=002500-median_rank_1=89-kaggle.ckpt",
                ):
        super().__init__()
        self.mulan_requires = None
        self.mulan_init_func = partial(init_mulan, hpath=mulan_ckpt, cache_dir=cache_dir, version=mulan_version)
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
        features1 = get_mulan_embeds(self.mulan_requires, waveform1, data_type="music", average=False, return_sequence=True)
        features2 = get_mulan_embeds(self.mulan_requires, waveform2, data_type="music", average=False, return_sequence=True)

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
                generated_output_fps = list(Path(output_dir).glob("**/*.generated.wav"))

                run_fad_mulan_distance_metrics(
                    generated_output_fps,
                    fad_mulan_distance_func=self.get_fad_mulan_distance,
                )

                run_fad_mulan_distance_metrics(
                    generated_output_fps,
                    fad_mulan_distance_func=self.get_fad_mulan_distance_sequence,
                    metric_name="fad_mulan_distance_sequence"
                )

                # run_fad_mulan_distance_metrics(
                #     generated_output_fps,
                #     fad_mulan_distance_func=self.get_fad_mulan_distance_stereo,
                #     metric_name="fad_mulan_distance_stereo"
                # )

        with open(
            Path(output_dir) / f"inference_params.json", "w", encoding="utf-8"
        ) as f:
            json.dump(pl_module.extra_params, f, indent=2)


def run_fad_mulan_distance_metrics(generated_output_fps, fad_mulan_distance_func, metric_name="fad_mulan_distance"):
    def compute_fad_mulan_distance(gen_path, gt_path, fad_mulan_distance_func):
        fad_mulan_distance = fad_mulan_distance_func(gen_path, gt_path)
        metadata = {metric_name: fad_mulan_distance}
        return metadata

    def merge_all_fad_mulan_distance(category2wer):
        all_fad_mulan_distance = []
        fad_mulan_distance_metadata_list = []
        for dir_path, fad_mulan_distances in category2wer.items():
            metrics_fp = Path(dir_path) / "metrics.json"
            mean = np.array(
                [d[metric_name] for d in fad_mulan_distances[0]]
            ).mean()
            metadata = {f"{metric_name}_avg": mean}
            fad_mulan_distance_metadata_list.append([metrics_fp, metadata])
            all_fad_mulan_distance.append(mean)

        all_metadata = {f"{metric_name}_avg": np.mean(all_fad_mulan_distance)}
        all_metrics_fp = Path(dir_path.rsplit("/", 1)[0]) / "all_metrics.json"
        fad_mulan_distance_metadata_list.append([all_metrics_fp, all_metadata])
        return fad_mulan_distance_metadata_list

    category2wer = defaultdict(list)

    for idx, generated_output_fp in tqdm.tqdm(enumerate(generated_output_fps), desc="Evaluating FAD Mulan Distance", total=len(generated_output_fps)):
        targetaudio_fp = str(generated_output_fp).replace(
            "generated.wav", "target_audio.wav"
        )
        metadata_fp = str(generated_output_fp).replace("generated.wav", "metadata.json")

        fad_mulan_distance_metadata = {}
        fad_mulan_distance_metadata[metric_name] = compute_fad_mulan_distance(
            generated_output_fp, targetaudio_fp, fad_mulan_distance_func
        )

        update_json(metadata_fp, fad_mulan_distance_metadata)
        if isinstance(generated_output_fp, str):
            import os

            category_dir = os.path.dirname(generated_output_fp)
        else:
            category_dir = generated_output_fp.parent.resolve()
        category2wer[str(category_dir)].append(
            [fad_mulan_distance_metadata[metric_name]]
        )  # append to base directory to calculate total wer

    for metrics_fp, wer_metadata in merge_all_fad_mulan_distance(category2wer):
        update_json(metrics_fp, {metric_name: wer_metadata})
        print(f"output_dir={metrics_fp}, fad_mulan_distance={wer_metadata}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    callback = FADMulanDistanceCallback()
    callback.load_mulan(0)
    
    if "," in args.input_dir:
        input_dirs = args.input_dir.split(",")
        for input_dir in input_dirs:
            print("Running distance for input_dir", input_dir)
            run_fad_mulan_distance_metrics(
                list(Path(input_dir).glob("**/*.generated.wav")),
                callback.get_fad_mulan_distance,
            )
            run_fad_mulan_distance_metrics(
                list(Path(input_dir).glob("**/*.generated.wav")),
                callback.get_fad_mulan_distance_sequence,
                metric_name="fad_mulan_distance_sequence"
            )
            # run_fad_mulan_distance_metrics(
            #     list(Path(input_dir).glob("**/*.generated.wav")),
            #     callback.get_fad_mulan_distance_stereo,
            #     metric_name="fad_mulan_distance_stereo"
            # )
    else:
        # output_fad_mulan_distance_for_all_samples_into_one_file(args.input_dir)
        run_fad_mulan_distance_metrics(
            list(Path(args.input_dir).glob("**/*.generated.wav")),
            callback.get_fad_mulan_distance,
        )
        run_fad_mulan_distance_metrics(
            list(Path(args.input_dir).glob("**/*.generated.wav")),
            callback.get_fad_mulan_distance_sequence,
            metric_name="fad_mulan_distance_sequence"
        )
        # run_fad_mulan_distance_metrics(
        #     list(Path(args.input_dir).glob("**/*.generated.wav")),
        #     callback.get_fad_mulan_distance_stereo,
        #     metric_name="fad_mulan_distance_stereo"
        # )


# if __name__ == "__main__":
#     callback = FADMulanDistanceCallback()
#     callback.load_mulan(0)
#     # output_fad_mulan_distance_for_all_samples_into_one_file(args.input_dir)
#     generated_fps = list(Path("/mnt/bn/ashaw-lq/eval/results_sacodec_diff/0416_zhongyi_metrics/quiet_wvae_8424_180k").glob("**/*.generated.wav"))
#     run_fad_mulan_distance_metrics(
#         generated_fps,
#         callback.get_fad_mulan_distance,
#     )


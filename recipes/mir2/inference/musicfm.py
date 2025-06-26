from __future__ import annotations
import os
import json
import pickle
from glob import glob
from typing import Dict, Any, List, Optional

import torch
from pydantic import BaseModel

from recipes.mir2.utils.musicfm_adapt import convert_nas_to_hdfs_in_glob
from recipes.bigmusic.inference.expins.base import BaseIns, BaseInsAgg, ExpInsConfig
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import MusicFMOutputMixin, PostprocessingMixin
from recipes.bigmusic.inference.expins.a2s import audio_to_bytes_with_clicks
from recipes.bigmusic.eval.mir_evaluator import MIREvaluator


# region Helper functions and classes

def download_from_hdfs(hdfs_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.split(hdfs_path)[-1]
    local_ckpt_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_ckpt_path):
        os.system(f'hdfs dfs -get {hdfs_path} {local_ckpt_path}')
    assert os.path.exists(local_ckpt_path), "Model does not exist or download failed"
    return local_ckpt_path


class MusicFMDfsDictBuilder(MusicFMOutputMixin, PostprocessingMixin):
    def create_output(self) -> Dict[str, Any]:
        self._check_error()
        return self.output_dict
#endregion


# region Inference instance

class MusicFMConfig(BaseModel):
    ckpt_path: str
    module_class: str = "LitFinetuneMultitask"
    load_deepspeed_folder: bool = False


MUSICFM_PRESET_DICT = {
    "20241118.musicfm.umm": MusicFMConfig(
        ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241118.musicfm.umm/checkpoints/epoch=84-step=8500.ckpt",
        module_class="MusicFMPLUMM2",
        load_deepspeed_folder=True,
    ),
    "20241120.musicfm.umm": MusicFMConfig(
        # ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241120.musicfm.umm/checkpoints/step=011000-train_loss=0.2941-val_summary=0.7155.ckpt",
        ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241120.musicfm.umm/checkpoints/step=006000-train_loss=0.2667-val_summary=0.7401.ckpt",
        module_class="MusicFMPLUMM2",
        load_deepspeed_folder=True,
    ),
    "20241120.musicfm.scratch": MusicFMConfig(
        ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241120.musicfm.scratch/checkpoints/step=030000-train_loss=0.1775-val_summary=0.6886.ckpt",
        module_class="MusicFMPLUMM2",
        load_deepspeed_folder=True,
    ),
    "20241120.musicfm.ft": MusicFMConfig(
        # ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241120.musicfm.ft/checkpoints/step=002000-train_loss=0.3518-val_summary=0.7521.ckpt",
        ckpt_path = "/home/byte_speech_sv/haonanchen/logs/20241120.musicfm.ft/checkpoints/step=010000-train_loss=0.1857-val_summary=0.7217.ckpt",
        module_class="LitFinetuneMultitask",
        load_deepspeed_folder=True,
    ),
}


class MusicFM:
    def __init__(self, config: MusicFMConfig, cache_dir: str):
        self.pl_module = MusicFM._init_module(config, cache_dir)

    @property
    def sample_rate(self):
        return self.pl_module._sample_rate

    @staticmethod
    def _init_module(config: MusicFMConfig, cache_dir: str):
        from recipes.mir_benchmark.pl_modules.multitask_finetune_pl import LitFinetuneMultitask
        from recipes.mir2.pl_modules.musicfm_pl_umm2 import MusicFMPLUMM2

        device = torch.device(f"cuda:0")
        local_ckpt_path = download_from_hdfs(config.ckpt_path, cache_dir)
        module_class = locals()[config.module_class]
        if config.load_deepspeed_folder:
        # if hasattr(module_class, "load_from_checkpoint_for_inference"):
            pl_module = module_class.load_from_checkpoint_deepspeed_folder(local_ckpt_path)
        else:
            pl_module = module_class.load_from_checkpoint(local_ckpt_path)
        pl_module.eval().to(device)
        return pl_module

    @torch.inference_mode()
    def run_audio(self, audio):
        return self.pl_module.predict_step(
            torch.tensor([audio], device=self.pl_module.device), None
        )
#endregion


# region Inference experiment management

class MusicFMInf(BaseIns):
    cache_root_dir = os.path.join(
        os.environ["DUMP_DIR"],
        "ai_music/20240312.symbolic.dump/logs/",
    )

    class Config(ExpInsConfig):
        save_ori_audio: bool = False
        add_pred_beat_click_to_ori_audio: bool = False

    def __init__(self, config: MusicFMInf.Config):
        super().__init__(config)
        musicfm_cache_dir = os.path.join(self.cache_root_dir, config.exp_id, "checkpoints")
        self.musicfm = MusicFM(MUSICFM_PRESET_DICT[config.exp_id], musicfm_cache_dir)

    def _exist(self, inf_id: str, fn_prefix: str) -> bool:
        output_dir = os.path.join(self.config.output_root_dir, inf_id)
        output_fns = [
            f"{fn_prefix}.dfs_dict_gt_or_srv.pkl",
            f"{fn_prefix}.dfs_dict.pkl",
            f"{fn_prefix}.metric.json",
            f"{fn_prefix}.ori_output.json",
        ]
        if self.config.save_ori_audio:
            output_fns.append(f"{fn_prefix}.ori.wav")
        for fn in output_fns:
            if not os.path.exists(os.path.join(output_dir, fn)):
                return False
        return True
    
    def _get_musicfm_result(self, audio) -> Dict[str, Any]:
        output = self.musicfm.run_audio(audio)
        output = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in output.items()}
        dfs_dict = (
            MusicFMDfsDictBuilder(output)
            .add_df_beat()
            .add_df_chord()
            .quantize_chord_to_beat()
            .add_df_section()
            .quantize_section_to_downbeat()
            .add_df_key()
            .quantize_key_to_section()
            .create_output()
        )
        return output, dfs_dict

    def _run_single(self, fn_prefix: str, batch: Dict[str, Any]) -> Dict[str, bytes]:
        output_dict = {}
        audio, sr = batch["audio"]
        assert sr == self.musicfm.sample_rate, "Sample rate mismatch"
        ori_output, dfs_dict = self._get_musicfm_result(audio)
        dump_keys = ["df_beat", "df_section"]
        dfs_dict_gt = {k: v for k, v in batch.items() if k in dump_keys}
        eval_tasks = batch.get("eval_tasks", [])
        mir_metric_result = MusicFMMetricCompute.run_single(dfs_dict, dfs_dict_gt, tasks=eval_tasks)
        if "dataset_name" in batch:
            mir_metric_result["dataset_name"] = batch["dataset_name"]
        output_dict[f"{fn_prefix}.metric.json"] = json.dumps(mir_metric_result)

        output_dict[f"{fn_prefix}.dfs_dict.pkl"] = pickle.dumps(dict(dfs_dict))
        if self.config.save_ori_audio:
            if self.config.add_pred_beat_click_to_ori_audio:
                bar_times = dfs_dict["df_beat"].time.values
            else:
                bar_times = None
            output_dict[f"{fn_prefix}.ori.wav"] = audio_to_bytes_with_clicks(audio[0], sr, bar_times)

        output_dict[f"{fn_prefix}.dfs_dict_gt_or_srv.pkl"] = pickle.dumps(dfs_dict_gt)
        output_dict[f"{fn_prefix}.ori_output.json"] = json.dumps(ori_output, ensure_ascii=False)

        return output_dict


class MusicFMInfAggMetric(BaseInsAgg, MusicFMInf):
    """Only save metric and aggregate them into one output.
    """
    def _run_single(self, fn_prefix: str, batch: Dict[str, Any]) -> Dict[str, bytes]:
        audio, sr = batch["audio"]
        assert sr == self.musicfm.sample_rate, "Sample rate mismatch"
        ori_output, dfs_dict_pred = self._get_musicfm_result(audio)
        # from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        eval_tasks = batch.get("eval_tasks", [])
        if "beat" not in eval_tasks:
            builder = MusicFMDfsDictBuilder(None)
            builder.output_dict = batch
            builder.output_dict["df_beat"] = dfs_dict_pred["df_beat"].copy()
            (
                builder.quantize_chord_to_beat()
                .quantize_section_to_downbeat()
                .quantize_key_to_beat()
            )
            builder.output_dict = dfs_dict_pred
            builder.quantize_key_to_beat()
        mir_metric_result = MusicFMMetricCompute.run_single(dfs_dict_pred, batch, tasks=eval_tasks)
        if "dataset_name" in batch:
            mir_metric_result["dataset_name"] = batch["dataset_name"]
        return mir_metric_result
#endregion


# region Metric compute

class MusicFMMetricCompute:
    """Compute metric for a inf_id
    suffix allows for customized suffix for sample finding
    """

    inf_root_dir = os.path.join(
        os.environ["AI_MUSIC_DIR"],
        f"../../dump/ai_music/20240312.symbolic.dump/inf/"
    )

    class Suffix(BaseModel):
        dfs_dict_pkl: str = ".dfs_dict.pkl"
        dfs_dict_gt_or_srv_pkl: str = ".dfs_dict_gt_or_srv.pkl"

    def __init__(
        self,
        inf_id: str,
        suffix: Optional[MusicFMMetricCompute.Suffix] = None
    ):
        self.inf_dir = os.path.join(self.inf_root_dir, inf_id)
        self.suffix = suffix or self.Suffix()
        self.file_prefixes = [
            os.path.relpath(fp, self.inf_dir)[:-len(self.suffix.dfs_dict_pkl)]
            for fp in glob(os.path.join(self.inf_dir, f"*{self.suffix.dfs_dict_pkl}"))
        ]
    
    def run(self, tasks: List[str] = []):
        result_dict = {}
        for prefix in self.file_prefixes:
            dfs_dict, dfs_dict_gt = self._load_data(prefix)
            result_dict[prefix] = MusicFMMetricCompute.run_single(
                dfs_dict, dfs_dict_gt, tasks=tasks
            )
        return result_dict

    def _load_data(self, prefix):
        with open(os.path.join(self.inf_dir, f"{prefix}{self.suffix.dfs_dict_pkl}"), "rb") as f:
            dfs_dict = pickle.load(f)
        with open(os.path.join(self.inf_dir, f"{prefix}{self.suffix.dfs_dict_gt_or_srv_pkl}"), "rb") as f:
            dfs_dict_gt = pickle.load(f)
        return dfs_dict, dfs_dict_gt
    
    @staticmethod
    def run_single(dfs_dict, dfs_dict_gt, tasks: List[str] =[]):
        mir_e = MIREvaluator(dfs_dict, dfs_dict_gt)
        result_dict = {}
        if "beat" in tasks:
            result_dict["beat"] = mir_e.beat_score()
            result_dict["beat"].update(mir_e.downbeat_score())
        if "chord" in tasks:
            result_dict["chord"] = mir_e.chord_beatwise_score()
        if "structure" in tasks:
            result_dict["structure"] = mir_e.structure_downbeatwise_score()
        if "key" in tasks:
            # result_dict["key"] = mir_e.key_sectionwise_score()
            result_dict["key"] = mir_e.key_beatwise_score()
        # if "vocal2midi" in tasks:
        #     result_dict["vocal2midi"] = mir_e.vocal2midi_beatwise_score()
        return result_dict
#endregion
from typing import Any, Dict, List
from collections import defaultdict
import cruise as crs
from cruise.trainer.callback import Callback
from cruise.utilities.types import STEP_OUTPUT
from cruise.trainer.logger.tracking import TrackingLogger
from cruise.utilities.rank_zero import rank_zero_info, rank_zero_warn, rank_zero_only
from io import BytesIO
import torch
import numpy as np
import torchaudio
import wandb as tk
from bytedance.easycycle import get_current_region
import bytedtos
import pickle
from recipes.bigmusic.utils.upload import upload_obj_to_tos
import os
from tasks.audio.audio_trainer import AudioTrainer

import yaml
try:
    from lite.module.datapath import datazone_hdfs_idc
except Exception:
    datazone_hdfs_idc = None

try:
    from bytedance.easycycle import get_training_data_config
except Exception:
    logger.warning("Could not import get_training_data_config from bytedance.easycycle, please upgrade your package.")
    get_training_data_config = None


class MusicTrainMeter:
    _instance = None
    _task_list = []
    _task_dataset_list = []

    def __init__(self):
        if not MusicTrainMeter._instance:
            MusicTrainMeter._instance = self
 
    
    @classmethod
    def initialize(cls, global_config):
        if cls._instance is None:
            cls()

        config = global_config.data.config
        dataset_version = config.dataset_version
        if dataset_version is not None and get_training_data_config is not None:
            # get dataset config from BigSpeech Platform
            dataset_version = str(dataset_version)
            try:
                dataset_config = get_training_data_config(dataset_version, force_hdfs_idc=datazone_hdfs_idc)
            except Exception:
                rank_zero_warn("can't support force_hdfs_idc, please make sure bytedance.easycycle greater than 1.1.43")
                dataset_config = get_training_data_config(dataset_version)
            dataset_config = yaml.safe_load(dataset_config)
            rank_zero_warn(f"update config with {dataset_version} {dataset_config}")

            def update_paths(target_paths, src_paths):
                for target_path in target_paths:
                    update_path = None
                    for src_path in src_paths:
                        if src_path['task'] == target_path['task']:
                            update_path = src_path
                    if update_path is None:
                        raise Exception(f"{target_path['task']} is not found in dataset_version:{dataset_version}")
                    else:
                        target_path.update(update_path)
            
            if "train_multitask_paths" in dataset_config["data"]:
                update_paths(config.train_multitask_paths, dataset_config["data"]["train_multitask_paths"])
            if "valid_multitask_paths" in dataset_config["data"]:
                update_paths(config.val_multitask_paths, dataset_config["data"]["valid_multitask_paths"])
            if "predict_multitask_paths" in dataset_config["data"]:
                update_paths(config.predict_multitask_paths, dataset_config["data"]["predict_multitask_paths"])

        task_list = []
        task_dataset_list = []

        for task in config.train_multitask_paths:
            task_list.append(task.task)
            for dataset in task.datasets:
                if 'dataset' in dataset:
                    task_dataset_list.append((task.task, dataset.get('dataset')))
        
        cls._task_list = task_list
        cls._task_dataset_list = task_dataset_list
        rank_zero_info(f"MusicTrainMeter init with task={cls._task_list} task_dataset={cls._task_dataset_list}")

    
    @classmethod
    def get_train_meters(cls):
        
        train_meters = []
    
        # consume_tokens(B)
        train_meters.extend(
            [
                (f'consume_tokens(B)/{task}', {'type': 'Sum', 'args': [f'consume_tokens(B)/{task}']})
                for task in cls._task_list
            ]
        )
        train_meters.extend(
            [
                (
                    f'consume_tokens(B)/{task}@@{dataset}',
                    {'type': 'Sum', 'args': [f'consume_tokens(B)/{task}@@{dataset}']},
                )
                for task, dataset in cls._task_dataset_list
            ]
        )

        # loss_tokens(B)
        train_meters.extend(
            [
                (f'loss_tokens(B)/{task}', {'type': 'Sum', 'args': [f'loss_tokens(B)/{task}']})
                for task in cls._task_list
            ]
        )
        train_meters.extend(
            [
                (
                    f'loss_tokens(B)/{task}@@{dataset}',
                    {'type': 'Sum', 'args': [f'loss_tokens(B)/{task}@@{dataset}']},
                )
                for task, dataset in cls._task_dataset_list
            ]
        )

        # loss_tokens
        train_meters.extend(
            [
                (f'loss_tokens/{task}', {'type': 'Sum', 'args': [f'loss_tokens/{task}']})
                for task in cls._task_list
            ]
        )
        train_meters.extend(
            [
                (
                    f'loss_tokens/{task}@@{dataset}',
                    {'type': 'Sum', 'args': [f'loss_tokens/{task}@@{dataset}']},
                )
                for task, dataset in cls._task_dataset_list
            ]
        )


        # acc
        train_meters.extend(
            [
                (f'acc/{task}', {'type': 'Weighted', 'args': [f'acc/{task}', f'loss_tokens/{task}']}) 
                for task in cls._task_list
            ]
        )

        train_meters.extend(
            [
                (
                    f'acc/{task}@@{dataset}',
                    {'type': 'Weighted', 'args': [f'acc/{task}@@{dataset}', f'loss_tokens/{task}@@{dataset}']},
                )
                for task, dataset in cls._task_dataset_list
            ]
        )

        # loss
        train_meters.extend(
            [
                (f'loss/{task}', {'type': 'Weighted', 'args': [f'loss/{task}', f'loss_tokens/{task}']})
                for task in cls._task_list
            ]
        )
        train_meters.extend(
            [
                (
                    f'loss/{task}@@{dataset}',
                    {'type': 'Weighted', 'args': [f'loss/{task}@@{dataset}', f'loss_tokens/{task}@@{dataset}']},
                )
                for task, dataset in cls._task_dataset_list
            ]
        )
        

        return train_meters

    @classmethod
    def calc_meters(
            cls,
            tasks : List[str],
            datasets : List[str],
            token_num : List[int], 
            loss_token_num : List[int], 
            loss : List[float], 
            acc : List[float],
            ):

        loss_dic, acc_dic, token_num_dic, loss_token_num_dic  = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

        if len(datasets) != len(tasks) or len(datasets) != len(loss) or len(datasets) != len(acc):
            rank_zero_warn(f"TokenNumPerCategory dataset({len(datasets)}), task({len(tasks)}), loss({len(loss)}), acc({len(acc)}) not same length")
            return {}
        
        for idx, (task, dataset) in enumerate(zip(tasks, datasets)):
            acc_dic[task] += acc[idx]
            acc_dic[f'{task}@@{dataset}'] += acc[idx]
            loss_dic[task] += loss[idx]
            loss_dic[f'{task}@@{dataset}'] += loss[idx]
            token_num_dic[task] += token_num[idx]
            token_num_dic[f'{task}@@{dataset}'] += token_num[idx]
            loss_token_num_dic[task] += loss_token_num[idx]
            loss_token_num_dic[f'{task}@@{dataset}'] += loss_token_num[idx]

        acc_dic = { k: v / loss_token_num_dic[k] + 1e-5 for k, v in acc_dic.items() }
        loss_dic = { k: v / loss_token_num_dic[k] + 1e-5 for k, v in loss_dic.items() }

        output = {}
        output.update({f'loss/{k}': v for k, v in loss_dic.items()})
        output.update({f'acc/{k}': v for k, v in acc_dic.items()})
        output.update({f'loss_tokens/{k}': v for k, v in loss_token_num_dic.items()})
        output.update({f'consume_tokens(B)/{k}': v * 1e-9 for k, v in token_num_dic.items()})
        output.update({f'loss_tokens(B)/{k}': v * 1e-9 for k, v in loss_token_num_dic.items()})

        return output

class SpeechDataCollectCallback(Callback):
    def __init__(self, keys_to_collect=None, audio_key="target_audio", audio_duration_key="duration", sample_rate=24000, audio_upload_to_tos=True, max_items_to_save=32, every_n_train_steps=100):
        super().__init__()
        keys_to_collect = keys_to_collect or []
        if "uttid" not in keys_to_collect:
            keys_to_collect = ["uttid"] + keys_to_collect
        if audio_key in keys_to_collect:
            keys_to_collect.remove(audio_key)
        self.keys_to_collect = keys_to_collect
        self.audio_key = audio_key
        self.audio_upload_to_tos = audio_upload_to_tos
        self.audio_duration_key = audio_duration_key
        self.sample_rate = sample_rate
        self.tracking_logger = None
        self.max_items_to_save = max_items_to_save
        self.every_n_train_steps = every_n_train_steps
        self.project = ""
        self.exp = ""
        self.version = ""

    def on_train_batch_end(
        self,
        trainer: "crs.CruiseTrainer",
        crs_module: "crs.CruiseModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        # TODO: sync dist
        if trainer.global_rank != rank_zero_only.rank_to_log:
            return
        if trainer.global_step % self.every_n_train_steps != 0:
            return
        if self.tracking_logger is None:
            if isinstance(trainer, AudioTrainer):
                logger_collection = trainer.cruise_logger
            else:
                logger_collection = trainer.logger
            for logger in logger_collection._logger_iterable:
                if isinstance(logger, TrackingLogger):
                    self.tracking_logger = logger
                    self.project = logger.project()
                    self.name = logger.name()
                    self.version = logger.version()
                    break
        if not isinstance(self.tracking_logger, TrackingLogger):
            rank_zero_warn(f"{self.__class__.__name__} only support TrackingLogger ({self.tracking_logger})")
            return    

        # infer batch size
        batch_size = len(batch[self.keys_to_collect[0]])
        if self.audio_key:
            if self.audio_key not in batch:
                rank_zero_warn(f"{self.__class__.__name__} audio_key={self.audio_key} not in batch ({list(batch.keys())})")

        rows_to_save = []
        media_to_save = {}
        columns = self.keys_to_collect + ["audio"]
        b_indices_choosen = np.random.choice(batch_size, min(self.max_items_to_save, batch_size), replace=False)
        for bidx in b_indices_choosen:
            row = []
            for key in self.keys_to_collect:
                if key not in batch:
                    row.append(None)
                    continue

                value = batch[key][bidx]
                if isinstance(value, torch.Tensor):
                    serialized_value = pickle.dumps(value.detach().cpu())
                elif isinstance(value, np.ndarray):
                    serialized_value = pickle.dumps(value)
                else:
                    try:
                        serialized_value = str(value)
                    except:
                        rank_zero_warn(f"{self.__class__.__name__} failed to convert {key}={value} to string")
                        serialized_value = None
                row.append(serialized_value)
                
            if self.audio_key and self.audio_key in batch:
                uttid = batch["uttid"][bidx]
                audio_samples = batch[self.audio_key][bidx].detach().cpu()
                if batch.get(self.audio_duration_key) is not None:
                    duration = batch[self.audio_duration_key][bidx]
                    audio_samples = audio_samples[..., :int(duration * self.sample_rate)]
                if audio_samples.ndim == 1:
                    audio_samples = audio_samples.unsqueeze(0)
                
                if self.audio_upload_to_tos:
                    audio_bytes = self.get_audio_bytes(audio_samples, self.sample_rate)
                    audio_url = upload_obj_to_tos(audio_bytes, os.path.join(self.project, self.name, str(self.version), str(trainer.global_step), f"{uttid}.wav"), verbose=True)
                else:
                    audio_url = None
                    if isinstance(audio_samples, torch.Tensor):
                        audio_samples = audio_samples.cpu().numpy()
                    assert isinstance(audio_samples, np.ndarray)
                    media_to_save[f"global_step={trainer.global_step}/{uttid}"] = tk.Audio(audio_samples, sample_rate=self.sample_rate, caption=uttid)
            else:
                audio_url = None
                
            row.append(audio_url)
            rows_to_save.append(row)

        log_dict = {
            f"global_step={trainer.global_step}": tk.Table(columns=columns, data=rows_to_save, allow_mixed_types=True),
            **media_to_save
        }
        tk.log(log_dict, step=trainer.global_step, commit=False)
        rank_zero_info(f"Logging {len(b_indices_choosen)} items to wandb in global_step={trainer.global_step}")

    @staticmethod
    def get_audio_bytes(audio_samples, sr=24000):
        assert isinstance(audio_samples, torch.Tensor)
        audio_bytes = BytesIO()
        torchaudio.save(audio_bytes, audio_samples, sr, format="wav")
        audio_bytes.seek(0)
        return audio_bytes.getvalue()

import os
import subprocess
import torch

from samantha.core import BaseModel

from recipes.chord.models.tcn import TCNStage
from recipes.beat.models.classifier import BeatClassifierStage


def get_tcn(sampling_rate=16000, n_fft=2048, hop_length=160, input_feature="hcqt", resnet_pools=[[2, 2]], n_channel=128, beat_pool=[1], n_layers=9, n_chan_conv=1024, front_conv=1, kernel_size=5):
    tcn_model_stage = TCNStage(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        input_feature=input_feature,
        resnet_pools=resnet_pools,
        n_channel=n_channel,
        n_layers=n_layers,
        n_chan_conv=n_chan_conv,
        front_conv=front_conv,
        dropout_rate=0.1,
        kernel_size=kernel_size
    )
  
    chord_classifier_stage = BeatClassifierStage(
        n_channel=n_channel,
        beat_pool=beat_pool
    )
    
    model = BaseModel(
        input_names=["audio", "aug_hop_size"],
        output_names=["beat_pred", "tempo_pred"],
        stages=[tcn_model_stage, chord_classifier_stage]
    )
    
    return model


def get_teacher(teacher_name, teacher_path):
    if not os.path.exists(f"recipes/beat/best_teacher/teacher_beat_{teacher_name}.ckpt"):
        os.makedirs("recipes/beat/best_teacher/", exist_ok=True)
        subprocess.run(f"hdfs dfs -get {teacher_path} recipes/beat/best_teacher/teacher_beat_{teacher_name}.ckpt", shell=True)
    
    if torch.cuda.is_available():
        teacher_checkpoint = torch.load(f'recipes/beat/best_teacher/teacher_beat_{teacher_name}.ckpt')
    else:
        teacher_checkpoint = torch.load(f'recipes/beat/best_teacher/teacher_beat_{teacher_name}.ckpt', 'cpu')

    if teacher_name == 'tcn':
        teacher = get_tcn()
        for key in list(teacher_checkpoint["state_dict"]):
            teacher_checkpoint["state_dict"][key[6:]] = teacher_checkpoint["state_dict"].pop(key)
    else:
        raise "teacher must be in [tcn]"
    
    teacher.load_state_dict(teacher_checkpoint["state_dict"])

    return teacher
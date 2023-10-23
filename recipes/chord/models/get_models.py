import os
import subprocess
import torch

from samantha.core import BaseModel
from recipes.beat.models.perceiver import BeatPerceiverModelStage
from recipes.chord.models.classifier import ChordClassifierStage, ChordPerceiverClassifierStage


def get_perceiver(
        sample_rate=16000, 
        sample_len=12, 
        hop_length=500, 
        n_layers=5, 
        spec_dim=128, 
        temporal_dim=128, 
        temporal_heads=4, 
        resnet_pools=[[2, 2], [2, 1]],
        n_fft=2048,
        semitone_scale=1,
        freq_pool_size=4,
        time_pool_size=2,
        num_fct=4,
        n_channel=512,
        chord_pool=[2],
        attn_dropout=0.3,
        ff_dropout=0.3,
        pretrain_path=None
    ):
    perceiver_model_stage=BeatPerceiverModelStage(
        sample_rate=sample_rate,
        sample_len=sample_len,
        hop_len=hop_length,
        n_layers=n_layers,
        spec_dim=spec_dim,
        temporal_dim=temporal_dim,
        temporal_heads=temporal_heads,
        resnet_pools=resnet_pools,
        n_fft=n_fft,
        semitone_scale=semitone_scale,
        freq_pool_size=freq_pool_size,
        time_pool_size=time_pool_size,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        num_fct=num_fct
    )

    chord_classifier_stage = ChordPerceiverClassifierStage(n_channel=n_channel, chord_pool=chord_pool)

    model = BaseModel(
        input_names=["audio", "aug_hop_size"],
        output_names=["chord_root", "chord_triad"],
        stages=[perceiver_model_stage, chord_classifier_stage]
    )

    return model


def get_teacher(teacher_name, teacher_path):
    if not os.path.exists(f"recipes/chord/best_teacher/teacher_chord_{teacher_name}.ckpt"):
        os.makedirs("recipes/chord/best_teacher/", exist_ok=True)
        subprocess.run(f"hdfs dfs -get {teacher_path} recipes/chord/best_teacher/teacher_chord_{teacher_name}.ckpt", shell=True)
    if torch.cuda.is_available():
        teacher_checkpoint = torch.load(f'recipes/chord/best_teacher/teacher_chord_{teacher_name}.ckpt')
    else:
        teacher_checkpoint = torch.load(f'recipes/chord/best_teacher/teacher_chord_{teacher_name}.ckpt', 'cpu')        

    if teacher_name == 'perceiver':
        teacher = get_perceiver()
        for key in list(teacher_checkpoint["state_dict"]):
            if "teacher" not in key:
                teacher_checkpoint["state_dict"][key[6:]] = teacher_checkpoint["state_dict"].pop(key)
            else:
                teacher_checkpoint["state_dict"].pop(key)
    else:
        raise "teacher must be in [tcn, perceiver]"

    
    teacher.load_state_dict(teacher_checkpoint["state_dict"])

    return teacher
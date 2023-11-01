import os
import subprocess

import torch

from recipes.beat.models.spectnt import SpecTNTModelStage
from recipes.structure.models.classifier import StructureClassifierStage
from samantha.core import BaseModel


def get_specTNT(
        sample_len=36,
        sample_rate=16000,
        n_fft=1024,
        resnet_pool_sizes=[[2, 3], [2, 1], [1, 2]],
        depth=5,
        temporal_dim=128,
        temporal_heads=8,
        spec_dim=96,
        spec_heads=4,
        n_boundary=2,
        n_function=7,
        semitone_scale=2
    ):

    freq_pool_size, time_pool_size = 1, 1
    for pool in resnet_pool_sizes:
        freq_pool_size *= pool[0]
        time_pool_size *= pool[1]

    spectnt_model_stage = SpecTNTModelStage(
        sample_len=sample_len,
        sample_rate=sample_rate,
        hop_len=n_fft // 2,
        n_layers=depth,
        spec_dim=spec_dim,
        temporal_dim=temporal_dim,
        temporal_heads=temporal_heads,
        spec_heads=spec_heads,
        resnet_pools=resnet_pool_sizes,
        n_fft=n_fft,
        semitone_scale=semitone_scale,
        freq_pool_size=freq_pool_size,
        time_pool_size=time_pool_size,
    )

    structure_classifier_stage = StructureClassifierStage(
        n_channel=temporal_dim, n_boundary=n_boundary, n_function=n_function
    )

    model = BaseModel(
        input_names=["audio", "aug_hop_size"],
        output_names=["boundary_pred", "function_pred"],
        stages=[spectnt_model_stage, structure_classifier_stage],
    )
    
    return model


def get_teacher(teacher_name, teacher_path):
    if not os.path.exists(
        f"recipes/structure/best_teacher/teacher_structure_{teacher_name}.ckpt"
    ):
        os.makedirs("recipes/structure/best_teacher/", exist_ok=True)
        subprocess.run(
            f"hdfs dfs -get {teacher_path} \
                recipes/structure/best_teacher/teacher_structure_{teacher_name}.ckpt",
            shell=True,
        )

    if torch.cuda.is_available():
        teacher_checkpoint = torch.load(
            f"recipes/structure/best_teacher/teacher_structure_{teacher_name}.ckpt"
        )
    else:
        teacher_checkpoint = torch.load(
            f"recipes/structure/best_teacher/teacher_structure_{teacher_name}.ckpt", "cpu"
        )

    if teacher_name == "spectnt":
        teacher = get_specTNT()
        
        ckpt_keys = list(teacher_checkpoint['model_state_dict'].keys())
        for key in ckpt_keys:
            if 'hstft' in key:
                teacher_checkpoint["model_state_dict"][key.replace('module', 'stages.0.preprocess')] = teacher_checkpoint["model_state_dict"].pop(key)
            elif 'conv_module' in key:
                teacher_checkpoint["model_state_dict"][key.replace('module.conv_module', 'stages.0.input_layer')] = teacher_checkpoint["model_state_dict"].pop(key)
            elif 'output_boundary' in key:
                teacher_checkpoint["model_state_dict"][key.replace('module', 'stages.1')] = teacher_checkpoint["model_state_dict"].pop(key)
            elif 'output_function' in key:
                teacher_checkpoint["model_state_dict"][key.replace('module', 'stages.1')] = teacher_checkpoint["model_state_dict"].pop(key)
            elif 'output_keysig' in key:
                teacher_checkpoint["model_state_dict"].pop(key)
            else:
                teacher_checkpoint["model_state_dict"][key.replace('module', 'stages.0')] = teacher_checkpoint["model_state_dict"].pop(key)
                    
    else:
        raise "teacher must be in [spectnt]"
    
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])

    return teacher

import torch
import logging
import os
import io
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from recipes.bigmusic.utils.audio_utils import audio_array_to_bytes
from recipes.bigmusic.utils.common_utils import ExternalModule
import samantha.utils.hdfs_helper as hh


class SamiMIRError(Exception):
    pass


class SamiMIRConfig:
    CKPT_HDFS_DIR = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/sami_models'
    LOCAL_DIR = "/opt/tiger/samantha/.module_cache/mir"  # TODO: make it configurable
    # Configurable
    MIR_REPO_DIR = os.getenv("BIGMUSIC_SAMI_MODELS_DIR", "/opt/tiger/samantha/bigmusic_sami_models")
    BASE_BATCH_SIZE = os.getenv("BIGMUSIC_SAMI_MODELS_BASE_BATCH_SIZE", 1)  # 1 for V100-32G; 4 for A100-80G


class SamiMIRTransform:
    def __init__(self, sample_rate: Optional[int] = None):
        """
        Arg:
            sample_rate: The default sample rate for tensor-to-bytes conversion.
        """
        self.logger = logging.getLogger(f"{__name__}#{id(self)}")
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.sample_rate = sample_rate
        self.model_paths = ExternalModule(
            SamiMIRConfig.MIR_REPO_DIR,
            "utils.bigmusic_sami_models_path",
        ).getattr("MODEL_WEIGHT_PATH")
        self.model = self._init_model()

    def _init_model(self):
        raise NotImplementedError()

    @torch.no_grad()
    def __call__(self, wav: Union[torch.Tensor, bytes], sample_rate: Optional[int] = None):
        """
        Arg:
            wav: A tensor with shape of (num_channels, num_samples) or audio bytes.
            sample_rate: Sample rate of the audio. It will override the default sample_rate if provided.
        """
        if isinstance(wav, torch.Tensor):
            sample_rate = sample_rate or self.sample_rate
            if sample_rate is None:
                raise SamiMIRError("sample_rate must be provided when wav is a tensor")
            wav= _tensor_to_bytes(wav, sample_rate)
        out_dict = self.model.predict(wav)
        json_temp = json.dumps(out_dict, cls=_json_serialize)
        output = json.loads(json_temp)
        return output


class Vocal2MidiTransform(SamiMIRTransform):
    def _init_model(self):
        # Make sure we can import the two modules
        mir_vocal2midi_model = ExternalModule(
            SamiMIRConfig.MIR_REPO_DIR,
            "sami_models.mir_models.vocal2midi.mir_vocal2midi_model",
        )
        hdfs_weight_path = self.model_paths["vocal2midi"]

        # Get the paths
        local_model_path = _download_to_dir(hdfs_weight_path)

        # Load the model
        return mir_vocal2midi_model.getattr("MirVocalMelodyExtractionModel")(
            model_path=local_model_path,
            batch_size=SamiMIRConfig.BASE_BATCH_SIZE * 2,
            map_location=self.device
        )


class DeepchorusTransform(SamiMIRTransform):
    def _init_model(self):
        # Make sure we can import the two modules
        mir_vocal2midi_model = ExternalModule(
            SamiMIRConfig.MIR_REPO_DIR,
            "sami_models.mir_models.structure.mir_structure_model",
        )
        hdfs_weight_path = self.model_paths["structure_downbeat"]
        config_path = self.model_paths["structure_config"]

        # Get the paths
        local_model_path = _download_to_dir(hdfs_weight_path)
        local_config_path = _download_to_dir(config_path)

        # Load the model
        return mir_vocal2midi_model.getattr("StructureSpecTntModel")(
            model_path=local_model_path,
            config_path=local_config_path,
            batch_size=SamiMIRConfig.BASE_BATCH_SIZE * 2,
            map_location=self.device
        )


def _download_to_dir(suffix_hdfs_path: str) -> str:
    local_path = Path(SamiMIRConfig.LOCAL_DIR) / Path(suffix_hdfs_path).name
    if local_path.exists():
        return str(local_path)
    hdfs_path = f"{SamiMIRConfig.CKPT_HDFS_DIR}/{suffix_hdfs_path}"
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    is_successful = hh.get(f"{hdfs_path}", str(local_path))
    if not is_successful:
        raise SamiMIRError("Failed downloading weight file")
    return str(local_path)


class _json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _tensor_to_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> io.BytesIO:
    """
    Args:
        audio_tensor: A tensor with shape of (num_channels, num_samples)
        sample_rate: The audio sample rate
    """
    audio_tensor = audio_tensor.T
    audio_numpy = audio_tensor.numpy()
    return audio_array_to_bytes(audio_numpy, sample_rate)

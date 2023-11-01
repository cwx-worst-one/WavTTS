import io
from abc import ABC

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

from recipes.beat.utils.augment_utils import get_augmentations
from recipes.structure.preprocess.utils import get_structure_training_labels
from recipes.beat.preprocess.common import HotGalaxyDatasetMixin, MCCDatasetMixin, MCCMSSDatasetMixin
from samantha.dataio.preprocess import AudioLengthModifier


class StructureDatasetMixin(ABC):
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_hop,
        enable_short,
        augmentation,
        chunk_per_sample,
        *args,
        **kwargs,
    ):
        super().__init__()  # forwards all unused arguments
        self._audio_length_modifier = AudioLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_hop = label_hop
        self._sampling_rate = sampling_rate
        self._enable_short = enable_short
        self._target_duration_sec = target_duration_sec
        self._augmentation = augmentation
        if augmentation:
            self.augmentations = get_augmentations(sampling_rate)
        self._chunk_per_sample = chunk_per_sample

    def train_preprocess(self, x):
        # get original data
        np_audio = x["audio.npy"] / 32768
        audio_duration_in_s = len(np_audio) / self._sampling_rate
        (
            boundary_label,
            function_label,
            chorus_only,
            chorus_interval,
            structure_intervals,
        ) = get_structure_training_labels(
            x, audio_duration_in_s, self._label_hop, self._enable_short
        )
        chunk_per_sample = min(1, int(audio_duration_in_s // 9))
        data_queue = []
        for i in range(chunk_per_sample):
            output_dict = {}
        
            # get chunked data
            audio, _, start_idx = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            sample_start_time = start_idx / self._sampling_rate
            label_len = int(self._target_duration_sec / self._label_hop)
            sample_idx = int(round(sample_start_time * (1 / self._label_hop)))
            boundary_label = boundary_label[sample_idx : sample_idx + label_len, :]
            function_label = function_label[sample_idx : sample_idx + label_len, :]
            pad_len = label_len - boundary_label.shape[0]
  
            if pad_len > 0:
                boundary_label = np.pad(boundary_label, [(0, pad_len), (0, 0)], "constant")
                function_label = np.pad(function_label, [(0, pad_len), (0, 0)], "constant")

            if self._augmentation:
                audio = self.augmentations(audio)

            output_dict["audio"] = audio.squeeze()
            output_dict["boundary_label"] = boundary_label
            output_dict["function_label"] = function_label
            output_dict["chorus_only"] = chorus_only
            output_dict["__key__"] = x["__key__"]

            data_queue.append(output_dict)

        return data_queue

    def val_preprocess(self, x):
        output_dict = {}

        x['intervals.pickle'].pop(-1)
        x['labels.pickle'].pop(-1)

        for i, interval in enumerate(x["intervals.pickle"]):
            start, end = float(interval[0]), float(interval[1])
            if end - start <= 0:
                x["intervals.pickle"].pop(i)
                x["labels.pickle"].pop(i)

        # get original data
        np_audio = x["audio.npy"] / 32768
        audio_duration_in_s = len(np_audio) / self._sampling_rate
        (
            boundary_label,
            function_label,
            chorus_only,
            chorus_interval,
            structure_intervals,
        ) = get_structure_training_labels(
            x, audio_duration_in_s, self._label_hop, self._enable_short
        )

        output_dict["audio"] = torch.from_numpy(np_audio).squeeze()
        output_dict["boundary_label"] = boundary_label
        output_dict["function_label"] = function_label
        output_dict["chorus_only"] = chorus_only
        output_dict["boundary_interval"] = structure_intervals
        output_dict["chorus_interval"] = chorus_interval
        output_dict["segment_type"] = x["segment_type.txt"]
        output_dict["__key__"] = x["__key__"]
        return output_dict


class StructurePreprocessor:
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        label_hop,
        augmentation=False,
        chunk_per_sample=1,
        enable_short=True,
        *args,
        **kwargs,
    ):
        self._dataset_preprocessors = {}
        self._dataset_preprocessors = StructureDatasetMixin(
            target_duration_sec,
            sampling_rate,
            label_hop,
            enable_short,
            augmentation,
            chunk_per_sample,
            *args,
            **kwargs,
        )
        self._mcc_preprocessors = MCCDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._hotgalaxy_preprocessors = HotGalaxyDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )

    def train_batch_preprocess(self, batch):
        for x in batch:
            if 'license-based_mcc' in x['__url__']:
                preprocessor = self._mcc_preprocessors
            elif 'hot_galaxy' in x['__url__']:
                preprocessor = self._hotgalaxy_preprocessors
            else:
                preprocessor = self._dataset_preprocessors
            
            data_queue = preprocessor.train_preprocess(x)
            while len(data_queue) > 0:
                yield data_queue.pop(0)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)

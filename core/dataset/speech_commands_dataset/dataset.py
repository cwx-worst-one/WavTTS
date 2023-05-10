from typing import Optional, Union
from pathlib import Path
import os
import random
import torch
import torch.utils.data as data
import numpy as np
import torchaudio
from tqdm import tqdm
from torchvision.transforms import Compose
from core.dataset.preprocess.wav import (
    ChangeAmplitude,
    FixAudioLength,
    ChangeSpeedAndPitchAudio,
    TimeshiftAudio,
)
from core.dataset.base import BaseDataset

URL = "speech_commands_v0.01"
NOISE_FOLDER = "_background_noise_"


class SpeechCommandV1(data.Dataset):
    """Create a Dataset for Speech Commands V1.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
        transform(None, optional):
            methods to transform the audio
        num_classes:
            support: 12
    """

    def __init__(
        self,
        root: Union[str, Path],
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: Optional[str] = None,
        silence_percent=0.1,
        transform=None,
        num_classes=12,
        noise_ratio=None,
        noise_max_scale=0.4,
        cache_origin_data=False,
    ) -> None:
        self.classes = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "bed",
            "bird",
            "cat",
            "dog",
            "happy",
            "house",
            "marvin",
            "sheila",
            "tree",
            "wow",
            "backward",
            "forward",
            "follow",
            "learn",
            "visual",
            "silence",
        ]
        self.classes_12 = [
            "unknown",
            "silence",
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root, URL, folder_in_archive, download, subset)
        data_path = os.path.join(root, folder_in_archive, URL)
        self.datas = []
        for fileid in dataset._walker:
            relpath = os.path.relpath(fileid, data_path)
            label, _ = os.path.split(relpath)
            self.datas.append([fileid, label])

        self.sample_rate = 16000

        # setup silence
        if silence_percent > 0:
            silence_data = [["", "silence"] for _ in range(int(len(dataset) * silence_percent))]
            self.datas.extend(silence_data)

        # setup noise
        self.noise_folder = os.path.join(root, folder_in_archive, URL, NOISE_FOLDER)
        self.noise_files = (
            sorted(str(p) for p in Path(self.noise_folder).glob("*.wav"))
            if subset is "training" and noise_ratio is not None
            else None
        )
        self.num_classes = num_classes
        self.transform = transform
        self.noise_ratio = noise_ratio
        self.noise_max_scale = noise_max_scale
        self.silence_ratio = silence_percent
        if noise_ratio is not None and subset is "training":
            assert 0 < noise_max_scale < 1
        assert num_classes == 12, "only support V1-12 now"
        self.cache_origin = cache_origin_data
        self.origin_datas = dict()
        self.origin_noise_datas = dict()

    def __len__(self):
        return len(self.datas)

    def label_to_name(self, label):
        if label >= 12:
            return "unknown"
        return self.classes_12[label]

    def name_to_label(self, name):
        if name not in self.classes_12:
            return 0
        return self.classes_12.index(name)

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.datas:
            index = self.classes.index(item[1])
            count[index] += 1

        N = float(sum(count))
        weight_per_class = [N / item if item > 0 else 0 for item in count]

        weight = np.zeros(len(self))
        for idx, item in enumerate(self.datas):
            index = self.name_to_label(item[1])
            weight[idx] = weight_per_class[index]
        return weight

    def __getitem__(self, index):
        """
        return feature and label
        """
        # Tensor, int, str, str, int
        if index in self.origin_datas.keys():
            [waveform, _, label] = self.origin_datas[index]
        else:
            waveform, sample_rate, label = self.pull_origin(index)
            if sample_rate != self.sample_rate:
                raise RuntimeError
            if self.cache_origin:
                self.origin_datas[index] = [waveform, sample_rate, label]

        if self.noise_files is not None and random.uniform(0, 1) < self.noise_ratio:
            noise_file = random.choice(self.noise_files)
            if noise_file in self.origin_noise_datas.keys():
                waveform_noise = self.origin_noise_datas[noise_file]
            else:
                waveform_noise, _ = torchaudio.load(noise_file)
                if self.cache_origin:
                    self.origin_noise_datas[noise_file] = waveform_noise
            noise_len = waveform_noise.shape[1]
            wav_len = waveform.shape[1]
            if noise_len >= wav_len:
                rand_start = random.randint(0, noise_len - wav_len - 1)
                waveform_noise = waveform_noise[:, rand_start : wav_len + rand_start]
            else:
                waveform_noise = torch.nn.functional.pad(waveform_noise, (0, wav_len - noise_len))
            random_scale = random.uniform(0, self.noise_max_scale)
            waveform = waveform * (1 - random_scale) + waveform_noise * random_scale

        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, self.name_to_label(label)

    def pull_origin(self, index):
        """
        get original item
        """
        [data_id, label] = self.datas[index]
        if data_id != "":
            waveform, sample_rate = torchaudio.load(data_id)
        else:
            waveform = torch.zeros(1, 16000)
            sample_rate = 16000
        return waveform, sample_rate, label


class SpeechCommandDataset(BaseDataset):
    """
    SpeechCommandDataset warp.
    """

    def __init__(self, cfg, dataset_type):
        super().__init__(
            path_list=list(),
            bucket_schedule=list(),
            cfg=cfg,
        )
        self.data_loader = self.get_data_loader(cfg, dataset_type)
        self.data_iter = None
        self.reset()
        self.batch_size = self.data_loader.batch_size

    def get_data_loader(self, cfg, dataset_type):
        """
        build the data loader.
        """

        train_transform = Compose(
            [
                ChangeAmplitude(),
                ChangeSpeedAndPitchAudio(),
                TimeshiftAudio(),
                FixAudioLength(),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=cfg.fbank_dim,
                    normalized=True,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            ]
        )
        valid_transform = Compose(
            [
                FixAudioLength(),
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=cfg.fbank_dim,
                    normalized=True,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            ]
        )

        if not os.path.exists(cfg.data_root):
            os.makedirs(cfg.data_root)

        dataset_train = SpeechCommandV1(
            cfg.data_root,
            subset="training",
            download=True,
            transform=train_transform,
            noise_ratio=0.3,
            noise_max_scale=0.3,
            cache_origin_data=False,
        )

        dataset_valid = SpeechCommandV1(
            cfg.data_root,
            subset="validation",
            download=True,
            transform=valid_transform,
            cache_origin_data=True,
        )

        dataset_test = SpeechCommandV1(
            cfg.data_root,
            subset="testing",
            download=True,
            transform=valid_transform,
            cache_origin_data=True,
        )
        dataset_dict = {
            "training": dataset_train,
            "validation": dataset_valid,
            "testing": dataset_test,
        }

        return data.DataLoader(
            dataset=dataset_dict[dataset_type],
            batch_size=96,
            shuffle=(dataset_type == "training"),
            sampler=None,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True,
        )

    def reset(self):
        """
        reset this dataset.
        """
        # self.data_iter = iter(tqdm(
        #         self.data_loader,
        #         unit="audios",
        #         unit_scale=self.data_loader.batch_size))
        self.data_iter = iter(self.data_loader)

    def terminate(self):
        super().terminate()

    def next(self):
        """
        iter the dataset.
        """
        item = next(self.data_iter, None)
        if item is None:
            return None

        batch_data = dict()
        feat, target = item
        batch_data["src"] = feat.cuda()
        batch_data["char"] = target.cuda()
        batch_data["src_mask"] = None

        return batch_data

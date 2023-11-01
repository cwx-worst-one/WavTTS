import os

import numpy as np

from samantha.dataio.webdataset import AbstractWebDatasetGeneratorBasedBuilder


class AbstractStructureDataset(AbstractWebDatasetGeneratorBasedBuilder):

    AUDIO_NUMPY_DTYPE = np.float32
    SAMPLING_RATE = 16000

    def __init__(self) -> None:
        super().__init__()

    def _import_segment_annotation(self, annotation_path, total_length):
        file = os.path.join(annotation_path)
        ss = open(file, "r").read().splitlines()
        times = []
        symbol = []

        for j in range(len(ss)):
            s = ss[j].split("\t")
            start = float(s[0])

            if (len(ss) - 1) == j:
                end = total_length
            else:
                s1 = ss[j + 1].split("\t")
                end = float(s1[0])

            times.append([start, end])
            tag = "".join(
                [i for i in s[1] if not i.isdigit()]
            )  # remove digits in funct name
            symbol.append(tag)

        entry = {"interval": times, "labels": symbol}
        return entry

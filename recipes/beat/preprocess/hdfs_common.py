import h5py

from recipes.beat.preprocess.common import BeatDatasetMixin


class BeatPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, label_hop, *args, **kwargs):
        self._dataset_preprocessors = BeatDatasetMixin(
            target_duration_sec, sampling_rate, label_hop, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec
        self._hop_factor = 1  # for beat

    def train_batch_preprocess(self, x):
        data_dict = {}
        with h5py.File(x["h5_file"], "r") as h5_file:
            data_dict["dataset.txt"] = h5_file[x["split"]][x["__key__"]].attrs[
                "dataset.txt"
            ]
            data_dict["__key__"] = x["__key__"]
            data_dict["beats.pickle"] = h5_file[x["split"]][x["__key__"]].attrs[
                "beats.pickle"
            ]
            data_dict["audio.npy"] = h5_file[x["split"]][x["__key__"]]["audio.npy"][:]

        output = self._dataset_preprocessors.train_preprocess(data_dict)

        return (output["audio.npy"], output["beats.pickle"], output["tempo_label"])

    def val_preprocess(self, x):
        data_dict = {}
        with h5py.File(x["h5_file"], "r") as h5_file:
            data_dict["dataset.txt"] = h5_file[x["split"]][x["__key__"]].attrs[
                "dataset.txt"
            ]
            data_dict["__key__"] = x["__key__"]
            data_dict["beats.pickle"] = h5_file[x["split"]][x["__key__"]].attrs[
                "beats.pickle"
            ]
            data_dict["audio.npy"] = h5_file[x["split"]][x["__key__"]]["audio.npy"][:]
        preprocessor = self._dataset_preprocessors
        output = preprocessor.val_preprocess(data_dict)
        return (
            output["audio.npy"].unsqueeze(0),
            output["beats.pickle"],
            output["tempo_label"],
            output["orig_beats"][None, :],
            output["dataset.txt"],
        )

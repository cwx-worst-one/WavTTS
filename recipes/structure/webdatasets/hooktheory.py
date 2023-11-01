"""
Billboard Structure Recognition Dataset.
"""

import os
import random

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/h5/hooktheory_merged_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/hooktheory_segments.zip",
}


HOOKTHEORY_VALIDATION_KEYS = [
    "07137_weRHyjj34ZE",
    "04252_M8fQ8H81RoA",
    "00292_N_lCmBvYMRs",
    "05483_aHtWKbR_P-c",
    "06713_9wfpXI5PKlw",
    "03778_9LmdO3TCzpM",
    "13343_zpHuG2hHmEo",
    "15545_k_JTV-6M1tQ",
    "11292_2Abk1jAONjw",
    "01789_CEjU9KVABao",
    "14442_ybaPCmfnApY",
    "05704_1NALkdPuHDQ",
    "08667_ZRGYNdxX2qU",
    "04061_Z6ai_0RnzYg",
    "03576_pcamjcoRmrQ",
    "01944_cs8rkoouPSU",
    "03206_L0bcRCCg01I",
    "14804_Abf8CQ_dxzc",
    "15117_IJiHDmyhE1A",
    "11020_82v0kX95EY0",
    "12243_cN9jTnxv0RU",
    "07141_wwCykGDEp7M",
    "05443_Qs7Fnr0c1Ng",
    "14437_khD4w3HzLl8",
    "03261__kHjDNHEU5o",
    "09226_m2DrKruxJUw",
    "02694_gGdGFtwCNBE",
    "04787_uINxIvwK1eo",
    "09431_9k-XrIGobsA",
    "02774_QStFk-dIzhU",
    "12324_IKaNodcBiLA",
    "14191_Ckxy-i7Dc5U",
    "14697_8tlZJvTijhY",
    "09860_m-80zLGzgXc",
    "00443__4IRMYuE1hI",
    "01762_HuS5NuXRb5Y",
    "05639_NE00ckCTwDo",
    "13797_l-sZyfFX4F0",
    "06735_FA5jsa1lR9c",
    "07386_HoRkntoHkIE",
    "07755_iFca32_7YUU",
    "01937_IG2zzxYMqDo",
    "05473_SXJGTnVfJic",
    "09639_ETxmCCsMoD0",
    "04306_3j9YFNx12I0",
    "07211_FI5xme5k5AQ",
    "09334_1bTHhXSpd4U",
    "00947_lXEQRVJRors",
    "16156_ZMScU2bRORE",
    "06468_emGri7i8Y2Y",
    "10941_Pu9rQ8lkQ5c",
    "15140_rnZoIAODQMA",
    "02545_Dxv0BTWxEq4",
    "15297_0bGjlvukgHU",
    "13564_WEvQxAxZ1YE",
    "05733_NeKTy5oXLoU",
    "14216_x9X2GN87F18",
    "03663_XfpEW8OOqKw",
    "14898_fyWp6t4fn_8",
    "09256_GWNDB1juaeo",
    "02528_Kz5scjSQ_WE",
    "10053_wEERFBI9eCg",
    "14807_bduVVl_xniE",
    "07345_qhCS7etNEbU",
    "15144_diTh-orgv8Q",
    "10094_kaSYvvfzYaE",
    "05886_Y5UVKmZpqao",
    "09191_ijituRe039w",
    "14479_wLUOlJV0uHA",
    "09459_JbXVNKtmWnc",
    "01346_DFI6cV9slfI",
    "00535_JXYn8nlHkDk",
    "11212_EqWRaAF6_WY",
    "00928_sf6LD2B_kDQ",
    "10659_MS1a-SQd79w",
    "01877_DPgmRAwl0ZA",
    "02666_W6TSOymYmPU",
    "15916_S1DWNsV2rYA",
    "08004_KWZGAExj-es",
    "15574_1oDHAgvUqzc",
    "13903_qktFsWuHzxE",
    "08020_uhG-vLZrb-g",
    "03237_kXYiU_JCYtU",
    "14289_4Twd965VzX4",
    "07353_JvUMV1N7eGM",
    "00857_xh5xDchVdPI",
    "13379_Suxkbl5R5yo",
    "06909_GlZxZ2n2zpw",
    "12820_kC2QK6KHnEA",
    "04637_CWnYIb2lqpo",
    "00602_O8CjdPDz-qY",
    "00463_5BS_rG_XZ0Y",
    "15344_Vt2YIpZWBqA",
    "02865_F57P9C4SAW4",
    "03347_nDyd_U40Lp8",
    "11244_BMcs4pbsmeg",
    "07791_bpOSxM0rNPM",
    "10983_oxqnFJ3lp5k",
    "02607_rrvFv6j3-sM",
    "07463_cuqKzBxAnE8",
]


class HooktheoryDataset(AbstractStructureDataset):
    """
    Hooktheory Dataset
    """

    DATASET_NAME = "hooktheory_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/hooktheory_structure/"

    def __init__(self) -> None:
        super().__init__()

    def _download_and_extract_data(self):
        # get audio
        h5_file = self._dl_manager.download_hdfs(_HDFS_DIRS["audio"])
        self.h5 = h5py.File(h5_file, "r")

        # get labels
        label_dir = self._dl_manager.download_hdfs(_HDFS_DIRS["structure_labels"])
        self.label_dir = self._dl_manager.extract(label_dir)

        return self.label_dir

    def _split_data(self, data_dir):
        train_keys, validation_keys = [], []
        for k in os.listdir(os.path.join(data_dir, "hooktheory_segments")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                if k in HOOKTHEORY_VALIDATION_KEYS:
                    validation_keys.append(k)
                else:
                    train_keys.append(k)

        return {"train": {"keys": train_keys}, "validation": {"keys": validation_keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "hooktheory_segments", key + ".txt"),
            len(np_audio) / self.SAMPLING_RATE,
        )
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "segment_type.txt": 's',
            "intervals.pickle": segs['interval'],
            "labels.pickle": segs['labels'],
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        for key in keys:
            yield self._load_example(key)

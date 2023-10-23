from abc import ABC
import io
import soundfile as sf

import numpy as np
import torch

from samantha.dataio.preprocess import AudioLengthModifier
from recipes.beat.preprocess.common import MCCDatasetMixin, HotGalaxyDatasetMixin
trial_dict = {
    "maj": [
        "maj",
        "7",
        "maj7",
        "add9",
        "add11",
        "maj6",
        "maj9",
        "9",
        "maj11",
        "11",
        "maj13",
        "13",
    ],
    "min": [
        "min",
        "min7",
        "madd9",
        "madd11",
        "min6",
        "min9",
        "min11",
        "min13",
        "minmaj7",
    ],
    "sus4": ["sus4", "maj7sus4", "7sus4", "maj9sus4", "9sus4"],
    "sus2": ["sus2", "maj7sus2", "7sus2"],
    "dim": ["dim", "dim7", "hdim", "hdim7"],
    "aug": ["aug"],
}


def num2seminote(num):
    return {
        "1": 0,
        "2": 2,
        "b3": 3,
        "3": 4,
        "4": 5,
        "b5": 6,
        "5": 7,
        "5#": 8,
        "6": 9,
        "bb7": 9,
        "b7": 10,
        "7": 11,
        "9": 2,
        "11": 5,
        "13": 9,
    }.get(num, 0)


def get_components_by_notation(type_name):
    return {
        "": (),
        # ************************* major and minor ************************
        "maj": ("1", "3", "5"),
        "min": ("1", "b3", "5"),
        # ************************ sevenths *****************************
        "7": ("1", "3", "5", "b7"),
        "maj7": ("1", "3", "5", "7"),
        "min7": ("1", "b3", "5", "b7"),
        # ************************ suspend and add ***************************
        "sus2": ("1", "2", "5"),
        "sus4": ("1", "4", "5"),
        "add9": ("1", "2", "3", "5"),
        "add11": ("1", "3", "4", "5"),
        "madd9": ("1", "2", "b3", "5"),
        "madd11": ("1", "b3", "4", "5"),
        "maj7sus2": ("1", "2", "5", "7"),
        "7sus2": ("1", "2", "5", "b7"),
        "maj7sus4": ("1", "4", "5", "7"),
        "7sus4": ("1", "4", "5", "b7"),
        "maj9sus4": ("1", "4", "5", "7", "9"),
        "9sus4": ("1", "4", "5", "b7", "9"),
        # ************************ sixth ******************************* #
        "maj6": ("1", "3", "5", "7", "9", "11", "13"),  # ('1', '3', '5', '6'),
        "min6": ("1", "b3", "5", "b7", "9", "11", "13"),  # ('1', 'b3', '5', '6'),
        # ************************ Extended ******************************#
        "maj9": ("1", "3", "5", "7", "9"),
        "min9": ("1", "b3", "5", "b7", "9"),
        "9": ("1", "3", "5", "b7", "9"),
        "maj11": ("1", "3", "5", "7", "9", "11"),
        "min11": ("1", "b3", "5", "b7", "9", "11"),
        "11": ("1", "3", "5", "b7", "9", "11"),
        "maj13": ("1", "3", "5", "7", "9", "11", "13"),
        "min13": ("1", "b3", "5", "b7", "9", "11", "13"),
        "13": ("1", "3", "5", "b7", "9", "11", "13"),
        # ******************** augmented and diminished ******************** #
        "aug": ("1", "3", "5#"),
        "dim": ("1", "b3", "b5"),
        "dim7": ("1", "b3", "b5", "bb7"),
        "hdim": ("1", "b3", "b5", "b7"),
        # HACK can't find its structure, but
        # it couldnt be converted to any MIREX category
        "hdim7": ("1", "b3", "b5", "b7"),
        "minmaj7": ("1", "b3", "5", "7"),
        # power chord
        "6": ("1", "6"),
        "5": ("1", "5"),
        "4": ("1", "4"),
        "1": ("1"),
    }.get(type_name)


def note2num(note):
    return {
        "C": "1",
        "C#": "2",
        "Db": "2",
        "D": "3",
        "D#": "4",
        "Eb": "4",
        "E": "5",
        "Fb": "5",
        "E#": "6",
        "F": "6",
        "F#": "7",
        "Gb": "7",
        "G": "8",
        "G#": "9",
        "Ab": "9",
        "A": "10",
        "A#": "11",
        "Bb": "11",
        "B": "12",
        "Cb": "12",
    }.get(note, 0)


def error_correct(chord):
    if ':(b5,b7,3)' in chord:
        return 'X'
    if ':(1,4,b5)' in chord:
        return 'X'
    if ':(b5,11)' in chord:
        return 'X'
    if ':7b5' in chord:
        return 'X'
    if ':(1,3,b5)/b5' in chord:
        return 'X'
    if ':(1,*3,*5)' in chord:
        return 'X'
    if ':(1,2,4,5)' in chord:
        return 'X'
    if ':(1,4)' in chord:
        return 'X'
    if ':(4,b7,9)' in chord:
        return 'X'
    if ':(1,2,4)' in chord:
        return 'X'
    if ':(1,2,#4,6)' in chord:
        return 'X'
    if ':(*5)' in chord:
        return 'X'
    if ':add2' in chord:
        return 'X'
    if ':(1,b3,4)' in chord:
        return 'X'
    if ':maj2' in chord:
        return 'X'
    if ':m7sus4' in chord:
        return 'X'

    chord = chord.replace(':(b3,b7,11,9)', ':min7').replace(':(1,b7)', ':7').replace(':(11,9)', ':maj') \
                 .replace(':(3)', ':maj').replace(':(3,7)', ':maj7').replace(':(b7)', ':7') \
                 .replace(':(13)', ':maj').replace(':(3)', ':maj').replace(':(1,4,b7)', ':7sus4') \
                 .replace(':(#5)', ':aug').replace(':min9(sus4)', ':9sus4').replace(':(1,2,5,b6)', ':sus2') \
                 .replace(':(1,5)', ':maj').replace(':(1,5,9)', ':maj').replace(':min7(sus4)', ':7sus4') \
                 .replace('Ab:Gb', 'Ab:maj').replace(':m7b5', ':hdim').replace('7:(sus4)', ':7sus4') \
                 .replace('m7?5', 'hdim').replace('7:(sus4）', '7sus4').replace(':(1,b3)', ':min') \
                 .replace(':(11)', ':maj').replace(':(1)', ':maj') \
                 .replace(':(1,b3,b5,6)', ':hdim').replace(':(7)', ':7').replace(':(6)', ':maj') \
                 .replace(':(b3,5)', ':min').replace(':(5)', ':maj').replace(':7ssu4', ':7su4') \
                 .replace(':min7b5', ':hdim').replace(':(b3,b7,11,9)', ':min7').replace('', '') \
                 .replace(':min//C#', ':min/C#').replace('E:bmaj', 'Eb:maj').replace(':m7♭5', ':hdim') \
                 .replace(':7♭9', ':7').replace('A::maj/C#', 'A:maj/C#').replace('G#:,maj', 'G#:maj') \
                 .replace(':min7C', ':min7').replace(':7b9', ':7').replace('C::maj/Bb', 'C:maj/Bb') \
                 .replace(':addj9', ':add9').replace('::9(sus4)', ':9sus4').replace('9:(sus4)', ':9sus4') \
                 .replace(':(7sus4)', ':7sus4').replace(':,min', ':min').replace(':dihdim', ':hdim') \
                 .replace(':m7add13', ':min7').replace(':m7(add13)', ':min7').replace(':majG', ':maj') \
                 .replace(':majF', ':maj').replace(':7su4', ':7sus4').replace(':,maj', ':maj') \
                 .replace('maj:/F#', ':maj').replace('dim7b5', 'dim7')
                  

    return chord


def split_chord(chord):
    root, type_name, adds, bass = chord, "", [], ""

    if "/" in chord:
        chord, bass = chord.split("/")
        root = chord
    if "(" in chord:
        adds = chord[chord.find("(") + 1 : chord.find(")")].split(",")
        chord = chord[: chord.find("(")] + chord[chord.find(")") + 1 :]
        root = chord
    
    if ':' in chord:
        root, type_name = chord.split(":")
    else:
        root = chord
        type_name = 'maj'

    if type_name == "7sus":
        type_name = "7sus4"
    if type_name == "sus":
        type_name = "sus4"

    return root.strip(), type_name, adds, bass


def get_chord_labels(chords, intervals, hop_in_sec):
    hop_per_sec = 1 / hop_in_sec
    if intervals[-1][1] < 1:
        end_idx = np.floor(intervals[-1][0] * hop_per_sec).astype(np.int)
    else:
        end_idx = np.floor(intervals[-1][1] * hop_per_sec).astype(np.int)
    oup = {}
    (
        oup["root_labels"],
        oup["triad_labels"],
        oup["bass_labels"],
        oup["note_labels"],
        oup["ignore_labels"],
        oup["seventh_labels"],
    ) = (
        np.zeros((end_idx, 13)),
        np.zeros((end_idx, 7)),
        np.zeros((end_idx, 13)),
        np.zeros((end_idx, 12)),
        np.ones(end_idx),
        np.zeros((end_idx, 4)),
    )

    for interval, chord in zip(intervals, chords):
        start_time, end_time = interval
        start = round(start_time * hop_per_sec)
        end = round(end_time * hop_per_sec)
        chord = chord.strip()
        chord = error_correct(chord)
        if chord == "N":
            root, triad, bass, seventh, _, _, _, ignore = (0, 0, 0, 0, 0, 0, 0, 0)
        elif chord == "X" or chord == "":
            root, triad, bass, seventh, _, _, _, ignore = (0, 0, 0, 0, 0, 0, 0, 1)
        else:
            root, type_name, adds, bass = split_chord(chord)
            if type_name in ["1", "4", "5", "6"]:
                root, triad, bass, seventh, _, _, _, ignore = (0, 0, 0, 0, 0, 0, 0, 1)
            else:
                # get root
                root = int(note2num(root))
                bass = int(note2num(bass)) if bass != "" else root

                # get triad
                if type_name in trial_dict["maj"]:
                    triad = 1
                elif type_name in trial_dict["min"]:
                    triad = 2
                elif type_name in trial_dict["sus4"]:
                    triad = 3
                elif type_name in trial_dict["sus2"]:
                    triad = 4
                elif type_name in trial_dict["dim"]:
                    triad = 5
                elif type_name in trial_dict["aug"]:
                    triad = 6
                else:
                    print(type_name, chord)
                    raise Exception("Type not in the dictionary.")

                # get seventh
                if "7" in get_components_by_notation(type_name):
                    seventh = 1
                elif "b7" in get_components_by_notation(type_name):
                    seventh = 2
                elif "bb7" in get_components_by_notation(type_name):
                    seventh = 3
                else:
                    seventh = 0

                for note in get_components_by_notation(type_name):
                    if "5" in note:
                        oup["note_labels"][
                            start:end, ((root - 1) + num2seminote(note)) % 12
                        ] = 2
                    elif "3" in note:
                        oup["note_labels"][
                            start:end, ((root - 1) + num2seminote(note)) % 12
                        ] = 2
                    else:
                        oup["note_labels"][
                            start:end, ((root - 1) + num2seminote(note)) % 12
                        ] = 1

                ignore = 0

        oup["root_labels"][start:end, root] = 1
        oup["triad_labels"][start:end, triad] = 1
        oup["bass_labels"][start:end, bass] = 1
        oup["seventh_labels"][start:end, seventh] = 1
        oup["ignore_labels"][start:end] = ignore

    return oup


def get_chord_training_label(
    audio_num_samples,
    oup,
    sample_start_time,
    sample_rate,
    label_hop,
    sample_len,
    is_train,
):
    label_len = int(sample_len / label_hop)
    chord_root, chord_triad, chord_bass, chord_note, chord_ignore, chord_seventh = (
        oup["root_labels"],
        oup["triad_labels"],
        oup["bass_labels"],
        oup["note_labels"],
        oup["ignore_labels"],
        oup["seventh_labels"],
    )
    if not is_train:
        root = chord_root[: int(audio_num_samples / sample_rate * (1 / label_hop))]
        triad = chord_triad[: int(audio_num_samples / sample_rate * (1 / label_hop))]
        seventh = chord_seventh[
            : int(audio_num_samples / sample_rate * (1 / label_hop))
        ]
        bass = chord_bass[: int(audio_num_samples / sample_rate * (1 / label_hop))]
        note = chord_note[: int(audio_num_samples / sample_rate * (1 / label_hop))]
        ignore = chord_ignore[: int(audio_num_samples / sample_rate * (1 / label_hop))]
        pad_len = sample_len * int(1 / label_hop) - root.shape[0]
    else:
        sample_idx = int(round(sample_start_time * (1 / label_hop)))
        root = chord_root[sample_idx : sample_idx + label_len, :]
        triad = chord_triad[sample_idx : sample_idx + label_len, :]
        seventh = chord_seventh[sample_idx : sample_idx + label_len, :]
        bass = chord_bass[sample_idx : sample_idx + label_len, :]
        note = chord_note[sample_idx : sample_idx + label_len, :]
        ignore = chord_ignore[sample_idx : sample_idx + label_len]
        pad_len = label_len - root.shape[0]
    pad_len = int(pad_len)

    if pad_len > 0:
        root = np.pad(root, [(0, pad_len), (0, 0)], "constant")
        triad = np.pad(triad, [(0, pad_len), (0, 0)], "constant")
        seventh = np.pad(seventh, [(0, pad_len), (0, 0)], "constant")
        bass = np.pad(bass, [(0, pad_len), (0, 0)], "constant")
        note = np.pad(note, [(0, pad_len), (0, 0)], "constant")
        ignore = np.pad(ignore, [(0, pad_len)], "constant")

    root_diff = np.argmax(root, -1) - np.roll(np.argmax(root, -1), 1, 0)
    root_diff[root_diff != 0] = 1
    triad_diff = np.argmax(triad, -1) - np.roll(np.argmax(triad, -1), 1, 0)
    triad_diff[triad_diff != 0] = 1
    boundary = np.logical_or(root_diff, triad_diff)[:, None]
    boundary[0] = 0
    boundary[-1] = 0

    return root, triad, seventh, bass, note, boundary, ignore


class ChordDatasetMixin(ABC):
    def __init__(
        self, target_duration_sec, sampling_rate, label_chord_hop, chunk_per_sample, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._audio_length_modifier = AudioLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._label_chord_hop = label_chord_hop
        self._chunk_per_sample = chunk_per_sample

    def train_preprocess(self, x):
        np_audio = x["audio.npy"]
        oup = get_chord_labels(x["chords.pickle"], x["intervals.pickle"], self._label_chord_hop)

        data_queue = []
        for i in range(self._chunk_per_sample):
            output_dict = {}
            audio, _, start_idx = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            sample_start_time = start_idx / self._audio_length_modifier.sampling_rate

            (
                output_dict["chord_root"],
                output_dict["chord_triad"],
                _,
                _,
                output_dict["chord_note"],
                output_dict["chord_boundary"],
                output_dict["chord_ignore"],
            ) = get_chord_training_label(
                len(np_audio),
                oup=oup,
                sample_start_time=sample_start_time,
                sample_rate=self._audio_length_modifier.sampling_rate,
                label_hop=self._label_chord_hop,
                sample_len=self._audio_length_modifier.target_duration_sec,
                is_train=True,
            )

            output_dict["audio"] = audio.squeeze()

            data_queue.append(output_dict)
        return data_queue

    def val_preprocess(self, x):
        oup = get_chord_labels(x["chords.pickle"], x["intervals.pickle"], self._label_chord_hop)
        output_dict = {}
        (
            output_dict["chord_root"],
            output_dict["chord_triad"],
            output_dict["chord_seventh"],
            output_dict["chord_bass"],
            output_dict["chord_note"],
            output_dict["chord_boundary"],
            output_dict["chord_ignore"],
        ) = get_chord_training_label(
            len(x["audio.npy"]),
            oup=oup,
            sample_start_time=0,
            sample_rate=self._audio_length_modifier.sampling_rate,
            label_hop=self._label_chord_hop,
            sample_len=self._audio_length_modifier.target_duration_sec,
            is_train=False,
        )
        output_dict["audio"] = torch.tensor(x["audio.npy"].squeeze())
        del x
        return output_dict


class MingusDatasetMixin(ChordDatasetMixin):
    def __init__(
        self, target_duration_sec, sampling_rate, label_chord_hop, chunk_per_sample, *args, **kwargs
    ):
        super().__init__(target_duration_sec, sampling_rate, label_chord_hop, chunk_per_sample, *args, **kwargs)  # forwards all unused arguments
    
    def train_preprocess(self, x):
        np_audio = x["audio.npy"]
        oup = get_chord_labels(x["chords.pickle"][1], x["chords.pickle"][0], self._label_chord_hop)

        data_queue = []
        for i in range(self._chunk_per_sample):
            output_dict = {}
            audio, _, start_idx = self._audio_length_modifier(
                torch.from_numpy(np_audio.reshape(1, -1))
            )
            sample_start_time = start_idx / self._audio_length_modifier.sampling_rate

            (
                output_dict["chord_root"],
                output_dict["chord_triad"],
                _,
                _,
                output_dict["chord_note"],
                output_dict["chord_boundary"],
                output_dict["chord_ignore"],
            ) = get_chord_training_label(
                len(np_audio),
                oup=oup,
                sample_start_time=sample_start_time,
                sample_rate=self._audio_length_modifier.sampling_rate,
                label_hop=self._label_chord_hop,
                sample_len=self._audio_length_modifier.target_duration_sec,
                is_train=True,
            )

            output_dict["audio"] = audio.squeeze()
            data_queue.append(output_dict)
        return data_queue


class ChordPreprocessor:
    def __init__(
        self,
        target_duration_sec,
        sampling_rate,
        hop_length,
        chord_pool,
        resnet_pools,
        chunk_per_sample=1,
        *args,
        **kwargs,
    ):
        self._dataset_preprocessors = {}
        label_chord_hop = (
            hop_length
            * np.prod([p for p in chord_pool])
            * np.prod([p[1] for p in resnet_pools])
        ) / sampling_rate
        self._dataset_preprocessors = ChordDatasetMixin(
            target_duration_sec, sampling_rate, label_chord_hop, chunk_per_sample, *args, **kwargs
        )
        self._mcc_preprocessors = MCCDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._hotgalaxy_preprocessors = HotGalaxyDatasetMixin(
            target_duration_sec, sampling_rate, chunk_per_sample, *args, **kwargs
        )
        self._mingus_preprocessors = MingusDatasetMixin(
            target_duration_sec, sampling_rate, label_chord_hop, chunk_per_sample, *args, **kwargs
        )

    def train_batch_preprocess(self, batch):
        for x in batch:
            if 'mcc' in x['__url__']:
                preprocessor = self._mcc_preprocessors
            elif 'mingus_dataset_mir_shard' in x['__url__']:
                preprocessor = self._mingus_preprocessors
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

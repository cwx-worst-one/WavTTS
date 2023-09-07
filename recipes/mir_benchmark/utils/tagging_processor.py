import logging
from abc import ABC
from random import randrange
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from samantha.dataio.preprocess import AudioLengthModifier

prefix = [
    ["Rhythm Acoustic Guitar (Arpeggio)", "3111", [8], [24], ["Guitar"]],
    ["Rhythm Electric Guitar (Arpeggio)", "3211", [9], [26], ["Guitar"]],
    ["Traditional Stringed Instruments", "8100", [35], [104], ["Other"]],
    ["East Asia Stringed Instruments", "8110", [35], [104], ["Other"]],
    ["Traditional Wind Instruments", "8200", [35], [104], ["Other"]],
    ["Indian Stringed Instruments", "8120", [35], [104], ["Other"]],
    ["Distorted Electric Guitar", "3240", [9], [29], ["Guitar"]],
    ["Rhythm Acoustic Guitar", "3110", [8], [24], ["Guitar"]],
    ["Rhythm Electric Guitar", "3210", [9], [26], ["Guitar"]],
    ["Electronic Snare Drum", "1122", [38], [128], ["Drums"]],
    ["Electronic Percussion", "1220", [36], [112], ["Percussive"]],
    ["Orchestral Percussion", "1250", [36], [112], ["Percussive"]],
    ["Female Backing Vocals", "9110", [39], [129], ["Vocal"]],
    [
        "Drums and Percussion",
        "1000",
        [36, 37, 38],
        [112, 120, 128],
        ["Drums", "Percussive"],
    ],
    ["Electronic Bass Drum", "1121", [38], [128], ["Drums"]],
    ["Acoustic Bass Guitar", "2110", [10], [33], ["Bass"]],
    ["Lead Acoustic Guitar", "3120", [8], [24], ["Guitar"]],
    ["Arr. Acoustic Guitar", "3130", [8], [24], ["Guitar"]],
    ["Lead Electric Guitar", "3220", [9], [29], ["Guitar"]],
    ["Arr. Electric Guitar", "3230", [9], [26], ["Guitar"]],
    ["Appalachian Dulcimer", "3350", [35], [104], ["Other"]],
    ["Other Electric Piano", "4240", [1], [4], ["Piano"]],
    ["Electronic Drum Kit", "1120", [38], [128], ["Drums"]],
    ["Organ and Accordion", "4400", [5, 6], [16], ["Organ"]],
    ["Brass and Harmonica", "5000", [23], [61], ["Brass"]],
    ["Male Backing Vocals", "9120", [39], [129], ["Vocal"]],
    ["Electronic Cymbals", "1125", [38], [128], ["Drums"]],
    ["Slide/Steel Guitar", "3250", [9], [26], ["Guitar"]],
    ["Pedal Steel Guitar", "3252", [9], [26], ["Guitar"]],
    ["Moog + Synthesizer", "4501", [33], [88], ["Synth Pad"]],
    ["Baritone Saxophone", "5251", [24], [67], ["Reed"]],
    ["Pitched Percussion", "7000", [4], [8], ["Chromatic Percussion"]],
    ["Acoustic Drum Kit", "1110", [38], [128], ["Drums"]],
    ["Electronic Hi-Hat", "1123", [38], [128], ["Drums"]],
    ["Ethnic Percussion", "1260", [36], [112], ["Percussive"]],
    ["Soprano Saxophone", "5254", [24], [64], ["Reed"]],
    ["Brass Instruments", "6200", [19, 20, 21, 22, 23], [61], ["Brass"]],
    ["Traditional Flute", "8210", [31], [73], ["Pipe"]],
    ["Female Lead Vocal", "9210", [39], [129], ["Vocal"]],
    ["Mixed Percussion", "1210", [36], [112], ["Percussive"]],
    ["Claves_Woodblock", "1213", [36], [112], ["Percussive"]],
    ["Latin Percussion", "1230", [36], [112], ["Percussive"]],
    ["Other Percussion", "1270", [36], [112], ["Percussive"]],
    ["Lap Steel Guitar", "3251", [9], [26], ["Guitar"]],
    ["Electronic Toms", "1124", [38], [128], ["Drums"]],
    ["Washboard_Güiro", "1236", [36], [112], ["Percussive"]],
    ["Body Percussion", "1240", [36], [112], ["Percussive"]],
    ["Acoustic Guitar", "3100", [8], [24], ["Guitar"]],
    ["Electric Guitar", "3200", [9], [26, 29], ["Guitar"]],
    ["Baritone Guitar", "3270", [9], [24], ["Guitar"]],
    ["Tenor Saxophone", "5252", [24], [66], ["Reed"]],
    ["Reed Aerophones", "8220", [35], [73], ["Reed"]],
    ["Sung Instrument", "9130", [39], [129], ["Vocal"]],
    ["Male Lead Vocal", "9220", [39], [129], ["Vocal"]],
    ["Repique de Mao", "1231.2", [36], [112], ["Percussive"]],
    ["Snap (Fingers)", "1242", [36], [112], ["Percussive"]],
    ["Finger Cymbals", "1264", [36], [112], ["Percussive"]],
    ["Electric Piano", "4200", [1], [4], ["Piano"]],
    ["Muted Trombone", "5220", [20], [57], ["Brass"]],
    ["Alto Saxophone", "5253", [24], [64], ["Reed"]],
    ["String Section", "6100", [11, 12, 13, 14, 15, 16], [48], ["Strings"]],
    ["Backing Vocals", "9100", [39], [129], ["Vocal"]],
    ["Human Beatbox", "1243", [36], [112], ["Percussive"]],
    ["Sound Effects", "1300", [37], [120], ["Sound Effects"]],
    ["Sound effects", "1310", [37], [120], ["Sound Effects"]],
    ["Noise effects", "1330", [37], [120], ["Sound Effects"]],
    ["Electric Bass", "2120", [10], [33], ["Bass"]],
    ["Fretless Bass", "2121", [10], [33], ["Bass"]],
    ["Vocal Bassist", "2140", [10], [33], ["Bass"]],
    ["Digital Piano", "4230", [1], [4], ["Piano"]],
    ["Synth Strings", "4520", [15], [50], ["Ensemble"]],
    ["Orchestra Hit", "4552", [15], [48], ["Synth Pad"]],
    ["Synth Ambiant", "4570", [33, 34], [88, 96], ["Synth Pad", "Synth Effects"]],
    ["Brass section", "5200", [23], [61], ["Brass"]],
    ["Muted Trumpet", "5210", [19], [56], ["Brass"]],
    ["Female voices", "9010", [39], [129], ["Vocal"]],
    ["Jingle Bells", "1217", [36], [112], ["Percussive"]],
    ["Upright Bass", "2111", [10], [33], ["Bass"]],
    ["Other Guitar", "3300", [35], [104], ["Other"]],
    ["Guitar Synth", "4590", [9], [26, 29], ["Guitar"]],
    ["English Horn", "6320", [26], [69], ["Reed"]],
    ["Glockenspiel", "7212", [4], [8], ["Chromatic Percussion"]],
    ["Tubular Bell", "7221", [4], [8], ["Chromatic Percussion"]],
    ["Blown Bottle", "8250", [31], [73], ["Reed"]],
    ["Wind Chimes", "1216", [36], [112], ["Percussive"]],
    ["Tres Cubano", "3347", [35], [104], ["Other"]],
    ["Harpsichord", "4300", [2], [4], ["Piano"]],
    [
        "Synthesizer",
        "4500",
        [32, 33, 34],
        [80, 88],
        ["Synth Lead", "Synth Pad", "Synth Effects"],
    ],
    ["Synth Voice", "4530", [18], [52], ["Ensemble"]],
    ["Musical Saw", "4541", [32], [80], ["Synth Lead"]],
    ["Synth Brass", "4560", [23], [61], ["Reed"]],
    ["Synth Flute", "4580", [29], [73], ["Pipe"]],
    ["Double Bass", "6140", [10], [33], ["Bass"]],
    ["French Horn", "6240", [22], [60], ["Brass"]],
    ["Wooden Bars", "7100", [4], [8], ["Chromatic Percussion"]],
    ["Wooden bars", "7110", [4], [8], ["Chromatic Percussion"]],
    ["Thumb Piano", "7215", [4], [8], ["Chromatic Percussion"]],
    ["Steel Drums", "7230", [4], [8], ["Chromatic Percussion"]],
    ["Irish Flute", "8212", [31], [73], ["Pipe"]],
    ["Male voices", "9020", [39], [129], ["Vocal"]],
    ["Snare Drum", "1112", [38], [128], ["Drums"]],
    ["Percussion", "1200", [36], [112], ["Percussive"]],
    ["Tambourine", "1211", [36], [112], ["Percussive"]],
    ["Jew's Harp", "1262", [36], [112], ["Percussive"]],
    ["Bacurinhas", "1272", [36], [112], ["Percussive"]],
    ["Synth Bass", "2130", [10], [33], ["Bass"]],
    ["Cavaquinho", "3348", [35], [104], ["Other"]],
    ["Synth Lead", "4540", [32], [80], ["Synth Lead"]],
    ["Synth Keys", "4550", [33], [88], ["Synth Pad"]],
    ["Arpegiator", "4551", [33], [88], ["Synth Pad"]],
    ["Flugelhorn", "6250", [22], [60], ["Brass"]],
    ["Metal bars", "7200", [4], [8], ["Chromatic Percussion"]],
    ["Metal Bars", "7210", [4], [8], ["Chromatic Percussion"]],
    ["Shakuhachi", "8217", [31], [73], ["Pipe"]],
    ["Guthbuinne", "8224", [35], [70], ["Reed"]],
    ["Didgeridoo", "8230", [35], [104], ["Other"]],
    ["Lead Vocal", "9200", [39], [129], ["Vocal"]],
    ["Bass Drum", "1111", [38], [128], ["Drums"]],
    ["Soalheira", "1212.2", [36], [112], ["Percussive"]],
    ["Sandpaper", "1212.2", [36], [112], ["Percussive"]],
    ["Vibraslap", "1219", [36], [112], ["Percussive"]],
    ["Castanets", "1231", [36], [112], ["Percussive"]],
    ["Hand Clap", "1241", [36], [112], ["Percussive"]],
    ["Tap dance", "1244", [36], [112], ["Percussive"]],
    ["Rainstick", "1261", [36], [112], ["Percussive"]],
    ["Explosion", "1316", [37], [120], ["Sound Effects"]],
    ["Tamburica", "3344", [35], [104], ["Other"]],
    ["Balalaïka", "3360", [35], [104], ["Other"]],
    ["Toy Piano", "4110", [0], [0], ["Piano"]],
    ["Wurlitzer", "4242", [1], [4], ["Piano"]],
    ["Dulcitone", "4243", [1], [4], ["Piano"]],
    ["Harmonium", "4412", [5], [16], ["Organ"]],
    ["Accordion", "4420", [6], [16], ["Organ"]],
    ["Bandonéon", "4421", [6], [16], ["Organ"]],
    ["Mellotron", "4430", [5], [16], ["Organ"]],
    ["Synth Pad", "4510", [33], [88], ["Synth Pad"]],
    ["Polysynth", "4553", [33], [88], ["Synth Pad"]],
    ["Harmonica", "5100", [7], [16], ["Brass"]],
    ["Saxophone", "5250", [24], [64], ["Reed"]],
    ["Orchestra", "6000", [15, 23], [48, 61], ["Strings", "Brass", "Reed"]],
    [
        "Woodwinds",
        "6300",
        [25, 26, 27, 28, 29, 30, 31],
        [68, 69, 70, 71, 73],
        ["Reed", "Pipe"],
    ],
    ["Xylophone", "7112", [4], [8], ["Chromatic Percussion"]],
    ["Music Box", "7213", [4], [8], ["Chromatic Percussion"]],
    ["Hank Drum", "7231", [4], [8], ["Chromatic Percussion"]],
    ["Pan Flute", "8211", [31], [73], ["Pipe"]],
    ["Drum Kit", "1100", [38], [128], ["Drums"]],
    ["Jamblock", "1213.1", [36], [112], ["Percussive"]],
    ["Triangle", "1214", [36], [112], ["Percussive"]],
    ["Berimbau", "1231.3", [36], [112], ["Percussive"]],
    ["Pandeiro", "1231.6", [36], [112], ["Percussive"]],
    ["Ambience", "1314.1", [37], [120], ["Sound Effects"]],
    ["Mandolin", "3340", [35], [104], ["Other"]],
    ["Bouzouki", "3341", [35], [104], ["Other"]],
    ["Charango", "3349", [35], [104], ["Other"]],
    ["Dulcimer", "3351", [35], [104], ["Other"]],
    ["Keyboard", "4000", [0, 1, 2, 3, 5], [4], ["Piano", "Organ"]],
    ["Clavinet", "4220", [3], [4], ["Piano"]],
    ["talk box", "4413", [5], [16], ["Organ"]],
    ["Theremin", "4542", [32], [80], ["Synth Lead"]],
    ["Melodica", "5110", [7], [16], ["Brass"]],
    ["Trombone", "6220", [20], [57], ["Brass"]],
    ["Clarinet", "6340", [27], [71], ["Reed"]],
    ["Calliope", "6370", [31], [73], ["Pipe"]],
    ["Shamisen", "8112", [35], [104], ["Other"]],
    ["Tamboura", "8123", [35], [104], ["Other"]],
    ["Bagpipes", "8221", [35], [73], ["Reed"]],
    ["Crumhorn", "8226", [35], [71], ["Reed"]],
    ["(ad lib)", "9200", [39], [129], ["Vocal"]],
    ["Cymbals", "1115", [38], [128], ["Drums"]],
    ["Cowbell", "1215", [36], [112], ["Percussive"]],
    ["Tambora", "1231.4", [36], [112], ["Percussive"]],
    ["Maracas", "1232", [36], [112], ["Percussive"]],
    ["Tan tan", "1239", [36], [112], ["Percussive"]],
    ["Timpani", "1251", [17], [47], ["Percussive"]],
    ["Rototom", "1271", [36], [112], ["Percussive"]],
    ["Foghorn", "1315.1", [37], [120], ["Sound Effects"]],
    ["Scratch", "1322", [37], [120], ["Sound Effects"]],
    ["Ukulele", "3320", [35], [104], ["Other"]],
    ["Vihuela", "3345", [35], [104], ["Other"]],
    ["Hammond", "4411", [5], [16], ["Organ"]],
    ["Vocoder", "4531", [18], [52], ["Ensemble"]],
    ["Trumpet", "6210", [19], [56], ["Brass"]],
    ["Bassoon", "6330", [26], [70], ["Reed"]],
    ["Piccolo", "6360", [28], [73], ["Pipe"]],
    ["Marimba", "7111", [4], [8], ["Chromatic Percussion"]],
    ["Celesta", "7211", [4], [8], ["Chromatic Percussion"]],
    ["Ocarina", "8214", [31], [73], ["Pipe"]],
    ["Bansuri", "8216", [31], [73], ["Pipe"]],
    ["Shehnai", "8223", [35], [73], ["Reed"]],
    ["Hi-Hat", "1113", [38], [128], ["Drums"]],
    ["Shaker", "1212", [36], [112], ["Percussive"]],
    ["Congas", "1234", [36], [112], ["Percussive"]],
    ["Bongos", "1235", [36], [112], ["Percussive"]],
    ["Cabasa", "1237", [36], [112], ["Percussive"]],
    ["Djembe", "1238", [36], [112], ["Percussive"]],
    ["Spoons", "1245", [36], [112], ["Percussive"]],
    ["Pleurs", "1312.2", [37], [120], ["Sound Effects"]],
    ["Breath", "1312.4", [37], [120], ["Sound Effects"]],
    ["filter", "1321", [37], [120], ["Sound Effects"]],
    ["TB-303", "2134", [10], [33], ["Bass"]],
    ["Guitar", "3000", [8, 9], [24, 26, 29], ["Guitar"]],
    ["Rhodes", "4210", [1], [4], ["Piano"]],
    ["Sample", "4700", [34], [96], ["Synth Effects"]],
    ["Violin", "6110", [11], [40], ["Strings"]],
    ["Fiddle", "6111", [35], [40], ["Strings"]],
    ["Zither", "6151", [16], [46], ["Strings"]],
    ["Ethnic", "8000", [35], [104], ["Other"]],
    ["Whistl", "8213", [31], [73], ["Pipe"]],
    ["Ganza", "1212.1", [36], [112], ["Percussive"]],
    ["Cajón", "1218", [36], [112], ["Percussive"]],
    ["Cuica", "1231.1", [36], [112], ["Percussive"]],
    ["Surdo", "1231.5", [36], [112], ["Percussive"]],
    ["Agogô", "1233", [36], [112], ["Percussive"]],
    ["Tabla", "1263", [36], [112], ["Percussive"]],
    ["Taiko", "1265", [36], [112], ["Percussive"]],
    ["Birds", "1311.4", [37], [120], ["Sound Effects"]],
    ["Crowd", "1312.3", [37], [120], ["Sound Effects"]],
    ["Phone", "1313.1", [37], [120], ["Sound Effects"]],
    ["clock", "1313.2", [37], [120], ["Sound Effects"]],
    ["vinyl", "1314.2", [37], [120], ["Sound Effects"]],
    ["Siren", "1315.2", [37], [120], ["Sound Effects"]],
    ["Sonar", "1325", [37], [120], ["Sound Effects"]],
    ["Bass0", "2000", [10], [32, 33], ["Bass"]],
    ["Dobro", "3140", [8], [24], ["Guitar"]],
    ["Banjo", "3310", [35], [104], ["Other"]],
    ["Tiple", "3346", [35], [104], ["Other"]],
    ["Piano", "4100", [0], [0], ["Piano"]],
    ["Organ", "4410", [5], [16], ["Organ"]],
    ["Viola", "6120", [12], [41], ["Strings"]],
    ["Cello", "6130", [13], [42], ["Strings"]],
    ["Flute", "6350", [29, 30], [73], ["Pipe"]],
    ["Vibes", "7214", [4], [8], ["Chromatic Percussion"]],
    ["Bells", "7220", [4], [8], ["Chromatic Percussion"]],
    ["Sitar", "8121", [35], [104], ["Other"]],
    ["Sarod", "8122", [35], [104], ["Other"]],
    ["Hurdy", "8130", [35], [104], ["Other"]],
    ["Quena", "8218", [31], [73], ["Pipe"]],
    ["Duduk", "8222", [35], [73], ["Reed"]],
    ["Tulum", "8225", [35], [73], ["Reed"]],
    ["Kazoo", "8240", [35], [104], ["Other"]],
    ["Voice", "9000", [39], [129], ["Vocal"]],
    ["Toms", "1114", [38], [128], ["Drums"]],
    ["Clap", "1221", [36], [112], ["Percussive"]],
    ["Drum", "1253", [36], [112], ["Percussive"]],
    ["Gong", "1256", [36], [112], ["Percussive"]],
    ["Wind", "1311.1", [37], [120], ["Sound Effects"]],
    ["Rain", "1311.3", [37], [120], ["Sound Effects"]],
    ["Kids", "1312.1", [37], [120], ["Sound Effects"]],
    ["yell", "1312.5", [37], [120], ["Sound Effects"]],
    ["Bowl", "1317", [37], [120], ["Sound Effects"]],
    ["Beep", "1323", [37], [120], ["Sound Effects"]],
    ["loop", "1324", [37], [120], ["Sound Effects"]],
    ["Bass", "2100", [10], [33], ["Bass"]],
    ["Moog", "2131", [10], [33], ["Bass"]],
    ["Lute", "3342", [35], [104], ["Other"]],
    ["Harp", "6150", [16], [46], ["Strings"]],
    ["Tuba", "6230", [21], [58], ["Brass"]],
    ["Oboe", "6310", [25], [68], ["Reed"]],
    ["Koto", "8111", [35], [104], ["Other"]],
    ["Erhu", "8113", [35], [104], ["Other"]],
    ["Pipa", "8114", [35], [104], ["Other"]],
    ["Kora", "8115", [35], [104], ["Other"]],
    ["Xiao", "8215", [31], [73], ["Pipe"]],
    ["Sea", "1311.2", [37], [120], ["Sound Effects"]],
    ["ARP", "2132", [10], [33], ["Bass"]],
    ["DX7", "2133", [10], [33], ["Bass"]],
    ["Sub", "2135", [10], [33], ["Bass"]],
    ["Oud", "3343", [35], [104], ["Other"]],
    ["FX", "1318", [37], [120], ["Sound Effects"]],
    ["CP", "4241", [1], [4], ["Piano"]],
]


def convert_inst(name):
    for pre, code, _, slakh_id, class_id in prefix:
        len_pre = len(pre)
        if name[:len_pre] == pre:
            return class_id
            # return pre, code, slakh_id, class_id

    name = name.replace("(", "")
    for pre, code, _, slakh_id, class_id in prefix:
        len_pre = len(pre)
        if name[:len_pre] == pre:
            return class_id
            # return pre, code, slakh_id, class_id

    # return None, '0000', [-1], [None]
    return []


class CustomLengthModifier(AudioLengthModifier):
    def __init__(
        self,
        sampling_rate: int,
        target_duration_sec: float,
        is_random_crop: bool = True,
    ):
        super().__init__(sampling_rate, target_duration_sec, is_random_crop)

        if target_duration_sec < 0:
            self._raise_config_error("target_duration_sec must be positive.")

    def __call__(self, audio: torch.Tensor):
        self._check_audio(audio)
        num_channels, source_frames = audio.shape
        target_frames = int(self.target_duration_sec * self.sampling_rate)

        start_idx = 0
        if source_frames < target_frames:
            output_audio = F.pad(
                input=audio,
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=0,
            )
            padding_mask = F.pad(
                input=torch.zeros_like(audio),
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=1,
            )
        else:
            if self.is_random_crop and (source_frames > target_frames):
                start_idx = randrange(source_frames - target_frames)
            output_audio = audio.clone()[:, start_idx : start_idx + target_frames]
            padding_mask = torch.zeros((num_channels, target_frames))
        return output_audio, padding_mask, start_idx


# Instrument
class InstrumentDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._train_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._val_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=False
        )
        self.instruments = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ensemble",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Synth Effects",
            "Synth Lead",
            "Synth Pad",
            "Vocal",
        ]

    def train_preprocess(self, x):
        out = x.copy()

        # mix audio
        vocals = x["vocal.npy"]
        accompaniment = x["acc.npy"]
        min_length = min(len(vocals), len(accompaniment))
        mixed_audio = vocals[:min_length] + accompaniment[:min_length]
        if abs(vocals).sum() > 100:
            is_vocal = True
        else:
            is_vocal = False

        # crop audio
        audio, _, _ = self._train_length_modifier(
            torch.from_numpy(mixed_audio.reshape(1, -1))
        )
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        instrument_binary = np.zeros(len(self.instruments)).astype("float32")
        for inst in x["metadata.json"]["instruments"]:
            inst_cls = convert_inst(inst[1:-1])
            for cls in inst_cls:
                if cls in self.instruments:
                    instrument_binary[self.instruments.index(cls)] = 1
        if is_vocal:
            instrument_binary[self.instruments.index("Vocal")] = 1
        out["binaries.npy"] = instrument_binary

        return out

    def val_preprocess(self, x):
        out = x.copy()

        # mix audio
        vocals = x["vocal.npy"]
        accompaniment = x["acc.npy"]
        min_length = min(len(vocals), len(accompaniment))
        mixed_audio = vocals[:min_length] + accompaniment[:min_length]
        if abs(vocals).sum() > 100:
            is_vocal = True
        else:
            is_vocal = False

        # crop audio
        audio, _, _ = self._val_length_modifier(
            torch.from_numpy(mixed_audio.reshape(1, -1))
        )
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        instrument_binary = np.zeros(len(self.instruments)).astype("float32")
        for inst in x["metadata.json"]["instruments"]:
            inst_cls = convert_inst(inst[1:-1])
            for cls in inst_cls:
                if cls in self.instruments:
                    instrument_binary[self.instruments.index(cls)] = 1
        if is_vocal:
            instrument_binary[self.instruments.index("Vocal")] = 1
        out["binaries.npy"] = instrument_binary

        return out


class InstrumentTransformFactory:
    """The factory class for creating beat transform objects based on dataset."""

    registry = {}
    """ Internal registry for available beat transform objects based on dataset. """

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class method to register MSS model classes to the internal registry.
        Args:
            name (str): The name of the MSS model.
        Returns:
            The MSS model class itself.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logging.warning("Executor %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Factory command to create the MSS model.
        This method gets the appropriate MSS model class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the MSS model to create.
        Returns:
            An instance of the MSS model that is created.
        """

        if name not in cls.registry:
            logging.warning("Executor %s does not exist in the registry", name)
            return None

        exec_class = cls.registry[name]
        executor = exec_class(*args, **kwargs)
        return executor


class InstrumentPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = InstrumentDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)


# Genre
class GenreDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._train_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._val_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=False
        )

    def train_preprocess(self, x):
        out = x.copy()

        # mix audio
        audio = x["audio.npy"]

        # crop audio
        audio, _, _ = self._train_length_modifier(
            torch.from_numpy(audio.reshape(1, -1))
        )
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        out["binaries.npy"] = x["tag_binary.npy"]

        return out

    def val_preprocess(self, x):
        out = x.copy()

        # mix audio
        audio = x["audio.npy"]

        # crop audio
        audio, _, _ = self._val_length_modifier(torch.from_numpy(audio.reshape(1, -1)))
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        out["binaries.npy"] = x["tag_binary.npy"]

        return out


class GenrePreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = GenreDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)


# Vocal
class VocalDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._train_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._val_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=False
        )
        self.vocal_tags = [
            "age_中老年",
            "age_中青年",
            "age_幼年",
            "age_青年",
            "gender_NO",
            "gender_女",
            "gender_男",
            "style_低沉和蔼",
            "style_厚实低沉",
            "style_嘹亮自信",
            "style_成熟明亮",
            "style_成熟磁性",
            "style_明亮细腻",
            "style_淘气萌娃",
            "style_甜美温柔",
            "style_磁性慵懒",
        ]

    def train_preprocess(self, x):
        out = x.copy()

        # mix audio
        audio = x["audio.npy"]

        # crop audio
        audio, _, _ = self._train_length_modifier(
            torch.from_numpy(audio.reshape(1, -1))
        )
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        vocal_binary = np.zeros(len(self.vocal_tags)).astype("float32")
        vocal_binary[self.vocal_tags.index("age_" + x["age"].decode("utf-8"))] = 1
        vocal_binary[self.vocal_tags.index("gender_" + x["gender"].decode("utf-8"))] = 1
        vocal_binary[self.vocal_tags.index("style_" + x["stype"].decode("utf-8"))] = 1
        out["binaries.npy"] = vocal_binary

        return out

    def val_preprocess(self, x):
        out = x.copy()

        # mix audio
        audio = x["audio.npy"]

        # crop audio
        audio, _, _ = self._val_length_modifier(torch.from_numpy(audio.reshape(1, -1)))
        out["audio.npy"] = audio.numpy().astype("float32")

        # get label
        vocal_binary = np.zeros(len(self.vocal_tags)).astype("float32")
        vocal_binary[self.vocal_tags.index("age_" + x["age"].decode("utf-8"))] = 1
        vocal_binary[self.vocal_tags.index("gender_" + x["gender"].decode("utf-8"))] = 1
        try:
            vocal_binary[
                self.vocal_tags.index("style_" + x["stype"].decode("utf-8"))
            ] = 1
        except ValueError as e:
            _ = 0
        out["binaries.npy"] = vocal_binary

        return out


class VocalPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = VocalDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)


# Multi
class MultiDatasetMixin(ABC):
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self._train_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=True
        )
        self._val_length_modifier = CustomLengthModifier(
            sampling_rate, target_duration_sec, is_random_crop=False
        )
        self.instrument_tags = [
            "Bass",
            "Brass",
            "Chromatic Percussion",
            "Drums",
            "Ensemble",
            "Guitar",
            "Organ",
            "Percussive",
            "Piano",
            "Pipe",
            "Reed",
            "Sound Effects",
            "Strings",
            "Synth Effects",
            "Synth Lead",
            "Synth Pad",
            "Vocal",
        ]
        self.vocal_tags = [
            "age_中老年",
            "age_中青年",
            "age_幼年",
            "age_青年",
            "gender_NO",
            "gender_女",
            "gender_男",
            "style_低沉和蔼",
            "style_厚实低沉",
            "style_嘹亮自信",
            "style_成熟明亮",
            "style_成熟磁性",
            "style_明亮细腻",
            "style_淘气萌娃",
            "style_甜美温柔",
            "style_磁性慵懒",
        ]

    def train_preprocess(self, x):
        out = x.copy()

        if not "dataset.txt" in x.keys():  # instrument
            # mix audio
            vocals = x["vocal.npy"]
            accompaniment = x["acc.npy"]
            min_length = min(len(vocals), len(accompaniment))
            mixed_audio = vocals[:min_length] + accompaniment[:min_length]
            if abs(vocals).sum() > 100:
                is_vocal = True
            else:
                is_vocal = False

            # crop audio
            audio, _, _ = self._train_length_modifier(
                torch.from_numpy(mixed_audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # get label
            instrument_binary = np.zeros(len(self.instruments)).astype("float32")
            for inst in x["metadata.json"]["instruments"]:
                inst_cls = convert_inst(inst[1:-1])
                for cls in inst_cls:
                    if cls in self.instruments:
                        instrument_binary[self.instruments.index(cls)] = 1
            if is_vocal:
                instrument_binary[self.instruments.index("Vocal")] = 1

            # return binaries
            out["instrument_binaries.npy"] = instrument_binary
            out["vocal_binaries.npy"] = np.zeros(len(self.vocal_tags)).astype("float32")
            out["genre_binaries.npy"] = np.zeros(34).astype("float32")
            out["task"] = "instrument"

        elif x["dataset.txt"][:5] == "vocal":  # vocal
            # mix audio
            audio = x["audio.npy"]

            # crop audio
            audio, _, _ = self._train_length_modifier(
                torch.from_numpy(audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # vocal binaries
            vocal_binary = np.zeros(len(self.vocal_tags)).astype("float32")
            vocal_binary[self.vocal_tags.index("age_" + x["age"].decode("utf-8"))] = 1
            vocal_binary[
                self.vocal_tags.index("gender_" + x["gender"].decode("utf-8"))
            ] = 1
            vocal_binary[
                self.vocal_tags.index("style_" + x["stype"].decode("utf-8"))
            ] = 1

            # return binaries
            out["instrument_binaries.npy"] = np.zeros(len(self.instrument_tags)).astype(
                "float32"
            )
            out["vocal_binaries.npy"] = vocal_binary
            out["genre_binaries.npy"] = np.zeros(34).astype("float32")
            out["task"] = "vocal"

        elif x["dataset.txt"][:5] == "genre":  # genre
            # mix audio
            audio = x["audio.npy"]

            # crop audio
            audio, _, _ = self._train_length_modifier(
                torch.from_numpy(audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # get label
            out["instrument_binaries.npy"] = np.zeros(len(self.instrument_tags)).astype(
                "float32"
            )
            out["vocal_binaries.npy"] = np.zeros(len(self.vocal_tags)).astype("float32")
            out["genre_binaries.npy"] = x["tag_binary.npy"]
            out["task"] = "genre"

        return out

    def val_preprocess(self, x):
        out = x.copy()
        if not "dataset.txt" in x.keys():  # instrument
            # mix audio
            vocals = x["vocal.npy"]
            accompaniment = x["acc.npy"]
            min_length = min(len(vocals), len(accompaniment))
            mixed_audio = vocals[:min_length] + accompaniment[:min_length]
            if abs(vocals).sum() > 100:
                is_vocal = True
            else:
                is_vocal = False

            # crop audio
            audio, _, _ = self._val_length_modifier(
                torch.from_numpy(mixed_audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # get label
            instrument_binary = np.zeros(len(self.instrument_tags)).astype("float32")
            for inst in x["metadata.json"]["instruments"]:
                inst_cls = convert_inst(inst[1:-1])
                for cls in inst_cls:
                    if cls in self.instrument_tags:
                        instrument_binary[self.instrument_tags.index(cls)] = 1
            if is_vocal:
                instrument_binary[self.instrument_tags.index("Vocal")] = 1

            # return binaries
            out["instrument_binaries.npy"] = instrument_binary
            out["vocal_binaries.npy"] = np.zeros(len(self.vocal_tags)).astype("float32")
            out["genre_binaries.npy"] = np.zeros(34).astype("float32")
            out["task"] = "instrument"

        elif x["dataset.txt"][:5] == "vocal":  # vocal
            # mix audio
            audio = x["audio.npy"]

            # crop audio
            audio, _, _ = self._val_length_modifier(
                torch.from_numpy(audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # get label
            vocal_binary = np.zeros(len(self.vocal_tags)).astype("float32")
            vocal_binary[self.vocal_tags.index("age_" + x["age"].decode("utf-8"))] = 1
            vocal_binary[
                self.vocal_tags.index("gender_" + x["gender"].decode("utf-8"))
            ] = 1
            try:
                vocal_binary[
                    self.vocal_tags.index("style_" + x["stype"].decode("utf-8"))
                ] = 1
            except ValueError as e:
                _ = 0

            # return binaries
            out["instrument_binaries.npy"] = np.zeros(len(self.instrument_tags)).astype(
                "float32"
            )
            out["vocal_binaries.npy"] = vocal_binary
            out["genre_binaries.npy"] = np.zeros(34).astype("float32")
            out["task"] = "vocal"

        elif x["dataset.txt"][:5] == "genre":  # genre
            # mix audio
            audio = x["audio.npy"]

            # crop audio
            audio, _, _ = self._val_length_modifier(
                torch.from_numpy(audio.reshape(1, -1))
            )
            out["audio.npy"] = audio.numpy().astype("float32")

            # get label
            out["instrument_binaries.npy"] = np.zeros(len(self.instrument_tags)).astype(
                "float32"
            )
            out["vocal_binaries.npy"] = np.zeros(len(self.vocal_tags)).astype("float32")
            out["genre_binaries.npy"] = x["tag_binary.npy"]
            out["task"] = "genre"

        return out


class MultiPreprocessor:
    def __init__(self, target_duration_sec, sampling_rate, *args, **kwargs):
        self._dataset_preprocessors = MultiDatasetMixin(
            target_duration_sec, sampling_rate, *args, **kwargs
        )
        self._sampling_rate = sampling_rate
        self._target_duration_sec = target_duration_sec

    def train_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.train_preprocess(x)

    def val_batch_preprocess(self, batch):
        for x in batch:
            preprocessor = self._dataset_preprocessors
            yield preprocessor.val_preprocess(x)

    def val_preprocess(self, x):
        preprocessor = self._dataset_preprocessors
        return preprocessor.val_preprocess(x)

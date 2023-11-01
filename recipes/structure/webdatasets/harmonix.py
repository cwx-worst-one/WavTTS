"""
Harmonix Structure Recognition Dataset.
"""

import os
import random

import h5py

from recipes.structure.webdatasets.structure_dataset import AbstractStructureDataset

_HDFS_DIRS = {
    "audio": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/h5/harmonix_audio.h5",
    "structure_labels": "hdfs://harunava/home/byte_speech_sv/data/"
    "SheetDoctor/labels/harmonix_segments.zip",
}

HARMONIX_VALIDATION_KEY = [
    "0460_numb",
    "0625_complicated",
    "0814_mistake",
    "0608_canttellmenothing",
    "0238_sambadejaneiro",
    "0995_youfoundme",
    "0268_supersonic",
    "0793_lovegame",
    "0321_wonderwall",
    "0742_iknowyouwantme",
    "0787_liveyourlife",
    "0831_noair",
    "0746_imnotyourhero",
    "0453_mygirl",
    "0434_lights",
    "0047_chinacatsunflower",
    "0132_iceicebaby",
    "0367_dontdreamitsover",
    "0918_starstrukk",
    "0447_mindyourmanners",
    "0400_heatofthemoment",
    "0077_dontsweat",
    "0386_footloose",
    "0282_thegreatsatan",
    "0682_forever",
    "0511_theonlyexception",
    "0043_callmemaybe",
    "0800_lovesexmagicjason",
    "0583_beautifulgirls",
    "0938_thebusiness",
    "0545_youkeepmehangingon",
    "0011_areyouexperienced",
    "0456_nohands",
    "0143_informer",
    "0301_unclejohnsband",
    "0290_thisishowwedoit",
    "0905_shocktoyoursystem",
    "0865_pocketfulofsunshine",
    "0596_bonbon",
    "0183_mynameisjonas",
    "0450_more",
    "0429_kissinu",
    "0725_hotncold",
    "0125_hellogoodmorning",
    "0310_whineup",
    "0106_getitshawty",
    "0986_winner",
    "0157_leanwitit",
    "0581_beautifulcarlyrae",
    "0994_youbelongwithme",
    "0494_somebodythatiusedtoknow",
    "0136_igotyoudancing",
    "0241_satellite",
    "0394_godgavemetoyou",
    "0570_aslongasyouloveme",
    "0737_ihatethispart",
    "0466_onthedarkside",
    "0084_dropitlikeitshot",
    "0501_stayawhile",
    "0779_letitrockfilthy",
    "0951_thunder",
    "0362_crazygirl",
    "0783_lighton",
    "0245_sayhey",
    "0965_wakingupinvegascalvin",
    "0423_iwannago",
    "0943_thetheayer",
    "0374_drunkonyou",
    "0780_letmebereal",
    "0642_dontconfess",
    "0808_makethemoney",
    "0765_jimmyiovine",
    "0219_pondereplay",
    "0119_gunpowderandlead",
    "0272_teachmehowtojerk",
    "0922_suffocate",
    "0343_banjo",
    "0329_youreajerk",
    "0504_suspiciousminds",
    "0533_whosays",
    "0178_mountainman",
    "0160_likeag6",
    "0898_sexyback",
    "0023_bewareoftheboys",
    "0472_partofme",
    "0827_needyounow",
    "0474_pleasedontgo",
    "0286_theragelive",
    "0372_drinkinmyhand",
    "0323_yeah",
    "0883_rocksatmywindow",
    "0977_wheelsnow",
    "0107_getlow",
    "0014_babaoriley",
    "0248_screamingfor",
    "0908_sober",
    "0036_breakingthegirl",
    "0189_neversaynever",
    "0041_calabria",
    "0530_whatthehell",
    "0361_countrymustbecountrywide",
    "0998_youregonnamissthis",
    "0209_paparazzi",
    "0803_lovestory",
    "0939_thecure",
    "0992_wrongthinwhiteduke",
    "0652_drovemewild",
    "0695_goodbyegoodbye",
    "0407_humpinaround",
    "0538_you",
    "0665_fall",
    "0842_obsessionstatic",
    "0482_redsolocup",
    "0633_dancingmachine",
    "0318_windup",
    "0112_giveitup",
    "0348_blackout",
    "0673_feelsliketonight",
    "0197_nothinonyou",
    "0443_magic",
    "0565_americanhoney",
    "0882_rightroundbenny",
    "0298_turnmeon",
    "0056_control",
    "0848_onething",
    "0850_onlyyoucanlovemethisway",
    "0659_everybody",
    "0490_showme",
    "0294_tonighttonight",
    "0541_youdroppedabombonme",
    "0590_betterintime",
    "0954_timeaftertimevan",
    "0198_nowthatwefoundlove",
    "0090_fearofthedarklive",
    "0352_boogiewonderland",
    "0092_fergalicious",
    "0700_gottabesomebody",
    "0149_johnnyguitar",
    "0421_ittakestwo",
    "0946_thinkingofyou",
    "0715_hellodadalife",
    "0645_dontstopthemusic",
    "0819_myblood",
    "0489_shakeyourbody",
    "0526_waitingoutsidethelines",
    "0867_pokerfacedos",
    "0959_turnthisclubaround",
    "0172_marrythenight",
    "0856_paperbackhead",
    "0660_everymorning",
    "0005_again",
]


class HarmonixDataset(AbstractStructureDataset):
    """
    Harmonix Dataset
    """

    DATASET_NAME = "harmonix_structure"
    DATASET_ROOT_DIR = "/mnt/bn/mir-tasks/structure/harmonix_structure/"

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
        for k in os.listdir(os.path.join(data_dir, "harmonix_segments")):
            k = k.split(".txt")[0]
            if k != "" and k in self.h5.keys():
                if k in HARMONIX_VALIDATION_KEY:
                    validation_keys.append(k)
                else:
                    train_keys.append(k)
        return {"train": {"keys": train_keys}, "validation": {"keys": validation_keys}}

    def _load_example(self, key):
        # read annotations
        np_audio = self.h5[key][:]
        segs = self._import_segment_annotation(
            os.path.join(self.label_dir, "harmonix_segments", key + ".txt"),
            len(np_audio) / self.SAMPLING_RATE,
        )
        return {
            "dataset.txt": self.DATASET_NAME,
            "__key__": key,
            "segment_type.txt": "r",
            "intervals.pickle": segs["interval"],
            "labels.pickle": segs["labels"],
            "audio.npy": np_audio.astype(self.AUDIO_NUMPY_DTYPE),
        }

    def _generate_examples(self, keys):
        """Yields examples as (key, example) tuples."""
        random.shuffle(keys)
        for key in keys:
            yield self._load_example(key)

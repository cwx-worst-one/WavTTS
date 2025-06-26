INST_TREE = {
    "Woodwinds": {
        "Piccolo": [],
        "Flute": [],
        "Clarinet": [],
        "Oboe": [],
        "English_Horn": [],
        "Pipe": [],
        "Saxophone": [
            "Soprano/Alto_Sax", 
            "Tenor_Sax", 
            "Baritone_Sax"
            ],
        "Bassoon": [],
    },
    "Brass": {
        "Trumpet": [],
        "French_Horn": [],
        "Trombone": [],
        "Tuba": [],
        "Brass_Section": [],
        "Synth_Brass": [],
    },
    "Percussion": {
        "Drums": [   # Change Drum_Set to Drums since it is invalid in v5 vocab
            "Bass_Drum", 
            "Snare", 
            "Tom", 
            "Hi_hat", 
            "Crash_Cymbal", 
            "Ride_Cymbal"
        ],
        "Chromatic_Percussion": [
            "Marimba",
            "Bells",
            "Belltree",
            "Xylophone",
            "Glockenspiel",
            "Vibraphone",
            "Hangdrum",
            "Church_Bells",
            "Tubular_Bells",
            "Singing_Bowl",
        ],
        "Percussive": [
            "Sandhammer",
            "Tambourine",
            "Castanets",
            "Triangle",
            "Claves",
            "Cowbell",
            "Mark_Tree",
            "Congas/Bongos",
            "Cajón/Box_drum",
            "Wind_Chimes",
        ],
        "Orchestral_Drums": [
            "Timpani",
            "Orchestral_Bassdrum",
            "Orchestral_Snare",
            "Orchestral_Cymbals",
            "Taiko_Drums",
            "Tam_Tam", 
        ],
        "Synth_Drums": [
            "Synth_Kick",
            "Synth_Snare",
            "Synth_Hi_hat",
            "Synth_Tom",
            "Synth_Cymbals",
            "Synth_Clave",
            "Fx",
        ],
        "Body_Percussion": [
            "Clap",
            "Finger_snaps",
            "Beat_box",
        ],
        "Chinese_Percussion": [
            "Chinese_Drums",
            "Bianzhong",
            "Gongs",
            "Wood_Block",
            "Tang_Drums",
            "Ban_Drums",
            "Chinese_Clappers",
            "Chinese_Cymbals",
            "Yunluo",
            "Bangzi",
        ],
    },
    "Keys": {
        "Acoustic_Piano": [],
        "Electric_piano": [],
        "Organ": [],
        "Accordion": [],
        "Harpsichord": [],
        "Pipe_Organ": [],
    },
    "Strings": {
        "Violin": [],
        "Viola": [],
        "Cello": [],
        "Double_Bass": [],
        "String_Ensemble": [],
        "Synth_Strings": [],
    },
    "Bass": {
        "Electric_Bass": [],
        "Synth_Bass": [],
        "Double_Bass_Pizzicato": [],
    },
    "Guitar": {
        "Acoustic_Guitar": [],
        "Electric_Guitar": [
            "Clean_Electric_Guitar",
            "Distorted_Electric_Guitar",
        ],
    },
    "Chinese_Traditional_Instruments": {
        "Di": [],
        "Xiao": [],
        "Suona": [],
        "Erhu": [],
        "Guzheng": [],
        "Pipa": [],
        "Yangqin": [],
        "Sheng": [],
        "Hulusi": [],
        "Panflute": [],
        "Xun": [],
        "Matouqin": [],
        "Ruan": [],
        "Sanxian": [],
        "Guqin": [],
    },
    "Synthesizers": {
        "Synth_Pluck": [],
        "Synth_Lead": [],
        "Synth_Pad": [],
        "Synth_Chord": [],
        "Synth_Bell": [],
        "Synth_Effects": [],
    },
    "Plucked_Strings": {
        "Ukelele": [],
        "Orchestral_Harp": [],
        "Banjo": [],
        "Mandolin": [],
    },
    "Ethnic": {
        "Bagpipe": [],
        "Dulcimer": [],
        "Harmonica": [],
        "Irishwhistle": [],
        "Ocarina": [],
        "Sitar": [],
    },
    "Whistle": {
        "Whistle": []
    },
    "Musicbox": {
        "Musicbox": []
    },
    "Sound_Effects": {
        "Sound_Effects": []
    },
    "Vocal": {
        "Backing_Vocals": [],
        "Chorus": ["Choir_and_Voice"],
        "Vocal_Chops": [],
    },
    "Others": {
        "Others": []
    },
}


INST_ALL_38_MAPPER = {
    "Woodwinds": "Pipe", 
    "Piccolo": "Pipe", 
    "Flute": "Pipe", 
    "Clarinet": "Clarinet", 
    "Oboe": "Oboe", 
    "English_Horn": "English Horn", 
    "Pipe": "Pipe", 
    "Saxophone": "Tenor Sax", 
    "Soprano/Alto_Sax": "Soprano/Alto Sax", 
    "Tenor_Sax": "Tenor Sax", 
    "Baritone_Sax": "Baritone Sax", 
    "Bassoon": "Bassoon", 
    "Brass": "Brass Section", 
    "Trumpet": "Trumpet", 
    "French_Horn": "French Horn", 
    "Trombone": "Trombone", 
    "Tuba": "Tuba", 
    "Brass_Section": "Brass Section", 
    "Synth_Brass": "Brass Section", 
    "Percussion": "Percussive", 
    "Drum_Set": "Drum_Set", 
    "Bass_Drum": "Drum_Set", 
    "Snare": "Drum_Set", 
    "Tom": "Drum_Set", 
    "Hi_hat": "Drum_Set", 
    "Crash_Cymbal": "Drum_Set", 
    "Ride_Cymbal": "Drum_Set", 
    "Chromatic_Percussion": "Chromatic Percussion", 
    "Marimba": "Chromatic Percussion", 
    "Bells": "Chromatic Percussion", 
    "Belltree": "Chromatic Percussion",
    "Xylophone": "Chromatic Percussion", 
    "Glockenspiel": "Chromatic Percussion", 
    "Vibraphone": "Chromatic Percussion", 
    "Hangdrum": "Chromatic Percussion", 
    "Church_Bells": "Chromatic Percussion", 
    "Tubular_Bells": "Chromatic Percussion", 
    "Singing_Bowl": "Chromatic Percussion", 
    "Percussive": "Percussive", 
    "Sandhammer": "Percussive", 
    "Tambourine": "Percussive", 
    "Castanets": "Percussive", 
    "Triangle": "Percussive", 
    "Claves": "Percussive", 
    "Cowbell": "Percussive", 
    "Mark_Tree": "Percussive", 
    "Congas/Bongos": "Percussive", 
    "Cajón/Box_drum": "Percussive", 
    "Wind_Chimes": "Percussive", 
    "Orchestral_Drums": "Drum_Set", 
    "Timpani": "Timpani", 
    "Orchestral_Bassdrum": "Timpani", 
    "Orchestral_Snare": "Drum_Set", 
    "Orchestral_Cymbals": "Drum_Set", 
    "Taiko_Drums": "Drum_Set", 
    "Tam_Tam": "Percussive", 
    "Synth_Drums": "Drum_Set", 
    "Synth_Kick": "Drum_Set", 
    "Synth_Snare": "Drum_Set", 
    "Synth_Hi_hat": "Drum_Set", 
    "Synth_Tom": "Drum_Set", 
    "Synth_Cymbals": "Drum_Set", 
    "Synth_Clave": "Percussive",
    "Fx": "Sound Effects", 
    "Body_Percussion": "Percussive", 
    "Clap": "Percussive", 
    "Finger_snaps": "Percussive", 
    "Beat_box": "Percussive", 
    "Chinese_Percussion": "Percussive", 
    "Chinese_Drums": "Percussive", 
    "Bianzhong": "Chromatic Percussion", 
    "Gongs": "Percussive", 
    "Wood_Block": "Percussive", 
    "Tang_Drums": "Percussive", 
    "Ban_Drums": "Percussive", 
    "Chinese_Clappers": "Percussive", 
    "Chinese_Cymbals": "Percussive", 
    "Yunluo": "Percussive", 
    "Bangzi": "Percussive", 
    "Keys": "Electric Piano", 
    "Acoustic_Piano": "Acoustic Piano", 
    "Electric_piano": "Electric Piano", 
    "Organ": "Organ", "Accordion": "Ethnic", 
    "Harpsichord": "Electric Piano", 
    "Pipe_Organ": "Pipe", 
    "Strings": "String Ensemble", 
    "Violin": "Violin", 
    "Viola": "Viola", 
    "Cello": "Cello", 
    "Double_Bass": "Contrabass", 
    "String_Ensemble": "String Ensemble", 
    "Synth_Strings": "Synth Strings", 
    "Bass": "Bass", 
    "Electric_Bass": "Bass", 
    "Synth_Bass": "Bass", 
    "Double_Bass_Pizzicato": "Contrabass", 
    "Guitar": "Acoustic Guitar", 
    "Acoustic_Guitar": "Acoustic Guitar", 
    "Electric_Guitar": "Clean Electric Guitar", 
    "Clean_Electric_Guitar": "Clean Electric Guitar", 
    "Distorted_Electric_Guitar": "Distorted Electric Guitar", 
    "Chinese Traditional Instruments": "Ethnic", 
    "Di": "Pipe", 
    "Xiao": "Pipe", 
    "Suona": "Pipe", 
    "Erhu": "Ethnic", 
    "Guzheng": "Ethnic", 
    "Pipa": "Ethnic", 
    "Yangqin": "Ethnic", 
    "Sheng": "Ethnic", 
    "Hulusi": "Pipe", 
    "Panflute": "Pipe", 
    "Xun": "Ethnic", 
    "Matouqin": "Ethnic", 
    "Ruan": "Ethnic", 
    "Sanxian": "Ethnic", 
    "Guqin": "Ethnic", 
    "Synthesizers": "Synth Lead", 
    "Synth_Pluck": "Synth Lead", 
    "Synth_Lead": "Synth Lead", 
    "Synth_Pad": "Synth Pad", 
    "Synth_Chord": "Synth Pad", 
    "Synth_Bell": "Chromatic Percussion", 
    "Synth_Effects": "Synth Effects", 
    "Plucked Strings": "Synth Strings", 
    "Ukelele": "Acoustic Guitar", 
    "Orchestral_Harp": "Orchestral Harp",
    "Banjo": "Acoustic Guitar", 
    "Mandolin": "Acoustic Guitar",
    "Ethnic": "Ethnic", 
    "Bagpipe": "Ethnic", 
    "Dulcimer": "Acoustic Guitar", 
    "Harmonica": "Ethnic", 
    "Irishwhistle": "Ethnic", 
    "Ocarina": "Ethnic", 
    "Sitar": "Ethnic", 
    "Whistle": "Ethnic", 
    "Musicbox": "Sound Effects", 
    "Sound_Effects": "Sound Effects", 
    "Vocal": "Vocal", 
    "Backing_Vocals": "Choir and Voice", 
    "Chorus": "Choir and Voice", 
    "Choir_and_Voice": "Choir and Voice", 
    "Vocal_Chops": "Vocal", 
    "Others": "", 
    "Cymbals": "Drum_Set", 
    "Chimes": "Chromatic Percussion", 
    "Bassdrum": "Drum_Set", 
    "soundeffects": "Sound Effects", 
    "frenchhorn": "French Horn", 
    "chinesedrums": "Percussive", 
    "doublebass": "Contrabass", 
    "pad": "Synth Pad", 
    "synthstrings": "Synth Pad", 
    "electricguitar": "Clean Electric Guitar", 
    "vocals": "Vocal", 
    "synthbass": "Bass", 
    "harp": "Orchestral Harp", 
    "synthdrums": "Drum_Set", 
    "percussions": "Percussive", 
    "electricpiano": "Electric Piano", 
    "drumset": "Drum_Set", 
    "lead": "Synth Lead",
    "syntheffects":	"Synth Effects",
    "pluck":	"Synth Lead",
    "others":	"Sound Effects",
    "smallpercussion": "Percussive",
    "drums": "Drum_Set",
}

INST_THRESHOLD_0p9 = {
    "Acoustic_Guitar": 0.62,
    "Acoustic_Piano": 0.5,
    "Bass": 0.262,
    "Brass_Section": 0.961,
    "Cello": 0.921,
    "Clean_Electric_Guitar": 0.198,
    "Electric_Piano": 0.639,
    "Ethnic": 0.988,            # too high...
    "French_Horn": 0.429,
    "Percussive": 0.963,
    "Synth_Lead": 0.412,
    "Synth_Strings": 0.821,
    "Tenor_Sax": 0.285,
    "Trumpet": 0.868,
    "Violin": 0.858,
    "Drums": 0.004,
    "Drum_Set": 0.004,
    "String_Ensemble": 0.775,
}

INST_SPECIAL_MAPPING = {
    'Ukulele': 'Ukelele',
    'Piano': 'Acoustic_Piano',
    "Synth": "Synth_Lead"
}
def is_valid_inst(inst, prob, version="0.9"):
    if version == "0.9":
        if inst not in INST_THRESHOLD_0p9:
            return True
        return prob >= INST_THRESHOLD_0p9[inst]
    else:
        raise NotImplementedError


def tagging_inst_to_38(inst):
    if "_" not in inst:
        inst = " ".join([x.capitalize() if x != "and" else x for x in inst.split(" ")])
        
    if inst in INST_SPECIAL_MAPPING:
        inst = INST_SPECIAL_MAPPING[inst] 

    if inst in INST_ALL_38_MAPPER:
        return INST_ALL_38_MAPPER[inst].replace(" ", "_")
    return None

def inst_38_to_tagging(inst):
    if inst in INST_ALL_38_MAPPER.values():
        return list(INST_ALL_38_MAPPER.keys())[list(INST_ALL_38_MAPPER.values()).index(inst)]
    return inst

def get_inst_family(name):
    for parent, parent_insts in INST_TREE.items():
        if name == parent and name not in parent_insts:
            return parent, None, None
        
        for subparent, children in parent_insts.items():
            if name == subparent:
                return parent, subparent, None

            if name in children:
                return parent, subparent, name
    return None, None, None

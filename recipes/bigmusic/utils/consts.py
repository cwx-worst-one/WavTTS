"""
Presets and configurations
"""

from recipes.bigmusic.datasets.utils.zh_vocab import VOCAB2ID_MIX_V3 as DEFAULT_VOCAB2ID

# Supported structure presets
# reference: https://bytedance.larkoffice.com/docx/DWCTdHkcaopMl5xto1CcnVqbnig
# Do NOT hard-code INTRO and OUTRO to the structure. They will be added automatically.
STRUCTURES = {
    "Pop": [
        [["verse", 4, 1], ["chorus", 4, 1]],
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4, 1]],
        [["verse", 4, 1], ["chorus", 4, 1], ["chorus", 4, 2]],
        [["chorus", 4, 1], ["verse", 4, 1], ["chorus", 4, 1], ["chorus", 4, 2]],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["chorus", 4, 1],
            ["inst"],
            ["chorus", 4, 2],
            ["verse", 4, 2],
        ],
        [
            ["chorus", 4, 1],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["inst"],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["chorus", 4, 1],
            ["inst"],
            ["verse", 4, 2],
            ["chorus", 4, 2],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["inst"],
            ["verse", 4, 3],
            ["verse", 4, 4],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
    ],
    "Chinese Style": [
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4]],
        [["verse", 4], ["chorus", 4, 1], ["chorus", 4, 2]],
        [["chorus", 4, 1], ["chorus", 4, 2]],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["chorus", 4, 1],
            ["inst"],
            ["verse", 4, 2],
            ["chorus", 4, 2],
        ],
        [
            ["chorus", 4, 1],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["inst"],
            ["verse", 4, 3],
            ["verse", 4, 4],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["bridge", 8],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["inst"],
            ["verse", 4, 3],
            ["verse", 4, 4],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["inst"],
            ["verse", 4, 3],
            ["chorus", 4, 2],
            ["chorus", 4, 3],
        ],
    ],
    "Hip Hop/Rap": [
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4, 1], ["chorus", 4, 2]],
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4, 1]],
        [["chorus", 4, 1], ["verse", 4, 1], ["chorus", 4, 1], ["chorus", 4, 2]],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["bridge", 8],
            ["chorus", 4, 1],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 3],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["bridge", 8],
            ["chorus", 4, 1],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["verse", 4, 3],
            ["chorus", 4, 1],
            ["bridge", 8],
            ["chorus", 4, 1],
        ],
        [
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["verse", 4, 3],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 3],
            ["verse", 4, 4],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
    ],
    "R&B/Soul": [
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4, 1]],
        [["verse", 4, 1], ["verse", 4, 2], ["chorus", 4, 1], ["chorus", 4, 2]],
        [
            ["chorus", 4, 1],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["bridge", 8],
            ["chorus", 4, 1],
            ["chorus", 3, 2],
        ],
        [
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["bridge", 8],
            ["chorus", 4, 1],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["verse", 4, 3],
            ["chorus", 4, 1],
            ["bridge", 8],
            ["chorus", 4, 1],
        ],
        [
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["inst"],
            ["verse", 4, 3],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
        [
            ["verse", 4, 1],
            ["verse", 4, 2],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
            ["verse", 4, 3],
            ["verse", 4, 4],
            ["chorus", 4, 1],
            ["chorus", 4, 2],
        ],
    ],
}


DEFAULT_TAG_COMBINATIONS = {
    "categories": ["genre", "mood", "gender", "timbre"],
    "combinations": [
        ["Chinese Style", "Chill", "*", "*"],
        ["Chinese Style", "Dynamic/Energetic", "*", "*"],
        ["Chinese Style", "Excited", "*", "*"],
        ["Chinese Style", "Happy", "*", "*"],
        ["Chinese Style", "Inspirational/Hopeful", "*", "*"],
        ["Chinese Style", "Miss", "*", "*"],
        ["Chinese Style", "Nostalgic/Memory", "*", "*"],
        ["Chinese Style", "Romantic", "*", "*"],
        ["Chinese Style", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["Chinese Style", "Sorrow/Sad", "*", "*"],
        ["DJ", "Dynamic/Energetic", "*", "*"],
        ["DJ", "Inspirational/Hopeful", "*", "*"],
        ["DJ", "Nostalgic/Memory", "*", "*"],
        ["DJ", "Sorrow/Sad", "*", "*"],
        ["Electronic", "Chill", "*", "*"],
        ["Electronic", "Dynamic/Energetic", "*", "*"],
        ["Electronic", "Excited", "*", "*"],
        ["Electronic", "Groovy/Funky", "*", "*"],
        ["Folk", "Chill", "*", "*"],
        ["Folk", "Happy", "*", "*"],
        ["Folk", "Miss", "*", "*"],
        ["Folk", "Nostalgic/Memory", "*", "*"],
        ["Folk", "Romantic", "*", "*"],
        ["Folk", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["Folk", "Sorrow/Sad", "*", "*"],
        ["Hip Hop/Rap", "Chill", "*", "*"],
        ["Hip Hop/Rap", "Dynamic/Energetic", "*", "*"],
        ["Hip Hop/Rap", "Excited", "*", "*"],
        ["Hip Hop/Rap", "Nostalgic/Memory", "*", "*"],
        ["Hip Hop/Rap", "Romantic", "*", "*"],
        ["Hip Hop/Rap", "Sorrow/Sad", "*", "*"],
        ["Jazz", "Chill", "*", "*"],
        ["Jazz", "Dynamic/Energetic", "*", "*"],
        ["Jazz", "Groovy/Funky", "*", "*"],
        ["Jazz", "Happy", "*", "*"],
        ["Jazz", "Miss", "*", "*"],
        ["Jazz", "Nostalgic/Memory", "*", "*"],
        ["Jazz", "Romantic", "*", "*"],
        ["Jazz", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["Jazz", "Sorrow/Sad", "*", "*"],
        ["Pop", "Chill", "*", "*"],
        ["Pop", "Dynamic/Energetic", "*", "*"],
        ["Pop", "Excited", "*", "*"],
        ["Pop", "Happy", "*", "*"],
        ["Pop", "Inspirational/Hopeful", "*", "*"],
        ["Pop", "Miss", "*", "*"],
        ["Pop", "Nostalgic/Memory", "*", "*"],
        ["Pop", "Romantic", "*", "*"],
        ["Pop", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["Pop", "Sorrow/Sad", "*", "*"],
        ["Punk", "Excited", "*", "*"],
        ["Punk", "Happy", "*", "*"],
        ["Punk", "Nostalgic/Memory", "*", "*"],
        ["Punk", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["R&B/Soul", "Chill", "*", "*"],
        ["R&B/Soul", "Dynamic/Energetic", "*", "*"],
        ["R&B/Soul", "Groovy/Funky", "*", "*"],
        ["R&B/Soul", "Happy", "*", "*"],
        ["R&B/Soul", "Nostalgic/Memory", "*", "*"],
        ["R&B/Soul", "Romantic", "*", "*"],
        ["R&B/Soul", "Sentimental/Melancholic/Lonely", "*", "*"],
        ["R&B/Soul", "Sorrow/Sad", "*", "*"],
        ["Reggae", "Chill", "*", "*"],
        ["Reggae", "Groovy/Funky", "*", "*"],
        ["Reggae", "Happy", "*", "*"],
        ["Reggae", "Nostalgic/Memory", "*", "*"],
        ["Reggae", "Romantic", "*", "*"],
        ["Rock", "Excited", "*", "*"],
        ["Rock", "Happy", "*", "*"],
        ["Rock", "Inspirational/Hopeful", "*", "*"],
        ["Rock", "Miss", "*", "*"],
        ["Rock", "Nostalgic/Memory", "*", "*"],
        ["Rock", "Romantic", "*", "*"],
        ["Rock", "Sorrow/Sad", "*", "*"],
    ],
}

DEFAULT_TAG_HIERARCHY = {
    "genre": {
        "Pop": ["Chinese Pop"],
        "Electronic": ["EDM", "Future Bass", "Disco"],
        "Chinese Style": ["China-Wave", "GuFeng Music", "Chinoiserie Electronic"],
        "Rock": ["Pop Rock", "Indie Rock"],
        "Jazz": ["Jazz Pop", "Bossa Nova", "Swing"],
        "Hip Hop/Rap": ["Trap Rap", "R&B Rap", "Pop Rap", "West Coast Hip Hop"],
        "R&B/Soul": ["Contemporary R&B"],
        "Folk": ["Folk Pop"],
        "Punk": ["Pop Punk"],
    }
}


# Distribution of syllable number - duration
SLB_DUR_DIST_CONFIG = {
    "Blues": {"mean": [107.62, 113.77], "cov": [[4012.61, 1821.3], [1821.3, 2440.4]]},
    "Chinese Style": {
        "mean": [182.84, 121.28],
        "cov": [[9246.41, 4267.93], [4267.93, 3136.86]],
    },
    "Chinese Tradition": {
        "mean": [132.9, 118.26],
        "cov": [[5527.91, 3336.71], [3336.71, 3290.87]],
    },
    "Country": {
        "mean": [150.51, 108.76],
        "cov": [[7506.62, 3287.85], [3287.85, 2546.54]],
    },
    "DJ": {"mean": [197.85, 112.58], "cov": [[11924.85, 4978.48], [4978.48, 3141.07]]},
    "Electronic": {
        "mean": [126.83, 114.75],
        "cov": [[7450.64, 2695.27], [2695.27, 3079.01]],
    },
    "Folk": {"mean": [140.4, 113.41], "cov": [[6006.98, 3293.09], [3293.09, 2861.66]]},
    "Hip Hop/Rap": {
        "mean": [351.95, 114.01],
        "cov": [[33064.87, 7535.43], [7535.43, 2506.91]],
    },
    "Jazz": {"mean": [101.91, 123.51], "cov": [[3123.34, 1170.75], [1170.75, 2074.57]]},
    "Latin": {"mean": [211.12, 132.49], "cov": [[10550.61, 634.73], [634.73, 2328.6]]},
    "MC": {"mean": [337.82, 98.32], "cov": [[23472.82, 4894.82], [4894.82, 1360.15]]},
    "Metal": {
        "mean": [111.52, 120.74],
        "cov": [[4788.06, 2688.75], [2688.75, 3221.91]],
    },
    "Pop": {"mean": [181.75, 119.2], "cov": [[10441.54, 4308.75], [4308.75, 3074.64]]},
    "Punk": {"mean": [128.02, 91.9], "cov": [[6136.39, 2341.74], [2341.74, 1879.58]]},
    "R&B/Soul": {
        "mean": [185.51, 121.66],
        "cov": [[11233.64, 3769.82], [3769.82, 2824.28]],
    },
    "Reggae": {
        "mean": [152.5, 105.94],
        "cov": [[11439.47, 4854.81], [4854.81, 3707.42]],
    },
    "Rock": {"mean": [133.5, 116.37], "cov": [[7040.24, 2971.83], [2971.83, 3064.09]]},
}


# Supported duration range
DURATION_RANGE = (30, 240)


# The tags supported by lyrics generation
LYRICS_GEN_TAGS = {
    "genre": [
        "Pop",
        "Chinese Style",
        "Chinese Tradition",
        "Folk",
        "DJ",
        "Rock",
        "Hip Hop/Rap",
        "Electronic",
        "R&B/Soul",
        "Jazz",
        "Punk",
        "Reggae",
    ]
    + [
        "Chinese Pop",
        "Country Pop",
        "EDM",
        "Future Bass",
        "Disco",
        "China-Wave",
        "GuFeng Music",
        "Chinoiserie Electronic",
        "Pop Rock",
        "Indie Rock",
        "Jazz Pop",
        "Pop Rap",
        "Contemporary R&B",
        "Folk Pop",
        "Pop Punk",
        "Chinese Folk",
        "Trap Rap",
        "R&B Rap",
        "West Coast Hip Hop",
        "Bossa Nova",
        "Swing",
    ],
    "mood": [
        "Happy",
        "Cute/Playful",
        "Excited",
        "Funny",
        "Inspirational/Hopeful",
        "Sweet",
        "Shocking/magnificent/epic",
        "Chill",
        "Calm/Relaxing",
        "Mysterious",
        "Groovy/Funky",
        "Dynamic/Energetic",
        "Romantic",
        "Nostalgic/Memory",
        "Miss",
        "Dreamy/Ethereal",
        "Healing",
        "Sorrow/Sad",
        "Sentimental/Melancholic/Lonely",
        "Weird",
        "Thrilling/Suspenseful/Tense",
        "Angry/Aggressive",
    ],
    "timbre": [
        "Loud",
        "Warm",
        "Husky",
        "Powerful",
        "Lazy",
        "Electrified",
        "Bright",
        "Sweet",
        "Cute",
    ],
    "gender": ["Male", "Female"],
}


LYRICS_TAG_TO_MUSIC_TAG_MAP = {
    "timbre": {
        "Loud": "Loud and sonorous",
        "Lazy": "Sexy/Lazy",
        "Electrified": "Electrified voice",
    }
}


MUSIC_TAG_TO_LYRICS_TAG_MAP = {
    "timbre": {
        "Ethereal": "Powerful",
        "Deep": "Husky",
        "Loud and sonorous": "Loud",  # rename
        "Extreme": "Powerful",
        "Sharp": "Bright",
        "Sexy/Lazy": "Lazy",  # rename
        "Magnetic": "Warm",
        "Electrified voice": "Electrified",  # rename
    }
}


DEFAULT_TAGS_FOR_LYRICS_GEN = {
    "genre": "Pop",
    "mood": "Sorrow/Sad",
    "timbre": "Warm",
    # Leave out gender
}


# Duration tag to duration range mapping for lyrics generation
LYRICS_GEN_DURATION_RANGE_MAP = {
    "1min": (30, 90),
    "2min": (90, 150),
    "3min+": (150, 240),
}


MIX_LANG_PROB = 0.3
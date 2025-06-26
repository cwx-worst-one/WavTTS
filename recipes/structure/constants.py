N_TOP_BOUND = 15

# SEGMENT_CLASSES = ["intro", "verse", "chorus", "inst", "bridge", "outro", "silence", "pre-chorus"]
#                 #    "prechorus", "head in", "solo", "head out", "exposition", "development", "recapitulation", 
#                 #    "cadenza", "coda", "hook", "breakdown", "build-up", "drop"]

# FUNCTION_LABEL = 8

# CLASSES_WEIGHTS = [1] * 19

SPEED_RATIO = 1

def init_speed_ratio(ratio):
    global SPEED_RATIO
    SPEED_RATIO = ratio

SEGMENT_CLASSES = ["intro", "verse", "pre-chorus", "chorus", "inst", "bridge", "outro", "silence"]
FUNCTION_LABEL = 8
CLASSES_WEIGHTS = [5, 1, 20, 1, 2, 20, 2, 2]

# for new taxonomy
seg_map = {
    "silence": "silence",
    "end": "end",
    "build": "verse",
    "fadein": "intro",
    "opening": "intro",
    "stutter": "chorus",
    "slow": "verse",
    "drumroll": "inst",
    "synth": "inst",
    "closing": "outro",
    "interlude": "inst",
    "mantra": "verse",
    "fade-out": "outro",
    "out": "outro",
    "guitar": "inst",
    "head": "inst",
    "loop": "inst",
}

substr_map = {
    "other": "other",
    "hook": "chorus",
    "pre-chorus-and-chorus": "chorus",
    "verse-and-chorus": "chorus",
    "intro": "intro",
    "verse": "verse",
    "prechorus": "pre-chorus",
    "refrain": "chorus",
    # "pre-chorus": "prechorus",
    "chorus": "chorus",
    "bridge": "bridge",
    "outro": "outro",
    "fadeout": "outro",
    "ending": "outro",
    "fadein": "intro",
    "inst": "inst",
    "solo": "inst",
    "break": "inst",
    "trans": "bridge",
    "gtr": "inst",
    "section": "verse",
    "riff": "inst",
    "rap": "verse",
    "coda": "outro",
    "interlude": "inst",
    "lead-in": "inst",
    "theme": "chorus",
    "development": "verse",
    "variation": "bridge",
    "impro": "inst",
    "guitar": "inst",
    "spoken": "inst",
    "trumpet": "inst",
    "applause": "inst",
    "voice": "inst",
    "stage": "inst",
    "banjo": "inst",
    "crowd": "inst",
    "pause": "inst",
    "tag": "inst",
}

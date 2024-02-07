import json
from pathlib import Path
from string import punctuation
import re
import random
from typing import List
from recipes.bigmusic.datasets.mir_data_util import chinese_genre1_vocab, chinese_genre2_vocab, chinese_scene_vocab


def rewrite_playlist_labels(label1, label2):
    genre = ""
    mood = ""
    scene = ""
    gender = ""
    lang = "普通话"
    if label1 == "中文场景":
        scene = label2        
    elif label1 == "中文心情":
        mood = label2
    elif label1 in chinese_genre1_vocab:        
        if label1 in {"金属", "儿童音乐", "宗教"}:
            genre = label1
        else:
            if label2 in chinese_genre2_vocab:
                genre = label2
            if genre == "粤语流行":
                lang = "粤语"
    else:
        print("Missing playlist metadata")    
    return "|".join([genre, mood, scene, gender, lang])


def rewrite_metadata(metadata, type="Vocal"):
    mood = metadata.get('final_mood', metadata.get('merge_mood'))
    genre = metadata.get('final_genre', metadata.get('merge_genre'))
    gender = metadata.get('merge_aed')
    text = ""
    if type == "Vocal":
        text = "A"
        if mood is not None and mood != 'nan' and mood.strip():
            text += " " + mood.lower()
        if genre is not None and genre != 'nan' and genre.strip():
            text += " " + genre.lower()
        text += " song"
        if gender is not None and gender != 'nan':
            if 'Female' in gender:
                text += " with female vocal"
            elif 'Male' in gender:
                text += " with male vocal"
        text += "."
    elif type == "Instrumental":
        text = ""
        if mood is not None and mood != 'nan' and mood.strip():
            text += mood.lower() + " "
        if genre is not None and genre != 'nan' and genre.strip():
            text += genre.lower() + " "
        text += "music."
    elif type == "Speech":
        text = "Speech."
    return text

def normalize_text(text, enable_punctuation=False, lowercase=False):
    if lowercase:
        text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")    
    text = text.replace("-", " ")
    text = text.replace(".", "\n")
    text = text.replace("。", "\n")
    text = text.replace("!", "\n")
    text = text.replace("！", "\n")
    text = text.replace("?", "\n")
    text = text.replace("？", "\n")
    text = re.sub(r'\n\s*\n', '\n', text) # remove double new lines
    nlp_punctuation = punctuation.replace("'", "") # allow single quotes (') for contractions
    nlp_punctuation = nlp_punctuation.replace("[", "") # allow brackets for structure tags
    nlp_punctuation = nlp_punctuation.replace("]", "") # allow brackets for structure tags    
    nlp_punctuation = punctuation.replace("<", "").replace(">", "") # allow <> for special tokens
    text = text.translate(str.maketrans("", "", nlp_punctuation))
    if enable_punctuation:
        text = text.replace("\n", " <n> ") # remove new lines
    else:
        text = text.replace("\n", " ") # remove new lines
    text = " ".join(text.split()) # remove spaces
    return text.strip()

def concat_metadata_list(existimg_metadata, metadata):
    if existimg_metadata is None:
        return metadata
    for idx, (m1, m2) in enumerate(zip(existimg_metadata, metadata)):
        existimg_metadata[idx] = { **m1, **m2 }
    return existimg_metadata

def update_json(metadata_fp, updates):
    if Path(metadata_fp).exists():
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata = { **metadata, **updates }
    with open(metadata_fp, 'w') as f:
        json.dump(metadata, f, indent=2)
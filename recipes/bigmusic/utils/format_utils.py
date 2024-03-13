import json
from pathlib import Path
from string import punctuation
import re
from typing import Dict, List, Optional, Tuple, Union

from recipes.bigmusic.datasets.mir_data_util import (
    CHINESE_GENRE1_VOCAB,
    CHINESE_GENRE2_VOCAB,
    CHINESE_TO_SA_MAPPING,
    SA_TAGS_SPECIAL_MAP,
    SA_CAT_VOCAB,
    VOICE_THRESHOLDS,
)
from recipes.datasets.mcc.sami_tokenizer import Phrase


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
    elif label1 in CHINESE_GENRE1_VOCAB:        
        if label1 in {"金属", "儿童音乐", "宗教"}:
            genre = label1
        else:
            if label2 in CHINESE_GENRE2_VOCAB:
                genre = label2
            if genre == "粤语流行":
                lang = "粤语"
    else:
        print("Missing playlist metadata")
    # TODO: (QQ) use SA sinking / language tags / voice.
    genre = CHINESE_TO_SA_MAPPING.get(genre, "")
    mood = CHINESE_TO_SA_MAPPING.get(mood, "")
    scene = CHINESE_TO_SA_MAPPING.get(scene, "")
    sinking = "Sinking" if (genre == "DJ" or genre == "MC") else "non-Sinking"
    lang = CHINESE_TO_SA_MAPPING.get(lang, "")
    return "|".join([genre, mood, scene, sinking, lang])


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


def parse_sa_music_tagging(music_tagging: Optional[Dict], sinking_threshold: float) -> Tuple[List[str], List[str], bool]:
    def parse_result(result: Union[List, str]) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        return result[0]

    def map_tag(tag: str) -> str:
        """Replace certain tags in the dataset"""
        return SA_TAGS_SPECIAL_MAP.get(tag, tag)

    # The order should match `mir_data_util`
    order = ["Genre20", "Mood", "Theme", "MusicLowQuality", "Language"]

    if music_tagging is None:
        return [""] * len(order), [], False
    sinking_prob = music_tagging["MusicLowQuality"]["Sinking"]
    is_sinking = sinking_prob >= sinking_threshold
    quality = "Sinking" if is_sinking else "non-Sinking"
    tags = [quality if item == "MusicLowQuality" else map_tag(parse_result(music_tagging[item]["result"])) for item in order]
    unfamiliar_tags = {cat_name: tag for tag, cat_vocab_tags, cat_name in zip(tags, SA_CAT_VOCAB, order) if tag not in cat_vocab_tags}
    return [(tag if tag in cat_vocab_tags else "") for tag, cat_vocab_tags in zip(tags, SA_CAT_VOCAB)], unfamiliar_tags, is_sinking


def parse_voice_tag(voice_probs: Optional[Dict[str, float]]) -> str:
    """Return the voice tag based on probablity thresholds. 'adult' tag is not used."""
    if voice_probs is None:
        return ""
    is_child = voice_probs["child"] >= VOICE_THRESHOLDS["Child"]
    if is_child:
        return "Child"
    is_female = voice_probs["female"] >= VOICE_THRESHOLDS["Female"]
    is_male = voice_probs["male"] >= VOICE_THRESHOLDS["Male"]
    if is_female and not is_male:
        return "Female"
    if is_male and not is_female:
        return "Male"
    if is_female and is_male:
        if voice_probs["female"] >= voice_probs["male"]:
            return "Female"
        return "Male"
    return ""


def parse_filter_label(filter_label: Optional[Dict[str, str]]) -> Tuple[bool, bool]:
    """Return (high_quality, popular_potential).
    Conservative filtering. Assume the song is high quality if it's not labeled.
    """
    if filter_label is None:
        return True, True
    return filter_label["high_quality"] == "yes", filter_label["popular_potential"] == "yes"


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
    nlp_punctuation = nlp_punctuation.replace("[", "").replace("]", "") # allow brackets for structure tags
    nlp_punctuation = nlp_punctuation.replace("<", "").replace(">", "") # allow <> for special tokens
    text = text.translate(str.maketrans("", "", nlp_punctuation))
    if enable_punctuation:
        text = text.replace("\n", " <n> ") # remove new lines
    else:
        text = text.replace("\n", " ") # remove new lines
    text = " ".join(text.split()) # remove spaces
    return text.strip()

def normalize_text_sami_tokenizer(text, enable_punctuation=False, lowercase=False):
    """Normalize text with special treatment for special tokens supported by sami_tokenizer."""
    sep = " <n> " if enable_punctuation else " "
    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        phrase = Phrase.parse(text=line)
        normalized_text = normalize_text(phrase.text, enable_punctuation, lowercase) if phrase.text else ""
        normalized_lines.append(phrase._replace(text=normalized_text).format_text())
    return sep.join([l for l in normalized_lines if l])  # remove empty lines

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

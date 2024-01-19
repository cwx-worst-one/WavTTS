import json
from pathlib import Path
from string import punctuation
import re
import random
from typing import List

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
    elif type == "mir_tags":
        # NOTE: randomly shuffle to diversify prompt
        random.shuffle(metadata["genres"])
        random.shuffle(metadata["vocals"]) 

        def multiple_choices_text_processor(text: List[str]) -> str:
            if len(text) > 1:
                text = ", ".join(text[:-1]) + f" and {text[-1]}"
            elif len(text) == 1:
                text = text[0]
            else:
                text = ""
            return text.lower()

        # example: 'rock, pop and blues'
        genre_text = multiple_choices_text_processor(metadata["genres"])
        genre_text = genre_text.replace("_", " ") # rnb_soul -> rnb soul
        if mood is not None and mood != 'nan':
            genre_text = f"{mood.lower()} {genre_text}"

        gender = {
            "gender_male": "male",
            "gender_female": "female",
        }
        vocal_gender_text = multiple_choices_text_processor([gender.get(v, "") for v in metadata["vocals"] if "gender_" in v])
        text = f"""A {genre_text} song with {vocal_gender_text} vocal."""
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
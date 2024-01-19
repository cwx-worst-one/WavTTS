import json
from pathlib import Path
from string import punctuation


def normalize_text(text, remove_newlines=True):
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("\n", " <n> ")  # remove new lines
    if remove_newlines:
        text = text.replace(" <n> ", " ")  # remove new lines
    nlp_punctuation = punctuation.replace(
        "'", ""
    )  # allow single quotes (') for contractions
    text = text.translate(str.maketrans("", "", nlp_punctuation))
    text = " ".join(text.split(" "))  # remove spaces
    return text


def concat_metadata_list(existimg_metadata, metadata):
    if existimg_metadata is None:
        return metadata
    for idx, (m1, m2) in enumerate(zip(existimg_metadata, metadata)):
        existimg_metadata[idx] = {**m1, **m2}
    return existimg_metadata


def update_json(metadata_fp, updates):
    if Path(metadata_fp).exists():
        with open(metadata_fp, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata = {**metadata, **updates}
    with open(metadata_fp, "w") as f:
        json.dump(metadata, f, indent=2)

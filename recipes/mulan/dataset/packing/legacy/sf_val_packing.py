# Download from hdfs://harunava/home/byte_speech_sv/mulan/
#   short_form_uio/{audio,text}_shards/shards_0_000000.tar
# Untar them to sf/audio and sf/text
# Rename all *.audio.npy to *.npy

# After this script
# tar sf/audio/*npy and sf/sf_val_meta.csv to sf_val.tar.gz (size: 2.2G)

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# List all files under sf/text
text_files = os.listdir("validation/sf/text")
text_files = [f"validation/sf/text/{f}" for f in text_files]

meta = []

for text_file in tqdm(text_files):
    if not text_file.endswith("json"):
        continue
    # Read file
    with open(text_file) as f:
        json_data = json.load(f)

    # Get music_id
    music_id = Path(text_file).stem.split(".")[0]

    source = json_data["data_source"]
    audio_file = text_file.replace("text", "audio").replace("meta.json", "npy")
    if source == "unknown":
        # remove corresponding audio file
        os.remove(audio_file)
        continue

    # Cut audio to 10seconds
    audio = np.load(audio_file)
    audio = audio[..., : 10 * 24000]
    np.save(audio_file, audio)
    # Get text
    text = []
    if source in ["both", "short_form"]:
        text += [
            json_data["meta_song_title"],
            json_data["meta_song_author"],
            json_data["meta_song_album_name"],
            json_data["genre"],
            json_data["mood"],
            json_data["theme"],
            json_data["language"],
        ]
    if source in ["both", "playlist"]:
        text += [json_data["playlist_name"], json_data["playlist_description"]]
    text = " ".join(text)
    meta.append({"music_id": music_id, "text": text})

meta_df = pd.DataFrame(meta)
meta_df.to_csv("validation/sf/sf_val_meta.csv", index=False)

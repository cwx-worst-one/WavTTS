# Download from hdfs://harunava/home/byte_speech_sv/daiyu.zhang/dataset/MusicLM_5k.zip
import glob
import multiprocessing as mp
import os
from ast import literal_eval

import numpy as np
import pandas as pd
from slice import STR_CH_FIRST, load_audio


def convert_mp3_to_npy(file):
    if not file.endswith(".mp3"):
        return
    # read in the audio file
    audio_file = os.path.join("validation/kaggle/MusicLM_5k/audio_clips", file)
    audio = load_audio(audio_file, STR_CH_FIRST, 24000, True)[0]
    # cut the audio to 10 seconds and add a dimension
    if audio.shape[1] < 240000:
        audio = np.pad(audio, ((0, 0), (0, 240000 - audio.shape[1])), "constant")
    else:
        audio = audio[:, :240000]
    # convert mp3 to npy
    audio = (audio * 32768.0).astype("int16")
    # save the npy file
    np.save(
        os.path.join("validation/kaggle/MusicLM_5k/audio", file[:-4] + ".npy"), audio
    )


if __name__ == "__main__":

    # List files under audio_clips
    audio_files = os.listdir("validation/kaggle/MusicLM_5k/audio_clips")
    os.makedirs("validation/kaggle/MusicLM_5k/audio", exist_ok=True)
    pool = mp.Pool(20)

    pool.map(convert_mp3_to_npy, audio_files)

    df = pd.read_csv("validation/kaggle/MusicLM_5k/musiccaps-public.csv")

    files = glob.glob("validation/kaggle/MusicLM_5k/audio/*.npy")

    idlist = [ff.split("/")[-1].split(".")[0] for ff in files]

    restdf = df[df["ytid"].isin(idlist)]

    to_drop = [
        "start_s",
        "end_s",
        "audioset_positive_labels",
        "author_id",
        "is_balanced_subset",
        "is_audioset_eval",
    ]
    restdf = restdf.drop(to_drop, axis=1)

    def get_aspect(row):
        return ",".join(literal_eval(row["aspect_list"]))

    restdf["aspect"] = restdf.apply(get_aspect, axis=1)
    restdf = restdf.drop(["aspect_list"], axis=1)
    restdf.to_csv("validation/kaggle/MusicLM_5k/mt_val.csv", index=False)

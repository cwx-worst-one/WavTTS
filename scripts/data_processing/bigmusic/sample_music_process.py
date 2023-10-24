
# import pandas as pd
# import subprocess
# import numpy as np

# df = pd.read_excel("authorized_chinese.xlsx", engine='openpyxl')
# df['audio_url'] = df['原始音频URL']
# df = df.drop(['原始音频URL'], axis=1)

# # Download the audio based on audio url, both curl or wget work
# # For simplicity, let's cut off first 100 rows
# df = df.head(100)


# output_dir = "/mnt/bn/audio-diffusion/unify_testing_sample_chinese_data"


# SAMPLE_RATE = 24000
# def download_audio(url):
#     ffmpeg_command = [
#         "ffmpeg",
#         "-i",
#         url,
#         "-ar",
#         str(SAMPLE_RATE),
#         "-ac",
#         "1",
#         "-f",
#         "s16le",
#         "-acodec",
#         "pcm_s16le",
#         "-",
#     ]
#     p = subprocess.Popen(
#         ffmpeg_command,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
#     out, _ = p.communicate()
#     if p.returncode != 0:
#         return None
#     npy = np.frombuffer(out, dtype=np.int16)
#     return npy

# for i, url in enumerate(df['audio_url']):

#     sample_audio = download_audio(url)
#     song_id = df['song_id'][i]
#     np.save(f"{output_dir}/{song_id}.npy", sample_audio)



import pandas as pd
import subprocess
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor

# Argument Parsing
parser = argparse.ArgumentParser(description="Download audio and save.")
parser.add_argument("--output_dir", type=str, help="Output directory to save audio data.")
args = parser.parse_args()
output_dir = args.output_dir

df = pd.read_excel("authorized_chinese.xlsx", engine='openpyxl')
df['audio_url'] = df['原始音频URL']
df = df.drop(['原始音频URL'], axis=1)
df = df.head(100)
SAMPLE_RATE = 24000

def download_audio(data):
    i, url = data
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        url,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    p = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = p.communicate()
    if p.returncode != 0:
        return None
    npy = np.frombuffer(out, dtype=np.int16)
    song_id = df['song_id'][i]
    np.save(f"{output_dir}/{song_id}.npy", npy)

with ProcessPoolExecutor() as executor:
    list(executor.map(download_audio, enumerate(df['audio_url'])))

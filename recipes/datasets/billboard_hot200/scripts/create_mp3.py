import os
import subprocess
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm

if __name__ == "__main__":

    audio_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200"
    mp3_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200_mp3_320kbps"
    audio_fps = glob(os.path.join(audio_dir, "*.flac"))
    def parallelize(fp):
        try:
            mp3_fn = os.path.basename(fp).replace(".flac", ".mp3")
            out_fp = os.path.join(mp3_dir, mp3_fn)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                fp,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                out_fp,
            ]
            # cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", fp, "-codec:a", "libmp3lame", "-q:a", "0", out_fp]
            p = subprocess.Popen(cmd)
            p.wait()
        except Exception as e:
            print(e)

    Parallel(n_jobs=32, prefer="threads")(
        delayed(parallelize)(fp) for fp in tqdm(audio_fps)
    )

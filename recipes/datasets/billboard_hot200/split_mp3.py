import os
import subprocess
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":

    in_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200_mp3"
    out_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200_mp3_30s"
    audio_fps = glob(os.path.join(in_dir, "*.mp3"))
    
    def parallelize(fp):
        try:
            fp = Path(fp)
            out_fp = f"{fp.stem}-%03d.mp3"
            out_fp = os.path.join(out_dir, str(out_fp))
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", fp, "-codec:a", "libmp3lame", "-b:a", "320k", "-f", "segment", "-segment_time", "30", "-min_seg_duration", "30", out_fp]
            p = subprocess.Popen(cmd)
            p.wait()
        except Exception as e:
            print(e)

    Parallel(n_jobs=64, prefer="threads")(delayed(parallelize)(fp) for fp in tqdm(audio_fps))
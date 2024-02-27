import sys
import os
from recipes.voicebox.datasets.utils import get_duration_frames

def get_duration(utt_id, duration_path, phonemes, hop_ms):
    durations = []
    with open(duration_path, "r") as f:
        for line in f.readlines():
            phone, duration = line.strip().split("\t")
            duration = float(duration)
            durations.append((phone, duration))
    duration_frames = get_duration_frames(durations, phonemes, hop_ms, utt_id)
    return duration_frames

if __name__ == "__main__":
    lab_dir=sys.argv[1]
    dur_dir=sys.argv[2]
    out_dir=sys.argv[3]

    hop_ms = 256 / 22050.0
    os.makedirs(out_dir, exist_ok=True)

    for item in os.listdir(lab_dir):
        lab_path = os.path.join(lab_dir, item)
        dur_path = os.path.join(dur_dir, item)
        out_path = os.path.join(out_dir, item)
        
        tacolabels = []
        phonemes = []
        for line in open(lab_path, "r").readlines():
            tacolabels.append(line)
            phonemes.append(line.split("\t")[0])
        
        duration = get_duration(item, dur_path, phonemes, hop_ms)
        
        fout = open(out_path, "w")
        
        for i, line in enumerate(tacolabels):
            fout.write(f"{line[:-1]}\t{duration[i]}\n")
        

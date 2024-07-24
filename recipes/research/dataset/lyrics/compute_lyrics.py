import argparse
import json
import os

import pandas as pd
import torch

# import whisper_timestamped as whisper
from faster_whisper import WhisperModel
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

from samantha.data.audio_utils import normalize_audio
from samantha.data.av_audio import audio_read, audio_write
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.hdfs_tools import hdfs_get, hdfs_mkdir, hdfs_put


def convert_to_dict(segments):
    d = []
    for s in segments:
        s = s._asdict()
        s["words"] = [w._asdict() for w in s["words"]]
        d.append(s)
    return d


def convert_file(row: pd.Series):
    global WHISPER
    global FAILED

    fp = row["fp"]

    tmp_path = f"/tmp/{fp}"
    tmp_dir = os.path.dirname(tmp_path)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        if not os.path.exists(fp):
            hdfs_get(fp, tmp_path)

        # WhisperX
        # audio = whisperx.load_audio(tmp_path)
        # result = WHISPER.transcribe(audio, batch_size=4)
        # print(result["segments"]) # before alignment
        # 2. Align whisper output
        # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        # result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda", return_char_alignments=False)
        # print(result["segments"]) # after alignment

        with torch.cuda.amp.autocast(enabled=True):

            segments, result = WHISPER.transcribe(
                tmp_path, word_timestamps=True, vad_filter=False
            )
            result = result._asdict()

            segments = list(segments)
            segments = convert_to_dict(segments)
            result["segments"] = segments
            # result = WHISPER(
            #     tmp_path,
            #     chunk_length_s=30,
            #     batch_size=32,
            #     return_timestamps=True,
            #     return_language=True,
            # )
            out_fn = f"{row['meta_song_id']}.json"
            out_fp = os.path.join(tmp_dir, out_fn)

            with open(out_fp, "w") as f:
                json.dump(result, f)

            hdfs_put(out_fp, os.path.join(OUT_FP, out_fn))

    except Exception as e:
        print(e)
        FAILED += 1
        print(f"Failed: {FAILED}/{len(files)}")

    os.remove(tmp_path)


def extract_meta_id(fp: str, parent_dir):
    meta_id = os.path.relpath(fp, parent_dir)
    return os.path.splitext(meta_id)[0].replace("/", "_")


def chunks(l, n):
    for i in range(0, n):
        yield l[i::n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--index", required=True, type=str)
    parser.add_argument("--existing_files", required=True, type=str)
    args = parser.parse_args()

    worker_id = int(os.getenv("ARNOLD_ID", 0)) + args.rank
    num_workers = int(os.getenv("WORLD_SIZE", 1))
    print(f"ARNOLD WORKER: {worker_id + 1}/{num_workers}")

    # WHISPER = whisperx.load_model("large-v2", "cuda", compute_type="float16")

    # WHISPER = pipeline(
    #     "automatic-speech-recognition",
    #     model="openai/whisper-large-v3",
    #     torch_dtype=torch.float16,
    #     device=f"cuda",
    #     model_kwargs=(
    #         {"attn_implementation": "flash_attention_2"}
    #         if is_flash_attn_2_available()
    #         else {"attn_implementation": "sdpa"}
    #     ),
    # )
    # assert is_flash_attn_2_available() == True
    # WHISPER = whisper.load_model("large-v3", device="cuda", backend="openai-whisper")
    # WHISPER = WhisperModel("large-v3", device="cuda", compute_type="float16")
    WHISPER = WhisperModel(
        "distil-large-v3", device="cuda", device_index=args.rank, compute_type="float16"
    )

    FAILED = 0
    SOURCE_PATH = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_normalised_-16LUFS_mp3"

    # files = hdfs_ls(os.path.join(SOURCE_PATH, "*"))
    # Generate file with: `hdfs dfs -ls hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_normalised_-16LUFS_mp3 | awk '{print $8}' > billboard_hot_200-v2.txt`
    files = open(args.index).read().splitlines()

    OUT_FP = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_lyrics_wordlevel"
    hdfs_mkdir(OUT_FP)

    # existing_files = hdfs_ls(os.path.join(OUT_FP, "*"))
    existing_files = open(args.existing_files).read().splitlines()

    meta_ids = list(map(lambda f: extract_meta_id(f, SOURCE_PATH), files))
    existing_meta_ids = list(map(lambda f: extract_meta_id(f, OUT_FP), existing_files))

    remainder_meta_ids = list(set(meta_ids) - set(existing_meta_ids))

    df = pd.DataFrame(files, columns=["fp"])
    df["meta_song_id"] = df["fp"].apply(lambda f: extract_meta_id(f, SOURCE_PATH))
    df["exists"] = ~df["meta_song_id"].isin(remainder_meta_ids)

    print("Percentage done:", df["exists"].sum() / len(df) * 100)

    df = df[df["exists"] == False]

    df = list(chunks(df, n=num_workers))
    print(f"NUM FILE LISTS: {len(df)}")

    df = df[worker_id]
    print(f"PROCESSING {len(df)} files")

    # for idx, row in tqdm(df.iterrows(), total=len(df)):
    #     convert_file(row)
    Parallel(n_jobs=16, prefer="threads")(
        delayed(convert_file)(row) for idx, row in tqdm(df.iterrows(), total=len(df))
    )

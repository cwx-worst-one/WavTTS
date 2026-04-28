import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm


DEFAULT_OLD_DIR = Path("data/Emilia_ZH_EN_pinyin")
DEFAULT_NEW_DIR = Path("data/Emilia_ZH_EN_pinyin_hdfs")
DEFAULT_OLD_ROOT = "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia"
DEFAULT_NEW_ROOT = "/mnt/hdfs/ssd_hldy/chenwenxi.sylvan/data/emilia"


def rewrite_audio_path(audio_path: str, old_root: str, new_root: str) -> str:
    if not audio_path.startswith(old_root):
        raise ValueError(f"audio_path does not start with old_root: {audio_path}")
    return new_root + audio_path[len(old_root) :]


def copy_metadata_files(src_dir: Path, dst_dir: Path) -> None:
    for name in ["duration.json", "vocab.txt"]:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing source file: {src}")
        shutil.copy2(src, dst)


def sample_verify(src_arrow: Path, old_root: str, new_root: str, sample_count: int) -> None:
    checked = 0
    ok = 0
    with open(src_arrow, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        while checked < sample_count:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            rows = batch.to_pylist()
            for row in rows:
                new_path = Path(rewrite_audio_path(row["audio_path"], old_root, new_root))
                exists = new_path.exists()
                checked += 1
                ok += int(exists)
                print(f"[{checked}] {exists} :: {new_path}")
                if checked >= sample_count:
                    break
    print(f"sample verification: {ok}/{checked} paths exist")
    if checked == 0:
        raise RuntimeError("no samples were checked")
    if ok != checked:
        raise RuntimeError("sample verification failed; some rewritten paths do not exist")


def migrate_arrow(src_arrow: Path, dst_arrow: Path, old_root: str, new_root: str) -> int:
    total = 0
    sink = pa.OSFile(dst_arrow.as_posix(), "wb")
    writer = None
    try:
        with open(src_arrow, "rb") as f:
            reader = ipc.RecordBatchStreamReader(f)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                rows = batch.to_pylist()
                for row in tqdm(rows, desc="Rewriting batch", leave=False):
                    row["audio_path"] = rewrite_audio_path(row["audio_path"], old_root, new_root)
                new_batch = pa.RecordBatch.from_pylist(rows, schema=batch.schema)
                if writer is None:
                    writer = ipc.new_stream(sink, new_batch.schema)
                writer.write_batch(new_batch)
                total += len(rows)
    finally:
        if writer is not None:
            writer.close()
        sink.close()
    return total


def print_summary(out_dir: Path) -> None:
    dur_path = out_dir / "duration.json"
    with open(dur_path, "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]
    print(f"output dir: {out_dir}")
    print(f"num durations: {len(durations)}")
    print(f"total hours: {sum(durations) / 3600:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HDFS-based Emilia index by copying an existing local index and rewriting audio_path."
    )
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--dst-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", default=DEFAULT_NEW_ROOT)
    parser.add_argument("--verify-only", action="store_true", help="Only sample-check rewritten paths; do not write files.")
    parser.add_argument("--sample-count", type=int, default=20, help="Number of rows to sample during verification.")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing dst-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    src_arrow = src_dir / "raw.arrow"
    dst_arrow = dst_dir / "raw.arrow"

    if not src_arrow.exists():
        raise FileNotFoundError(f"missing source arrow file: {src_arrow}")

    print(f"source index: {src_dir}")
    print(f"target index: {dst_dir}")
    print(f"rewrite root: {args.old_root} -> {args.new_root}")

    sample_verify(src_arrow, args.old_root, args.new_root, args.sample_count)
    if args.verify_only:
        print("verify-only mode; no files written")
        return

    if dst_dir.exists() and any(dst_dir.iterdir()) and not args.force:
        raise FileExistsError(f"destination exists and is not empty: {dst_dir}; use --force to overwrite files inside it")

    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(src_dir, dst_dir)
    total = migrate_arrow(src_arrow, dst_arrow, args.old_root, args.new_root)
    print(f"rewrote {total} rows into {dst_arrow}")
    print_summary(dst_dir)


if __name__ == "__main__":
    main()

# python src/f5_tts/train/datasets/migrate_emilia_index_to_hdfs.py
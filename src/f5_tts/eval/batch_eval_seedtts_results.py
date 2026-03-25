#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT_DEFAULT = REPO_ROOT / "results"
EVAL_SCRIPT_DEFAULT = REPO_ROOT / "src" / "f5_tts" / "eval" / "eval_seedtts_param.sh"
SUMMARY_DIR_DEFAULT = RESULTS_ROOT_DEFAULT / "seedtts_eval_summary"
EXPECTED_COUNTS = {
    "seedtts_test_zh": REPO_ROOT / "data" / "seedtts_testset" / "zh" / "meta.lst",
    "seedtts_test_en": REPO_ROOT / "data" / "seedtts_testset" / "en" / "meta.lst",
}
METRIC_TO_FILE = {
    "wer": "_wer_results.jsonl",
    "sim": "_sim_results.jsonl",
    "utmos": "_utmos_results.jsonl",
}
TASK_TO_LANG = {
    "seedtts_test_zh": "zh",
    "seedtts_test_en": "en",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run SeedTTS evaluation on existing inference results under results/."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--eval-script", type=Path, default=EVAL_SCRIPT_DEFAULT)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR_DEFAULT)
    parser.add_argument("--models", nargs="*", default=None, help="Result subdirectories to evaluate. Default: all.")
    parser.add_argument("--steps", nargs="+", type=int, default=[200000, 400000])
    parser.add_argument("--tasks", nargs="+", default=["seedtts_test_zh", "seedtts_test_en"])
    parser.add_argument("--metrics", nargs="+", default=["wer", "sim", "utmos"])
    parser.add_argument("--eval-gpus", type=str, default="[0,1,2,3,4,5,6,7]")
    parser.add_argument("--master-port", type=int, default=53721)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-audio-count",
        action="store_true",
        help="Exit non-zero if any inference directory has incomplete audio count.",
    )
    return parser.parse_args()


def load_expected_counts(tasks: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in tasks:
        meta_path = EXPECTED_COUNTS.get(task)
        if meta_path is None:
            raise ValueError(f"Unsupported task for this batch evaluator: {task}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta list for {task}: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            counts[task] = sum(1 for line in f if line.strip())
    return counts


def resolve_model_dirs(results_root: Path, models: Sequence[str] | None) -> List[Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"results root not found: {results_root}")

    if not models:
        return sorted(
            path
            for path in results_root.iterdir()
            if path.is_dir() and path.name != "exp_libritts"
        )

    resolved: List[Path] = []
    for model in models:
        model_path = Path(model)
        if not model_path.is_absolute():
            model_path = results_root / model
        if not model_path.exists() or not model_path.is_dir():
            print(f"[WARN] Skip missing model dir: {model_path}", flush=True)
            continue
        if model_path.name == "exp_libritts":
            print(f"[INFO] Skip excluded model dir: {model_path}", flush=True)
            continue
        resolved.append(model_path)
    return sorted(resolved)


def iter_gen_dirs(model_dir: Path, steps: Sequence[int], tasks: Sequence[str]) -> Iterable[tuple[int, str, Path]]:
    for step in steps:
        step_dir = model_dir / str(step)
        if not step_dir.is_dir():
            print(f"[INFO] Missing step dir, skip: {step_dir}", flush=True)
            continue
        for task in tasks:
            task_dir = step_dir / task
            if not task_dir.is_dir():
                print(f"[INFO] Missing task dir, skip: {task_dir}", flush=True)
                continue
            gen_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
            if not gen_dirs:
                print(f"[INFO] No inference subdir found, skip: {task_dir}", flush=True)
                continue
            for gen_dir in gen_dirs:
                yield step, task, gen_dir


def count_wavs(gen_dir: Path) -> int:
    return sum(1 for _ in gen_dir.rglob("*.wav"))


def missing_metrics(gen_dir: Path, metrics: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for metric in metrics:
        metric_path = gen_dir / METRIC_TO_FILE[metric]
        if not metric_path.exists() or metric_path.stat().st_size == 0:
            missing.append(metric)
    return missing


def run_eval(
    eval_script: Path,
    gen_dir: Path,
    task: str,
    metrics: Sequence[str],
    eval_gpus: str,
    master_port: int,
    local: bool,
) -> int:
    env = os.environ.copy()
    env["LANG_OVERRIDE"] = TASK_TO_LANG[task]
    env["GEN_WAV_DIR_OVERRIDE"] = str(gen_dir)
    env["EVAL_METRICS_OVERRIDE"] = " ".join(metrics)
    env["GPUS_OVERRIDE"] = eval_gpus
    env["MASTER_PORT_OVERRIDE"] = str(master_port)
    if local:
        env["LOCAL_OVERRIDE"] = "1"

    cmd = ["bash", str(eval_script)]
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False).returncode


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.summary_dir / "batch_eval_seedtts_results.jsonl"

    expected_counts = load_expected_counts(args.tasks)
    model_dirs = resolve_model_dirs(args.results_root, args.models)

    total_gen_dirs = 0
    evaluated = 0
    skipped_complete = 0
    skipped_incomplete = 0
    failed = 0

    with progress_path.open("a", encoding="utf-8") as progress_file:
        for model_dir in model_dirs:
            for step, task, gen_dir in iter_gen_dirs(model_dir, args.steps, args.tasks):
                total_gen_dirs += 1
                wav_count = count_wavs(gen_dir)
                expected_count = expected_counts[task]
                missing = missing_metrics(gen_dir, args.metrics)
                record = {
                    "model_dir": str(model_dir),
                    "step": step,
                    "task": task,
                    "gen_dir": str(gen_dir),
                    "wav_count": wav_count,
                    "expected_wav_count": expected_count,
                    "missing_metrics": missing,
                    "status": "",
                }

                if wav_count != expected_count:
                    skipped_incomplete += 1
                    record["status"] = "skip_incomplete_audio"
                    print(
                        f"[SKIP] incomplete audio count: {gen_dir} ({wav_count}/{expected_count})",
                        flush=True,
                    )
                    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    continue

                if not missing:
                    skipped_complete += 1
                    record["status"] = "skip_metrics_exist"
                    print(f"[SKIP] metrics already exist: {gen_dir}", flush=True)
                    progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    progress_file.flush()
                    continue

                print(
                    f"[RUN] model={model_dir.name} step={step} task={task} gen_dir={gen_dir.name} "
                    f"metrics={','.join(missing)}",
                    flush=True,
                )
                record["status"] = "dry_run" if args.dry_run else "running"
                progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress_file.flush()

                if args.dry_run:
                    continue

                rc = run_eval(
                    eval_script=args.eval_script,
                    gen_dir=gen_dir,
                    task=task,
                    metrics=missing,
                    eval_gpus=args.eval_gpus,
                    master_port=args.master_port,
                    local=args.local,
                )
                if rc == 0:
                    evaluated += 1
                    status = "ok"
                else:
                    failed += 1
                    status = f"fail({rc})"
                progress_file.write(
                    json.dumps(
                        {
                            **record,
                            "status": status,
                            "returncode": rc,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                progress_file.flush()

    print(
        "[SUMMARY] "
        f"gen_dirs={total_gen_dirs} "
        f"evaluated={evaluated} "
        f"skipped_complete={skipped_complete} "
        f"skipped_incomplete={skipped_incomplete} "
        f"failed={failed}",
        flush=True,
    )

    if failed > 0:
        return 1
    if args.strict_audio_count and skipped_incomplete > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

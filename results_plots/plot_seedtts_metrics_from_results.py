#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

DEFAULT_TASKS = ("seedtts_test_zh", "seedtts_test_en")
DEFAULT_METRICS = ("wer", "sim", "utmos")

METRIC_FILE_MAP = {
    "wer": "_wer_results.jsonl",
    "sim": "_sim_results.jsonl",
    "utmos": "_utmos_results.jsonl",
}

METRIC_PRECISION = {
    "wer": 5,
    "sim": 5,
    "utmos": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SeedTTS zh/en metrics from results/<model>/<step>/<task>/<gen_dir>/_xxx_results.jsonl."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Root results directory.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model directory names.")
    parser.add_argument("--steps", type=str, required=True, help="Comma-separated step numbers.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks. Default: seedtts_test_zh,seedtts_test_en",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics. Default: wer,sim,utmos",
    )
    parser.add_argument(
        "--gen-subdir",
        type=str,
        default="",
        help="Optional exact generation subdir name under each task directory. Empty means auto-select.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results_plots"), help="Output directory.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def score_gen_dir_name(name: str) -> int:
    score = 0
    if name.startswith("seed0_"):
        score += 100
    if "nfe32" in name:
        score += 50
    if "cfg3.0" in name:
        score += 30
    if "speed1.0" in name:
        score += 20
    if "ss-1.0" in name:
        score += 10
    if "no_vocoder" in name:
        score += 5
    return score


def find_metric_file(
    results_root: Path,
    model: str,
    step: int,
    task: str,
    metric: str,
    gen_subdir: str,
) -> Path | None:
    task_root = results_root / model / str(step) / task
    if not task_root.exists():
        return None

    file_name = METRIC_FILE_MAP[metric]
    if gen_subdir:
        candidate = task_root / gen_subdir / file_name
        return candidate if candidate.exists() else None

    # Auto-select: choose the best-matching generation directory by heuristic.
    candidates = sorted(task_root.rglob(file_name))
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (
            score_gen_dir_name(p.parent.name),
            p.parent.name,
            str(p),
        ),
        reverse=True,
    )
    return candidates[0]


def parse_metric_value(path: Path, metric: str) -> float:
    last_line = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                last_line = line

    if not last_line:
        raise ValueError(f"Empty file: {path}")

    metric_upper = metric.upper()
    m_named = re.search(rf"{metric_upper}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", last_line)
    if m_named:
        return float(m_named.group(1))

    m_last = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", last_line)
    if m_last:
        return float(m_last.group(1))

    raise ValueError(f"Cannot parse value from last line in {path}: {last_line}")


def collect_metric_data(
    results_root: Path,
    models: list[str],
    steps: list[int],
    tasks: list[str],
    metrics: list[str],
    gen_subdir: str,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, dict[str, dict[int, str]]]]:
    data = {
        task: {metric: {model: [] for model in models} for metric in metrics}
        for task in tasks
    }
    picked_gen_dir = {
        task: {model: {} for model in models}
        for task in tasks
    }

    for task in tasks:
        for model in models:
            for step in steps:
                for metric in metrics:
                    metric_file = find_metric_file(
                        results_root=results_root,
                        model=model,
                        step=step,
                        task=task,
                        metric=metric,
                        gen_subdir=gen_subdir,
                    )
                    if metric_file is None:
                        print(f"[WARN] Missing file | task={task} metric={metric} model={model} step={step}")
                        data[task][metric][model].append(float("nan"))
                        continue

                    picked_gen_dir[task][model][step] = metric_file.parent.name
                    try:
                        value = parse_metric_value(metric_file, metric=metric)
                    except Exception as e:
                        print(f"[WARN] Parse failed | {metric_file} | {e}")
                        value = float("nan")
                    data[task][metric][model].append(value)

    return data, picked_gen_dir


def plot_metric(
    steps: list[int],
    model_to_values: dict[str, list[float]],
    task: str,
    metric_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for model, values in model_to_values.items():
        plt.plot(steps, values, marker="o", linewidth=2, label=model)

    plt.title(f"{task} - {metric_name.upper()} vs Training Step")
    plt.xlabel("Step")
    plt.ylabel(metric_name.upper())
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def format_value(value: float, precision: int) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{value:.{precision}f}"


def build_metric_table(
    task: str,
    metric_name: str,
    steps: list[int],
    models: list[str],
    model_to_values: dict[str, list[float]],
) -> str:
    precision = METRIC_PRECISION.get(metric_name, 5)
    headers = ["step"] + models
    rows: list[list[str]] = []

    for idx, step in enumerate(steps):
        row = [str(step)]
        for model in models:
            row.append(format_value(model_to_values[model][idx], precision))
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines = [f"[{task} | {metric_name.upper()}]"]
    lines.append("  ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * col_widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(col_widths[i]) for i in range(len(row))))
    return "\n".join(lines)


def export_txt_summary(
    output_path: Path,
    steps: list[int],
    models: list[str],
    tasks: list[str],
    metrics: list[str],
    metric_data: dict[str, dict[str, dict[str, list[float]]]],
    picked_gen_dir: dict[str, dict[str, dict[int, str]]],
) -> None:
    sections = []

    sections.append("[Picked Generation Subdir]")
    for task in tasks:
        sections.append(f"- task={task}")
        for model in models:
            parts = []
            for step in steps:
                gen_name = picked_gen_dir.get(task, {}).get(model, {}).get(step, "N/A")
                parts.append(f"{step}:{gen_name}")
            sections.append(f"  {model} -> " + ", ".join(parts))

    for task in tasks:
        for metric in metrics:
            sections.append(
                build_metric_table(
                    task=task,
                    metric_name=metric,
                    steps=steps,
                    models=models,
                    model_to_values=metric_data[task][metric],
                )
            )

    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]

    if not models:
        raise ValueError("No valid models provided.")
    if not steps:
        raise ValueError("No valid steps provided.")
    if not tasks:
        raise ValueError("No valid tasks provided.")
    if not metrics:
        raise ValueError("No valid metrics provided.")

    for metric in metrics:
        if metric not in METRIC_FILE_MAP:
            raise ValueError(f"Unsupported metric: {metric}. Supported: {sorted(METRIC_FILE_MAP.keys())}")

    metric_data, picked_gen_dir = collect_metric_data(
        results_root=results_root,
        models=models,
        steps=steps,
        tasks=tasks,
        metrics=metrics,
        gen_subdir=args.gen_subdir.strip(),
    )

    can_plot = True
    try:
        import matplotlib  # noqa: F401
    except Exception as e:
        can_plot = False
        print(f"[WARN] matplotlib unavailable, skip plotting: {e}")

    if can_plot:
        for task in tasks:
            for metric in metrics:
                out_file = output_dir / f"{task}_{metric}_line.png"
                plot_metric(
                    steps=steps,
                    model_to_values=metric_data[task][metric],
                    task=task,
                    metric_name=metric,
                    output_path=out_file,
                    dpi=args.dpi,
                )
                print(f"[OK] Saved: {out_file}")

    txt_path = output_dir / "metrics_summary.txt"
    export_txt_summary(
        output_path=txt_path,
        steps=steps,
        models=models,
        tasks=tasks,
        metrics=metrics,
        metric_data=metric_data,
        picked_gen_dir=picked_gen_dir,
    )
    print(f"[OK] Saved: {txt_path}")


if __name__ == "__main__":
    main()

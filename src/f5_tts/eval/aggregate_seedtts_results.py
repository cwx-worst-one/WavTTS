#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT_DEFAULT = REPO_ROOT / 'results'
OUT_DIR_DEFAULT = RESULTS_ROOT_DEFAULT / 'seedtts_eval_summary'


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate SeedTTS zh/en WER/SIM/UTMOS from results directory.')
    p.add_argument('--results-root', type=Path, default=RESULTS_ROOT_DEFAULT)
    p.add_argument('--out-dir', type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument('--tasks', nargs='+', default=['seedtts_test_zh', 'seedtts_test_en'])
    p.add_argument(
        '--include-exp-libritts',
        action='store_true',
        help='Include results/exp_libritts subtree in aggregation (default: exclude).',
    )
    return p.parse_args()


def parse_metric(file_path: Path, metric_upper: str) -> Optional[float]:
    if not file_path.exists() or file_path.stat().st_size == 0:
        return None
    txt = file_path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(rf'{metric_upper}\s*:\s*([0-9]+(?:\.[0-9]+)?)', txt)
    if not m:
        return None
    return float(m.group(1))


def parse_result_triplet(
    metric_file: Path,
    results_root: Path,
    wanted_tasks: set[str],
    include_exp_libritts: bool,
) -> Optional[Tuple[str, int, str, Path]]:
    try:
        rel = metric_file.relative_to(results_root)
    except ValueError:
        return None

    # Expected: <model>/<step>/<task>/<gen_dir>/_xxx_results.jsonl
    parts = rel.parts
    if len(parts) < 5:
        return None
    model, step_s, task = parts[0], parts[1], parts[2]
    gen_dir = results_root / model / step_s / task / parts[3]

    if not include_exp_libritts and model == 'exp_libritts':
        return None
    if task not in wanted_tasks:
        return None
    if not re.fullmatch(r'\d+', step_s):
        return None

    return model, int(step_s), task, gen_dir


def score_gen_dir_name(name: str) -> int:
    score = 0
    if name.startswith('seed0_'):
        score += 100
    if 'nfe32' in name:
        score += 50
    if 'cfg3.0' in name:
        score += 30
    if 'speed1.0' in name:
        score += 20
    if 'ss-1.0' in name:
        score += 10
    if 'no_vocoder' in name:
        score += 5
    return score


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    task_set = set(args.tasks)

    # Collect candidate generation dirs from actual metric files.
    candidates: Dict[Tuple[str, int, str, str], Dict] = {}
    metric_names = {'_wer_results.jsonl': 'wer', '_sim_results.jsonl': 'sim', '_utmos_results.jsonl': 'utmos'}
    for metric_file in args.results_root.rglob('*_results.jsonl'):
        metric_key = metric_names.get(metric_file.name)
        if metric_key is None:
            continue
        parsed = parse_result_triplet(metric_file, args.results_root, task_set, args.include_exp_libritts)
        if parsed is None:
            continue
        model, step, task, gen_dir = parsed
        key = (model, step, task, str(gen_dir))
        if key not in candidates:
            candidates[key] = {
                'model': model,
                'step': step,
                'task': task,
                'gen_wav_dir': str(gen_dir),
                'gen_tag': gen_dir.name,
                'wer': None,
                'sim': None,
                'utmos': None,
            }
        candidates[key][metric_key] = parse_metric(metric_file, metric_key.upper())

    rows = list(candidates.values())
    for r in rows:
        r['complete'] = int(r['wer'] is not None and r['sim'] is not None and r['utmos'] is not None)
    rows.sort(key=lambda x: (x['model'], x['step'], x['task'], x['gen_tag']))

    all_csv = args.out_dir / 'seedtts_metrics_all.csv'
    with all_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['model', 'step', 'task', 'gen_tag', 'wer', 'sim', 'utmos', 'complete', 'gen_wav_dir'],
        )
        w.writeheader()
        w.writerows(rows)

    # markdown summary
    md_path = args.out_dir / 'seedtts_metrics_all.md'
    with md_path.open('w', encoding='utf-8') as f:
        f.write('# SeedTTS Metrics (scanned from results)\n\n')
        f.write('| model | step | task | gen_tag | WER | SIM | UTMOS | complete |\n')
        f.write('|---|---:|---|---|---:|---:|---:|---:|\n')
        for r in rows:
            def fmt(v):
                return '' if v is None else f'{v:.5f}'
            f.write(
                f"| {r['model']} | {r['step']} | {r['task']} | {r['gen_tag']} | {fmt(r['wer'])} | {fmt(r['sim'])} | {fmt(r['utmos'])} | {r['complete']} |\n"
            )

    # Keep one preferred run per (model, step, task) for pivot tables/plots.
    chosen_by_triplet: Dict[Tuple[str, int, str], Dict] = {}
    dup_triplets = 0
    for r in rows:
        k = (r['model'], r['step'], r['task'])
        if k not in chosen_by_triplet:
            chosen_by_triplet[k] = r
            continue
        dup_triplets += 1
        old = chosen_by_triplet[k]
        old_score = score_gen_dir_name(old['gen_tag'])
        new_score = score_gen_dir_name(r['gen_tag'])
        if (new_score, r['gen_tag']) > (old_score, old['gen_tag']):
            chosen_by_triplet[k] = r
    if dup_triplets > 0:
        print(f'[WARN] found multiple gen dirs for some (model,step,task); kept best by heuristic. duplicates={dup_triplets}')

    chosen_rows = list(chosen_by_triplet.values())
    chosen_rows.sort(key=lambda x: (x['model'], x['step'], x['task']))

    chosen_csv = args.out_dir / 'seedtts_metrics_selected.csv'
    with chosen_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['model', 'step', 'task', 'gen_tag', 'wer', 'sim', 'utmos', 'complete', 'gen_wav_dir'],
        )
        w.writeheader()
        w.writerows(chosen_rows)

    # per-task+metric pivot CSVs (based on selected rows)
    metrics = ['wer', 'sim', 'utmos']
    tasks = args.tasks

    models = sorted(set(r['model'] for r in chosen_rows))
    steps = sorted(set(r['step'] for r in chosen_rows))

    for task in tasks:
        for metric in metrics:
            out_csv = args.out_dir / f'table_{task}_{metric}.csv'
            with out_csv.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['step'] + models)
                for step in steps:
                    line = [step]
                    for m in models:
                        v = None
                        for r in chosen_rows:
                            if r['model'] == m and r['step'] == step and r['task'] == task:
                                v = r[metric]
                                break
                        line.append('' if v is None else f'{v:.6f}')
                    writer.writerow(line)

    # plot six figures
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[WARN] matplotlib unavailable, skip plotting: {e}')
        print(f'[INFO] table files are ready in: {args.out_dir}')
        return 0

    for task in tasks:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            any_line = False
            for m in models:
                xs = []
                ys = []
                for step in steps:
                    v = None
                    for r in chosen_rows:
                        if r['model'] == m and r['step'] == step and r['task'] == task:
                            v = r[metric]
                            break
                    if v is not None:
                        xs.append(step)
                        ys.append(v)
                if xs:
                    any_line = True
                    plt.plot(xs, ys, marker='o', linewidth=1.5, label=m)

            plt.title(f'{task} - {metric.upper()} vs Step')
            plt.xlabel('Step')
            plt.ylabel(metric.upper())
            plt.grid(True, linestyle='--', alpha=0.35)
            if any_line:
                plt.legend(fontsize=7, loc='best')
            plt.tight_layout()
            out_png = plot_dir / f'curve_{task}_{metric}.png'
            plt.savefig(out_png, dpi=200)
            plt.close()

    print(f'[INFO] Aggregation done: {all_csv}')
    print(f'[INFO] Selected rows CSV: {chosen_csv}')
    print(f'[INFO] Markdown table: {md_path}')
    print(f'[INFO] Plot dir: {plot_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

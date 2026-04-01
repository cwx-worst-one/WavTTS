#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
EMILIA_ROOT_DEFAULT = Path('/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia')
RESULTS_ROOT_DEFAULT = REPO_ROOT / 'results'
CONFIG_ROOT_DEFAULT = REPO_ROOT / 'src' / 'f5_tts' / 'configs'


@dataclass
class Combo:
    model_dir: Path
    exp_name: str
    ckpt_step: int
    ckpt_path: Path
    task: str
    mel_spec_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Overnight SeedTTS infer+eval runner for Emilia ckpts (10w steps).')
    parser.add_argument('--emilia-root', type=Path, default=EMILIA_ROOT_DEFAULT)
    parser.add_argument('--results-root', type=Path, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT_DEFAULT)
    parser.add_argument('--tasks', nargs='+', default=['seedtts_test_zh', 'seedtts_test_en'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nfe-step', type=int, default=32)
    parser.add_argument('--cfg-strength', type=float, default=3.0)
    parser.add_argument('--swaysampling', type=float, default=-1.0)
    parser.add_argument('--eval-gpus', type=str, default='[0,1,2,3,4,5,6,7]')
    parser.add_argument('--master-port', type=int, default=53721)
    parser.add_argument('--cfg-scale-interval-min', type=float, default=0.0)
    parser.add_argument('--cfg-scale-interval-max', type=float, default=1.0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--load-dtype', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--infer-dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--force', action='store_true', help='Force rerun infer+eval even if result files exist.')
    parser.add_argument('--continue-on-error', action='store_true', default=True)
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument('--max-combos', type=int, default=0, help='Debug only; 0 means no limit.')
    parser.add_argument('--dry-run', action='store_true', help='Only print planned combos and exit.')
    parser.add_argument('--summary-dir', type=Path, default=RESULTS_ROOT_DEFAULT / 'seedtts_eval_summary')
    return parser.parse_args()


def load_mel_spec_type(config_path: Path) -> Optional[str]:
    txt = config_path.read_text(encoding='utf-8')
    m = re.search(r'^\s*mel_spec_type\s*:\s*([A-Za-z0-9_\-\.]+)\b', txt, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def discover_combos(args: argparse.Namespace) -> List[Combo]:
    combos: List[Combo] = []
    for model_dir in sorted(args.emilia_root.iterdir()):
        if not model_dir.is_dir():
            continue
        exp_name = model_dir.name.split('-emilia-')[0]
        config_path = args.config_root / f'{exp_name}.yaml'
        if not config_path.exists():
            print(f'[WARN] Missing config for {exp_name}, skip.', flush=True)
            continue

        mel_spec_type = load_mel_spec_type(config_path)
        if mel_spec_type is None:
            print(f'[WARN] Cannot parse mel_spec_type for {exp_name}, skip.', flush=True)
            continue

        ckpt_dir = model_dir / 'ckpts'
        if not ckpt_dir.exists():
            continue

        steps = []
        for p in ckpt_dir.glob('model_*.pt'):
            if p.name == 'model_last.pt':
                continue
            m = re.match(r'model_(\d+)\.pt$', p.name)
            if not m:
                continue
            step = int(m.group(1))
            if step % 200000 == 0:
                steps.append(step)

        for step in sorted(set(steps)):
            ckpt_path = ckpt_dir / f'model_{step}.pt'
            for task in args.tasks:
                combos.append(
                    Combo(
                        model_dir=model_dir,
                        exp_name=exp_name,
                        ckpt_step=step,
                        ckpt_path=ckpt_path,
                        task=task,
                        mel_spec_type=mel_spec_type,
                    )
                )

    if args.max_combos > 0:
        combos = combos[: args.max_combos]
    return combos


def metric_files(gen_wav_dir: Path):
    return {
        'wer': gen_wav_dir / '_wer_results.jsonl',
        'sim': gen_wav_dir / '_sim_results.jsonl',
        'utmos': gen_wav_dir / '_utmos_results.jsonl',
    }


def is_complete(gen_wav_dir: Path) -> bool:
    files = metric_files(gen_wav_dir)
    return all(p.exists() and p.stat().st_size > 0 for p in files.values())


def run_cmd(cmd: List[str], env: dict) -> int:
    print('[CMD]', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
    p.wait()
    return p.returncode


def run_combo(combo: Combo, args: argparse.Namespace, env: dict) -> dict:
    # keep result path consistent with eval_infer_batch.py
    gen_wav_dir = (
        args.results_root
        / combo.exp_name
        / str(combo.ckpt_step)
        / combo.task
        / (
            f'seed{args.seed}_euler_nfe{args.nfe_step}_{combo.mel_spec_type}_ss{args.swaysampling}'
            f'_cfg{args.cfg_strength}_speed1.0_load-{args.load_dtype}_infer-{args.infer_dtype}'
            f'_cfgitv{args.cfg_scale_interval_min}-{args.cfg_scale_interval_max}'
        )
    )
    status = {
        'exp_name': combo.exp_name,
        'step': combo.ckpt_step,
        'task': combo.task,
        'gen_wav_dir': str(gen_wav_dir),
        'infer': 'skipped',
        'wer': 'skipped',
        'sim': 'skipped',
        'utmos': 'skipped',
        'ok': True,
        'error': '',
    }

    completed = is_complete(gen_wav_dir)
    if completed and not args.force and not args.eval_only:
        status['infer'] = 'done_exists'
        status['wer'] = 'done_exists'
        status['sim'] = 'done_exists'
        status['utmos'] = 'done_exists'
        return status

    local_flag = ['--local'] if args.local else []

    if not args.eval_only:
        infer_cmd = [
            'accelerate', 'launch',
            '--main_process_port', str(args.master_port),
            'src/f5_tts/eval/eval_infer_batch.py',
            '-s', str(args.seed),
            '-n', combo.exp_name,
            '-t', combo.task,
            '-c', str(combo.ckpt_step),
            '--ckpt_path', str(combo.ckpt_path),
            '--nfe_step', str(args.nfe_step),
            '--swaysampling', str(args.swaysampling),
            '--cfg_strength', str(args.cfg_strength),
            '--cfg_scale_interval_min', str(args.cfg_scale_interval_min),
            '--cfg_scale_interval_max', str(args.cfg_scale_interval_max),
            '--load_dtype', args.load_dtype,
            '--infer_dtype', args.infer_dtype,
        ] + local_flag
        rc = run_cmd(infer_cmd, env)
        status['infer'] = 'ok' if rc == 0 else f'fail({rc})'
        if rc != 0:
            status['ok'] = False
            status['error'] = f'infer failed rc={rc}'
            return status

    if args.infer_only:
        return status

    if not gen_wav_dir.exists():
        status['ok'] = False
        status['error'] = f'gen_wav_dir missing: {gen_wav_dir}'
        return status

    eval_lang = 'zh' if combo.task == 'seedtts_test_zh' else 'en'

    files = metric_files(gen_wav_dir)

    if args.force or not (files['wer'].exists() and files['wer'].stat().st_size > 0):
        wer_cmd = [
            'python', 'src/f5_tts/eval/eval_seedtts_testset.py',
            '-e', 'wer',
            '-l', eval_lang,
            '-g', str(gen_wav_dir),
            '-n', args.eval_gpus,
        ] + local_flag
        rc = run_cmd(wer_cmd, env)
        status['wer'] = 'ok' if rc == 0 else f'fail({rc})'
        if rc != 0:
            status['ok'] = False
            status['error'] = f'wer failed rc={rc}'
            return status
    else:
        status['wer'] = 'done_exists'

    if args.force or not (files['sim'].exists() and files['sim'].stat().st_size > 0):
        sim_cmd = [
            'python', 'src/f5_tts/eval/eval_seedtts_testset.py',
            '-e', 'sim',
            '-l', eval_lang,
            '-g', str(gen_wav_dir),
            '-n', args.eval_gpus,
        ] + local_flag
        rc = run_cmd(sim_cmd, env)
        status['sim'] = 'ok' if rc == 0 else f'fail({rc})'
        if rc != 0:
            status['ok'] = False
            status['error'] = f'sim failed rc={rc}'
            return status
    else:
        status['sim'] = 'done_exists'

    if args.force or not (files['utmos'].exists() and files['utmos'].stat().st_size > 0):
        utmos_cmd = ['python', 'src/f5_tts/eval/eval_utmos.py', '--audio_dir', str(gen_wav_dir)]
        rc = run_cmd(utmos_cmd, env)
        status['utmos'] = 'ok' if rc == 0 else f'fail({rc})'
        if rc != 0:
            status['ok'] = False
            status['error'] = f'utmos failed rc={rc}'
            return status
    else:
        status['utmos'] = 'done_exists'

    return status


def main() -> int:
    args = parse_args()
    if args.stop_on_error:
        args.continue_on_error = False

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.summary_dir / 'overnight_progress.jsonl'

    env = os.environ.copy()
    env.setdefault('PYTHONWARNINGS', 'ignore::UserWarning,ignore::FutureWarning')
    env.setdefault('MASTER_ADDR', '127.0.0.1')
    env.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    combos = discover_combos(args)
    total = len(combos)
    done = 0
    failed = 0

    print(f'[INFO] total combos={total}', flush=True)
    if args.dry_run:
        for i, c in enumerate(combos, start=1):
            print(f'[PLAN] ({i}/{total}) {c.exp_name} step={c.ckpt_step} task={c.task}')
        return 0

    with progress_path.open('a', encoding='utf-8') as f:
        for i, combo in enumerate(combos, start=1):
            print(
                f'\n[INFO] ({i}/{total}) {combo.exp_name} step={combo.ckpt_step} task={combo.task}',
                flush=True,
            )
            status = run_combo(combo, args, env)
            if status['ok']:
                done += 1
            else:
                failed += 1
                print(f"[ERROR] {status['error']}", flush=True)
                if not args.continue_on_error:
                    f.write(json.dumps(status, ensure_ascii=False) + '\n')
                    break
            f.write(json.dumps(status, ensure_ascii=False) + '\n')
            f.flush()

    print(f'\n[INFO] finished. success={done}, failed={failed}, total={total}', flush=True)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

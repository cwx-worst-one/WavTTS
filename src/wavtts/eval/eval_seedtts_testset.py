# Evaluate with Seed-TTS testset

import argparse
import ast
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from wavtts.eval.utils_eval import get_seed_tts_test, run_asr_wer, run_sim


rel_path = str(files("wavtts").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en", choices=["zh", "zh_hard", "en"])
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument(
        "-n", "--gpu_nums", type=str, default="8", help="Number of GPUs to use (e.g., 8) or GPU list (e.g., [0,1,2,3])"
    )
    parser.add_argument(
        "--wavlm_ckpt_dir",
        type=str,
        default="",
        help="Path to wavlm_large_finetune.pth for SIM evaluation.",
    )
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def parse_gpu_nums(gpu_nums_str):
    try:
        if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
            gpu_list = ast.literal_eval(gpu_nums_str)
            if isinstance(gpu_list, list):
                return gpu_list
        return list(range(int(gpu_nums_str)))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            f"Invalid GPU specification: {gpu_nums_str}. Use a number (e.g., 8) or a list (e.g., [0,1,2,3])"
        )


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    asr_lang = "zh" if lang == "zh_hard" else lang
    gen_wav_dir = args.gen_wav_dir
    if lang == "zh_hard":
        metalst = rel_path + "/data/seedtts_testset/zh/hardcase.lst"
    else:
        metalst = rel_path + f"/data/seedtts_testset/{lang}/meta.lst"  # seed-tts testset

    # NOTE. paraformer-zh result will be slightly different according to the number of gpus, cuz batchsize is different
    #       zh 1.254 seems a result of 4 workers wer_seed_tts
    gpus = parse_gpu_nums(args.gpu_nums)
    test_set = get_seed_tts_test(metalst, gen_wav_dir, gpus)

    wavlm_ckpt_dir = args.wavlm_ckpt_dir
    if eval_task == "sim" and not wavlm_ckpt_dir:
        raise ValueError(
            "--wavlm_ckpt_dir is required for SIM evaluation. "
            "Download wavlm_large_finetune.pth and pass its path."
        )

    local = args.local
    if local:  # use local custom checkpoint dir
        if asr_lang == "zh":
            asr_ckpt_dir = "../checkpoints/funasr"  # paraformer-zh dir under funasr
        elif asr_lang == "en":
            asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, asr_lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                full_results.extend(r)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
    with open(result_path, "w") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()

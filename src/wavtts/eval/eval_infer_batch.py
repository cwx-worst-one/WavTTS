import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from wavtts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from wavtts.infer.utils_infer import load_checkpoint
from wavtts.model import CFM
from wavtts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


target_rms = 0.1    # 0.1, 0.12


rel_path = str(files("wavtts").joinpath("../../"))
DEFAULT_CKPT_FILE = "hf://worstchan/WavTTS/model_1200000.pt"
DEFAULT_VOCAB_FILE = str(files("wavtts").joinpath("infer/examples/vocab.txt"))


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, type=int)

    parser.add_argument("-nfe", "--nfe_step", default=50, type=int)
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)
    parser.add_argument("--timestep_mapping", default="power", choices=["uniform", "sway_sampling", "power"])
    parser.add_argument("--timestep_power", default=2.0, type=float)
    parser.add_argument("--shift", default=3.0, type=float)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument(
        "-p",
        "--librispeech_test_clean_path",
        default=f"{rel_path}/data/LibriSpeech/test-clean",
        type=str,
    )
    parser.add_argument(
        "--ljspeech_inset_meta",
        default=f"{rel_path}/data/ljspeech_inset_test_9s.lst",
        type=str,
        help="Metadata list for ljspeech_inset_test_9s. Format: ref_utt ref_dur ref_txt gen_utt gen_dur gen_txt.",
    )
    parser.add_argument(
        "--ljspeech_wav_dir",
        default=f"{rel_path}/data/ljspeech/LJSpeech-1.1/wavs",
        type=str,
        help="Directory containing original LJSpeech wavs. Only used when truth duration is enabled.",
    )

    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT_FILE, type=str)
    parser.add_argument(
        "--vocab_file",
        default=DEFAULT_VOCAB_FILE,
        type=str,
        help="Path to vocab.txt used to build the model tokenizer.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Optional directory to save generated wavs. Defaults to the standard results/... directory.",
    )
    parser.add_argument(
        "--fixed_prompt_wav",
        default=None,
        type=str,
        help="Use one fixed prompt wav for all samples instead of each test sample's prompt wav.",
    )
    parser.add_argument(
        "--fixed_prompt_text",
        default=None,
        type=str,
        help="Text corresponding to --fixed_prompt_wav. Required when --fixed_prompt_wav is set.",
    )
    parser.add_argument(
        "--result_expname",
        default=None,
        type=str,
        help="Optional result directory name override. Defaults to --expname.",
    )
    parser.add_argument("--cfg_strength", default=3.0, type=float)
    parser.add_argument(
        "--load_dtype",
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Model checkpoint load precision. Default fp32 for more stable weight loading.",
    )
    parser.add_argument(
        "--infer_dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Inference precision: bf16/fp16 for autocast mixed precision, fp32 to disable autocast.",
    )

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    result_exp_name = args.result_expname or exp_name
    ckpt_step = args.ckptstep

    nfe_step = args.nfe_step
    sway_sampling_coef = args.swaysampling
    timestep_mapping = args.timestep_mapping
    timestep_power = args.timestep_power
    shift = args.shift

    if timestep_mapping == "power" and timestep_power is None:
        raise ValueError("--timestep_power must be provided when --timestep_mapping power is used")

    testset = args.testset
    load_dtype_name = args.load_dtype
    infer_dtype_name = args.infer_dtype

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = args.cfg_strength
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    amp_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    autocast_dtype = amp_dtype_map.get(infer_dtype_name)

    model_cfg = OmegaConf.load(str(files("wavtts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"wavtts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    target_sample_rate = model_cfg.model.waveform.target_sample_rate
    wav_frame_len = model_cfg.model.waveform.wav_frame_len

    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = args.librispeech_test_clean_path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_zh_hard":
        metalst = rel_path + "/data/seedtts_testset/zh/hardcase.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "ljspeech_inset_test_9s":
        metalst = args.ljspeech_inset_meta
        if not os.path.exists(metalst):
            raise FileNotFoundError(f"LJSpeech inset metadata not found: {metalst}")

        metainfo = []
        with open(metalst, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                ref_utt, _ref_dur, ref_txt, gen_utt, _gen_dur, gen_txt = parts[:6]
                ref_wav = os.path.join(args.ljspeech_wav_dir, ref_utt + ".wav")
                gen_wav = os.path.join(args.ljspeech_wav_dir, gen_utt + ".wav")
                metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    else:
        raise ValueError(f"Unknown testset: {testset}")

    if (args.fixed_prompt_wav is None) != (args.fixed_prompt_text is None):
        raise ValueError("--fixed_prompt_wav and --fixed_prompt_text must be set together")

    if args.fixed_prompt_wav is not None:
        if not os.path.exists(args.fixed_prompt_wav):
            raise FileNotFoundError(f"Fixed prompt wav not found: {args.fixed_prompt_wav}")
        fixed_prompt_wav = args.fixed_prompt_wav
        fixed_prompt_text = args.fixed_prompt_text
        metainfo = [
            (utt, fixed_prompt_text, fixed_prompt_wav, gt_text, gt_wav)
            for utt, _prompt_text, _prompt_wav, gt_text, gt_wav in metainfo
        ]
        

    # path to save genereted wavs
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = (
            f"{rel_path}/"
            f"results/{result_exp_name}/{ckpt_step}/{testset}/"
            f"seed{seed}_nfe{nfe_step}_wav"
            f"{'_uniform' if timestep_mapping == 'uniform' else ''}"
            f"{f'_ss{sway_sampling_coef}' if timestep_mapping == 'sway_sampling' and sway_sampling_coef else ''}"
            f"{f'_power{timestep_power}' if timestep_mapping == 'power' else ''}"
            f"{f'_shift{shift}' if shift != 1.0 else ''}"
            f"_cfg{cfg_strength}"
            f"{'_gt-dur' if use_truth_duration else ''}"
            f"{'_no-ref-audio' if no_ref_audio else ''}"
        )

    # -------------------------------------------------#
    wav_frame_len = int(model_cfg.model.waveform.get("wav_frame_len", 160))

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
        wav_frame_len=wav_frame_len,
    )

    # Tokenizer
    if args.vocab_file:
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_file, "custom")
    else:
        vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # waveform geometry
    waveform_kwargs = model_cfg.model.waveform
    if waveform_kwargs is None:
        waveform_kwargs = dict(
            wav_frame_len=wav_frame_len,
            target_sample_rate=target_sample_rate,
        )

    # CFM kwargs
    cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}

    # Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, wav_frame_len=wav_frame_len),
        waveform_kwargs=waveform_kwargs,
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    ).to(device)

    # ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}"
    # if os.path.exists(ckpt_prefix + ".pt"):
    #     ckpt_path = ckpt_prefix + ".pt"
    # elif os.path.exists(ckpt_prefix + ".safetensors"):
    #     ckpt_path = ckpt_prefix + ".safetensors"
    # else:
    #     print("Loading from self-organized training checkpoints rather than released pretrained.")
    #     ckpt_prefix = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}"
    #     if os.path.exists(ckpt_prefix + ".pt"):
    #         ckpt_path = ckpt_prefix + ".pt"
    #     elif os.path.exists(ckpt_prefix + ".safetensors"):
    #         ckpt_path = ckpt_prefix + ".safetensors"
    #     else:
    #         raise ValueError("The checkpoint does not exist or cannot be found in given location.")
    ckpt_path = args.ckpt_path or DEFAULT_CKPT_FILE
    if ckpt_path.startswith(("hf://", "http://", "https://")):
        ckpt_path = str(cached_path(ckpt_path))

    load_dtype = amp_dtype_map.get(load_dtype_name, torch.float32)
    model = load_checkpoint(model, ckpt_path, device, dtype=load_dtype, use_ema=True)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_wavs, ref_wav_lens, total_wav_lens, final_text_list = prompt
            ref_wavs = ref_wavs.to(device)
            ref_wav_lens = torch.tensor(ref_wav_lens, dtype=torch.long).to(device)
            total_wav_lens = torch.tensor(total_wav_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    generated, _ = model.sample(
                        cond=ref_wavs,
                        text=final_text_list,
                        duration=total_wav_lens,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        timestep_mapping=timestep_mapping,
                        timestep_power=timestep_power,
                        shift=shift,
                        use_epss=timestep_mapping == "sway_sampling",
                        no_ref_audio=no_ref_audio,
                        seed=seed,
                    )
                    # Final result
                    for i, gen in enumerate(generated):
                        start_idx = ref_wav_lens[i].item()
                        end_idx = total_wav_lens[i].item()
                        generated_wave = gen[start_idx:end_idx].cpu()

                        if ref_rms_list[i] < target_rms:
                            generated_wave = generated_wave * ref_rms_list[i] / target_rms

                        wave_to_save = generated_wave.to(torch.float32).cpu()

                        if wave_to_save.ndim == 1:
                            wave_to_save = wave_to_save.unsqueeze(0)
                        elif wave_to_save.ndim > 2:
                            wave_to_save = wave_to_save.squeeze()
                            if wave_to_save.ndim == 1:
                                wave_to_save = wave_to_save.unsqueeze(0)

                        torchaudio.save(f"{output_dir}/{utts[i]}.wav", wave_to_save, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()

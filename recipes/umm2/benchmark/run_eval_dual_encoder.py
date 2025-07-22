import torch
assert torch.cuda.is_available()
# from recipes.umm.requires.model_initializer import init_stage3, init_stage3_rvq
import matplotlib.pyplot as plt
from recipes.umm.tests.datasets.test_dataset import set_random_seed
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.benchmark.evaluator import TokenEvaluator
import numpy as np
from time import time
import torchaudio
from torchaudio.transforms import Resample
import torch, os
import sys, importlib
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.scripts.test_stage3_wav2tokens_RVQ import ModelLoader, init_model



@torch.no_grad()
def evaluate_audio_slice(it, 
                        evaluator: TokenEvaluator, 
                        model_name = "default",
                        text_tokenizer=None,
                        max_i=1000):
    
    # slice_modes = ["full", "even", "max"]
    slice_modes = ["section"]
    slice_loss_dicts = {k: {} for k in slice_modes}

    i = 0
    for batch in it:
        if i == max_i:
            break
        input_batch = {
            "audio": batch["audio"].to("cuda"), 
            "audio_padding_mask": None,
            "token_padding_mask": None,
            "token": batch["token"].to("cuda"),
            "slice": batch["slice"],
            }
        
        for slice_mode in slice_modes:
            tokens, sliced_audios, sliced_texts = evaluator.get_tokens(input_batch["audio"], slice_mode=slice_mode, 
                                                         slice_info=input_batch["slice"], chunk_dur=60)
            slice_loss_dict = {}
            for a in range(len(sliced_audios)):
                if sliced_texts is not None:
                    text = text_tokenizer(
                        sliced_texts[a], add_special_tokens=False, return_tensors="pt")
                    slice_text_token = text["input_ids"].squeeze(dim=0)
                else:
                    slice_text_token = input_batch["token"]
                slice_input_batch = {"audio": sliced_audios[a], "token": slice_text_token}    # max/even的ctc算不了,因为无法正确截断text,只有section/full可以算
                slice_input_dict = evaluator.pl_module.prepare_feature(slice_input_batch)
                slice_output_dict = evaluator.model(slice_input_dict)
                this_loss_dict = evaluator.get_loss_dict(slice_input_dict, slice_output_dict)
                for key in this_loss_dict.keys():
                    if key not in slice_loss_dict.keys():
                        slice_loss_dict[key] = 0
                    slice_loss_dict[key] += this_loss_dict[key].item()
            print(slice_mode)
            for key in slice_loss_dict.keys():
                slice_loss_dict[key] = slice_loss_dict[key] / len(sliced_audios)
            print(slice_loss_dict)

            for key in slice_loss_dict.keys():
                if key not in slice_loss_dicts[slice_mode].keys():
                    slice_loss_dicts[slice_mode][key] = 0
                slice_loss_dicts[slice_mode][key] += slice_loss_dict[key]
        i += 1
        
    print("[slice loss dict]")
    for mode in slice_loss_dicts.keys():
        for key2 in slice_loss_dicts[mode].keys():
            slice_loss_dicts[mode][key2] = slice_loss_dicts[mode][key2] / max_i
        print(mode, slice_loss_dicts[mode])

    return slice_loss_dicts

def print_stat(i, evaluator, loss_dicts, locality_dict, ctc_wers, tiktok, tasks=["loss", "locality", "ctc_wer"]):
    print(f"temporary statistics [{i}]:")
    code_rate = evaluator.compute_code_rate()
    print(f"\n[code rate]\n{code_rate}\n")
    
    _ld = {}
    if "loss" in tasks:
        for key in loss_dicts.keys():
            _ld[key] = loss_dicts[key] / i
        print("\n[loss dict]\n", _ld)

    _ld = {}
    if "locality" in tasks:
        print("\n[locality]")
        for key in locality_dict.keys():
            _ld[key] = torch.tensor(locality_dict[key]).mean(0)  # [n_slice, n_locality_metrics]
            print(key, _ld[key])

    if "ctc_wer" in tasks:
        print("\n[ctc wer]")
        _cd = torch.tensor(ctc_wers)
        print("wer | ins | del | sub ", _cd.mean(0))
        
    print("[time]")
    print(tiktok)


def get_gt_wav(this_output_dict, dt_config):
    est_spec = this_output_dict["spec_out"].cpu().reshape(1, dt_config.num_bins, 2, -1).float()
    est_mag = (est_spec[:, :, 0] ** 2 + est_spec[:, :, 1] ** 2).pow(0.5)
    est_spec_with_gt_phase = est_mag * torch.exp(1j * torch.angle(this_output_dict["spec"].cpu()))
    est_wav_with_gt_phase = torch.istft(est_spec_with_gt_phase, n_fft=dt_config.win, hop_length=dt_config.stride, 
                            window=torch.hann_window(dt_config.win))
    return est_wav_with_gt_phase

@torch.no_grad()
def evaluate(it, 
             f,
             evaluator: TokenEvaluator, 
             model_name = "default",
             out_dir=None,
             max_i=1000, 
             sample_rate=44100,):
    loss_dicts, locality_dict = {}, {}
    nsample, mean_sample_dur, time_cnt, i = 0, 0, 0, 0
    ctc_wers = []
    evaluator.h = evaluator.model.stages[1].dt_config.rvq
    dt_config = evaluator.model.stages[1].dt_config
    evaluator.reset_code_count()
    resampler = Resample(sample_rate, 24000)

    for batch in it:
        if i == max_i:
            break
        
        input_batch = {
            "audio_"+str(sample_rate): batch["audio"].to("cuda"), 
            "audio": resampler(batch["audio"]).to("cuda"), 
            "audio_padding_mask": None,
            "token_padding_mask": None,
            "token": batch["token"].to("cuda"),
            "slice": batch["slice"],
            "inference_R": 0
            }
        ct = time()
        this_output_dict = evaluator.model(input_batch)
        tokens = this_output_dict["vq_ids"]
        evaluator.update_code_usage(tokens)

        time_cnt += time() - ct
        sample_dur = input_batch['audio'].shape[-1] / evaluator.sample_rate
        nsample += tokens.shape[0]
        mean_sample_dur += (sample_dur * tokens.shape[0])
        print("=" * 50)
        print(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}")
        f.writelines("=" * 50 + "\n")
        f.writelines(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}\n")

        i += 1
        tiktok = [time_cnt, nsample, mean_sample_dur / nsample]
        
        torchaudio.save(f"{out_dir}/test_out_R0_{i}.wav", this_output_dict["audio_recon_"+str(sample_rate)].cpu(), sample_rate)
        torchaudio.save(f"{out_dir}/test{i}.wav", input_batch["audio_"+str(sample_rate)].cpu(), sample_rate)
        # use ground truth phase
        est_wav_with_gt_phase = get_gt_wav(this_output_dict, dt_config)
        torchaudio.save(f"{out_dir}/test_gt_R0_{i}.wav", est_wav_with_gt_phase, sample_rate)

        input_batch["inference_R"] = None
        this_output_dict = evaluator.model(input_batch)
        torchaudio.save(f"{out_dir}/test_out_R4_{i}.wav", this_output_dict["audio_recon_"+str(sample_rate)].cpu(), sample_rate)
        # use ground truth phase
        est_wav_with_gt_phase = get_gt_wav(this_output_dict, dt_config)
        torchaudio.save(f"{out_dir}/test_gt_R4_{i}.wav", est_wav_with_gt_phase, sample_rate)



    output_dict = {}
    code_rate = evaluator.compute_code_rate()
    print(f"[code rate]\n{code_rate}\n")
    if out_dir is None:
        out_dir = "./recipes/umm/tests/eval_tokenizers"
    code_distribution = evaluator.plot_token_distribution(f"{out_dir}/{model_name}.png")
    output_dict["code_rate"] = code_rate

    # if "locality" in tasks:
    #     print("[locality]")
    #     for key in locality_dict.keys():
    #         locality_dict[key] = torch.tensor(locality_dict[key]).mean(0)  # [n_slice, n_locality_metrics]
    #         print(key, locality_dict[key])
    #     output_dict["locality"] = locality_dict

    output_dict["time_count"] = [time_cnt, nsample, mean_sample_dur / nsample]
    print("[time]")
    print(time_cnt)
    print(nsample, mean_sample_dur / nsample)
    return output_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_cls", type=str, default="recipes.umm2.modules.stages.stage3.Stage3", 
    help="training pytorch lightning module")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--nsample", type=int, default=None, help="number of evaluated samples (full)")
    parser.add_argument("--seed", type=int, default=20250101, help="seed")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    out_dir = args.out_dir
    max_i = args.nsample
    pl_module_string = args.model_cls
    
    cache_dir = "./.module_cache/umm2/"
    if max_i is None:
        max_i = 500
    if out_dir is None:
        out_dir = cache_dir
    os.makedirs(out_dir, exist_ok=True)

    local_rank = 0
    num_workers = 4
    batch_size = 1
    min_duration = 60
    max_duration = 600  # full song
    device = torch.device(f"cuda:{local_rank}")
    set_random_seed(args.seed)

    # ===== load model =====
    model = init_model(model_path, cache_dir, device, pl_module_string)
    pl_module = model["pl_module"]
    config = pl_module.model.stages[0].config   # same configs shared between stages
    try:
        text_tokenizer = pl_module.hparams.extra_params["tokenizer"]
    except:
        text_tokenizer = "bert-base-multilingual-uncased"
    sample_rate = 32000 #44100
    frame_rate = config.frame_rate

    # ===== load evaluator =====
    evaluator = TokenEvaluator(pl_module, config,
                                model_type=model_name, 
                                segment_size=60, 
                                slice_length=[15, 30, 60])

    # ===== build dataset =====
    dataset = EvalDataset(
        data_id=7295,
        url_pattern=None,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        num_worker=num_workers,
        batch_size=batch_size,
        min_duration=min_duration,
        max_duration=max_duration,
        tokenizer=text_tokenizer,
    )
    dataloader = dataset.dataloader    

    f = open(f"{out_dir}/{model_name}.rlt", "w")
    f.write(f"{model_name}: {args.model_path}\n")

    eval_output_dict = evaluate(dataloader, 
                                f, evaluator, 
                                max_i=max_i, 
                                out_dir=out_dir, 
                                model_name=model_name,
                                sample_rate=sample_rate) 
    f.write("="*50 + "\n")
    for k, v in eval_output_dict.items():
        f.write(f"{k}\n{v}" + "\n")
    f.close()

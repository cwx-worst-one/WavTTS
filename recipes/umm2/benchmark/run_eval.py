import torch
assert torch.cuda.is_available()
# from recipes.umm.requires.model_initializer import init_stage3, init_stage3_rvq
import matplotlib.pyplot as plt
from recipes.umm.tests.datasets.test_dataset import set_random_seed
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.benchmark.evaluator import TokenEvaluator
import numpy as np
from time import time

import torch, os, ast
from recipes.umm2.loaders.base import BaseModelLoader, local_zero_first
import sys, importlib
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.scripts.test_stage3_wav2tokens_RVQ import ModelLoader, init_model


@torch.no_grad()
def evaluate_audio_slice(it, 
                        f,
                        evaluator: TokenEvaluator, 
                        tasks=["loss", "locality", "ctc_wer"],
                        model_name = "default",
                        out_dir=None, 
                        max_i=1000,
                        slice_mode='even',
                        chunk_dur=45,
                        text_tokenizer=None
                        ):
    nsample, mean_sample_dur, time_cnt, i = 0, 0, 0, 0
    slice_loss_dicts, slice_locality_dicts, slice_repetition_dicts, slice_ctc_wers = {}, {}, {}, []
    slice_loss_dicts_vocal, slice_loss_dicts_inst = {}, {}
    evaluator.reset_code_count()

    i, inst_i, vocal_i = 0, 0, 0
    for batch in it:
        # if batch["no_lyric_flag"]:
        #     continue
        if i == max_i:
            break

        input_ids = batch["token"].to("cuda")
        attention_mask = batch["token_padding_mask"].to("cuda")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_batch = {
            "audio": batch["audio"].to("cuda"), 
            "audio_length": batch['audio_length'].to("cuda"),
            "token": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "slice": batch["slice"],
        }
        
        tokens, sliced_audios, sliced_texts = evaluator.get_tokens(input_batch["audio"], slice_mode=slice_mode, 
                                                                  slice_info=input_batch["slice"], chunk_dur=chunk_dur)
        ct = time()
        time_cnt += time() - ct
        evaluator.update_code_usage(tokens)

        txt_f = open(f"test_{model_name}.txt", "w")
        txt_f.writelines("\n".join(tokens[0].cpu().reshape(-1).numpy().astype(str).tolist()))
        txt_f.close()

        sample_dur = batch['audio'].shape[-1] / evaluator.sample_rate
        nsample += tokens.shape[0]
        mean_sample_dur += (sample_dur * tokens.shape[0])
        data_type = "inst" if batch['no_lyric_flag'] else "vocal"
        print("=" * 50)
        print(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}, data_type={data_type}")
        f.writelines("=" * 50 + "\n")
        f.writelines(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}\n")

        slice_loss_dict, slice_locality_dict, slice_repetition_dict = {}, {}, {}
        # iter over slices
        for a in range(len(sliced_audios)):
            if sliced_texts is not None:
                text = text_tokenizer(
                    sliced_texts[a], add_special_tokens=False, return_tensors="pt")
                slice_text_token = text["input_ids"].to("cuda")
                slice_text_token_attn_mask = text["attention_mask"].to("cuda")

                if slice_text_token.ndim == 1:
                    slice_text_token = slice_text_token.unsqueeze(0)
                if slice_text_token_attn_mask.ndim == 1:
                    slice_text_token_attn_mask = slice_text_token_attn_mask.unsqueeze(0)
                slice_text_token = {
                    "input_ids": slice_text_token,
                    "attention_mask": slice_text_token_attn_mask,
                }


            else:
                slice_text_token = input_batch["token"]

            slice_length = sliced_audios[0].shape[-1]
            slice_audio_length_tensor = torch.tensor([slice_length], device=sliced_audios[a].device)
      
            
            slice_input_batch = {
                        "audio": sliced_audios[a], 
                        "audio_length": slice_audio_length_tensor,
                        "token": slice_text_token
                        }    # max/even的ctc算不了,因为无法正确截断text,只有section/full可以算

            
            slice_output_dict = evaluator.model(slice_input_batch)

            if "locality" in tasks:
                this_locality = evaluator.locality(slice_input_batch["audio"])
                for key in this_locality.keys():
                    if key not in slice_locality_dict:
                        slice_locality_dict[key] = [this_locality[key]]
                    else:
                        slice_locality_dict[key].append(this_locality[key])
            if "token_repetition" in tasks:
                this_repetition = evaluator.token_repetition(tokens)
                for key in this_repetition.keys():
                    if key not in slice_repetition_dict:
                        slice_repetition_dict[key] = 0
                    else:
                        slice_repetition_dict[key] += this_repetition[key]

            this_loss_dict = evaluator.get_loss_dict(slice_output_dict)
            this_output_dict = evaluator.get_outputs(slice_output_dict)
            # record loss & output
            for key in this_loss_dict.keys():
                if key not in slice_loss_dict.keys():
                    slice_loss_dict[key] = 0
                slice_loss_dict[key] += this_loss_dict[key].item()
            for key in this_output_dict.keys():
                if key not in slice_loss_dict.keys():
                    slice_loss_dict[key] = []
                slice_loss_dict[key].append(this_output_dict[key])

        # combine results of slices
        for key in slice_loss_dict.keys():
            if key.endswith("out"):
                slice_loss_dict[key] = torch.cat(slice_loss_dict[key], dim=-2)
            else:
                slice_loss_dict[key] = slice_loss_dict[key] / len(sliced_audios)
        

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2, 1, 2)
        # plt.imshow(slice_loss_dict["mel_out"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
        # plt.savefig("./mel_chunk_stage3.png")
        # import pdb; pdb.set_trace()

        if "locality" in tasks:
            for key in slice_locality_dict.keys():
                slice_locality_dict[key] = torch.tensor(slice_locality_dict[key]).mean(0)  # [n_slice, n_locality_metrics]
                if key not in slice_locality_dicts.keys():
                    slice_locality_dicts[key] = []
                slice_locality_dicts[key].append(slice_locality_dict[key])

        if "token_repetition" in tasks:
            for key in slice_repetition_dict.keys():
                slice_repetition_dict[key] = torch.tensor(slice_repetition_dict[key]).mean(0)  # [n_slice, n]
                if key not in slice_repetition_dicts.keys():
                    slice_repetition_dicts[key] = []
                slice_repetition_dicts[key].append(slice_repetition_dict[key])

        # ctc wer   
        if "ctc_wer" in tasks:
            ctc_wer, transcript = evaluator.ctc_wer(slice_loss_dict["ctc_out"], input_batch["token"], text_tokenizer)
            if len(ctc_wer) > 0:
                this_ctc_wer = torch.tensor(ctc_wer).mean(0)
                print("wer | ins | del | sub ", this_ctc_wer)
                f.writelines(f"wer | ins | del | sub | {this_ctc_wer}\n")
                f.writelines(f"transcript: {transcript} \nground truth: {batch['text']}\n")
                print("transcript: ", transcript)
                print("ground truth: ", batch["text"])
                slice_ctc_wers.append(this_ctc_wer)

        # combine results of all batches
        for key in slice_loss_dict.keys():
            if key.endswith("out"):
                continue
            if key not in slice_loss_dicts:
                slice_loss_dicts[key] = 0
                slice_loss_dicts_inst[key] = 0
                slice_loss_dicts_vocal[key] = 0
            slice_loss_dicts[key] += slice_loss_dict[key]
            if batch["no_lyric_flag"]:
                slice_loss_dicts_inst[key] += slice_loss_dict[key]
            else:
                slice_loss_dicts_vocal[key] += slice_loss_dict[key]
        if data_type == 'inst':
            inst_i += 1
        else:
            vocal_i += 1

        i += 1  

        tiktok = [time_cnt, nsample, mean_sample_dur / nsample]
        if i % 5 == 0:
            print_stat(i, evaluator, slice_loss_dicts, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)
            print("instrumental music")
            print_stat(inst_i, evaluator, slice_loss_dicts_inst, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)
            print("vocal music")
            print_stat(vocal_i, evaluator, slice_loss_dicts_vocal, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)

    print_stat(i, evaluator, slice_loss_dicts, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)
    print("instrumental music")
    print_stat(inst_i, evaluator, slice_loss_dicts_inst, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)
    print("vocal music")
    print_stat(vocal_i, evaluator, slice_loss_dicts_vocal, slice_locality_dicts, slice_ctc_wers, slice_repetition_dicts, tiktok, tasks=tasks)


    output_dict = {}
    if "code_rate" in tasks:
        code_rate = evaluator.compute_code_rate()
        print(f"[code rate]\n{code_rate}\n")
        if out_dir is None:
            out_dir = "./recipes/umm/tests/eval_tokenizers"
        code_distribution = evaluator.plot_token_distribution(f"{out_dir}/{model_name}.png")
        output_dict["code_rate"] = code_rate

    if "loss" in tasks:
        for key in slice_loss_dicts.keys():
            if key.endswith("out"):
                del slice_loss_dicts[key]
            slice_loss_dicts[key] = slice_loss_dicts[key] / i
        print("[loss dict]\n", slice_loss_dicts)
        output_dict["loss"] = slice_loss_dicts
    
    if "locality" in tasks:
        print("[locality]")
        for key in slice_locality_dicts.keys():
            slice_locality_dicts[key] = torch.stack(slice_locality_dicts[key]).mean(0)  # [n_slice, n_locality_metrics]
            print(key, slice_locality_dicts[key])
        output_dict["locality"] = slice_locality_dicts

    if "ctc_wer" in tasks:
        print("[ctc wer]")
        slice_ctc_wers = torch.from_numpy(np.stack(slice_ctc_wers))
        print("wer | ins | del | sub ", slice_ctc_wers.mean(0))
        output_dict["ctc_wer"] = slice_ctc_wers.mean(0)
    
    if "token_repetition" in tasks:
        print("[token repetition]")
        for key in slice_repetition_dicts.keys():
            print(key, slice_repetition_dicts[key])
        output_dict["token_repetition"] = slice_repetition_dicts

    output_dict["time_count"] = [time_cnt, nsample, mean_sample_dur / nsample]
    print("[time]")
    print(time_cnt)
    print(nsample, mean_sample_dur / nsample)
    return output_dict
        
    
def print_stat(i, evaluator, loss_dicts, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=["loss", "locality", "ctc_wer"]):
    if i == 0:
        return
    
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
        for hh in range(evaluator.h):
            print("** hierarchy {} **".format(hh))
            for key in locality_dict.keys():
                if isinstance(locality_dict[key][hh][0], list):
                    _ld[key] = torch.tensor(locality_dict[key][hh]).mean(0)
                else:
                    _ld[key] = torch.stack(locality_dict[key][hh]).mean(0)  # [n_slice, n_locality_metrics]
                print(key, f"{_ld[key].item() * 100:.2f}%")  

    if "ctc_wer" in tasks:
        print("\n[ctc wer]")
        _cd = torch.from_numpy(np.stack(ctc_wers))
        print("wer | ins | del | sub ", _cd.mean(0))
        
    if "token_repetition" in tasks:
        print("\n[token_repetition]")
        for hh in range(len(repetition_dict)):
            cnt = 0
            print("** hierarchy {} **".format(hh))
            all_tokens = sum([k*v for k, v in repetition_dict[hh].items()])
            for key in sorted(repetition_dict[hh].keys()):
                print(key, repetition_dict[hh][key], "{:.2f}%".format(repetition_dict[hh][key] / all_tokens * 100))
                cnt += 1
                if cnt > 10:
                    break

    print("\n[time]")
    print(tiktok)


@torch.no_grad()
def evaluate(it, 
             f,
             evaluator: TokenEvaluator, 
             tasks=["loss", "locality", "ctc_wer"],
             model_name = "default",
             out_dir=None,
             max_i=1000, 
             slice_mode="max",
             chunk_dur=60,
             text_tokenizer=None,
             chunk_locality_dur=10,
             ):
    loss_dicts, locality_dict, repetition_dict = {}, {}, [{} for _ in range(evaluator.h)]
    nsample, mean_sample_dur, time_cnt, i, inst_i, vocal_i = 0, 0, 0, 0, 0, 0
    loss_dicts_vocal, loss_dicts_inst = {}, {}
    ctc_wers = []
    evaluator.reset_code_count()

    for batch in it:
        if i == max_i:
            break

        input_ids = batch["token"].to("cuda")
        attention_mask = batch["token_padding_mask"].to("cuda")

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        

        input_batch = {
            "audio": batch["audio"].to("cuda"), 
            "audio_length": batch['audio_length'].to("cuda"),
            "token": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "slice": batch["slice"],
        }

        ct = time()
        tokens, sliced_audios, sliced_texts = evaluator.get_tokens(input_batch["audio"], 
                                                                input_batch['audio_length'], slice_mode=slice_mode, 
                                                                slice_info=input_batch["slice"], 
                                                                chunk_dur=chunk_dur)
        time_cnt += time() - ct
        evaluator.update_code_usage(tokens)

        sample_dur = batch['audio'].shape[-1] / evaluator.sample_rate
        nsample += tokens.shape[0]
        mean_sample_dur += (sample_dur * tokens.shape[0])
        data_type = "inst" if batch['no_lyric_flag'] else "vocal"
        print("=" * 50)
        print(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}, data_type={data_type}")
        f.writelines("=" * 50 + "\n")
        f.writelines(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}, no_lyric_flag={batch['no_lyric_flag']}")

        this_output_dict = evaluator.model(input_batch)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(this_output_dict["mel"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
        # plt.subplot(2, 1, 2)
        # plt.imshow(this_output_dict["mel_out"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
        # plt.savefig("./mel_full_stage3.png")
        # import pdb; pdb.set_trace()
        
        if "loss" in tasks:
            loss_dict = evaluator.get_loss_dict(this_output_dict)
            if len(loss_dicts.keys()) == 0:
                loss_dicts.update(loss_dict)
            else:
                for key in loss_dict.keys():
                    loss_dicts[key] += loss_dict[key]
            print(loss_dict)
            for key in loss_dict.keys():
                if key.endswith("out"):
                    continue
                if data_type == 'inst':
                    if key not in loss_dicts_inst:
                        loss_dicts_inst[key] = 0
                    loss_dicts_inst[key] += loss_dict[key]
                else:
                    if key not in loss_dicts_vocal:
                        loss_dicts_vocal[key] = 0
                    loss_dicts_vocal[key] += loss_dict[key]
            if data_type == 'inst':
                inst_i += 1
            else:
                vocal_i += 1

        if "ctc_wer" in tasks:
            ctc_wer, transcript = evaluator.ctc_wer(this_output_dict["ctc_out"], input_batch["token"]['input_ids'], text_tokenizer)

            if len(ctc_wer) > 0:
                ctc_wers.append(ctc_wer)
            
            print("wer | ins | del | sub ", torch.tensor(ctc_wers).mean(0))
            f.writelines(f"wer | ins | del | sub | {torch.tensor(ctc_wers).mean(0)}\n")
            f.writelines(f"transcript: {transcript} \nground truth: {batch['text']}\n")
            print("transcript: ", transcript)
            print("ground truth: ", batch["text"])


        if "locality" in tasks:
            this_locality = evaluator.locality(input_batch["audio"])
            chunk_locality = evaluator.locality_chunk(input_batch["audio"], chunk_size=chunk_locality_dur)
            this_locality.update(chunk_locality)
            for key in this_locality.keys():
                if key not in locality_dict:
                    locality_dict[key] = [this_locality[key]]
                else:
                    locality_dict[key].append(this_locality[key])
            
        if "token_repetition" in tasks:
            this_repetition = evaluator.token_repetition(tokens)
            for hh in range(evaluator.h):
                for key in this_repetition[hh].keys():
                    if key not in repetition_dict[hh]:
                        repetition_dict[hh][key] = 0
                    repetition_dict[hh][key] += this_repetition[hh][key]

        i += 1
        

        tiktok = [time_cnt, nsample, mean_sample_dur / nsample]
        if i % 5 == 0:
            print_stat(i, evaluator, loss_dicts, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)
            print("instrumental music")
            print_stat(inst_i, evaluator, loss_dicts_inst, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)
            print("vocal music")
            print_stat(vocal_i, evaluator, loss_dicts_vocal, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)


    print_stat(i, evaluator, loss_dicts, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)
    print("instrumental music")
    print_stat(inst_i, evaluator, loss_dicts_inst, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)
    print("vocal music")
    print_stat(vocal_i, evaluator, loss_dicts_vocal, locality_dict, ctc_wers, repetition_dict, tiktok, tasks=tasks)
    output_dict = {}
    if "code_rate" in tasks:
        code_rate = evaluator.compute_code_rate()
        print(f"[code rate]\n{code_rate}\n")
        if out_dir is None:
            out_dir = "./recipes/umm/tests/eval_tokenizers"
        code_distribution = evaluator.plot_token_distribution(f"{out_dir}/{model_name}.png")
        output_dict["code_rate"] = code_rate

    if "loss" in tasks:
        for key in loss_dict.keys():
            loss_dicts[key] = loss_dicts[key] / i
        print("[loss dict]\n", loss_dicts)
        output_dict["loss"] = loss_dicts

    if "locality" in tasks:
        print("[locality]")
        _ld = {}
        for hh in range(evaluator.h):
            print("** hierarchy {} **".format(hh))
            for key in locality_dict.keys():
                if isinstance(locality_dict[key][hh][0], list):
                    _ld[key] = torch.tensor(locality_dict[key][hh]).mean(0).item()
                else:
                    _ld[key] = torch.stack(locality_dict[key][hh]).mean(0).item()  # [n_slice, n_locality_metrics]
                print(key, f"{_ld[key] * 100:.2f}%")
        output_dict["locality"] = _ld

    if "ctc_wer" in tasks:
        print("[ctc wer]")
        ctc_wers = torch.tensor(ctc_wers)
        print("wer | ins | del | sub ", ctc_wers.mean(0))
        output_dict["ctc_wer"] = ctc_wers.mean(0)

    if "token_repetition" in tasks:
        print("\n[token_repetition]")
        cnt = 0
        for hh in range(len(repetition_dict)):
            print("** hierarchy {} **".format(hh))
            all_tokens = sum([k*v for k, v in repetition_dict[hh].items()])
            for key in sorted(repetition_dict[hh].keys()):
                print(key, repetition_dict[hh][key], "{:.2f}%".format(repetition_dict[hh][key] / all_tokens * 100))
                cnt += 1
        output_dict["token_repetition"] = repetition_dict

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
    parser.add_argument("--tasks", type=str, default="loss,locality,ctc_wer", help="tasks to evaluate")
    parser.add_argument("--seed", type=int, default=20250101, help="seed")
    parser.add_argument("--slice_modes", type=str, default='full', help="slice modes of tokenizer input audio, now support [full, max, even, section]")
    parser.add_argument("--slice_dur", type=int, default=45, help="slice duration of tokenizer input audio")
    parser.add_argument("--locality_chunk", type=int, default=10, help="locality chunk duration")
    parser.add_argument("--inference_R", type=int, default=None, help="valid inference hierarchy for RVQ/HVQ tokens")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    out_dir = args.out_dir
    max_i = args.nsample
    slice_modes = args.slice_modes.split(",")
    chunk_dur = args.slice_dur
    inference_R = args.inference_R
    tasks = args.tasks.split(",")
    tasks.append("code_rate")
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
    text_tokenizer = "/opt/tiger/tokenizer/bert-base-multilingual-uncased"

    sample_rate = config.sample_rate
    frame_rate = config.frame_rate

    # ===== load evaluator =====
    evaluator = TokenEvaluator(pl_module, config,
                                model_type=model_name, 
                                segment_size=60, 
                                slice_length=[15, 30, 60],
                                inference_R=inference_R)

    # ===== build dataset =====
    if any([x in ["loss", "locality", "token_repetition"] for x in tasks]):
        dataset = EvalDataset(
            data_id="CN:7562",
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

        if 'full' in slice_modes:
            f = open(f"{out_dir}/{model_name}.full.rlt", "w")
            f.write(f"{model_name}: {args.model_path}\n")
            eval_output_dict = evaluate(dataloader, 
                                        f, evaluator, 
                                        max_i=max_i, 
                                        out_dir=out_dir, 
                                        model_name=model_name,
                                        slice_mode="full", 
                                        chunk_dur=chunk_dur,
                                        chunk_locality_dur=args.locality_chunk,
                                        tasks=[task for task in tasks if task != 'ctc_wer'],
                                        text_tokenizer=dataset.tokenizer) 
            f.write("="*50 + "\n")
            for k, v in eval_output_dict.items():
                f.write(f"{k}\n{v}" + "\n")
            f.write("="*50 + "\n")

        for slice_mode in slice_modes:
            if slice_mode == 'full':
                continue
            f = open(f"{out_dir}/{model_name}.{slice_mode}.rlt", "w")
            f.write(f"{model_name}: {args.model_path}\n")
            eval_output_dict = evaluate_audio_slice(dataloader,
                                        f, evaluator,
                                        max_i=max_i,
                                        out_dir=out_dir,
                                        model_name=model_name,
                                        slice_mode=slice_mode, chunk_dur=chunk_dur,
                                        tasks=[task for task in tasks if task!= 'ctc_wer'],
                                        text_tokenizer=dataset.tokenizer)
            f.write("="*50 + "\n")
            for k, v in eval_output_dict.items():
                f.write(f"{k}\n{v}" + "\n")
            f.write("="*50 + "\n")
        f.close()

    if any([x in ["ctc_wer"] for x in tasks]):
        lyric_dataset = EvalDataset(
            data_id="CN:7602",
            url_pattern=None,
            frame_rate=frame_rate,
            sample_rate=sample_rate,
            num_worker=num_workers,
            batch_size=batch_size,
            min_duration=min_duration,
            max_duration=max_duration,
            tokenizer="/opt/tiger/tokenizer/bert-base-multilingual-uncased",
        )
        dataloader = lyric_dataset.dataloader


        if 'full' in slice_modes:
            f2 = open(f"{out_dir}/{model_name}.full.ctc_wer.rlt", "w")
            f2.write(f"{model_name}: {args.model_path}\n")
            eval_output_dict = evaluate(dataloader, 
                                        f2, evaluator, 
                                        max_i=max_i, 
                                        out_dir=out_dir, 
                                        model_name=model_name,
                                        slice_mode="full", chunk_dur=chunk_dur,
                                        text_tokenizer=lyric_dataset.tokenizer,
                                        tasks=['ctc_wer'],) 
            f2.write("="*50 + "\n")
            for k, v in eval_output_dict.items():
                f2.write(f"{k}\n{v}" + "\n")
            f2.write("="*50 + "\n")

        for slice_mode in slice_modes:
            if slice_mode == 'full':
                continue
            f2 = open(f"{out_dir}/{model_name}.{slice_mode}.ctc_wer.rlt", "w")
            f2.write(f"{model_name}: {args.model_path}\n")
            eval_output_dict = evaluate_audio_slice(dataloader,
                                        f2, evaluator,
                                        max_i=max_i,
                                        out_dir=out_dir,
                                        model_name=model_name,
                                        slice_mode=slice_mode, chunk_dur=chunk_dur,
                                        text_tokenizer=lyric_dataset.tokenizer,
                                        tasks=['ctc_wer'])
            f2.write("="*50 + "\n")
            for k, v in eval_output_dict.items():
                f2.write(f"{k}\n{v}" + "\n")
            f2.write("="*50 + "\n")
        f2.close()


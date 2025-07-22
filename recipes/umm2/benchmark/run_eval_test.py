import torch
assert torch.cuda.is_available()
# from recipes.umm.requires.model_initializer import init_stage3, init_stage3_rvq
import matplotlib.pyplot as plt
from recipes.umm.tests.datasets.test_dataset import set_random_seed
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.benchmark.evaluator import TokenEvaluator
import numpy as np
from time import time

import torch, os
from recipes.umm2.loaders.base import BaseModelLoader, local_zero_first
import sys, importlib
from recipes.umm2.benchmark.dataset import EvalDataset
from recipes.umm2.scripts.test_stage3_wav2tokens_RVQ import ModelLoader, init_model
from recipes.datasets.mcc.mix_mkii import MixDataModule


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
        _cd = torch.from_numpy(np.array(ctc_wers))
        print("wer | ins | del | sub ", _cd.mean(0))
        
    print("[time]")
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
             text_tokenizer=None):
    loss_dicts, locality_dict = {}, {}
    nsample, mean_sample_dur, time_cnt, i = 0, 0, 0, 0
    ctc_wers = []
    evaluator.reset_code_count()

    for batch in it:
        if i == max_i:
            break
        input_batch = {
            "audio": batch["audio"].to("cuda"), 
            "audio_padding_mask": None,
            "token_padding_mask": None,
            "text": batch["text"],
            "token": batch["token"].to("cuda"),
            # "slice": batch["slice"],
            }
        ct = time()
        print(batch["audio"].shape, batch["audio"].mean(), batch["audio"].std())
        # tokens, sliced_audios, sliced_texts = evaluator.get_tokens(input_batch["audio"], slice_mode=slice_mode, 
        #                                                         slice_info=input_batch["slice"], chunk_dur=chunk_dur)
        # time_cnt += time() - ct
        # evaluator.update_code_usage(tokens)

        sample_dur = batch['audio'].shape[-1] / evaluator.sample_rate
        nsample += batch['audio'].shape[0]
        mean_sample_dur += (sample_dur * batch['audio'].shape[0])
        print("=" * 50)
        print(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}")
        f.writelines("=" * 50 + "\n")
        f.writelines(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}\n")

        this_output_dict = evaluator.model(input_batch)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(this_output_dict["mel"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
        # plt.subplot(2, 1, 2)
        # plt.imshow(this_output_dict["mel_out"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
        # plt.savefig("./mel.png")
        tokens = evaluator.get_vq_id_from_dict(this_output_dict)
        evaluator.update_code_usage(tokens)

        print(this_output_dict["mel"].mean(), this_output_dict["mel"].std())
        print(this_output_dict["mel_out"].mean(), this_output_dict["mel_out"].std())

        if "loss" in tasks:
            loss_dict = evaluator.get_loss_dict(this_output_dict)
            if len(loss_dicts.keys()) == 0:
                loss_dicts.update(loss_dict)
            else:
                for key in loss_dict.keys():
                    loss_dicts[key] += loss_dict[key]
            print(loss_dict)

        if "ctc_wer" in tasks:
            ctc_wer, transcript = evaluator.ctc_wer(this_output_dict["ctc_out"], input_batch["token"], text_tokenizer)
            if len(ctc_wer) > 0:
                ctc_wers.extend(ctc_wer)

            print("wer | ins | del | sub ", torch.from_numpy(np.array(ctc_wer)).mean(0))
            f.writelines(f"wer | ins | del | sub | {torch.from_numpy(np.array(ctc_wer)).mean(0)}\n")
            f.writelines(f"transcript: {transcript} \nground truth: {batch['text']}\n")

            print("transcript: ", transcript)
            print("ground truth: ", batch["text"])
            
            # print(batch["text"], loss_dict, ctc_wer)
            # import torchaudio; torchaudio.save(f"{out_dir}/{i}.wav", batch["audio"], sample_rate=evaluator.sample_rate)

            # slice_batch={"audio": batch["audio"][...,:24000*30].to("cuda"), 
            #              "token": batch["token"].to("cuda")}
            # slice_dict = evaluator.pl_module.prepare_feature(slice_batch)
            # slice_out = evaluator.model(slice_dict)
            # _, slice_transcript = evaluator.ctc_wer(slice_out["ctc_out"], slice_dict["text_ids"], text_tokenizer)
            # print("slice: \n", slice_transcript)


        if "locality" in tasks:
            this_locality = evaluator.locality(input_batch["audio"])
            for key in this_locality.keys():
                if key not in locality_dict:
                    locality_dict[key] = [this_locality[key]]
                else:
                    locality_dict[key].append(this_locality[key])

        i += 1
        tiktok = [time_cnt, nsample, mean_sample_dur / nsample]
        if i % 5 == 0:
            print_stat(i, evaluator, loss_dicts, locality_dict, ctc_wers, tiktok, tasks=tasks)


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
        for key in locality_dict.keys():
            locality_dict[key] = torch.tensor(locality_dict[key]).mean(0)  # [n_slice, n_locality_metrics]
            print(key, locality_dict[key])
        output_dict["locality"] = locality_dict

    if "ctc_wer" in tasks:
        print("[ctc wer]")
        ctc_wers = torch.from_numpy(np.array(ctc_wer))
        print("wer | ins | del | sub ", ctc_wers.mean(0))
        output_dict["ctc_wer"] = ctc_wers.mean(0)
        
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
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    out_dir = args.out_dir
    max_i = args.nsample
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
    min_duration = 1
    max_duration = 60  # full song
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
    sample_rate = config.sample_rate
    frame_rate = config.frame_rate

    # ===== load evaluator =====
    evaluator = TokenEvaluator(pl_module, config,
                                model_type=model_name, 
                                segment_size=60, 
                                slice_length=[15, 30, 60])

    # ===== build dataset =====
    datamodule = MixDataModule(
        data_ids=[2255],
        val_data_id=2255,
        data_weights=[1],
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        num_workers=num_workers,
        batch_size=max_duration*batch_size*sample_rate,
        max_duration=60,
        min_duration=1,
        tokenizer=text_tokenizer,
    )
    dataloader = datamodule.val_dataloader() 

    f = open(f"{out_dir}/{model_name}.rlt", "w")
    f.write(f"{model_name}: {args.model_path}\n")

    eval_output_dict = evaluate(dataloader, 
                                f, evaluator, 
                                max_i=max_i, 
                                out_dir=out_dir, 
                                model_name=model_name,
                                slice_mode="full", chunk_dur=60,
                                tasks=[task for task in tasks],
                                text_tokenizer=datamodule.tokenizer) 
    f.write("="*50 + "\n")
    for k, v in eval_output_dict.items():
        f.write(f"{k}\n{v}" + "\n")
    f.close()

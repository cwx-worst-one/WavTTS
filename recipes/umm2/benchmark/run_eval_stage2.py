from recipes.umm2.benchmark.run_eval import *
import math
import matplotlib.pyplot as plt
from samantha.utils.hdfs_helper import exists

import pprint

pp = pprint.PrettyPrinter(indent=4)

def print_stat(i, loss_dicts, ctc_wers, tasks=["loss", "locality", "ctc_wer"]):
    print(f"======= Statistics [Iteration {i}] =======")
    
    _ld = {}
    if "loss" in tasks:
        print("\n[Average Loss]")
        for key, value in loss_dicts.items():
            # 使用 .item() 获取纯数值，使字典更干净
            # Use .item() to get the raw number, making the dictionary cleaner
            if isinstance(value, torch.Tensor):
                _ld[key] = value.detach().cpu().item() / i
            else: # Fallback for non-tensor values
                _ld[key] = value / i
        pp.pprint(_ld)

    if "ctc_wer" in tasks:
        print("\n[Average CTC WER]")
        # .detach().cpu() 是处理用于分析的张量的好习惯
        # .detach().cpu() is a good practice for tensors used in analysis
        _cd = torch.from_numpy(np.stack(ctc_wers)).detach().cpu()
        
        # 将张量转换为列表，然后用 f-string 格式化，更清晰
        # Convert tensor to a list and format with an f-string for more clarity
        wer, ins, del_rate, sub = _cd.mean(0).tolist()
        print(f"  WER: {wer:.2f}% | Ins: {ins:.3f} | Del: {del_rate:.3f} | Sub: {sub:.3f}")
    
    print("=" * 40 + "\n")

@torch.no_grad()
def get_slice(audio, slice_mode="full", chunk_dur=60, sample_rate=24000):
    """
    audio shape: [B, n_signal_sample]
    """
    sliced_audios = []
    slice_texts = None
    n_samples = audio.shape[-1]
    n_secs = float(audio.shape[-1]) / 24000

    if slice_mode == 'full':
        sliced_audios.append(audio)
    
    elif slice_mode in ['max', 'even']:
        # evenly slice the audio in a way that the `chunk_size` is as close as possible to `chunk_dur`
        if slice_mode == 'even':  
            chunk_num = math.ceil(n_secs / chunk_dur)
            chunk_size = math.ceil(n_secs / chunk_num)
        # always slice the audio with the maximum `chunk_dur`, combine the tail audio < 1s
        elif slice_mode == 'max':
            chunk_size = chunk_dur
        
        st = 0
        while st < n_samples:
            _st, _et = int(st*sample_rate), int((st+chunk_size)*sample_rate)
            # merge the tail if the remaining chunk is too short (< 1s) 
            if n_samples - _et < sample_rate * 1 or audio[...,_et:].shape[-1] < sample_rate * 1:
                _et = n_samples
            sliced_audios.append(audio[..., _st:_et])
            if _et >= n_samples:
                break
            st += chunk_size
    else:
        raise NotImplementedError(f"slice_mode {slice_mode} not implemented")
    
    return sliced_audios, slice_texts



@torch.no_grad()
def evaluate_stage2(it, 
                    evaluator: TokenEvaluator,
                    slice_mode='full',
                    text_tokenizer=None,
                    chunk_dur=45,
                    max_i=100):
    nsample, mean_sample_dur, time_cnt, i = 0, 0, 0, 0
    evaluator.reset_code_count()
    loss_dicts = {}
    ctc_wers = []

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

        sample_dur = batch['audio'].shape[-1] / evaluator.sample_rate
        nsample += batch["audio"].shape[0]
        print("=" * 50)
        print(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}")
        f.writelines("=" * 50 + "\n")
        f.writelines(f"batch {i}, length={sample_dur}, avg_length={mean_sample_dur/nsample}\n")

        sliced_audios, sliced_texts = get_slice(input_batch["audio"], slice_mode=slice_mode, chunk_dur=chunk_dur)

        if slice_mode == 'full':
            this_output_dict = evaluator.model(input_batch)

            loss_dict = evaluator.get_loss_dict(this_output_dict)
            # plt.figure()
            # plt.subplot(2, 1, 1)
            # plt.imshow(this_output_dict["mel"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
            # plt.subplot(2, 1, 2)
            # plt.imshow(this_output_dict["mel_out"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
            # plt.savefig("./mel_full.png")
            # import pdb; pdb.set_trace()
            
            # print(loss_dict)
            if len(loss_dicts.keys()) == 0:
                loss_dicts.update(loss_dict)
            else:
                for key in loss_dict.keys():
                    loss_dicts[key] += loss_dict[key]


            ctc_wer, transcript = evaluator.ctc_wer(this_output_dict["ctc_out"], input_batch["token"]['input_ids'], text_tokenizer)
            if len(ctc_wer) > 0:
                this_ctc_wer = torch.tensor(ctc_wer).mean(0)
                print("wer | ins | del | sub ", this_ctc_wer)
                f.writelines(f"wer | ins | del | sub | {this_ctc_wer}\n")
                f.writelines(f"transcript: {transcript} \nground truth: {batch['text']}\n")
                print("transcript: ", transcript)
                print("ground truth: ", batch["text"])
                ctc_wers.append(this_ctc_wer)
        else:
            slice_loss_dict = {}
            # iter over slices
            for a in range(len(sliced_audios)):
                if sliced_texts is not None:
                    text = text_tokenizer(
                        sliced_texts[a], add_special_tokens=False, return_tensors="pt")
                    slice_text_token = text["input_ids"].squeeze(dim=0)
                else:
                    slice_text_token = input_batch["token"]
                slice_input_batch = {"audio": sliced_audios[a], "token": slice_text_token}    # max/even的ctc算不了,因为无法正确截断text,只有section/full可以算
                slice_output_dict = evaluator.model(slice_input_batch)

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
                # print(a, this_loss_dict)

            # combine results of slices
            for key in slice_loss_dict.keys():
                if key.endswith("out"):
                    slice_loss_dict[key] = torch.cat(slice_loss_dict[key], dim=-2)
                else:
                    slice_loss_dict[key] = slice_loss_dict[key] / len(sliced_audios)

            # plt.figure()
            # plt.imshow(slice_loss_dict["mel_out"][0, :1000].cpu().numpy().T, aspect="auto", origin='lower')
            # plt.savefig("./mel_chunk.png")
            # print(slice_loss_dict)
            # import pdb; pdb.set_trace()

            # ctc wer   
            ctc_wer, transcript = evaluator.ctc_wer(slice_loss_dict["ctc_out"], input_batch["token"], text_tokenizer)
            if len(ctc_wer) > 0:
                this_ctc_wer = torch.tensor(ctc_wer).mean(0)
                print("wer | ins | del | sub ", this_ctc_wer)
                f.writelines(f"wer | ins | del | sub | {this_ctc_wer}\n")
                f.writelines(f"transcript: {transcript} \nground truth: {batch['text']}\n")
                print("transcript: ", transcript)
                print("ground truth: ", batch["text"])
                ctc_wers.append(this_ctc_wer)

            # combine results of all batches
            for key in slice_loss_dict.keys():
                if key.endswith("out"):
                    continue
                if key not in loss_dicts:
                    loss_dicts[key] = 0
                loss_dicts[key] += slice_loss_dict[key]

        i += 1

    #print(f"Reulst: {model_file}")
    #print_stat(i, loss_dicts, ctc_wers, tasks=["loss", "ctc_wer"])
    return ctc_wers, loss_dicts


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
    parser.add_argument("--slice_modes", type=str, default="full", help="slice mode")
    parser.add_argument("--slice_dur", type=int, default=45, help="chunk duration")
    parser.add_argument("--seed", type=int, default=20250101, help="seed")
    parser.add_argument("--start" ,required=True, type=int, default=10000, help="start ckpt")
    parser.add_argument("--end" ,required=True, type=int, default=30000, help="start ckpt")
    parser.add_argument("--text_tokenizer", type=str, default="bert-base-multilingual-uncased", help="text tokenizer")
    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    out_dir = args.out_dir
    max_i = args.nsample
    slice_mode = args.slice_modes
    chunk_dur = args.slice_dur
    tasks = args.tasks.split(",")
    tasks.append("code_rate")
    pl_module_string = args.model_cls
    
    cache_dir = "./.module_cache/umm2_stage2/"
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


    ctc_wers = {}
    loss_dicts = {}
    ckpt_idx = list(range(args.start, args.end + 1, 10000))
    ckpt_names = ["step=" + str(idx).zfill(7) + ".ckpt" for idx in ckpt_idx]

    for ckpt_name in ckpt_names:
        model_path = os.path.join(args.model_path, ckpt_name)
        print(f"Decoding {model_path}")

        if not exists(model_path):
            print(f"{model_path} Not existis")
            continue


        # ===== load model =====
        model = init_model(model_path, cache_dir, device, pl_module_string)
        pl_module = model["pl_module"]
        config = pl_module.model.stages[0].config   # same configs shared between stages
        try:
            text_tokenizer = args.text_tokenizer
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
        dataset = EvalDataset(
            data_id=7602,
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

        f = open(f"{out_dir}/{model_name}.rlt", "w", encoding="utf-8") 
        f.write(f"{model_name}: {args.model_path}\n")

        ctc_wer, loss_dict = evaluate_stage2(dataloader, 
                                            evaluator, max_i=max_i,
                                            text_tokenizer=dataset.tokenizer,
                                            slice_mode=slice_mode, chunk_dur=chunk_dur ) 


        ctc_wers[ckpt_name] = ctc_wer
        loss_dicts[ckpt_name] = loss_dict



    for ckpt_name in ctc_wers.keys():
        print("=========================")
        print(ckpt_name)
        print_stat(len(ctc_wers[ckpt_name]), loss_dicts[ckpt_name], ctc_wers[ckpt_name], tasks=["loss", "ctc_wer"])
        print("=========================")
    
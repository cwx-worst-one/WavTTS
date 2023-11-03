import argparse
import math
import os
import numpy as np
import torch
from jiwer import cer, wer
from mir_eval import key
from sklearn import metrics

from recipes.mir.datasets.mix import MODE_MAJMIN, TAG_NAMES, TestDataModule
from recipes.mir.requires.model_initializer import init_stage3, init_unified_decoder
from recipes.musiclm.requires.model_initializer import init_mulan
from samantha.utils.hdfs_tools import hdfs_put


#############################
#  Unified decoder related  #
#############################
def load_decoder(name, step, hdfs_root):
    print("Loading UnifiedMirModel...")
    hpath = os.path.join(hdfs_root, name, f"checkpoints/step={step}.ckpt")
    decoder = init_unified_decoder(
        hpath=hpath,
        local_rank=RANK,
        cache_dir=f".module_cache/{name}/umm_mir/",
    )["unified_decoder"]
    print("UnifiedMirModel loaded.")
    print("Loading Stage3...")
    stage3 = init_stage3(
        hpath=decoder.hparams.required_modules["stage3"]["ckpt_path"],
        local_rank=RANK,
        cache_dir=f".module_cache/{name}/umm/",
    )
    decoder.requires.update(stage3)
    print("Stage3 loaded.")
    decoder.extra_params.dynamic_batch = True
    return decoder


def decode(batch, decoder, tokenizer, max_length):
    batch["mono_audio"] = batch["mono_audio"].to(decoder.device)
    batch["mono_text"] = batch["mono_text"].to(decoder.device)
    token_seq, token_length, target_length = decoder.prepare_feature(batch)
    model_input = []
    for j, (tok_len, tar_len) in enumerate(zip(token_length, target_length)):
        model_input.append(token_seq[j, : tok_len - tar_len])
    model_input = torch.stack(model_input, dim=0)
    model_output = decoder.predict(model_input, 1, 0, max_length=max_length)

    refs = []
    ests = []
    for i, out in enumerate(model_output):
        tokenizer_input = [token for token in out if token < decoder.extra_params.vocab_size_text]
        if len(tokenizer_input) == 0:
            est = ""
        else:
            tokenizer_input = torch.stack(tokenizer_input, dim=0)
            est = tokenizer.decode(tokenizer_input, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        ref = tokenizer.decode(batch["mono_text"][batch["mono_map"][i][4]], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if VERBOSE:
            print(f"{ref=}")
            print(f"{est=}")
        refs.append(ref)
        ests.append(est)
    return refs, ests


#############################
#        MIR9 metrics       #
#############################

def tempo_acc2(reference, estimated, tol=0.04):
    if tol < 0 or tol > 1:
        raise ValueError()
    estimated_tempi = np.array([estimated / 3, estimated / 2, estimated, estimated * 2, estimated * 3])
    relative_error = np.min(np.abs(reference - estimated_tempi) / reference)
    return relative_error <= tol

def key_mirex(reference, estimated):
    score = key.weighted_score(reference, estimated)
    return score


#############################
#       aggregate func      #
#############################
def tempo_func(refs, ests):
    count = 0
    hit = 0
    for ref, est in zip(refs, ests):
        try:
            if tempo_acc2(int(ref), int(est)):
                hit += 1
        except Exception:
            pass
        count += 1
    score = hit / count
    return score


def tag_func(refs, ests):
    t2i = {tag: i for i, tag in enumerate(TAG_NAMES)}
    ref_array = np.zeros((len(refs), len(t2i)))
    est_array = np.zeros((len(ests), len(t2i)))
    for i, (ref, est) in enumerate(zip(refs, ests)):
        ref = ref.split(", ")
        est = est.split(", ")
        for t in ref:
            if t in t2i:
                ref_array[i, t2i[t]] = 1
        for t in est:
            if t in t2i:
                est_array[i, t2i[t]] = 1
    score = []
    for c in range(len(t2i)):
        try:
            score.append(metrics.roc_auc_score(ref_array[:, c], est_array[:, c]))
        except Exception as e:
            continue
    score = np.mean(score)
    return score


def structure_func(refs, ests):
    score = []
    for i, (ref, est) in enumerate(zip(refs, ests)):
        ref = " ".join(ref.split(", "))
        est = " ".join(est.split(", "))
        try:
            score.append(wer(ref.lower(), est.lower()))
        except Exception as e:
            score.append(0)
    score = np.mean(score)
    return score


def key_func(refs, ests):
    def _clean(name):
        key_mode = name.split(" ")
        if len(key_mode) == 2:
            key = key_mode[0]
            mode = key_mode[1]
        elif len(key_mode) == 3:
            if key_mode[1] == "#":
                key = key_mode[0] + key_mode[1]
                mode = key_mode[2]
            else:
                raise ValueError(f"Invalid key {name}")
        else:
            ValueError(f"Invalid key {name}")
        mode = MODE_MAJMIN[mode.lower()]
        return f"{key} {mode}"
            
    score = []
    for ref, est in zip(refs, ests):
        try:
            score.append(key_mirex(_clean(ref), _clean(est)))
        except Exception:
            score.append(0)
    score = np.mean(score)
    return score


def gender_func(refs, ests):
    ref_array = np.zeros(len(refs))
    est_array = np.zeros(len(ests))
    for i, (ref, est) in enumerate(zip(refs, ests)):
        if ref.lower() == "male":
            ref_array[i] = 1
        if est.lower() == "male":
            est_array[i] = 1
    score = metrics.roc_auc_score(ref_array, est_array)
    return score

def dialect_func(refs, ests):
    ref_array = np.zeros((len(refs), 3))
    est_array = np.zeros((len(ests), 3))
    for i, (ref, est) in enumerate(zip(refs, ests)):
        if ref.lower() == "mandarin":
            ref_array[i, 0] = 1
        elif ref.lower() == "cantonese":
            ref_array[i, 1] = 1
        elif ref.lower() == "hokkien":
            ref_array[i, 2] = 1
        if est.lower() == "mandarin":
            est_array[i, 0] = 1
        elif est.lower() == "cantonese":
            est_array[i, 1] = 1
        elif est.lower() == "hokkien":
            est_array[i, 2] = 1
    score = []
    for i in range(3):
        try:
            score.append(metrics.roc_auc_score(ref_array[:, i], est_array[:, i]))
        except Exception as e:
            continue
    score = np.mean(score)
    return score


def caption_func(refs, ests):
    print("Loading MuLan...")
    mulan = init_mulan(
        hpath="hdfs://haruna/home/byte_speech_sv/zongyu.yin/ckpts/mulan/mulan-step=014000-median_rank_1=160-kaggle.ckpt",
        local_rank=RANK,
        cache_dir=".module_cache/mulan/",
        version="g4",
    )
    print("MuLan loaded.")
    score = []
    bs = 20
    for i in range(math.ceil(len(refs) / bs)):
        ref = refs[i * bs : (i + 1) * bs]
        est = ests[i * bs : (i + 1) * bs]
        ref_embeds = mulan["mulan_infer_fn"](
            model=mulan["mulan"], text=ref, device=mulan["mulan"].device
        ).cpu()
        est_embeds = mulan["mulan_infer_fn"](
            model=mulan["mulan"], text=est, device=mulan["mulan"].device
        ).cpu()
        cos_sim = torch.nn.functional.cosine_similarity(ref_embeds, est_embeds, dim=1).tolist()
        score += cos_sim
    score = np.mean(score)
    return score


def asr_func(refs, ests):
    score = []
    for i, (ref, est) in enumerate(zip(refs, ests)):
        try:
            score.append(cer(ref.lower(), est.lower()))
        except Exception as e:
            score.append(0)
    score = np.mean(score)
    return score

eval_func_map = {
    "tempo": tempo_func,
    "tag": tag_func,
    "structure": structure_func,
    "key": key_func,
    "gender": gender_func,
    "dialect": dialect_func,
    "caption": caption_func,
    "asr-music": asr_func,
    "asr-speech": asr_func, 
}

def main(args):
    max_duration=300
    res_log = open(args.out, "w")
    task_weights = {
        "tempo": 1,
        "tag": 1,
        "structure": 1,
        "key": 1,
        "gender": 1,
        "dialect": 1,
        "caption": 1,
        "asr-music": 1,
        "asr-speech": 1,
    }
    pl_datamodule = TestDataModule(
        sample_rate=24000,
        batch_size=24000 * max_duration,
        min_duration=5,
        max_duration=max_duration,
        shuffle_buffer_size=50,
        num_workers=4,
        weights=list(task_weights.values()),
        dynamic_batch=True,
        max_length=max_duration * 25,
    )
    tokenizer = pl_datamodule.tokenizer
    loader_dict = {}
    loader_idx = 0
    for task, w in task_weights.items():
        if w > 0:
            loader_dict[task] = pl_datamodule.test_dataloader()[loader_idx]
            loader_idx += 1

    # Load unfied MIR decoder
    decoder = load_decoder(name=args.exp, step=args.exp_step, hdfs_root=args.in_hdfs_path)

    for task, loader in loader_dict.items():
        ref_bucket = []
        est_bucket = []
        for i, batch in enumerate(loader):
            # unfied MIR decode
            print(f"Batch index => {i}")
            refs, ests = decode(batch, decoder, tokenizer, max_length=None)
            ref_bucket += refs
            est_bucket += ests
        score = eval_func_map[task](ref_bucket, est_bucket)
        print(f"[{task}] {score}")
        res_log.write(f"[{task}] {score}")
        res_log.write("\n")
        res_log.flush()

    res_log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, default="mir9_log.txt", help="Output log name."
    )
    parser.add_argument(
        "--out-hdfs-path", type=str, default="hdfs:///home/byte_speech_sv/zongyu.yin/", help="Output HDFS path."
    )
    parser.add_argument(
        "--in-hdfs-path", type=str, default="hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm_mir/", help="Input HDFS path."
    )
    parser.add_argument(
        "--exp", type=str, default="decoder_mix_cvq-65536_bert-base-uncased-30522", help="Exp name."
    )
    parser.add_argument(
        "--exp-step", type=str, default="0010000", help="Step number of ckpt."
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Device rank."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print output."
    )
    args = parser.parse_args()
    RANK = args.rank
    VERBOSE = args.verbose
    for name in [
        "decoder_mix_base-32768_bert-base-uncased-30522",
        "decoder_mix_cvq-65536_bert-base-uncased-30522",
        "decoder_mix_lfq-131072_bert-base-uncased-30522"
    ]:
        args.exp = name
        for i in range(1, 10):
            args.exp_step = f"{(i * 10000):>07}"
            args.out = f"mir9_log_{args.exp}_{args.exp_step}"
            main(args)
            hdfs_put(args.out, args.out_hdfs_path, force=True)

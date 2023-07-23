import argparse
import os

import torch
import numpy as np

from recipes.text2semantic.datasets import PhoneTokenizerWithAudioTokensSpk
from recipes.text2semantic.datasets.text_converter import  TextToTacolabID
from recipes.text2semantic.scripts.infer_utils import prepare_models_llama, save_wav, to_device, get_step_epoch_from_ckpt
import librosa
import json


@torch.no_grad()
def main(args):
    device = args.device
    ar_model, nar_model, vqgan =  prepare_models_llama(args, device)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # utt|prompt_text|prompt_wav_path|text|wav_path
    text2id = TextToTacolabID(args.metaid_to_textid_path)
    tokenizer = PhoneTokenizerWithAudioTokensSpk(args.phone_token_num, args.audio_token_num, 1252)
    symbol_sets = ['.', ',', '?', '!', '，', '。', '？', '！']

    # spk_dict_path='/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/datasets/dict/spk_en_TTS.json'
    f = open(args.spk_dict_path)
    spk_dict = json.load(f)
    spk_id = spk_dict[args.spkid]
    spk_id = tokenizer.tokenize(spk_id, "spks")
    
    for line in open(args.meta_file):
        temp = line.strip().split("|")
        if len(temp) == 5:
            utt, prompt_text, prompt_wav_path, text, wav_path = line.strip().split("|")
        elif len(temp) == 4:
            utt, prompt_text, prompt_wav_path, text = line.strip().split("|")
        if prompt_text[-1] in symbol_sets:
            text = prompt_text + ' ' + text.strip()
        else:
            text = prompt_text + ', ' + text
        text_id = text2id(text)
        text_id[text_id==4775]=2 # hard-code map "sp" to ",""
        wav, sr = librosa.load(prompt_wav_path, sr=24000)
        wav = torch.from_numpy(wav).unsqueeze(0).to(device)
        wav_id = vqgan["encoder"](wav)[2]
        wav_id = torch.stack(wav_id, dim=2)[0].cpu().numpy()

        prompt_text_id = text2id(prompt_text)
        pred_text_id_len = text_id.shape[0] - prompt_text_id.shape[0]
        min_pred_wav_id_len = int(pred_text_id_len * 3.5)

        # Prepare datasets.
        text_id = tokenizer.tokenize(text_id, "inputs")
        wav_id = tokenizer.tokenize(wav_id, "targets")
        wav_len, num_rvqs = wav_id.shape[0], wav_id.shape[1]
        text_len = text_id.shape[0]
        seq = (
            [tokenizer.bos]
            + list(text_id)
            + [tokenizer.sep]
            + [spk_id]
            + list(wav_id[:, 0])
        )
        pos_id = np.asarray(list(range(text_len + 2)) + list(range(wav_len + 1)))
        seq_sen_id = np.asarray([1] * (text_len + 2) + [2] * (wav_len + 1))
        seq = np.asarray(seq)
        full_text_seq = np.stack([text_id] * num_rvqs, axis=1)
        sep = np.stack([np.asarray([tokenizer.sep])] * num_rvqs, axis=1)
        bos = np.stack([np.asarray([tokenizer.bos])] * num_rvqs, axis=1)
        full_spk_seq = np.stack([np.asarray([spk_id])] * num_rvqs, axis=1)
        full_seq = np.concatenate([bos, full_text_seq, sep, wav_id], axis=0)
        
        seqs = torch.from_numpy(seq).unsqueeze(0)
        seq_lens = torch.from_numpy(np.array([seq.shape[0]]))
        pos_ids = torch.from_numpy(pos_id).unsqueeze(0)
        seq_sen_ids = torch.from_numpy(seq_sen_id).unsqueeze(0)
        full_seqs = torch.from_numpy(full_seq).unsqueeze(0)
        utts = [utt]

        seqs, seq_lens, pos_ids, seq_sen_ids, init_full_seqs, utts = to_device(
            (seqs,seq_lens,pos_ids,seq_sen_ids,full_seqs,utts), device=device
        )
        text_len = (seq_sen_ids == 1).sum(dim=1)
        unmask_len = (seq_sen_ids == 2).sum(dim=1)
        num_res = init_full_seqs.shape[2]

        # ar
        seqs, pos_ids, seq_sen_ids = ar_model.inference_from_text(
            (seqs, seq_lens, pos_ids, seq_sen_ids, utts), tokenizer, thres=args.thres, temp=args.temp, mode=args.mode, min_pred_wav_id_len=min_pred_wav_id_len)
        b, t = seqs.shape

        seqs = torch.cat((seqs[:, :text_len[0]], seqs[:, text_len[0] + 1:]), dim=-1)
        pos_ids = pos_ids[:, :-1]
        seq_sen_ids = seq_sen_ids[:, :-1]
        full_seqs = torch.stack([seqs] * num_res, dim=1)  # [b, n_codebook, t]
        full_seqs[:, :, text_len[0] : text_len[0] + unmask_len[0] - 1] = \
            init_full_seqs[:, text_len[0] :, :].transpose(1, 2)
        layer_index = torch.ones(size=[b], device=device)  # [b,]
        mask1 = layer_index.unsqueeze(1) > torch.arange(num_res, device=device).unsqueeze(0)  # [b, 1] > [1, n_codebook] = [b, n_codebook]
        mask2 = (text_len + unmask_len - 1).unsqueeze(1) > torch.arange(t - 1, device=device).unsqueeze(0)  # [b, 1] > [1, t] = [b, t]
        mask = mask1.unsqueeze(2) + mask2.unsqueeze(1)  # [b, n_codebbok, 1] + [b, 1, t]
        full_seqs = torch.where(mask, full_seqs, torch.zeros_like(full_seqs))

        
        # nar
        full_seqs = nar_model.generate(full_seqs,
                                       unmask_len, 
                                       seq_sen_ids, 
                                       pos_ids, 
                                       mask2, 
                                       tokenizer, 
                                       temperature=args.temperature)

        full_seq = full_seqs[:, :, text_len[0] :]  # [b, n_codbook, t]
        full_seq = (full_seq - tokenizer.phone_token_num - 1)[:, :, :-1]

        wav_gen = vqgan["decoder"](full_seq)
        wav_gen = wav_gen.cpu().squeeze(1).squeeze(0).numpy()
        save_wav(wav_gen, os.path.join(out_dir, "prompt-{}.wav".format(utts[0])),sr=24000)
        wav_gen = wav_gen[unmask_len * 300 :]
        save_wav(wav_gen, os.path.join(out_dir, "{}.wav".format(utts[0])),sr=24000)

if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_ckpt_path", type=str, required=True)
    parser.add_argument("--nar_ckpt_path", type=str, required=True)
    parser.add_argument("--phone_token_num", type=int, default=200)
    parser.add_argument("--audio_token_num", type=int, default=1024)
    parser.add_argument("--metaid_to_textid_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--codec_ckpt_path", type=str, 
                        default="hdfs://haruna/home/byte_speech_sv/user/congjian/2023-01-17_causal_x300_1024_6book_doubleG_export/", 
                        help="codec model ckpt path")
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True, help="text file")
    parser.add_argument("--spkid", type=str, required=True, help="text file")
    parser.add_argument("--thres", type=float, default=0.0)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--mode", type=str, default="naive")
    parser.add_argument("--spk_dict_path", type=str, 
                        default="/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/datasets/dict/spk_en_TTS.json", 
                        help="spk dict")

    args = parser.parse_args()
    main(args)

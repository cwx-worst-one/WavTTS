import argparse
import os
import numpy as np

import torch
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from recipes.text2semantic.datasets.continuous_dataset import ContinuousTTSDataset, ContinuousCollator
from recipes.text2semantic.lit_modules.llama.lit_vae_t2s_ctiga import VAET2SModule
from samantha.utils.hparams import DotDict
from recipes.text2semantic.datasets.text_converter_v2 import  TextToTacolabID
from recipes.text2semantic.datasets.continuous_dataset import PhoneTokenizerWithAudioTokens
from transformers import LlamaTokenizer
import random
import json

def setup_seed(seed=1996):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    return

def save_wav(audio, output_file, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            # if tensor.dtype == torch.float16:
            #     tensor = tensor.bfloat16()
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch

def prepare_models(args, device):
    ar_model = VAET2SModule.load_from_checkpoint(
        args.ar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    ar_model.eval()
    return ar_model


@torch.no_grad()
def main(args):
    if args.seed:
        print("seed: ", args.seed)
        setup_seed(args.seed)

    devices = args.device
    # 这里务必要和训练一致！！！！！
    ar_model_hps = DotDict({"phone_tokens_num": 7370, "speaker_tokens_num": 8192})
    print(ar_model_hps)
    ar_model = prepare_models(args, devices)

    if args.use_bpe:
        bpe_tokenizer = LlamaTokenizer.from_pretrained(
            args.bpe_tokenizer
        )
        bpe_tokens_num = len(bpe_tokenizer)
    else:
        bpe_tokens_num = 0
    print("bpe_tokens_num: ", bpe_tokens_num)

#    ar_step, ar_epoch = get_step_epoch_from_ckpt(args.ar_ckpt_path)
#    nar_step, nar_epoch = get_step_epoch_from_ckpt(args.nar_ckpt_path)
#    out_dir = os.path.join(args.out_dir, 
#        f"ar_step{ar_step//1000}k_epoch{ar_epoch}-nar_step{nar_step//1000}k_epoch{nar_epoch}")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    #### prepare dataset for inference !!!!
    # test_dataset = ContinuousTTSDataset(
    #     args.meta_file, hp=ar_model_hps, return_full_seq=True, inference=True
    # )
    # test_data_loader = DataLoader(
    #     test_dataset,
    #     num_workers=0,
    #     shuffle=False,
    #     sampler=None,
    #     batch_size=1,
    #     collate_fn=ContinuousCollator(tokenizer_pad=0),
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # spk_name = ['BaileyP', 'daily_life_F01-phone_prompt1', 'daily_life_F02-phone_prompt1', 'daily_life_F03-phone_prompt1',
    #     'daily_life_M01-phone_prompt1', 'daily_life_M02-phone_prompt1', 'daily_life_M03-phone_prompt1', 
    #     'Isabelle.npy', 'James', 'NayS', 'StellaY', 'TiaC']
    text2id = TextToTacolabID(args.lab2id_path, args.textid_version, args.use_sy)
    tokenizer = PhoneTokenizerWithAudioTokens(ar_model_hps.phone_tokens_num, ar_model_hps.speaker_tokens_num, bpe_tokens_num=bpe_tokens_num)
    symbol_sets = ['.', ',', '?', '!', '，', '。', '？', '！']

    if args.use_spkid:
        f = open(args.spk2id)
        spk_dict = json.load(f)
        spk_id = spk_dict[args.spkname]
        spk_id = tokenizer.tokenize(int(spk_id), "spk")

    for line in open(args.meta_file):
        if not args.use_spkid:
            infer_utt, prompt_text, prompt_wav_path, infer_text = line.strip().split("|")
            prompt_utt = prompt_wav_path.split('/')[-1][:-4]
        else:
            infer_utt, infer_text = line.strip().split("|")

        # text_id: text2id([prompt_tacolab, infer_tacolab])
        if not args.use_spkid:
            prompt_tacolab_path = os.path.join(args.prompt_tacolab_dir, prompt_utt + '.lab')
            infer_tacolab_path = os.path.join(args.infer_tacolab_dir, infer_utt + '.lab')
            prompt_tacolab = open(prompt_tacolab_path).read().strip('\n ').split('\n')
            infer_tacolab = open(infer_tacolab_path).read().strip('\n ').split('\n')
            tacolab = '\n'.join(prompt_tacolab + infer_tacolab[1:])
            tacolab = list(filter(lambda x: x != "", tacolab.split('\n')))
            text_id = text2id(tacolab) #[N]
        else:
            infer_tacolab_path = os.path.join(args.infer_tacolab_dir, infer_utt + '.lab')
            infer_tacolab = open(infer_tacolab_path).read().strip('\n ').split('\n')
            tacolab = '\n'.join(infer_tacolab)
            tacolab = list(filter(lambda x: x != "", tacolab.split('\n')))
            text_id = text2id(tacolab) #[N]

        text_id = tokenizer.tokenize(text_id, "inputs")

        # print("text_id: ", text_id.shape)
        # text_id_type = text_id.dtype
        # text_id = np.zeros([0]).astype(text_id_type)

        if not args.use_spkid:
            if prompt_text[-1] in symbol_sets:
                text = prompt_text + ' ' + infer_text.strip()
            else:
                text = prompt_text + ', ' + infer_text
        else:
            text = infer_text

        if args.use_bpe:
            bpe_id = np.asarray(bpe_tokenizer(
                text, max_length=4096, truncation=True
            ).input_ids)
            bpe_id = tokenizer.tokenize(bpe_id, "bpe")

            # print("bpe_id: ", bpe_id.shape)
            # bpe_id_type = bpe_id.dtype
            # bpe_id = np.zeros([0]).astype(bpe_id_type)

            text_id = np.concatenate([
                bpe_id,
                [tokenizer.sep],
                text_id
            ])

        text_id_len = text_id.shape[0]

        # bn
        if not args.use_spkid:
            prompt_bn_path = os.path.join(args.prompt_bn_dir, prompt_utt + '.npy')
            bn = np.load(prompt_bn_path)
        else:
            bn = np.zeros([0, 32])
        
        bn_T, bn_C = bn.shape[0], bn.shape[1]
        if not args.use_spkid:
            # print("bn: ", bn.shape)
            # bn_type = bn.dtype
            # bn = np.zeros([0, bn.shape[1]]).astype(bn_type)

            # seq
            seq = (
                [tokenizer.bos]
                + list(text_id)
                + [tokenizer.sep]
                + [0] * bn_T # place holder
            )
        else:
            seq = (
                [tokenizer.bos]
                + list(text_id)
                + [tokenizer.sep]
                + [spk_id]
                + [0] * bn_T # place holder
            )

        text_ids = torch.from_numpy(text_id).unsqueeze(0)
        text_id_lens = torch.from_numpy(np.array([text_id.shape[0]]))
        bns = torch.from_numpy(bn).unsqueeze(0)
        bn_lens = torch.from_numpy(np.array([bn.shape[0]]))
        seq = np.asarray(seq)
        seqs = torch.from_numpy(seq).unsqueeze(0)
        seq_lens = torch.from_numpy(np.array([seq.shape[0]]))

        utts = [infer_utt]

        text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, utts = to_device(
            (text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, utts), device=devices
        )
        
        # ar
        z_outputs, semantic_outputs = ar_model.inference_from_text(
            (text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, utts), tokenizer
        )
        # b, t, c = semantic_outputs.shape
        # semantic_outputs = semantic_outputs.cpu().numpy()
        b, t, c = z_outputs.shape
        z_outputs = z_outputs.cpu().numpy()

        save_path = os.path.join(args.out_dir, '%s.npy'%utts[0])
        # save_path = os.path.join(args.out_dir, '%s_%s.npy'%(spk_name[i//20], utts[0]))

        print("save %s" % save_path)
        np.save(save_path, z_outputs)
        


        # for j in range(3):
        #     text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, pos_ids, seq_sen_ids, init_full_seqs, utts = to_device(
        #         loaded_data, device=devices
        #     )
            
        #     text_len = (seq_sen_ids == 1).sum(dim=1)
        #     unmask_len = (seq_sen_ids == 2).sum(dim=1)
        #     # num_res = init_full_seqs.shape[2]
        #     # ar
        #     semantic_outputs, pos_ids, seq_sen_ids = ar_model.inference_from_text(
        #         (text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, pos_ids, seq_sen_ids, utts), test_dataset.tokenizer
        #     )
        #     b, t, c = semantic_outputs.shape
        #     semantic_outputs = semantic_outputs.cpu().numpy()

        #     np.save(os.path.join(args.out_dir, '%s_%d.npy'%(utts[0], j)), semantic_outputs)

        # if i > 1 :
        #     break
        # exit()

        

if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_ckpt_path", type=str, required=True)
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument("--prompt_tacolab_dir", type=str, required=True)
    parser.add_argument("--infer_tacolab_dir", type=str, required=True)
    parser.add_argument("--prompt_bn_dir", type=str, required=True)
    parser.add_argument("--lab2id_path", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True, help="text file")
    parser.add_argument("--bpe_tokens_num", type=int, 
                        default=32000)
    parser.add_argument("--use_bpe", type=bool, const=True,
                        default=False, nargs="?")
    parser.add_argument("--bpe_tokenizer", type=str, 
                        default="recipes/text2semantic/datasets/dict/llama_7B_tokenizer", 
                        help="bpe tokenizer model path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_spkid", type=bool, const=True,
                        default=False, nargs="?")
    parser.add_argument("--spk2id", type=str, default=None)
    parser.add_argument("--spkname", type=str, default=None)
    parser.add_argument("--textid_version", type=str, default="v2")
    parser.add_argument("--use_sy", type=bool, const=True,
                        default=False, nargs="?")

    args = parser.parse_args()
    main(args)

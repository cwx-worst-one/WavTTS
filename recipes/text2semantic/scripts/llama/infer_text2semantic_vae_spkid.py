import argparse
import os
import numpy as np

import torch
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from recipes.text2semantic.datasets.continuous_spkid_dataset import ContinuousTTSDataset, ContinuousCollator
from recipes.text2semantic.lit_modules.llama.lit_vae_t2s_ctiga import VAET2SModule
from recipes.text2semantic.scripts.infer_utils import setup_seed
from samantha.utils.hparams import DotDict


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
    devices = args.device
    # 这里务必要和训练一致！！！！！
    ar_model_hps = DotDict({"phone_tokens_num": 7370, "speaker_tokens_num": 8192})
    ar_model = prepare_models(args, devices)
        
#    ar_step, ar_epoch = get_step_epoch_from_ckpt(args.ar_ckpt_path)
#    nar_step, nar_epoch = get_step_epoch_from_ckpt(args.nar_ckpt_path)
#    out_dir = os.path.join(args.out_dir, 
#        f"ar_step{ar_step//1000}k_epoch{ar_epoch}-nar_step{nar_step//1000}k_epoch{nar_epoch}")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    #### prepare dataset for inference !!!!
    test_dataset = ContinuousTTSDataset(
        args.meta_file, hp=ar_model_hps, return_full_seq=True, inference=True
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=0,
        shuffle=False,
        sampler=None,
        batch_size=1,
        collate_fn=ContinuousCollator(tokenizer_pad=0),
        pin_memory=True,
        drop_last=True,
    )


    for i, loaded_data in enumerate(test_data_loader):
        # seqs, seq_lens, pos_ids, seq_sen_ids, init_full_seqs, utts
        if loaded_data is None:
            print('ignore ...')
            continue

        # if i % 50 != 0:
        #     continue

        
        text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, pos_ids, seq_sen_ids, init_full_seqs, utts, spk_ids = to_device(
            loaded_data, device=devices
        )
        
        text_len = (seq_sen_ids == 1).sum(dim=1)
        unmask_len = (seq_sen_ids == 2).sum(dim=1)
        # num_res = init_full_seqs.shape[2]
        # ar
        z_outputs, semantic_outputs = ar_model.inference_from_text(
            (text_ids, text_id_lens, bns, bn_lens, seqs, seq_lens, utts), test_dataset.tokenizer
        )
        # b, t, c = semantic_outputs.shape
        # semantic_outputs = semantic_outputs.cpu().numpy()
        b, t, c = z_outputs.shape
        z_outputs = z_outputs.cpu().numpy()

        save_path = os.path.join(args.out_dir, '%s.npy'%utts[0])

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

        

if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_ckpt_path", type=str, required=True)
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True, help="text file")
    parser.add_argument("--seed", type=int, default=1996)

    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)

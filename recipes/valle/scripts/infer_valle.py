import argparse
import os

import torch
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from recipes.valle.datasets import GPT2TTSDataset, ValleCollator
from recipes.valle.lit_modules import ValleCoarse, ValleFine
from recipes.soundstream.models.vqgan_res import VQGAN
from recipes.soundstream.utils.utils import get_config_from_file
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
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device


def prepare_models(args, device):
    ar_model = ValleCoarse.load_from_checkpoint(
        args.ar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    ar_model.eval()

    nar_model = ValleFine.load_from_checkpoint(
        args.nar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    nar_model.eval()
    
    hp = get_config_from_file(args.codec_config).hparams
    vqgan_model = VQGAN(hp).to(device)
    ckpt = torch.load(args.codec_ckpt, map_location=device)
    vqgan_model.load_state_dict(ckpt["G"])
    vqgan_model.eval()

    return ar_model, nar_model, vqgan_model


@torch.no_grad()
def main(args):
    devices = args.device
    ar_model_hps = DotDict({"phone_tokens_num": 200, "audio_tokens_num": 1024})
    ar_model, nar_model, vqgan_model = prepare_models(args, devices)
    os.makedirs(args.out_dir, exist_ok=True)
    #### prepare dataset for inference !!!!
    test_dataset = GPT2TTSDataset(
        args.meta_file, hp=ar_model_hps, return_full_seq=True, inference=True
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=0,
        shuffle=False,
        sampler=None,
        batch_size=1,
        collate_fn=ValleCollator(tokenizer_pad=0),
        pin_memory=True,
        drop_last=True,
    )

    temperature = 1.0
    for i, loaded_data in enumerate(test_data_loader):
        seqs, seq_lens, pos_ids, seq_sen_ids, init_full_seqs, utts = to_device(
            loaded_data, device=devices
        )
        text_len = (seq_sen_ids == 1).sum(dim=1)
        unmask_len = (seq_sen_ids == 2).sum(dim=1)
        num_res = init_full_seqs.shape[2]

        # ar
        seqs, pos_ids, seq_sen_ids, _ = ar_model.generate(
            (seqs, seq_lens, pos_ids, seq_sen_ids), test_dataset.tokenizer
        )
        b, t = seqs.shape
        
        full_seqs = torch.stack([seqs] * num_res, dim=1)  # [b, n_codebook, t]
        full_seqs[:, :, text_len[0] : text_len[0] + unmask_len[0]] = \
            init_full_seqs[:, text_len[0] :, :].transpose(1, 2)
        layer_index = torch.ones(size=[b], device=devices)  # [b,]
        mask1 = layer_index.unsqueeze(1) > torch.arange(num_res, device=devices).unsqueeze(0)  # [b, 1] > [1, n_codebook] = [b, n_codebook]
        mask2 = (text_len + unmask_len).unsqueeze(1) > torch.arange(t, device=devices).unsqueeze(0)  # [b, 1] > [1, t] = [b, t]
        mask = mask1.unsqueeze(2) + mask2.unsqueeze(1)  # [b, n_codebbok, 1] + [b, 1, t]
        full_seqs = torch.where(mask, full_seqs, torch.zeros_like(full_seqs))

        # nar
        full_seqs = nar_model.generate(full_seqs,
                                       unmask_len, 
                                       seq_sen_ids, 
                                       pos_ids, 
                                       mask2, 
                                       test_dataset.tokenizer, 
                                       temperature=temperature)

        full_seq = full_seqs[:, :, text_len[0] :]  # [b, n_codbook, t]
        full_seq = (full_seq - test_dataset.tokenizer.phone_token_num - 1)[:, :, :-1]

        wav2 = vqgan_model.decode(vqgan_model.get_quant_output_from_index(full_seq))
        wav2 = wav2.cpu().squeeze(1).squeeze(0).numpy()
        save_wav(wav2, os.path.join(args.out_dir, "{}.wav".format(utts[0])), sr=24000)
        wav_gen = wav2[unmask_len * 300 :]
        save_wav(wav_gen, os.path.join(args.out_dir, "{}.wav".format(utts[0] + "-gen")),sr=24000)

if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_ckpt_path")
    parser.add_argument("--nar_ckpt_path", type=str, required=True)
    parser.add_argument("--codec_config", type=str, required=True)
    parser.add_argument(
        "--codec_ckpt", type=str, required=True, help="codec model ckpt path"
    )
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True, help="text file")
    args = parser.parse_args()
    main(args)

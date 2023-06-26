import argparse
import os

import torch
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

from recipes.valle.datasets import GPT2TTSDataset, ValleCollator
from recipes.valle.lit_modules.lit_valle_fine import ValleFine
from recipes.soundstream.models.vqgan_res import VQGAN
from recipes.soundstream.utils.utils import get_config_from_file
from samantha.utils.hparams import DotDict
from recipes.valle.utils.model_init import init_sound_stream_decoder
from recipes.valle.scripts.infer_utils import prepare_models_llama, save_wav, to_device, get_step_epoch_from_ckpt, prepare_models_ctiga_llama


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

def get_step_epoch_from_ckpt(ckpt_path):
    data = torch.load(ckpt_path)
    global_step = data["global_step"]
    epoch = data["epoch"]
    return global_step, epoch

def prepare_models(args, device):
    ar_model = ValleCoarseModule.load_from_checkpoint(
        args.ar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    ar_model.eval()

    nar_model = ValleFine.load_from_checkpoint(
        args.nar_ckpt_path, device=torch.device(device)
    ).to(args.device)
    nar_model.eval()

    assert device.split(':')[0] == 'cuda'
    vqgan_model = init_sound_stream_decoder(
        args.codec_ckpt_path, device.split(':')[1], cache_dir=".cache_dir")["ss_dec"]
    vqgan_model.eval()

    return ar_model, nar_model, vqgan_model


@torch.no_grad()
def main(args):
    devices = args.device
    ar_model_hps = DotDict({"phone_tokens_num": 8000, "audio_tokens_num": 1024})
    # ar_model, nar_model, vqgan_model = prepare_models(args, devices)
    if args.ctiga:
        ar_model, nar_model, vqgan_model =  prepare_models_ctiga_llama(args, devices)
    else:
        ar_model, nar_model, vqgan_model =  prepare_models_llama(args, devices)
        
#    ar_step, ar_epoch = get_step_epoch_from_ckpt(args.ar_ckpt_path)
#    nar_step, nar_epoch = get_step_epoch_from_ckpt(args.nar_ckpt_path)
#    out_dir = os.path.join(args.out_dir, 
#        f"ar_step{ar_step//1000}k_epoch{ar_epoch}-nar_step{nar_step//1000}k_epoch{nar_epoch}")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
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
        seqs, pos_ids, seq_sen_ids = ar_model.inference_from_text(
            (seqs, seq_lens, pos_ids, seq_sen_ids, utts), test_dataset.tokenizer
        )
        b, t = seqs.shape
        
        full_seqs = torch.stack([seqs] * num_res, dim=1)  # [b, n_codebook, t]
        full_seqs[:, :, text_len[0] : text_len[0] + unmask_len[0]] = \
            init_full_seqs[:, text_len[0] :, :].transpose(1, 2)
        layer_index = torch.ones(size=[b], device=devices)  # [b,]
        mask1 = layer_index.unsqueeze(1) > torch.arange(num_res, device=devices).unsqueeze(0)  # [b, 1] > [1, n_codebook] = [b, n_codebook]
        mask2 = (text_len + unmask_len).unsqueeze(1) > torch.arange(t, device=devices).unsqueeze(0)  # [b, 1] > [1, t] = [b, t]
        mask = mask1.unsqueeze(2) + mask2.unsqueeze(1)  # [b, n_codebbok, 1] + [b, 1, t]
        # print("mask: ", mask.shape, mask)
        full_seqs = torch.where(mask, full_seqs, torch.zeros_like(full_seqs))
        # print("full_seqs: ", full_seqs.shape, full_seqs)

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

        wav2 = vqgan_model["decoder"](full_seq)
        wav2 = wav2.cpu().squeeze(1).squeeze(0).numpy()

        ### wav_prompt_infer
        os.makedirs(os.path.join(args.out_dir, 'gen_with_prompt'), exist_ok=True)
        save_wav(wav2, os.path.join(out_dir, 'gen_with_prompt', "{}.wav".format(utts[0])), sr=24000)

        ### wav_infer
        os.makedirs(os.path.join(args.out_dir, 'gen'), exist_ok=True)
        wav_gen = wav2[unmask_len * 300 :]
        save_wav(wav_gen, os.path.join(out_dir, 'gen', "{}.wav".format(utts[0])),sr=24000)

if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_ckpt_path", type=str, required=True)
    parser.add_argument("--nar_ckpt_path", type=str, required=True)
    parser.add_argument("--codec_ckpt_path", type=str, 
                        default="hdfs://harunava/home/byte_speech_sv/litang/2023-01-17_causal_x300_1024_6book_doubleG_export", 
                        help="codec model ckpt path")
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True, help="text file")
    parser.add_argument("--ctiga", type=bool, default=False, const=True, nargs="?")

    args = parser.parse_args()
    main(args)

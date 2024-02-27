import argparse
from recipes.voicebox.vocoder.wvae import Wave
from recipes.voicebox.utils.infer_utils import save_wav, load_torch_script
import numpy as np
import torch
import os

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    wvae_encoder = load_torch_script(
        model_path=args.wvae_encoder_path,
        rank=0,
        cache_dir=args.wvae_cache_dir
    )
    wvae_decoder = load_torch_script(
        model_path=args.wvae_decoder_path,
        rank=0,
        cache_dir=args.wvae_cache_dir
    )
    wvae = Wave(
        wvae_encoder, 
        wvae_decoder, 
        version=args.version,
        hop_size=args.wvae_encoder_hop_size,
        win_size=args.wvae_encoder_win_size
    )
    wav = np.zeros(60 * 24000)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    wav = wav.to(args.device)

    with torch.no_grad():
        bn = wvae.encode(wav)
        output_bn_path = os.path.join(args.out_dir, "silence_wvae.npy")
        np.save(output_bn_path, bn.squeeze().cpu().numpy())
        
        reconstruct_wav = wvae.decode(bn)
        output_wav_path = os.path.join(args.out_dir, "silence_reconstruct.wav")
        save_wav(reconstruct_wav.squeeze().cpu().numpy(), output_wav_path)
        
        output_wav_path = os.path.join(args.out_dir, "silence.wav")
        save_wav(wav.squeeze().cpu().numpy(), output_wav_path)


if __name__ == "__main__":
    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--wvae_encoder_path", type=str, required=True)
    parser.add_argument("--wvae_decoder_path", type=str, required=True)
    parser.add_argument("--wvae_cache_dir", type=str, required=True)
    parser.add_argument("--wvae_encoder_hop_size", type=int, default=300)
    parser.add_argument("--wvae_encoder_win_size", type=int, default=1200)
    parser.add_argument("--version", type=float, required=True)
    parser.add_argument(
        "--device", type=str, default="cpu", help='Inference device, "cpu" or "cuda"'
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

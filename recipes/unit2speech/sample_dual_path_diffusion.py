"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from loader.audio_loader import load_data
from pydub import AudioSegment
from scipy.io.wavfile import write
from tqdm import tqdm

from recipes.unit2speech.models.autoencoder.utils import get_config_from_file
from recipes.unit2speech.models.diffusion.embedder import FrozenSSLEmbedder
from recipes.unit2speech.models.diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

MAX_WAV_VALUE = 32768.0


def remove_ddp_module(ckpt):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key in ckpt:
        new_key = key.replace('module.', '', 1)
        new_dict[new_key] = ckpt[key]
    return new_dict


def cos_sim(x, y):
    return F.cosine_similarity(x, y, dim=-1)


def save_wav(audio, output_file, sr=24000):
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    write(output_file, sr, audio)


def load_prompt_wav(path):
    wav = AudioSegment.from_file(path)
    wav = wav.set_channels(1)
    wav = wav.set_frame_rate(24000)
    wav = np.asarray(wav.get_array_of_samples())
    wav = wav.ravel().astype(np.float32)
    wav_abs_max = np.abs(wav).max()
    wav = wav / wav.std() * 0.13
    if np.abs(wav).max() >= 0.99:
        wav = wav / (np.abs(wav).max() + 1e-2)
    return wav


def main():
    args = create_argparser().parse_args()

    output_dir = os.path.join(args.model_ckpt_dir, 'samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # DDP sampling setting
    accelerator = Accelerator()
    #device = torch.device('cpu')
    device = accelerator.device
    
    # conditional on audio/text embedding
    embedder = FrozenSSLEmbedder(device=accelerator.device)
    embedder = embedder.to(accelerator.device)
    accelerator.print("Condition Embedder ({}) Params: {:.4f}M".format(
       args.cond_embedder, sum(p.numel() for p in embedder.parameters()) / 1e6))
    
    # prepare diffusion model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if not isinstance(diffusion, nn.Module):
        diffusion.cuda = accelerator.process_index
        diffusion = diffusion.to(device)
    
    # prepare audio encoder
    if args.autoencoder_config:
        hp = get_config_from_file(f"{args.autoencoder_config}").hparams
    else:
        hp = get_config_from_file(f"{args.autoencoder}/config.yaml").hparams
    if args.autoencoder == 'autoencoder':
        from recipes.unit2speech.models.autoencoder.autoencoder_kl import AutoencoderKL
        autoencoder = AutoencoderKL(hp, stage='enc', device=device)
        ckpt = torch.load(args.autoencoder_path, map_location='cpu')
        state = remove_ddp_module(ckpt['G'])
        for k in state:
            if 'encoder.cnt' in k:
                del state[k]
                break
        autoencoder.load_state_dict(state)
        autoencoder.eval()
    else:
        raise NotImplementedError
    if autoencoder is not None:
        accelerator.print("Audio Encoder ({}) Params: {:.4f}M".format(
            args.autoencoder, sum(p.numel() for p in autoencoder.parameters()) / 1e6))
    
    if args.use_ema:
        ema_ckpt = torch.load(os.path.join(args.model_ckpt_dir, 'ema_diffusion_model.pt'), map_location='cpu')
        ema_state_dict = remove_ddp_module(ema_ckpt)
    ckpt = torch.load(os.path.join(args.model_ckpt_dir, 'diffusion_model.pt'), map_location='cpu')
    state_dict = remove_ddp_module(ckpt)
    init_dict = model.state_dict()
    for k in init_dict:
        if args.use_ema and k in ema_state_dict:
            state_dict[k] = ema_state_dict[k]
        if k not in state_dict:
            state_dict[k] = init_dict[k]
    for k in state_dict:
        if k not in init_dict:
            accelerator.print(k, 'in checkpoint not in model!', flush=True)
    model.load_state_dict(state_dict)
    model.eval()
    accelerator.print("Diffusion Model Params: {:.4f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    
    #model = accelerator.prepare(model)
    model = model.to(device)
    if autoencoder is not None:
        autoencoder = autoencoder.to(device)

    accelerator.print("start sampling...")
    
    # prepare dataset
    data_loader = load_data(
        wav_list=args.wav_list,
        accelerator=accelerator,
        batch_size=args.batch_size,
        return_path=True,
        use_formant_shift=False,
    )
    
    classifier_free_guidance = args.classifier_free_guidance
    i = 0
    cnt_audios = 0
    with torch.no_grad():
        random.seed(0)
        #start_noise = torch.randn(1, 8, 2500, device=device)
        for data in tqdm(data_loader):
            (wavs, wav_lens), tokens, paths = data
            if tokens is None:
                tokens = embedder((wavs, wav_lens))
                token_lens = torch.tensor([tokens.shape[-1],]).long()
            else:
                tokens, token_lens = tokens
            wavs, tokens, token_lens = wavs.to(device), tokens.to(device), token_lens.to(device)
            if args.test_prompt and i % 2 == 0:
                prompt = wavs
                i += 1
                continue
            i += 1
            zs, z_lens = autoencoder(wavs, wav_lens)
            if not args.test_prompt:
                prompt = wavs
            start_time = time.time()
            sample = diffusion(
                args.batch_size,
                args.diffusion_steps,
                xT=None,
                skip_steps=0,
                show_progress=True,
                classifier_free_guidance=classifier_free_guidance,
                model_kwargs={"context": (tokens, token_lens),
                              "z_lens": z_lens,
                              "prompt": prompt[..., :3 * 24000]}
            )
            dpd_time = time.time() - start_time
            #accelerator.print('RTF: ', dpd_time / (wav_lens[0] / 24000), flush=True)
            start_time = time.time()
            for i in range(len(sample)):
                wav_len = wav_lens[i]
                wav_name = paths[i].split('/')[-1].split('.')[0]
                audio = autoencoder.decode(sample[i, :, :z_lens[i]][None]).detach().cpu()
                audio = audio / audio.abs().max(-1, keepdim=True)[0] * 0.6
                audio = audio.numpy().ravel()
                filename = f'{output_dir}/{wav_name}.wav'
                sf.write(filename, audio, 24000)
                ori_sample = wavs[i].cpu()[..., :wav_len]
                ori_sample = ori_sample / ori_sample.abs().max(-1, keepdim=True)[0] * 0.6
                audio_ori = ori_sample.numpy().ravel()
                filename = f'{output_dir}/{wav_name}_gt.wav'
                sf.write(filename, audio_ori, 24000)
                if args.test_prompt:
                    audio_prompt = prompt[i, :3 * 24000]
                    audio_prompt = audio_prompt / audio_prompt.abs().max(-1, keepdim=True)[0] * 0.6
                    filename = f'{output_dir}/{wav_name}_prompt.wav'
                    sf.write(filename, audio_prompt.cpu().numpy().ravel(), 24000)
            cnt_audios += len(sample) * accelerator.num_processes
            torch.cuda.empty_cache()
        accelerator.print("sampling complete")


def create_argparser():
    defaults = dict(
        wav_list="",
        autoencoder="autoencoder",  # supports {autoencoder, soundstream}
        autoencoder_path="",
        autoencoder_config="",
        cond_embedder="ssl",  # supports {t5, mulan, clap, clip}
        test_prompt=False,
        use_ema=False,
        classifier_free_guidance=1.,  # 1 for conditional; 0 for unconditional
        batch_size=2,
        model_ckpt_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

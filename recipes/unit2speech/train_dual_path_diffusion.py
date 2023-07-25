"""
Train a diffusion model on images.
"""

import argparse

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs

from recipes.unit2speech.loader.audio_loader import load_data
from recipes.unit2speech.models.diffusion import logger
from recipes.unit2speech.models.diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from recipes.unit2speech.modules.trainer import TrainLoop
from recipes.unit2speech.modules.utils import get_config_from_file


def remove_ddp_module(ckpt):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key in ckpt:
        new_key = key.replace('module.', '', 1)
        new_dict[new_key] = ckpt[key]
    return new_dict


def main():
    args = create_argparser().parse_args()
    logger.configure()

    # prepare DDP trainer
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
   
    embedder = None
    # conditional on audio/text embedding
    #if args.cond_embedder.lower() == 'ssl':
    #    embedder = FrozenSSLEmbedder(device=accelerator.device)
    #    args.context_dim = 1
    #else:
    #    embedder = None
    #if embedder is not None:
    #    accelerator.print("Condition Embedder ({}) Params: {:.4f}M".format(
    #       args.cond_embedder, sum(p.numel() for p in embedder.parameters()) / 1e6))
    
    # prepare diffusion model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.cuda = accelerator.process_index

   
    # prepare audio encoder
    if args.autoencoder_config:
        hp = get_config_from_file(f"{args.autoencoder_config}").hparams
    else:
        hp = get_config_from_file(f"{args.autoencoder}/config.yaml").hparams
    if args.autoencoder == 'autoencoder':
        from recipes.unit2speech.models.autoencoder.autoencoder_kl import AutoencoderKL
        autoencoder = AutoencoderKL(hp, device=accelerator.device, stage='enc')
        ckpt = torch.load(args.autoencoder_path, map_location='cpu')
        state = remove_ddp_module(ckpt['G'])
        autoencoder.load_state_dict(state)
        autoencoder.eval()
        del autoencoder.decoder
    else:
        raise NotImplementedError
    if autoencoder is not None:
        accelerator.print("Audio Encoder ({}) Params: {:.4f}M".format(
            args.autoencoder, sum(p.numel() for p in autoencoder.parameters()) / 1e6))
    accelerator.print("Diffusion Model Params: {:.4f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    
    # prepare dataset
    data_loader = load_data(
        wav_list=args.wav_list,
        accelerator=accelerator,
        batch_size=args.batch_size,
        use_formant_shift=True,
    )
    accelerator.print("Dataset contains {} samples".format(len(data_loader.dataset)), flush=True)
    
    TrainLoop(
        accelerator=accelerator,
        model=model,
        autoencoder=autoencoder,
        diffusion=diffusion,
        embedder=embedder,
        data_loader=data_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_ckpt_dir=args.resume_ckpt_dir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        end2end=args.end2end,
    ).run_loop()


def create_argparser():
    defaults = dict(
        wav_list="",
        autoencoder="autoencoder",  # supports {autoencoder, soundstream}
        autoencoder_path="",
        autoencoder_config="",
        cond_embedder="ssl",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate=0.999,
        log_interval=10,
        save_interval=10000,
        resume_ckpt_dir="",
        use_fp16=False,
        fp16_scale_growth=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

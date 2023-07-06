import os
import torch
from recipes.soundstream.models.vqgan import VQGAN_KL
from recipes.soundstream.modules.pl_module_vae import VocoderModule

def init_vocoder(trainer, path, device, cache_dir=None):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    local_path = f"{cache_dir}/{os.path.basename(path)}"

    if path.startswith("hdfs://") or path.startswith("/home"):
        if trainer.local_rank == 0:
            if not os.path.exists(local_path):
                try:
                    os.system(f"hdfs dfs -get {path} {cache_dir}")
                except Exception:
                    raise ConnectionError(f"Cannot retrieve file from {path}.")
        trainer.strategy.barrier()
    
    # TODO: put model confic somewhere else
    gen = VQGAN_KL(
        model_type='bytewave_wn_causal',
        quant_token_dim=128,
        down_rates=[2, 3, 4, 4],
        upsample_rates=[4, 4, 3, 2],
        encoder_initial_channel=8,
        decoder_initial_channel=768,
        trunc_noise=False,
        smaller_encoder=True,
        init_cluster_size=32,
        dist=False,
    )
    model = VocoderModule.load_from_checkpoint(local_path, generator=gen, discriminator=None, balancer=None, strict=False)
    model.to(device)
    model.eval()

    return {
        "model": model.generator, # model.generator.encoder
    }

def init_vocoder_yongye(trainer, path, device, cache_dir=None):
    def remove_ddp_module(ckpt):
        from collections import OrderedDict
        new_dict = OrderedDict()
        for key in ckpt:
            new_key = key.replace('module.', '', 1)
            new_dict[new_key] = ckpt[key]
        return new_dict

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    local_path = f"{cache_dir}/{os.path.basename(path)}"

    if path.startswith("hdfs://") or path.startswith("/home"):
        if trainer.local_rank == 0:
            if not os.path.exists(local_path):
                try:
                    os.system(f"hdfs dfs -get {path} {cache_dir}")
                except Exception:
                    raise ConnectionError(f"Cannot retrieve file from {path}.")
        trainer.strategy.barrier()
    
    # TODO: put model confic somewhere else
    model = VQGAN_KL(
        model_type='bytewave_wn',
        quant_token_dim=256,
        down_rates=[2, 3, 4, 4],
        upsample_rates=[4, 4, 3, 2],
        encoder_initial_channel=16,
        decoder_initial_channel=768,
        trunc_noise=False,
        smaller_encoder=True,
        init_cluster_size=32,
        dist=False,
    )
    ckpt = torch.load(local_path, map_location='cpu')
    state = remove_ddp_module(ckpt['G'])
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return {
        "model": model,
    }

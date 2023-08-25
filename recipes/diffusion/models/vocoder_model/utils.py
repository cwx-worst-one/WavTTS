import os
import torch
from recipes.soundstream.models.vqgan import VQGAN_KL, VQGAN_KL_new
from recipes.soundstream.modules.pl_module_vae import VocoderModule

def load_ema_checkpoint(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # divide param group
    no_decay = [
        "bn",
        "bias",
        "norm"
        "rotary",
        "embedding",
        ".g", # g in RMSNorm
    ]

    base_params = {}
    no_decay_params = {}
    for name, param in model.named_parameters(): 
        _found = False
        for k in no_decay:
            if k in name:
                no_decay_params[name] = param
                _found = True
                break
        if not _found:
            base_params[name] = param
    # combine the two dictionaries into one
    new_state_dict = {}
    new_keys = []
    for k, v in base_params.items():
        new_state_dict[k] = v
        new_keys.append(k)
    for k, v in no_decay_params.items():
        new_state_dict[k] = v
        new_keys.append(k)

    for idx, k in enumerate(new_keys):
        assert new_state_dict[k].shape == ckpt["optimizer_states"][0]["ema"][idx].shape
        new_state_dict[k] = ckpt["optimizer_states"][0]["ema"][idx]

    model.load_state_dict(new_state_dict)
    return model

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
    model = generator=VQGAN_KL_new(
            latent_dim=32,
            downsample_rates=[2, 3, 4, 8],
            upsample_rates=[8, 4, 3, 2],
            encoder_base_dim=96,
            decoder_base_dim=2560,
        )

    vocoder_model = load_ema_checkpoint(local_path, model).eval().to(device)

    return {
        "model": vocoder_model, # model.generator.encoder
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


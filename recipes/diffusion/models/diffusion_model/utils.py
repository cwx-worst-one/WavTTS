import os
import torch
from recipes.musiclm.utils.dist import local_zero_first
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.diffusion.models.tnt_mulan_free import TNTDiffusionNetwork
from recipes.diffusion.models.tnt_gru import TNTDiffusionNetwork as ZhTNTDiffusionNetwork

VOCODER_HZ = 125

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
        shape1 = new_state_dict[k].shape
        shape2 = ckpt["optimizer_states"][0]["ema"][idx].shape
        assert shape1 == shape2, f"idx={idx}, k={k}, shape1={shape1}, shape2={shape2}"
        
        new_state_dict[k] = ckpt["optimizer_states"][0]["ema"][idx]

    model.load_state_dict(new_state_dict)
    return model

def init_diffusion(checkpoint_path, local_rank, cache_dir, is_zh_token=False):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        if is_zh_token:
            diffusion_network = ZhTNTDiffusionNetwork(
                input_dim=32,
                feature_dim=1024,
                context_dim=1,
                depth=16,
                segment_size=32,
                segment_stride=32,
                dropout=0,
                semantic_cfg_prob=0.10,
                use_checkpoint=False
            )
        else:
            diffusion_network = TNTDiffusionNetwork(
                input_dim=32,
                feature_dim=1024,
                context_dim=1,
                depth=16,
                segment_size=32,
                segment_stride=32,
                dropout=0,
                semantic_cfg_prob=0.10,
                use_checkpoint=False
            )            
        diffusion_model = load_ema_checkpoint(
            local_path,
            diffusion_network,
        )
        diffusion_model.eval().to(device)
        return { "diffusion": diffusion_model }

@torch.no_grad()
def run_diffusion(requires, samples, params):
    diffusion_model = requires['diffusion']
    sampler = requires['sampler']
    vocoder = requires['vocoder']

    num_chunks = params.get('num_chunks', 1)
    diffusion_steps = params.get('diffusion_steps', 25)
    schedule_slope = params.get('schedule_slope', 2.5)
    guidance_scale = params.get('guidance_scale', 2.5)
    bf16_portion = params.get('bf16_portion', 0.0)

    pred_emb = sampler(
        model=diffusion_model,
        semantic_context=samples,
        num_items=samples.shape[0],
        num_chunks=num_chunks,
        num_steps=diffusion_steps,
        bf16_portion=bf16_portion,
        start=None,
        show_progress=True,
        angle_schedule='linear',
        schdeule_slope=schedule_slope,
        classifier_free_guidance=guidance_scale,
    ).detach()

    pred_emb = pred_emb.float()
    # torch.interpolate causes OOM for large batch sizes > 24. chunking to batch of 8 instead.
    # If you see this error, lower batch size: "RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false."
    duration = pred_emb.shape[-1] // VOCODER_HZ
    batch_chunks = 8 if duration < 60 else 2
    wavs_g = torch.cat([vocoder.decode(c).detach() for c in torch.split(pred_emb, batch_chunks)])
    # wavs_g = vocoder.decode(pred_emb.float()).detach()

    # For bigmusic: [bs, c, seq] -> [bs, seq] 
    if len(wavs_g.shape) == 3:
        wavs_g = wavs_g.squeeze(1)

    return wavs_g



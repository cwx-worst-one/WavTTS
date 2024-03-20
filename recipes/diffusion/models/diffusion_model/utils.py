import os
from urllib.parse import non_hierarchical
import torch
from recipes.musiclm.utils.dist import local_zero_first
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.diffusion.models.vocoder_model.utils import vocode_in_chunks
from recipes.diffusion.models.tnt_mulan_free import TNTDiffusionNetwork
from recipes.diffusion.models.tnt_mss import TNTDiffusionNetwork as TNTDiffusionNetworkMSS
from recipes.diffusion.models.tnt_gru import TNTDiffusionNetwork as ZhTNTDiffusionNetwork
from recipes.diffusion.models.tnt_v2 import TNTDiffusionNetwork as TNTDiffusionNetworkV2
from recipes.diffusion.models.tnt_v3 import TNTDiffusionNetwork as TNTDiffusionNetworkV3
from recipes.diffusion.modules.pl_module import DiffusionModule

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

def init_diffusion(checkpoint_path, local_rank, cache_dir, is_zh_token=False, sstk=False, mixv2=False, sample_rate=24000, version=None):
    # TODO (weitsung) read the model version from ckpt, remove the arguments: is_zh_token, sstk, mixv2.
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        if mixv2:
            diffusion_model = DiffusionModule.load_from_checkpoint(
                checkpoint_path=local_path,
                diffusion_model=TNTDiffusionNetworkV2(
                    input_dim=32,
                    feature_dim=1024,
                    context_dim=32,
                    depth=20,
                    segment_size=32,
                    segment_stride=32,
                    unet=True,
                    dropout=0,
                    semantic_cfg_prob=0.1,
                    use_checkpoint=False,        
                    vc=False,
                    vc_cfg_prob=0.1,
                    lora=False,
                )
            ).model
        if sstk:
            if version == 'sstk_v5':
                diffusion_network = TNTDiffusionNetworkV3(
                    input_dim=64,
                    feature_dim=1024,
                    context_dim=1,
                    depth=20,
                    segment_size=8,
                    segment_stride=8,
                    unet=False,
                    dropout=0,
                    semantic_cfg_prob=0.10,
                    use_checkpoint=False
                )
            else:
                # 24k models
                if version == 'sstk_v1':
                    input_dim = 32
                    depth = 16
                    unet =True
                    segment_size = 32
                    segment_stride=32
                    unet_stages=[4,8,4]
                elif version == 'sstk_v2':
                    input_dim = 32
                    depth = 20
                    unet = False
                    segment_size = 32
                    segment_stride = 32
                    unet_stages = [6,8,6]
                # 44.1k models
                elif version in ['sstk_v3', 'sstk_v4']:
                    input_dim = 64
                    depth = 20
                    if version == 'sstk_v3':
                        unet = True
                    else:
                        unet = False
                    segment_size = 8
                    segment_stride = 8
                    unet_stages = [6,8,6]

                diffusion_network = TNTDiffusionNetworkV2(
                    input_dim=input_dim,
                    feature_dim=1024,
                    context_dim=1,
                    depth=depth,
                    segment_size=segment_size,
                    segment_stride=segment_stride,
                    unet=unet,
                    unet_stages=unet_stages,
                    dropout=0,
                    semantic_cfg_prob=0.10,
                    use_checkpoint=False
                )
            diffusion_model = DiffusionModule.load_from_checkpoint(
                checkpoint_path=local_path,
                diffusion_model=diffusion_network,
                strict=False
            ).model
        else:
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

def init_diffusio_mss(checkpoint_path, local_rank, cache_dir):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        diffusion_model = load_ema_checkpoint(
            local_path,
            TNTDiffusionNetworkMSS(
                input_dim=256,
                output_dim=128,
                feature_dim=1024,
                context_dim=128,
                depth=16,
                segment_size=32,
                segment_stride=32,
                dropout=0,
                cfg_prob=0.10,
                use_checkpoint=False
            ),
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
    vocoder_hz = params.get('vocoder_hz', VOCODER_HZ)

    pred_emb = sampler(
        model=diffusion_model,
        semantic_context=samples,
        num_items=samples.shape[0],
        num_chunks=num_chunks,
        num_steps=diffusion_steps,
        bf16_portion=bf16_portion,
        # start=None,
        # show_progress=True,
        angle_schedule='linear',
        schdeule_slope=schedule_slope,
        classifier_free_guidance=guidance_scale,
    ).detach()

    pred_emb = pred_emb.float()
    duration = pred_emb.shape[-1] // vocoder_hz
    
    wavs_g = vocode_in_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=1)

    # For bigmusic: [bs, c, seq] -> [bs, seq] 
    if len(wavs_g.shape) == 3:
        wavs_g = wavs_g.squeeze(1)

    return wavs_g

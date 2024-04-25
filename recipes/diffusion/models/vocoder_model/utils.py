import os
import torch
from recipes.musiclm.utils.dist import local_zero_first
from recipes.soundstream.models.vqgan import VQGAN_KL, VQGAN_KL_new, VQGAN_KL_mix
from recipes.soundstream.modules.pl_module_vae import VocoderModule
from recipes.diffusion.utils.utils import download_checkpoint

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

def init_vocoder(checkpoint_path, local_rank, cache_dir=None, sample_rate=24000, adapt_hopper=False, version='24k_125hz_dim32_baseline'):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
    
        if sample_rate == 24000:
            _24k_setups = {
                # latent_dim, downsample_rates, upsample_rates
                '24k_125hz_dim32_baseline': (32, [2, 3, 4, 8], [8, 4, 3, 2]),
                '24k_40hz_dim64_sa': (64, [2, 5, 6, 10], [10, 6, 5, 2]),
                '24k_125hz_dim64_sa': (64, [2, 3, 4, 8], [8, 4, 3, 2]),
            }
            if version in _24k_setups:
                latent_dim, downsample_rates, upsample_rates = _24k_setups[version]
            else:
                raise NotImplementedError(f"unsupported vocoder version: {version}")

            vocoder_model = VQGAN_KL_new(
                latent_dim=latent_dim,
                downsample_rates=downsample_rates,
                upsample_rates=upsample_rates,
                encoder_base_dim=96,
                decoder_base_dim=2560,
                adapt_hopper=adapt_hopper
            )
            try: 
                vocoder_model = load_ema_checkpoint(local_path, vocoder_model).eval().to(device)

            except Exception as ex: 
                print(f"EMA loading failed: {ex}, trying non-EMA load")
                vocoder_model_pl = VocoderModule.load_from_checkpoint(
                    local_path,
                    generator=vocoder_model,
                    discriminator=None,
                    n_channels=None,
                    dataloader_samplerate=sample_rate,
                    encoder_samplerate=sample_rate,
                    decoder_samplerate=sample_rate,
                    sample_pool_size=None,
                    batch_size=None,
                    sample_length=None,
                    strict=False,
                )
                vocoder_model = vocoder_model_pl.generator.eval().to(device)
        elif sample_rate == 44100 or sample_rate == 48000:
            if version == '24k_to_48k_stereo':
                vocoder_model = VQGAN_KL_mix(
                    in_channels=1,
                    out_channels=2,
                    latent_dim=32,
                    downsample_rates=[2, 3, 4, 8],
                    upsample_rates=[8, 6, 4 ,2],
                    encoder_base_dim=96,
                    decoder_base_dim=2560,
                    adapt_hopper=adapt_hopper,
                )
            elif version in ['44.1k_sa', '44.1k_vocal']:
                if version == '44.1k_sa':
                    latent_dim = 64
                elif version == '44.1k_vocal':
                    latent_dim = 128
                vocoder_model = VQGAN_KL_new(
                    n_channels=2,
                    latent_dim=latent_dim,
                    downsample_rates=[2, 5, 9, 10],
                    upsample_rates=[10, 9, 5 ,2],
                    encoder_base_dim=96,
                    decoder_base_dim=2560,
                    adapt_hopper=adapt_hopper,
                )
            else:
                raise NotImplementedError(f"unsupported vocoder version: {version}")

            vocoder_model_pl = VocoderModule.load_from_checkpoint(
                    local_path,
                    strict=False,
                    generator=vocoder_model,
                    discriminator=None,
                    n_channels=None,
                    dataloader_samplerate=sample_rate,
                    encoder_samplerate=sample_rate,
                    decoder_samplerate=sample_rate,
                    sample_pool_size=None,
                    train_batch_size=None,
                    valid_batch_size=None,
                    sample_length=None,

                )
            vocoder_model = vocoder_model_pl.generator.eval().to(device)

        return {
            "vocoder": vocoder_model, 
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


@torch.no_grad()
def vocode_in_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=4):
    "Vocode in chunks to prevent OOM"
    # If you see this error, lower batch size and chunk size:
    # RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.
    
    # pred_emb = bs x emb x seq_len
    items = []
    for item in torch.split(pred_emb, mini_bs): # mini_bs x emb x seq_len
        chunks = []
        for chunk in item.chunk(chunk_size, dim=-1): # mini_bs x emb x seq_len / chunk_size
            x = vocoder.decode(chunk).detach().cpu() # mini_bs x audio_seq_len / chunk_size
            chunks.append(x)
        chunks = torch.cat(chunks, dim=-1) # mini_bs x audio_seq_len
        items.append(chunks) # bs x audio_seq_len
    return torch.cat(items)

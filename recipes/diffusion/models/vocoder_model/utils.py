import os
import math
import torch
from recipes.musiclm.utils.dist import local_zero_first
from recipes.soundstream.models.vqgan import VQGAN_KL, VQGAN_KL_new, VQGAN_KL_mix
from recipes.soundstream.modules.pl_module_vae import VocoderModule
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.diffusion.models.vocoder_model.stream import islice


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
        elif sample_rate == 44100 or sample_rate == 48000 or sample_rate == 32000:
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
            else:
                _setups = {
                    # latent_dim, downsample_rates, upsample_rates, last_act
                    '44.1k_sa': (64, [2, 5, 9, 10], [10, 9, 5 ,2], True),
                    '44.1k_vocal': (128, [2, 5, 9, 10], [10, 9, 5 ,2], True),
                    '32k': (128, [2, 5, 8, 10], [10, 8, 5, 2], False),
                }
                if version in _setups:
                    latent_dim, downsample_rates, upsample_rates, last_act = _setups[version]
                else:
                    raise NotImplementedError(f"unsupported vocoder version: {version}")
               
                vocoder_model = VQGAN_KL_new(
                    n_channels=2,
                    latent_dim=latent_dim,
                    downsample_rates=downsample_rates,
                    upsample_rates=upsample_rates,
                    encoder_base_dim=96,
                    decoder_base_dim=2560,
                    adapt_hopper=adapt_hopper,
                    last_act=last_act
                )

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
def vocode_in_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=4, device=None):
    "Vocode in chunks to prevent OOM"
    # If you see this error, lower batch size and chunk size:
    # RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.
    
    # pred_emb = bs x emb x seq_len
    items = []
    for item in torch.split(pred_emb, mini_bs): # mini_bs x emb x seq_len
        chunks = []
        for chunk in item.chunk(chunk_size, dim=-1): # mini_bs x emb x seq_len / chunk_size
            x = vocoder.decode(chunk).detach() # mini_bs x audio_seq_len / chunk_size
            if device is not None:
                x = x.to(device)
            chunks.append(x)
        chunks = torch.cat(chunks, dim=-1) # mini_bs x audio_seq_len
        items.append(chunks) # bs x audio_seq_len
    return torch.cat(items)


@torch.no_grad()
def vocode_in_ovl_chunks(pred_emb, vocoder, mini_bs=1, chunk_size=4, overlap_ratio=0.2, border_padding=1, padding_value=-5,
                            vocoder_frame_rate=49, sample_rate=44100, device=None, vocoder_type='v1'):
    "Vocode in overlapped chunks to prevent OOM"
    # pred_emb = bs x emb x seq_len
    items = []
    overlap_len = math.ceil(chunk_size * overlap_ratio)
    overlap_wav_len = math.ceil(overlap_len / vocoder_frame_rate * sample_rate)
    pad_wav_len = math.ceil(1/vocoder_frame_rate * sample_rate)
    # n_chunks = math.ceil(pred_emb.shape[-1] / (chunk_size - overlap_len)) 

    for item in torch.split(pred_emb, mini_bs): # mini_bs x emb x seq_len
        chunks = None
        chunk_idx, chunk_st = 0, 0
        final_flag = False
        while not final_flag: #for chunk_idx in range(n_chunks): # mini_bs x emb x seq_len / chunk_size
            # import pdb; pdb.set_trace()
            this_chunk = item[..., chunk_st:chunk_st+chunk_size]
            final_flag = True if chunk_st + chunk_size >= pred_emb.shape[-1] else False
            # print(chunk_idx, chunk_st, chunk_st+chunk_size, final_flag)
            if chunk_idx == 0:
                this_chunk = torch.nn.functional.pad(this_chunk, [border_padding, 0], value=padding_value) # padding left
            elif final_flag:
                this_chunk = torch.nn.functional.pad(this_chunk, [0, border_padding], value=padding_value) # padding right
            
            if vocoder_type == 'v1':
                x = vocoder.decode(this_chunk).detach() # mini_bs x audio_seq_len / chunk_size
            else:
                x = vocoder.decode_latents(this_chunk.transpose(1,2)).detach()
                
            if device is not None:
                x = x.to(device)

            if chunk_idx == 0:
                x = x[..., pad_wav_len:] # remove padding
                x[..., -overlap_wav_len:] *= torch.linspace(1, 0, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)  # fade out only
                chunks = x
            else:
                if final_flag:
                    x = x[..., :-pad_wav_len] # remove padding
                    x[..., :overlap_wav_len] *= torch.linspace(0, 1, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device) # fade in only
                else:
                    # fade in & out
                    x[..., :overlap_wav_len] *= torch.linspace(0, 1, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)
                    x[..., -overlap_wav_len:] *= torch.linspace(1, 0, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)
                chunks = torch.cat([chunks[..., :-overlap_wav_len], 
                                    chunks[..., -overlap_wav_len:] + x[..., :overlap_wav_len],
                                    x[..., overlap_wav_len:]
                                    ], dim=-1) 
            
            chunk_st += (chunk_size - overlap_len)
            chunk_idx += 1
                
            
        items.append(chunks) # bs x audio_seq_len
    return torch.cat(items)


@torch.no_grad()
def vocode_in_ovl_chunks_v2(pred_emb, vocoder, chunk_size=100, overlap=8, device=None, 
                            vocoder_frame_rate=49, sample_rate=44100, vocoder_type='v1', dim=-1, *args, **kwargs):
    "Vocode in overlapped chunks with boundary on both sides"
    # pred_emb: bs x emb x seq_len
    items = []
    for item in torch.split(pred_emb, 1):  #mini_bs x emb x seq_len
        def emb_generator():
            yield item
        
        # 2. cut into overlapped chunks
        chunk_generator = islice(
            emb_generator(), 
            step=chunk_size, 
            overlap=overlap, 
            dim=dim,
        )
        
        chunk_sample = int(chunk_size / vocoder_frame_rate*sample_rate)
        # 3. decode and cut overlapped chunks
        audio_chunks = []
        for i, chunk_info in enumerate(chunk_generator):
            chunk = chunk_info["data"] # mini_bs x emb x chunk_size_with_overlap

            if vocoder_type == 'v1':
                _audio = vocoder.decode(chunk).detach()  # mini_bs x 1 x audio_len
            else:
                _audio = vocoder.decode_latents(chunk.transpose(1, 2)).detach()
            
            # remove overlap at both sides
            start_sample = int(overlap/vocoder_frame_rate*sample_rate)
            if i == 0: # first chunk
                start_sample = 0
            if start_sample > _audio.shape[-1]: # last chunk
                audio = _audio[..., -chunk_sample:]
            else:
                audio = _audio[..., start_sample:start_sample+chunk_sample]

            if device is not None:
                audio = audio.to(device)
            
            audio_chunks.append(audio)

        items.append(torch.cat(audio_chunks, dim=-1))
    
    return torch.cat(items, dim=0)



def process_eos_indexes(semantic_samples, eos_id=32769, semantic_frame_rate=25, output_frame_rate=50):
    eos_padding_id = -10000
    eos_mask = torch.cumsum(semantic_samples == eos_id, 1) > 0
    semantic_samples[eos_mask] = eos_padding_id
    semantic2output_rate = output_frame_rate / semantic_frame_rate
    eos_index_list = ((semantic_samples == eos_padding_id).bool().cumsum(axis=1) == 0).bool().sum(
        axis=1)
    eos_index_list = torch.round(eos_index_list * semantic2output_rate) # convert 
    return eos_index_list.long().to(semantic_samples.device)


def pad_sequence_dim(tensors, dim=1, padding_value=0):
    max_length = max(tensor.size(dim) for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        pad_length = max_length - tensor.size(dim)
        padding = [0 for _ in range(len(tensor.shape))]
        padding[dim] = pad_length
        padded = torch.nn.functional.pad(
            tensor, padding, mode='constant', value=padding_value
        )
        padded_tensors.append(padded)
    return torch.stack(padded_tensors, dim=0)

import argparse
import librosa
from pydub import AudioSegment
import os
from samantha.utils.hparams import DotDict
import torch
import torch.nn.functional as F
import random
import numpy as np
import unicodedata
import re
from tqdm import tqdm
import time


def set_seed(seed=1996):
    # reproduction setting
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    return


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def save_wav(audio, output_file, sr=24000):
    from scipy.io.wavfile import write
    audio = audio * 32768.0
    audio = audio.astype('int16')
    write(output_file, sr, audio)
    return


def load_wav(path):
    if path.endswith(".npy"):
        wav = np.load(path)
    elif path.endswith(".wav"):
        wav, sr = librosa.load(path, sr=24000)
    else:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(24000)
        wav = np.asarray(audio.get_array_of_samples())
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav


def sample(predict_logits, temp, mode="naive"):
    if mode == "naive":
        predict_logits = predict_logits / (temp)
        probs = predict_logits.softmax(dim=1)  # [b, d]
        dist = torch.distributions.categorical.Categorical(probs=probs)
        samples = dist.sample().unsqueeze(1).to(args.device)
    elif mode == "gumbel":
        predict_logits = top_k(predict_logits, thres=args.gt)
        samples = gumbel_sample(predict_logits, temp).unsqueeze(dim=1)
    else:
        raise NotImplementedError()
    return samples


def log(t, eps=1e-5):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.95):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def text2semantic(semantic_model, mulan_tokens):
    bs = mulan_tokens.size(0)
    eos_ids = torch.zeros([bs, 1], dtype=torch.long, device=args.device) + 1024 # offset: w2v-bert
    mulan_tokens = mulan_tokens + 1024 + 1 # [b, 12] # offset: w2-vert + EOS 1

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * 25 - 3
        if end >= args.duration * 25 - 3:
            end = args.duration * 25 - 3
            beg = end - (10 * 25 - 3)
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += 5 * 25
    prev_end = 0
    semantic_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        if cache_len == 0:
            input_tokens = torch.cat([mulan_tokens, eos_ids], dim=1)
        else:
            prefix_semantic_samples = semantic_samples[:, cur_beg: cur_beg + cache_len]
            input_tokens = torch.cat([mulan_tokens, eos_ids, prefix_semantic_samples], dim=1)
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for _ in pbar:
            pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
            semantic_outputs = semantic_model(input_tokens, past_key_values=past_key_values, use_cache=True)
            logits = semantic_outputs['logits'] # [b, t, d]
            predict_logits = logits[:, -1, 0 : 1024]
            samples = sample(predict_logits, temp=args.st, mode=args.sample_mode)
            past_key_values = semantic_outputs['past_key_values']
            input_tokens = samples
            if semantic_samples is None:
                semantic_samples = samples
            else:
                semantic_samples = torch.cat([semantic_samples, samples], dim=1)
    return semantic_samples


def semantic2coarse(coarse_model, mulan_tokens, semantic_samples):
    bs = mulan_tokens.size(0)
    eos_ids = torch.zeros([bs, 1], dtype=torch.long, device=args.device) + 1024 + num_coarse * 1024 # offset: w2v-bert + coarse
    mulan_tokens = mulan_tokens + 1024 + num_coarse * 1024 + 2 # offset: w2v-bert + coarse + EOS 2

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * frequency * num_coarse
        if end >= args.duration * frequency * num_coarse:
            end = args.duration * frequency * num_coarse
            beg = end - 10 * frequency * num_coarse
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += 5 * frequency * num_coarse
    
    prev_end = 0
    coarse_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        semantic_beg = int(cur_beg / frequency / num_coarse * 25)
        semantic_slice = semantic_samples[:, semantic_beg : semantic_beg + 247]
        if cache_len == 0:
            input_tokens = torch.cat([mulan_tokens, eos_ids, semantic_slice, eos_ids + 1], dim=1)
        else:
            prefix_coarse_samples = coarse_samples[:, cur_beg: cur_beg + cache_len]
            input_tokens = torch.cat([mulan_tokens, eos_ids, semantic_slice, eos_ids + 1, prefix_coarse_samples], dim=1)
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for i in pbar:
            pbar.set_description(f"Coarse [{cur_beg} - {cur_end}]")
            coarse_outputs = coarse_model(input_tokens, past_key_values=past_key_values, use_cache=True)
            logits = coarse_outputs['logits'] # [b, t, d]
            layer_idx = i % num_coarse
            predict_logits = logits[:, -1, 1024 + layer_idx * 1024: 1024 + (layer_idx + 1) * 1024] # [b,d] # offset: w2v-bert
            samples = sample(predict_logits, temp=args.ct, mode=args.sample_mode)
            samples = samples + 1024 + layer_idx * 1024 # [b, 1], # offset: w2v-bert
            past_key_values = coarse_outputs['past_key_values']
            input_tokens = samples
            if coarse_samples is None:
                coarse_samples = samples
            else:
                coarse_samples = torch.cat([coarse_samples, samples], dim=1)
    coarse_samples = coarse_samples - 1024 # remove offset: w2v-bert
    return coarse_samples


def coarse2fine(fine_model, coarse_samples):
    bs = coarse_samples.size(0)
    eos_ids = torch.zeros(size=[bs, 1], dtype=torch.long, device=args.device) + num_res * 1024 # [b, 1]

    slice_range = []
    beg = 0
    while True:
        end = beg + 10 * frequency * num_fine
        if end >= args.duration * frequency * num_fine:
            end = args.duration * frequency * num_fine
            beg = end - 10 * frequency * num_fine
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += 5 * frequency * num_fine

    prev_end = 0
    fine_samples = None
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        coarse_beg = int(cur_beg / frequency / num_fine * num_coarse)
        coarse_slice = coarse_samples[:, coarse_beg: coarse_beg + 2000] # [b, coarse*t]
        if cache_len == 0:
            input_tokens = torch.cat([coarse_slice, eos_ids], dim=1)
        else:
            prefix_fine_samples = fine_samples[:, cur_beg: cur_beg + cache_len]
            input_tokens = torch.cat([coarse_slice, eos_ids, prefix_fine_samples], dim=1)
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))
        for i in pbar:
            pbar.set_description(f"Fine [{cur_beg} - {cur_end}]")
            fine_outputs = fine_model(input_tokens, past_key_values=past_key_values, use_cache=True)
            logits = fine_outputs['logits'] # [b, t, d]
            layer_idx = i % num_fine + num_coarse
            predict_logits = logits[:, -1, layer_idx * 1024: (layer_idx + 1) * 1024] # [b, d]
            samples = sample(predict_logits, temp=args.ft, mode=args.sample_mode)
            samples = samples + layer_idx * 1024
            past_key_values = fine_outputs['past_key_values']
            input_tokens = samples
            if fine_samples is None:
                fine_samples = samples
            else:
                fine_samples = torch.cat([fine_samples, samples], dim=1)
    return fine_samples


def ss_decode(ss_dec, coarse_samples, fine_samples):
    bs = coarse_samples.size(0)
    coarse_samples = coarse_samples.view([bs, -1, num_coarse])
    fine_samples = fine_samples.view([bs, -1, num_fine])
    vqgan_inputs = torch.cat([coarse_samples, fine_samples], dim=2) - torch.arange(num_res).to(args.device) * 1024 # [b, t, n_codebook]
    vqgan_inputs = vqgan_inputs.transpose(1, 2) # [b, t, n_codebook] -> [b, n_codebook, t]
    wavs = ss_dec(vqgan_inputs).squeeze(1)
    return wavs


def gather_prompts(mulan_model):
    items = []
    if args.prompts_group in ["google", "all"]:
        import pandas as pd
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/google_prompts/google_prompts.csv")
        for _, row in df.iterrows():
            items.append([row["category"], row["text"]])
    
    if args.prompts_group in ["musiccaps-asp", "all"]:
        import pandas as pd
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["aspect_list"]])
    
    if args.prompts_group in ["musiccaps-cap", "all"]:
        import pandas as pd
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["musiccaps", row["text"]])

    if args.prompts_group in ["sami", "all"]:
        from webdataset import WebDataset
        tar_file_paths = [
            '/mnt/bn/audio-diffusion/data/mulan_prompts/sami_prompts.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/genre_stats.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/mood_stats.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_long_prompts.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_short_prompts.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/painting_desc.tar.mulan149',
            '/mnt/bn/audio-diffusion/data/mulan_prompts/theme_stats.tar.mulan149'
        ]
        for tar_file_path in tar_file_paths:
            prefix = tar_file_path.split('/')[-1].split('.')[0]
            tar_file = WebDataset(tar_file_path).decode()
            for item in tar_file:
                items.append([prefix, item["text.txt"]])

    if args.prompts_group in ["gpt", "all"]:
        with open("/mnt/bn/audio-diffusion/data/mulan_prompts/gpt_text_prompts.txt", "r") as fp:
            for line in fp.readlines():
                items.append(["gpt", line.strip()])

    if args.prompts_group in ["image", "all"]:
        with open("/mnt/bn/audio-diffusion/data/mulan_prompts/image_captions.txt", "r") as fp:
            for line in fp.readlines():
                items.append(["image captions", line.strip()])

    if args.prompts_group in ["painting", "all"]:
        with open("/mnt/bn/audio-diffusion/data/mulan_prompts/image2text2music.txt", "r") as fp:
            for line in fp.readlines():
                items.append(["image2text2music", line.strip()])
    
    if args.prompts_group == "direct_prompt":
        for i in range(args.bs):
            items.append(["direct_prompt", args.direct_prompt + f"_copy_{i}", args.direct_prompt])
    
    if args.prompts_group == "audio_prompt":
        import pandas as pd
        df = pd.read_csv("/mnt/bn/audio-diffusion/data/musiccaps/kaggle_val_meta.csv")
        for _, row in df.iterrows():
            items.append(["audio prompt (musiccaps)", row["text"]])


    text_embs = []
    prompts = []
    categories = []
    pbar = tqdm(items)
    for item in pbar:
        pbar.set_description("Extracting mulan text embeds...")
        text_emb = mulan_inference(mulan_model, text=item[1], device=args.device)
        text_embs.append(text_emb)
        prompts.append(item[1])
        categories.append(item[0])
    if args.prompts_group == "audio_prompt":
        print("Loading musiccaps_audio.pyt...")
        audio_embs = torch.load("/mnt/bn/audio-diffusion/data/musiccaps/musiccaps_audio.pyt").to(args.device)
        print("Loaded")
    else:
        audio_embs = torch.zeros([len(items), 128]).to(args.device)
    text_embs = torch.cat(text_embs, dim=0)
    return text_embs, audio_embs, prompts, categories

@torch.no_grad()
def main():
    set_seed(seed=args.seed)
    print("Loading semantic_model...")
    semantic_model = SemanticModule.load_from_checkpoint(ckpts.semantic, args.device).eval().to(args.device).model
    print("Loading coarse_model...")
    coarse_model = CoarseModule.load_from_checkpoint(ckpts.coarse, args.device).eval().to(args.device).model
    print("Loading fine_model...")
    fine_model = FineModule.load_from_checkpoint(ckpts.fine, args.device).eval().to(args.device).model
    
    ss_dec = torch.jit.load(ckpts.ss_dec).eval()
    mulan_model = create_mulan_model(ckpts.mulan, args.device).eval()
    mulan_centers = torch.from_numpy(np.load(ckpts.mulan_centers)).float().to(args.device)
    text_embs, audio_embs, prompts, categories = gather_prompts(mulan_model)
    for rd in range(args.rounds):
        text_batch = torch.split(text_embs, args.bs)
        audio_batch = torch.split(audio_embs, args.bs)
        start_time = time.time()
        for batch_idx, (text_embeds_batch, audio_embeds_batch) in enumerate(zip(text_batch, audio_batch)):
            if args.prompts_group == "audio_prompt":
                mulan_embeds = mulan_inference(mulan_model, music=audio_embeds_batch, device=args.device)
            else:
                mulan_embeds = text_embeds_batch
            mulan_tokens, ds = mulan_rvq_indexs(mulan_embeds, mulan_centers)
            semantic_samples = text2semantic(semantic_model, mulan_tokens)
            print("Semantic samples: ", semantic_samples.size())
            coarse_samples = semantic2coarse(coarse_model, mulan_tokens, semantic_samples)
            print("Coarse samples: ", coarse_samples.size())
            fine_samples = coarse2fine(fine_model, coarse_samples)
            print("Fine samples: ", fine_samples.size())
            wavs = ss_decode(ss_dec, coarse_samples, fine_samples)
            mulan_audio_embeds = mulan_inference(mulan_model, music=wavs, device=args.device)
            scores = F.cosine_similarity(mulan_audio_embeds, text_embeds_batch, dim=1)
            if args.prompts_group == "audio_prompt":
                orig_scores = F.cosine_similarity(mulan_embeds, text_embeds_batch, dim=1)
            for wav_idx, wav in enumerate(wavs):
                prompt_idx = batch_idx * args.bs + wav_idx
                wav_dir = os.path.join(filepath_prefix, categories[prompt_idx])
                os.makedirs(wav_dir, exist_ok=True)
                if args.prompts_group == "audio_prompt":
                    fp = os.path.join(wav_dir, f"{slugify(prompts[prompt_idx])[:128]}.{rd + 1}.[{scores[wav_idx]:.4f} - {orig_scores[wav_idx]:.4f}]")
                else:
                    fp = os.path.join(wav_dir, f"{slugify(prompts[prompt_idx])[:128]}.{rd + 1}.cs{scores[wav_idx]:.4f}")
                print(f"[Saving] {fp}")
                save_wav(wav.cpu().numpy(), fp + ".wav", sr=sample_rate)
                with open(fp + ".txt", "w") as prompt_txt:
                    prompt_txt.write(prompts[prompt_idx])
        print(f"[Elapsed Time] {time.time() - start_time}")


if __name__ == "__main__":

    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=3, help="How many rounds to run")
    parser.add_argument('-b', '--bs', type=int, default=5, help="Batch size for each forward pass")
    parser.add_argument('-d', '--duration', type=int, default=10, help="Duration in seconds")
    parser.add_argument('--sample_mode', choices=["naive", "gumbel"])
    parser.add_argument('--st', type=float, default=1.0, help="Semantic decoder sampling temperature")
    parser.add_argument('--ct', type=float, default=0.9, help="Coarse decoder sampling temperature")
    parser.add_argument('--ft', type=float, default=0.8, help="Fine decoder sampling temperature")
    parser.add_argument('--gt', type=float, default=0.9, help="Gumbel sample threshold")
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--prompts_group', choices=[
        "all", "musiccaps-cap", "musiccaps-asp", "google", "sami",
        "painting", "gpt", "image", "direct_prompt", "audio_prompt"
    ])
    parser.add_argument('--direct_prompt', type=str, default="jazz")

    # parser.add_argument("--save_tensor", action="store_true", help="Save logits and sampled tokens")
    args = parser.parse_args()

    # Hparams
    num_res = 12
    num_coarse = 4
    num_fine = num_res - 4
    sample_rate = 24000
    frequency = 24000 // 480

    # Checkpoint paths
    ckpts = {}
    from recipes.speech_qa.lit_modules.v3_1.lit_semantic import SemanticModule
    from recipes.speech_qa.lit_modules.v3_1.lit_coarse_3ar import CoarseModule
    from recipes.speech_qa.lit_modules.v4.lit_fine import FineModule
    # sparse 2.4B
    # ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_sparse/version_3.1/checkpoints/epoch=03-step=100000-accu=36.39.ckpt'
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse/version_3.1/checkpoints/epoch=00-step=46000-accu=27.13.ckpt'
    # sparse 0.3B
    # ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_sparse/version_3.2/checkpoints/epoch=18-step=39000-accu=35.62.ckpt'
    # ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_sparse/version_3.2/checkpoints/epoch=06-step=70000-accu=28.64.ckpt'
    # llama 0.4B - mulan191
    ckpts['semantic'] = '/mnt/bn/zongyu-lq/logs/semantic_llama/406M/checkpoints/epoch=22-step=61000-accu=37.32.ckpt'
    ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_llama/415M/checkpoints/epoch=02-step=120000-accu=30.45.ckpt'
    # llama 0.4B - mulan115
    ckpts['semantic'] = "/mnt/bn/zongyu-lq/logs/semantic_llama_mulan115/406M/checkpoints/epoch=08-step=22000-accu=36.29.ckpt"
    ckpts['coarse'] = '/mnt/bn/zongyu-lq/logs/coarse_llama_mulan115/415M/checkpoints/epoch=01-step=50000-accu=30.41.ckpt'

    ckpts['fine'] = '/mnt/bn/zongyu-lq/ckpts/musiclm/epoch=03-step=193000-accu=18.92.ckpt'
    ckpts['ss_dec'] = '/mnt/bn/zongyu-lq/ckpts/soundstream/190k/ss_decoder_0.pt'

    from recipes.speech_qa.requires.mulan.mulan_infer_newer import create_mulan_model, mulan_inference, mulan_rvq_indexs
    # mulan 149
    # ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan_149.pt'
    # ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook-1024x12.npy'
    # mulan 191
    ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan-step=042400-median_rank_0=191-kaggle.ckpt'
    ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook_mulan_multi_dset_191.npy'
    # mulan 115
    ckpts['mulan'] = '/mnt/bn/zongyu-lq/ckpts/mulan/mulan-step=006400-median_rank_1=115-kaggle.ckpt'
    ckpts['mulan_centers'] = '/mnt/bn/zongyu-lq/ckpts/mulan/kmeans_minibatch_codebook_mulan-1b-151_6M_ninit3.npy'

    ckpts = DotDict(ckpts)
    filepath_prefix = f'recipes/speech_qa/'
    filepath_prefix += f'MusicLM_llama_mulan115'
    filepath_prefix += f'_Dur_{args.duration}'
    filepath_prefix += f'_{args.prompts_group}'
    filepath_prefix += f'_Mode_{args.sample_mode}'
    filepath_prefix += f'_St_{args.st}'
    filepath_prefix += f'_Ct_{args.ct}'
    filepath_prefix += f'_Ft_{args.ft}'
    filepath_prefix += f'_Gt_{args.gt}'
    filepath_prefix += f'_Seed_{args.seed}'

    main()

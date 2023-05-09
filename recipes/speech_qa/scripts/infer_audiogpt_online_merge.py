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
from transformers import Wav2Vec2FeatureExtractor, HubertModel


sr=16000
huggingface_hubert = True

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


def hubert_cal_conv_out_dim(length):
    for _, k, s in eval('[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'):
        length = (length - k) // s + 1
    return length


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Examples: With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))
    bs = lengths.size()[0]
    maxlen = lengths.max()
    # if not isinstance(lengths, list):
    #     lengths = lengths.tolist()
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths.cpu()).unsqueeze(-1)

    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask



def hubert_tokenization(hubert_model, wavs, wav_lens, centers, device, huggingface_hubert=False):
    b, t = wavs.size()
    
    if huggingface_hubert:
        km_layer = 24
        hubert_embeds = hubert_model(wavs.to(device),
                output_hidden_states=True).hidden_states[km_layer].detach()
    else:
        batch_data_feat_length = hubert_cal_conv_out_dim(wav_lens)
        batch_data_feat_mask = make_pad_mask(batch_data_feat_length)

        hubert_embeds, mask = hubert_model(
            wavs.to(device), wav_lens.to(device), batch_data_feat_mask.to(device))

    # kmeans
    b, t, d = hubert_embeds.shape
    # dataset = hubert_embeds.view([b * t, d])
    dataset = hubert_embeds.reshape([b * t, d])
    num_points = dataset.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)  # [1024, 1024]
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    inertia = 0
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
        inertia += distances[range(distances.shape[0]), min_ind].sum()
    codes = codes.view([b, t])
    return codes



token_len = 850

def speech_continuation_simple(semantic_model, prompt_tokens, token_size):
    input_tokens = torch.cat([prompt_tokens], dim=1)
    print('prompt size: %d' % prompt_tokens.shape[1])
    pbar = tqdm(range(token_len - prompt_tokens.shape[1]))
    past_key_values = None
    semantic_samples = None
    
    for _ in pbar:
        # pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
        semantic_outputs = semantic_model(input_tokens, past_key_values=past_key_values, use_cache=True)
        logits = semantic_outputs['logits'] # [b, t, d]
        predict_logits = logits[:, -1, 0 : token_size]
        samples = sample(predict_logits, temp=args.st, mode=args.sample_mode)
        past_key_values = semantic_outputs['past_key_values']
        # print(input_tokens, '  ', samples)
        input_tokens = samples
        # input_tokens = torch.cat([input_tokens, samples], dim=1)
        if semantic_samples is None:
            semantic_samples = samples
        else:
            semantic_samples = torch.cat([semantic_samples, samples], dim=1)

    return semantic_samples


win_token_len = 800
hop_token_len = 400

def speech_continuation(semantic_model, prompt_tokens, token_size):
    slice_range = []
    beg = 0
    while True:
        end = beg + win_token_len
        if end >= args.duration:
            end = args.duration
            beg = end - win_token_len
            slice_range.append([beg, end])
            break
        else:
            slice_range.append([beg, end])
        beg += hop_token_len
    prev_end = 0
    semantic_samples = None
    
    
    for cur_beg, cur_end in slice_range:
        cache_len = prev_end - cur_beg
        prev_end = cur_end
        if cache_len == 0:
            input_tokens = torch.cat([prompt_tokens], dim=1)
        else:
            prefix_semantic_samples = semantic_samples[:, cur_beg: cur_beg + cache_len]
            input_tokens = torch.cat([prompt_tokens, prefix_semantic_samples], dim=1)
        past_key_values = None

        pbar = tqdm(range(cur_end - cur_beg - cache_len))

        for _ in pbar:
            pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
            semantic_outputs = semantic_model(input_tokens, past_key_values=past_key_values, use_cache=True)
            logits = semantic_outputs['logits'] # [b, t, d]
            predict_logits = logits[:, -1, 0 : token_size]
            samples = sample(predict_logits, temp=args.st, mode=args.sample_mode)
            past_key_values = semantic_outputs['past_key_values']
            # print(input_tokens, '  ', samples)
            input_tokens = samples
            # input_tokens = torch.cat([input_tokens, samples], dim=1)
            if semantic_samples is None:
                semantic_samples = samples
            else:
                semantic_samples = torch.cat([semantic_samples, samples], dim=1)
                
    # return torch.cat([prompt_tokens, semantic_samples], dim=1)
    return semantic_samples




@torch.no_grad()
def main():
    set_seed(seed=args.seed)
    print("Loading semantic_model...")
    semantic_model = SemanticModule.load_from_checkpoint(ckpts.semantic, args.device).eval().to(args.device).model

    
    # hubert_larget_24l_km1024
    if huggingface_hubert:
        processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-large-ll60k')
        hubert_model = HubertModel.from_pretrained('facebook/hubert-large-ll60k').to(args.device).eval()

        centers = torch.from_numpy(
            np.load('/mnt/bn/cyz-lq-nas/project/samantha/.module_cache/musiclm_3ar_1024x12/centroids_epoch_428.npy')).to(args.device)
    else:
        # mhubert 1024
        hubert_model = torch.jit.load(
            '/mnt/bn/cyz-lq-nas/project/samantha/.module_cache/musiclm_3ar_1024x12/mhubert_v1_1024_semantic.jit.pt'
                , map_location=args.device).to(args.device).eval()
        centers = torch.from_numpy(
            np.load('/mnt/bn/cyz-lq-nas/project/samantha/.module_cache/musiclm_3ar_1024x12/centroids_epoch_55.npy')).to(args.device)
    

    prompt_point_num = 400  

    for _wav_file in os.listdir(args.input_dir):
        wav_file =os.path.join(args.input_dir, _wav_file)
        print(wav_file)
        # 输入wav并确保采样率转换到16k
        audio = AudioSegment.from_file(wav_file)
        audio = audio.set_channels(1).set_frame_rate(sr)
        wav = np.asarray(audio.get_array_of_samples())
        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        if len(wav.shape) >= 2:
            wav = wav[0]
        wav = wav.astype(np.float32)

        # wav = np.concatenate((np.zeros([16000*5]).astype(np.float32), wav), axis=0) #确保长度大于5s
        
        wav = torch.from_numpy(wav).to(args.device)
        wav_len = torch.Tensor([wav.shape[0]]).long().to(args.device)

        if huggingface_hubert:
            wav = processor(wav, return_tensors="pt", sampling_rate=sr).input_values
            print(wav.shape)
            wav_len = wav.shape[1]
            prompt_tokens = hubert_tokenization(hubert_model, wav, wav_len, centers,
                                     device=args.device, huggingface_hubert=True)
        else:
            print('wav_len: ', wav_len)
            wav = wav.unsqueeze(0)
            prompt_tokens = hubert_tokenization(hubert_model, wav, wav_len, centers, device=args.device)

        prompt_tokens = prompt_tokens[:,-prompt_point_num:]

        print('prompt tokens: ', prompt_tokens)
        print('prompt tokens num: ', prompt_tokens.shape)
        # merge
        cur_input_tokens = prompt_tokens[0]
        offset_input_token = cur_input_tokens[1:]
        cur_merge_tokens = cur_input_tokens[:1]
        judge = ~(cur_input_tokens[:-1] == offset_input_token)
        cur_merge_tokens = torch.cat([cur_merge_tokens, offset_input_token[judge]]).long()
        cur_len = cur_merge_tokens.shape[0]

        print('prompt merge_tokens num: ', cur_len)
        print('prompt merge_tokens: ', cur_merge_tokens)

        cur_merge_tokens = cur_merge_tokens.unsqueeze(0)

        semantic_samples = speech_continuation_simple(semantic_model, cur_merge_tokens, args.token_size)

        ####### cat with prompt #######
        semantic_samples = torch.cat([cur_merge_tokens, semantic_samples], dim=1)

        save_path = os.path.join(args.output_dir, _wav_file.replace('.wav', '.npy'))

        semantic_samples = semantic_samples.squeeze(0).cpu().numpy().astype(np.int32)

        if os.path.exists(save_path):
            os.remove(save_path)
        np.save(save_path, semantic_samples, allow_pickle=False)
        print('output: ', semantic_samples)


    
    # ss_dec = torch.jit.load(ckpts.ss_dec).eval()
    # mulan_model = create_mulan_model(ckpts.mulan, args.device).eval()
    # mulan_centers = torch.from_numpy(np.load(ckpts.mulan_centers)).float().to(args.device)
    # text_embs, audio_embs, prompts, categories = gather_prompts(mulan_model)
    # for rd in range(args.rounds):
    #     text_batch = torch.split(text_embs, args.bs)
    #     audio_batch = torch.split(audio_embs, args.bs)
    #     start_time = time.time()
    #     for batch_idx, (text_embeds_batch, audio_embeds_batch) in enumerate(zip(text_batch, audio_batch)):
    #         if args.prompts_group == "audio_prompt":
    #             mulan_embeds = mulan_inference(mulan_model, music=audio_embeds_batch, device=args.device)
    #         else:
    #             mulan_embeds = text_embeds_batch
    #         mulan_tokens, ds = mulan_rvq_indexs(mulan_embeds, mulan_centers)
    #         semantic_samples = text2semantic(semantic_model, mulan_tokens)
    #         print("Semantic samples: ", semantic_samples.size())
            
            
            
    #         for wav_idx, wav in enumerate(wavs):
    #             prompt_idx = batch_idx * args.bs + wav_idx
    #             wav_dir = os.path.join(filepath_prefix, categories[prompt_idx])
    #             os.makedirs(wav_dir, exist_ok=True)
    #             if args.prompts_group == "audio_prompt":
    #                 fp = os.path.join(wav_dir, f"{slugify(prompts[prompt_idx])[:128]}.{rd + 1}.[{scores[wav_idx]:.4f} - {orig_scores[wav_idx]:.4f}]")
    #             else:
    #                 fp = os.path.join(wav_dir, f"{slugify(prompts[prompt_idx])[:128]}.{rd + 1}.cs{scores[wav_idx]:.4f}")
    #             print(f"[Saving] {fp}")
    #             save_wav(wav.cpu().numpy(), fp + ".wav", sr=sr)
    #             with open(fp + ".txt", "w") as prompt_txt:
    #                 prompt_txt.write(prompts[prompt_idx])
    #     print(f"[Elapsed Time] {time.time() - start_time}")


if __name__ == "__main__":

    # Generation configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--token_size", type=int)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("-r", "--rounds", type=int, default=3, help="How many rounds to run")
    parser.add_argument('-b', '--bs', type=int, default=5, help="Batch size for each forward pass")
    parser.add_argument('-d', '--duration', type=int, default=800, help="Duration in seconds")
    parser.add_argument('--sample_mode', choices=["naive", "gumbel"])
    parser.add_argument('--st', type=float, default=1.0, help="Semantic decoder sampling temperature")
    parser.add_argument('--ct', type=float, default=0.9, help="Coarse decoder sampling temperature")
    parser.add_argument('--ft', type=float, default=0.8, help="Fine decoder sampling temperature")
    parser.add_argument('--gt', type=float, default=0.9, help="Gumbel sample threshold")
    parser.add_argument('--seed', type=int, default=9527)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--prompts_group', choices=[
        "all", "musiccaps-cap", "musiccaps-asp", "google", "sami",
        "painting", "gpt", "image", "direct_prompt", "audio_prompt"
    ])
    parser.add_argument('--direct_prompt', type=str, default="jazz")



    # parser.add_argument("--save_tensor", action="store_true", help="Save logits and sampled tokens")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Hparams
    num_res = 12
    num_coarse = 4
    num_fine = num_res - 4

    # Checkpoint paths
    ckpts = {}
    from recipes.speech_qa.lit_modules.audiogpt.lit_audiogpt_online_merge import SemanticModule

    ckpts['semantic'] = args.ckpt


    ckpts = DotDict(ckpts)
    filepath_prefix = f'recipes/speech_qa/'
    filepath_prefix += f'bshall_hubert_lm'
    filepath_prefix += f'_Dur_{args.duration}'
    filepath_prefix += f'_{args.prompts_group}'
    filepath_prefix += f'_Mode_{args.sample_mode}'
    filepath_prefix += f'_St_{args.st}'
    filepath_prefix += f'_Gt_{args.gt}'
    filepath_prefix += f'_Seed_{args.seed}'

    main()

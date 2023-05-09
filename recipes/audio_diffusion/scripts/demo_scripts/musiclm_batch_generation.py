import argparse
import math
import re
import unicodedata

import torch
import torchaudio
from mulan_infer import mulan_inference
from webdataset import WebDataset

from recipes.audio_diffusion.modules.model_types.musiclm_transformer import (
    CoarseTransformerDecoder,
    FineTransformerDecoder,
    SoundStreamTransformerDecoder,
    Text2SemanticTransformerDecoder,
)
from recipes.audio_diffusion.modules.pl_module import (
    MusicLMCoarseModule,
    MusicLMFineModule,
    MusicLMFineModuleLegacy,
    MusicLMText2SemanticModule,
)
from recipes.audio_diffusion.modules.transforms.encoder import SoundStreamTransform


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
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def load_text2semantic_model(ckpt_path):
    pl_module = (
        MusicLMText2SemanticModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            semantic_token_model=Text2SemanticTransformerDecoder(
                semantic_codebook_size=1024,
                transformer_emb_dim=768,
                transformer_heads=12,
                transformer_layers=12,
                transformer_ff_dim=3072,
                semantic_seq_len=250,
                use_text_emb_rvq=False,
            ),
            map_location=device,
            # strict=False
        )
        .to(device)
        .eval()
    )
    return pl_module


def load_coarse_model(ckpt_path):
    pl_module = (
        MusicLMCoarseModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            acoustic_token_model=CoarseTransformerDecoder(
                acoustic_codebook_size=1024,
                semantic_codebook_size=1024,
                transformer_emb_dim=1024,
                transformer_heads=16,
                transformer_layers=14,
                transformer_ff_dim=4096,
                acoustic_seq_len=800,
                semantic_seq_len=250,
                use_text_emb_rvq=False,
                start_channel=0,
                end_channel=2,
            ),
            cuda_transforms=soundstream_encoder,
            map_location=device,
            # strict=False
        )
        .to(device)
        .eval()
    )
    return pl_module


def load_fine_model(ckpt_path):
    if args.model == "mix":
        pl_module = (
            MusicLMFineModuleLegacy.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                acoustic_token_model=SoundStreamTransformerDecoder(
                    max_sequence_length=240,
                    transformer_emb_dim=1024,
                    transformer_heads=16,
                    transformer_layers=16,
                    transformer_ff_dim=4096,
                    start_channel=2,
                    end_channel=6,
                    use_gpt=True,
                    use_cond=False,
                ),
                cuda_transforms=soundstream_encoder,
                map_location=device,
                # strict=False
            )
            .to(device)
            .eval()
        )
    else:
        pl_module = (
            MusicLMFineModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                acoustic_token_model=FineTransformerDecoder(
                    acoustic_codebook_size=1024,
                    transformer_emb_dim=1024,
                    transformer_heads=16,
                    transformer_layers=14,
                    transformer_ff_dim=4096,
                    acoustic_seq_len=240,
                    start_channel=2,
                    end_channel=6,
                ),
                cuda_transforms=soundstream_encoder,
                map_location=device,
                # strict=False
            )
            .to(device)
            .eval()
        )
    return pl_module


def generate_semantic(text_emb, semantic_temp):
    sample_length = 250  # 10 seconds
    with torch.no_grad():
        print(f"Generating {sample_length} semantic tokens...")
        semantic_tokens = text2semantic_model.generate_samples(
            text_input=text_emb,
            temperature=semantic_temp,
            sequence_length=sample_length,
        )
    return semantic_tokens.long()


def generate_coarse(text_emb, semantic_tokens, coarse_temp):
    sample_length = 800  # 10 seconds
    with torch.no_grad():
        print(f"Generating {sample_length} coarse tokens...")
        coarse_tokens = coarse_model.generate_samples(
            text_input=text_emb,
            semantic_tokens=semantic_tokens,
            temperature=coarse_temp,
            sequence_length=sample_length,
        )
    return coarse_tokens.long()


def generate_fine(coarse_tokens, fine_temp, fine_step_size=80):
    window_size = 240
    assert fine_step_size <= window_size, "Step size can't be larger than window size!"

    prefix_size = window_size - fine_step_size

    with torch.no_grad():
        fine_tokens = torch.ones(coarse_tokens.size(0), 4, 0, device=device)
        start_idx = 0
        samples_to_generate = window_size
        seed_sequence = None
        while fine_tokens.shape[2] < coarse_tokens.shape[2]:
            print(f"Generating {samples_to_generate} fine tokens...")
            sub_coarse_tokens = coarse_tokens[:, :, start_idx : start_idx + window_size]
            sub_fine_tokens = fine_model.generate_samples(
                coarse_tokens=sub_coarse_tokens,
                temperature=fine_temp,
                sequence_length=window_size,
                seed_sequence=seed_sequence,
            )
            selected_fine_tokens = sub_fine_tokens[:, :, -samples_to_generate:]
            fine_tokens = torch.cat([fine_tokens, selected_fine_tokens], dim=2)
            # Prepare for next round
            start_idx += fine_step_size
            samples_to_generate = fine_step_size
            if prefix_size > 0:
                dummy_coarse_tokens = coarse_tokens[:, :, -prefix_size:]
                prefix_fine_tokens = sub_fine_tokens[:, :, -prefix_size:]
                tmp_model = fine_model.acoustic_token_model
                seed_sequence = tmp_model.extract_channel_group_and_interleave(
                    labels=torch.cat(
                        [dummy_coarse_tokens, prefix_fine_tokens], dim=1
                    ).permute(0, 2, 1),
                    start_channel=2,
                    end_channel=6,
                    apply_offset=True,
                )
            else:
                seed_sequence = None
    return fine_tokens[:, :, : coarse_tokens.size(2)].long()


def tokenize(text, tokenizer):
    token_outputs = tokenizer(text, return_tensors="pt")
    token_outputs["input_ids"] = token_outputs["input_ids"].to(device)
    token_outputs["token_type_ids"] = token_outputs["token_type_ids"].to(device)
    token_outputs["attention_mask"] = token_outputs["attention_mask"].to(device)
    return token_outputs


def generate_by_musiclm(text_input, semantic_temp, coarse_temp, fine_temp):
    if args.retrieval:
        text_embs = text_input.unsqueeze(1).to(device)
    else:
        text_embs = []
        with torch.no_grad():
            for tp in text_input:
                text_embs.append(
                    mulan_inference(mulan_model, text=tp, device=device).unsqueeze(0)
                )
        text_embs = torch.cat(text_embs, dim=0)

    output_demos = []
    for text_emb in torch.chunk(text_embs, 2, dim=0):
        generated_semantic_tokens = generate_semantic(text_emb, semantic_temp)
        generated_coarse_tokens = generate_coarse(
            text_emb, generated_semantic_tokens, coarse_temp
        )
        generated_fine_tokens = generate_fine(generated_coarse_tokens, fine_temp)
        generated_acoustic_tokens = torch.cat(
            [generated_coarse_tokens, generated_fine_tokens], dim=1
        )
        print("SoundStream is decoding acoustic tokens...")
        with torch.no_grad():
            demos = soundstream_encoder.decode(generated_acoustic_tokens).cpu()
        for i in range(demos.size(0)):
            demos[i] = demos[i] / demos[i].abs().max()
        print("Outputting the generated audio...")
        output_demos.append(demos)
    output_demos = torch.cat(output_demos, dim=0)
    return output_demos


##################################
#     For manual generating   #
##################################


def get_retrieved_emb(tar_path):
    sub_dir = "retrieval" if args.retrieval else "text"
    retrieval_embs = WebDataset(tar_path).decode()
    prompts = []
    embs = []
    for item in retrieval_embs:
        prompts.append(item["text.txt"])
        embs.append(torch.from_numpy(item["emb.npy"]))
        top_wavs = torch.from_numpy(item["wav.npy"])
        for i, wav in enumerate(top_wavs[: int(args.topk)]):
            torchaudio.save(
                (
                    f"demo/{sub_dir}/{args.model}/"
                    + f'{slugify(item["text.txt"][:40])}_TOP{i + 1}.wav'
                ),
                wav.unsqueeze(0),
                24000,
            )
    embs = torch.stack(embs, dim=0)

    return prompts, embs


def get_prompts_and_embs():
    total_prompts = []
    total_embs = []
    txt_files = [
        "musiccaps_short_prompts",
        "musiccaps_long_prompts",
        "painting_desc",
        "genre_stats",
    ]
    for file in txt_files:
        prompts, embs = get_retrieved_emb(
            "/mnt/bn/audio-diffusion/data/mulan_prompts/"
            + f"{file}.tar.mulan149.retrieved_with_wav.top_{args.topk}"
        )
        for prompt, emb in zip(prompts, embs):
            total_prompts.append(prompt)
            total_embs.append(emb)
    total_embs = torch.stack(total_embs, dim=0)
    return total_prompts, total_embs


def run_transforms(transforms, x):
    for transform in transforms:
        x = transform(x)
    # Pad time dimension to 1024
    x = torch.nn.functional.pad(x, ((0, 1024 - x.shape[-1])))
    # batch_size x time x freq
    x = x.transpose(1, 2)
    return x


def get_transforms():
    to_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        # 400 (25ms) in AST with 16K.
        # 42.6ms if using 1024 in 24K. 46.43ms in music tagging (1024, 22050Hz)
        n_fft=1024,
        # 10ms in AST with 16K.
        # 21.3ms if using 512 in 24K. 23.2ms in music tagging task (512, 22050Hz)
        hop_length=240,
        f_min=0,
        f_max=24000 // 2,
        n_mels=128,
        window_fn=torch.hann_window,
        power=2.0,
        center=True,
        pad_mode="reflect",
    ).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
    return to_mel, to_db


@torch.no_grad()
def get_audio_emb(audio, model, transforms, batch_size=1):
    sample_rate = 24000  # 24kHz
    segment_length = sample_rate * 10  # 10s
    x = torch.from_numpy(audio).squeeze(0)
    x = x.to(device)
    # Pad to 10s if needed
    if x.shape[0] < segment_length:
        x = torch.nn.functional.pad(x, ((0, segment_length - x.shape[0])))
    # Pad into multiples of 1s and unfold with overlap
    num_segments = math.ceil(float(x.shape[0]) / sample_rate)
    x = torch.nn.functional.pad(x, ((0, sample_rate * num_segments - x.shape[0])))
    x = x.unfold(0, segment_length, sample_rate)
    x = run_transforms(transforms, x)
    audio_emb = None
    for i in range(0, x.shape[0], batch_size):
        sub_audio_emb = model.model.music_encoder(x[i : i + batch_size])
        if audio_emb is None:
            audio_emb = sub_audio_emb
        else:
            audio_emb = torch.cat([audio_emb, sub_audio_emb], dim=0)
    return audio_emb.detach().cpu().float().squeeze(0)


def evaluate(text_embs, generated_demos):
    with torch.no_grad():
        # for text_emb, generated_demo in zip(text_embs, generated_demos):
        audio_embs = mulan_inference(mulan_model, music=generated_demos, device=device)
        eval_stats = hit_score_rank(text_embs, audio_embs)

        print(f"{args.model} eval stats: ")
        print(f'hit_score: {eval_stats["hit_score"]}')
        print(f'median_rank: {eval_stats["median_rank"]}')


def hit_score_rank(source_embed, target_embed):
    assert source_embed.shape[1] == target_embed.shape[1]
    sample_size = source_embed.shape[0]
    mat = torch.matmul(source_embed, target_embed.T)
    smat, indx = mat.sort(dim=1, descending=True)
    score, ranklst = list(), list()
    for i in range(sample_size):
        rank = (indx[i] == i).nonzero(as_tuple=True)[0]
        ranklst.append(rank)
        score.append((sample_size - rank) / sample_size)

    # print("ranklst:  ", sorted(ranklst)[len(ranklst) // 2: len(ranklst) // 2 + 5])
    median_rank = sorted(ranklst)[len(ranklst) // 2]

    # print("ranklst:  ", sorted(ranklst)[len(ranklst) // 2: len(ranklst) // 2 + 5])
    return {"hit_score": sum(score) / len(score), "median_rank": median_rank}


def gen(gen_fn, params):
    sub_dir = "retrieval" if args.retrieval else "text"
    demos = gen_fn(retrieved_embs if args.retrieval else text_prompts, *params)
    for tp, demo in zip(text_prompts, demos):
        print(f"[SAVING] demo/{sub_dir}/{args.model}/{slugify(tp[:40])}.wav")
        torchaudio.save(
            f"demo/{sub_dir}/{args.model}/{slugify(tp[:40])}.wav", demo, 24000
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument("--model", default="resso")
    args = parser.add_argument("--topk", default="5")
    args = parser.add_argument("--retrieval", action="store_true")
    args = parser.parse_args()

    configs = {
        "mulan_ckpt": "/mnt/bn/audio-diffusion/pretrained_models/"
        "mulan/young-mulan-shortform-step=028000-median_rank_1=149-kaggle_merged.pt"
    }
    if args.model == "resso":
        configs["semantic_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326226/"
            "musiclm_text2semantic/karaoke_acc/checkpoints/epoch=0-step=26500.ckpt"
        )
        configs["coarse_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326275/"
            "musiclm_coarse/karaoke_acc/checkpoints/epoch=0-step=191000.ckpt"
        )
        configs["fine_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2334007/"
            "musiclm_fine/karaoke_resso_acc/checkpoints/epoch=0-step=100000.ckpt"
        )
    elif args.model == "karaoke":
        configs["semantic_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326140/"
            "musiclm_text2semantic/karaoke_acc/checkpoints/epoch=0-step=9500.ckpt"
        )
        configs["coarse_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326137/"
            "musiclm_coarse/karaoke_acc/checkpoints/epoch=0-step=142000.ckpt"
        )
        configs["fine_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2333890/"
            "musiclm_fine/karaoke_acc/checkpoints/epoch=0-step=82000.ckpt"
        )
    elif args.model == "mix":
        configs["semantic_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326226/"
            "musiclm_text2semantic/karaoke_acc/checkpoints/epoch=0-step=26500.ckpt"
        )
        configs["coarse_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/377594/trials/2326275/"
            "musiclm_coarse/karaoke_acc/checkpoints/epoch=0-step=191000.ckpt"
        )
        configs["fine_ckpt"] = (
            "/mnt/bn/audio-diffusion/logs/373893/trials/2327995/"
            "musiclm_fine/resso_mss_uncond/ckpt_53000.pt"
        )

    ####################
    #    model init   #
    ####################
    breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ####################
    #    text model   #
    ####################
    mulan_model = None
    print("Loading MuLan model...")
    # mulan_model = create_mulan_model(configs["mulan_ckpt"], device=device)

    ####################
    #     MusicLM     #
    ####################
    soundstream_encoder = (
        SoundStreamTransform(sample_rate=24000, return_continuous_latents=False)
        .to(device)
        .eval()
    )
    print("Loading semantic2text model...")
    text2semantic_model = load_text2semantic_model(configs["semantic_ckpt"])
    print("Loading coarse model...")
    coarse_model = load_coarse_model(configs["coarse_ckpt"])
    print("Loading fine model...")
    fine_model = load_fine_model(configs["fine_ckpt"])

    text_prompts, retrieved_embs = get_prompts_and_embs()
    print(f"Text Prompts: {text_prompts}")

    gen(generate_by_musiclm, (1, 1, 0.9))

import math
import re
import sys
import unicodedata

import torch
import torchaudio
from mulan_infer import create_mulan_model, mulan_inference
from transformers import BertTokenizer

from recipes.audio_diffusion.modules.diffusion_schemes import VDiffusionScheme
from recipes.audio_diffusion.modules.model_types.DAG_unet import DAGNext
from recipes.audio_diffusion.modules.pl_module import DiffusionLitModule
from recipes.audio_diffusion.modules.transforms.encoder import SoundStreamTransform

sys.path.append("/mnt/bn/audio-diffusion/pretrained_models/bytegen/models")
sys.path.append("/mnt/bn/audio-diffusion/pretrained_models/mtpretrain")


def slugify(value, allow_unicode=False):
    """
    Taken from django/text.py at main · django/django
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


configs = {
    "mulan_ckpt": (
        "/mnt/bn/audio-diffusion/pretrained_models/mulan/"
        "young-mulan-shortform-step=028000-median_rank_1"
        "=149-kaggle_merged.pt"
    ),
    "diffusion_ckpt": (
        "/mlx_devbox/users/julian.parker/playground/"
        "samantha/logs/diffusion/DAG_700M_new_mulan/"
        "checkpoints/last-v2.ckpt"
    ),
}


def load_diffusion_model(ckpt_path):
    pl_module = (
        DiffusionLitModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=DAGNext(
                num_signal_channels=256,
                downsampling_factors=[1, 2, 2, 2, 2, 5],
                hidden_sizes=[512, 512, 1024, 2048, 2048, 2048],
                middle_bypassed=[False, False, False, False, False, False],
                resampling_block_kernel_sizes=[15, 15, 15, 15, 15, 7],
                resampling_block_depths=[8, 4, 2, 2, 2, 2],
                text_cond_dim=128,
                text_cond_hidden_dim=512,
                text_cond_noise_coeff=0.000,
            ),
            diffusion_scheme=VDiffusionScheme(),
            map_location=device,
            strict=False,
        )
        .to(device)
        .eval()
    )
    return pl_module


def tokenize(text, tokenizer):
    token_outputs = tokenizer(text, return_tensors="pt")
    token_outputs["input_ids"] = token_outputs["input_ids"].to(device)
    token_outputs["token_type_ids"] = token_outputs["token_type_ids"].to(device)
    token_outputs["attention_mask"] = token_outputs["attention_mask"].to(device)
    return token_outputs


def generate_by_diffusion(raw_text_prompts, guidance_scale, num_steps):
    text_emb = []
    print("Starting_mulan_embedding...")
    with torch.no_grad():
        for tp in raw_text_prompts:
            text_emb.append(
                mulan_inference(mulan_model, text=tp, device=device).unsqueeze(0)
            )
    text_emb = torch.cat(text_emb, dim=0).permute(0, 2, 1)

    initial_latents = torch.randn(text_emb.shape[0], 256, 800, device=device)
    print("Diffusing...")
    with torch.no_grad():
        generated_latents = diffusion_model.generate_samples(
            initial_latents,
            text_emb,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
        )

    print("SoundStream is decoding acoustic tokens...")
    with torch.no_grad():
        demos = soundstream_encoder.decode(generated_latents).cpu()
    for i in range(demos.size(0)):
        demos[i] = demos[i] / demos[i].abs().max()
    print("Outputting the generated audio...")
    return demos


######################
#    model init   #
######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
#    text model   #
######################
print("Loading BERT tokenizer...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Loading MuLan model...")
mulan_model = create_mulan_model(configs["mulan_ckpt"], device=device)

######################
#     MusicLM     #
######################
soundstream_encoder = (
    SoundStreamTransform(sample_rate=24000, return_continuous_latents=True)
    .to(device)
    .eval()
)
print("Loading diffusion model...")
diffusion_model = load_diffusion_model(configs["diffusion_ckpt"])

##################################
#     For manual generating   #
##################################


def get_prompts():
    prompts = []
    txt_files = [
        "/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_short_prompts.txt",
        "/mnt/bn/audio-diffusion/data/mulan_prompts/musiccaps_long_prompts.txt",
        "/mnt/bn/audio-diffusion/data/mulan_prompts/painting_desc.txt",
        "/mnt/bn/audio-diffusion/data/mulan_prompts/genre_stats.txt",
    ]
    for file in txt_files:
        with open(file, "r") as fp:
            prompts += [t.strip() for t in fp.readlines()][:10]
    return prompts


text_prompts = get_prompts()
print(f"Text Prompts: {text_prompts}")


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
        n_fft=1024,
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


def evaluate(text_prompt, generated_audio):
    with torch.no_grad():
        token_outputs = tokenize(text_prompt, bert_tokenizer)
        text_emb = mulan_model.model.text_encoder(**token_outputs)[0]

        audio_emb = get_audio_emb(generated_audio, mulan_model, get_transforms())
        distance = sum((text_emb.cpu() - audio_emb.cpu()) ** 2).item()
    return distance


def gen(gen_fn, params):
    demos = gen_fn(text_prompts, *params)
    for tp, demo in zip(text_prompts, demos):
        torchaudio.save(f"demo_text/{slugify(tp[:40])}.wav", demo, 24000)


if __name__ == "__main__":
    gen(generate_by_diffusion, (6.0, 200))

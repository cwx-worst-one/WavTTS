import re
import sys
import unicodedata

import torch
import torchaudio
from webdataset import WebDataset

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
    "diffusion_ckpt": (
        "/mlx_devbox/users/julian.parker/playground/samantha/"
        "logs/diffusion/DAG_700M_new_mulan/checkpoints/last-v2.ckpt"
    )
}


def get_retrieved_emb(tar_path):
    retrieval_embs = WebDataset(tar_path).decode()
    prompts = []
    embs = []
    for item in retrieval_embs:
        prompts.append(item["text.txt"])
        embs.append(torch.from_numpy(item["emb.npy"]))
    embs = torch.stack(embs, dim=0)
    return prompts, embs


def get_prompts_and_embs(topk="1000"):
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
            (
                f"/mnt/bn/audio-diffusion/data/mulan_prompts/{file}.tar."
                f"mulan149.retrieved.top_{topk}.interpolated"
            )
        )
        for prompt, emb in zip(prompts, embs):
            total_prompts.append(prompt)
            total_embs.append(emb)
    total_embs = torch.stack(total_embs, dim=0)
    return total_prompts, total_embs


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


def generate_by_diffusion(text_emb, guidance_scale, num_steps):
    text_emb = text_emb.unsqueeze(-1).to(device)
    # text_emb = text_emb[0:4,...]
    print(text_emb.shape)

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
#     model init     #
######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################
#      MusicLM       #
######################
soundstream_encoder = (
    SoundStreamTransform(sample_rate=24000, return_continuous_latents=True)
    .to(device)
    .eval()
)
print("Loading diffusion model...")
diffusion_model = load_diffusion_model(configs["diffusion_ckpt"])

##################################
#      For manual generating     #
##################################

text_prompts, embs = get_prompts_and_embs()
print(f"Text Prompts: {text_prompts}")


def gen(gen_fn, params):
    demos = gen_fn(embs, *params)
    for tp, demo in zip(text_prompts, demos):
        torchaudio.save(f"demo/{slugify(tp[:40])}.wav", demo, 24000)


if __name__ == "__main__":
    gen(generate_by_diffusion, (8.0, 100))

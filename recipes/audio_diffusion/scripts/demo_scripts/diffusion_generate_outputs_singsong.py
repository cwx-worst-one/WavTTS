import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from samantha.utils.hparams import DotDict

sys.path.append("/mnt/bn/audio-diffusion/pretrained_models/bytegen/models")
ckpt_path = (
    "/mnt/bn/audio-diffusion/logs/281298/trials/2336145/diffusion/"
    "DAG_700M_singsong/checkpoints/last.ckpt"
)
config = "conf/latent_diffusion_reboot/DAG_700M_resso_mix_singsong.yaml"
device = "cuda"


def load_config_from_file(config_fp: str):
    with open(config_fp, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides="")
    return DotDict(hparams)


def load_model(config_fp, ckpt_path, device):
    cfg = load_config_from_file(config_fp)
    pl_module = cfg.pl_module
    pl_datamodule = cfg.pl_datamodule
    encoder = cfg.encoder_transform.to(device)
    pl_module = (
        pl_module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=cfg.model,
            cuda_transforms=encoder,
            map_location=device,
        )
        .to(device)
        .eval()
    )
    return pl_module, pl_datamodule, encoder


pl_module, pl_datamodule, encoder = load_model(config, ckpt_path, device)
print("Loaded! \n")

audio, cond, vocal = next(iter(pl_datamodule.val_dataloader()))

num_parallel_examples = audio.shape[0]

output_dir = "./test_output/"

initial_latents = torch.randn(num_parallel_examples, 256, 800 * 1, device=device)

text_cond = cond.to(device)[0:num_parallel_examples, ...]  # .unsqueeze(1)
print(f"text_cond{text_cond.shape}")
print("Generating... \n")
with torch.no_grad():
    result = pl_module.generate_samples(
        initial_latents, text_cond, guidance_scale=6.0, num_steps=100
    )
    result = encoder.decode(result)
    result += vocal[0:num_parallel_examples, ...].to(device)
print("Saving... \n")
result = (result / result.abs().max()).detach().cpu()
for i in range(num_parallel_examples):
    torchaudio.save(output_dir + f"test_{i}.wav", result[i, :, :], 24000)
    torchaudio.save(output_dir + f"ref_{i}.wav", audio[i, :, :] + vocal[i, :, :], 24000)

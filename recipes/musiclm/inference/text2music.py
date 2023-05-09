import os
import pickle

import torch
from pytorch_lightning import seed_everything

from recipes.musiclm.inference.utils import load_config, load_model
from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.acoustic_modeling.semantic_acoustic_model import (
    SemanticAcousticModel,
)
from recipes.musiclm.lightning.semantic_modeling.mulan_semantic_model import (
    MulanSemanticModel,
)
from recipes.musiclm.utils.prompts import gather_prompts


def inference_pipeline(
    mulan_token_ids,
    semantic_token_ids,
    coarse_model: SemanticAcousticModel,
    fine_model: AcousticModel,
    coarse_temperature: float,
    fine_temperature: float,
):
    # coarse_token_ids = coarse_model.sample_with_audio_conditioning(
    coarse_token_ids = coarse_model.sample_with_semantic_conditioning(
        mulan_token_ids, semantic_token_ids, temperature=coarse_temperature
    )

    fine_token_ids = []
    split_coarse_tokens = coarse_token_ids.split(
        fine_model.hparams.max_sequence_length, dim=2
    )

    for s in split_coarse_tokens:
        fine_token_ids.append(
            fine_model.sample_with_audio_conditioning(
                acoustic_token_ids=s.clone(), temperature=fine_temperature
            )
        )

    fine_token_ids = torch.cat(fine_token_ids, dim=2)

    coarse_token_ids = coarse_token_ids[..., : fine_token_ids.shape[2]]
    acoustic_token_ids = torch.cat((coarse_token_ids, fine_token_ids), dim=1)

    with torch.no_grad():
        sampled_audio = fine_model.audio_model.decode(acoustic_token_ids)
    return sampled_audio


if __name__ == "__main__":
    device = "cuda"

    max_samples = 200
    seed_everything(52)

    cfg = load_config(
        "recipes/musiclm/conf/semantic_modeling/musiclm_mulan_semantic.yaml"
    )
    pl_datamodule = cfg.pl_datamodule
    val_loader = pl_datamodule.val_dataloader()
    mulan_semantic_model: MulanSemanticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/pretrained_models/musiclm/janne/semantic/last.ckpt",  # noqa
        device=device,
    )
    mulan_semantic_model.on_train_epoch_start()  # cast to devices
    mulan_semantic_model.hparams.max_sequence_length = (
        mulan_semantic_model.semantic_model.n_frames
    )

    cfg = load_config(
        "recipes/musiclm/conf/acoustic_modeling/musiclm_mulan_semantic_coarse.yaml"
    )
    coarse_model: SemanticAcousticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/pretrained_models/musiclm/janne/coarse/last.ckpt",  # noqa
        device=device,
    )
    coarse_model.on_train_epoch_start()  # cast to devices

    cfg = load_config("./recipes/musiclm/conf/acoustic_modeling/musiclm_fine.yaml")
    fine_model: AcousticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/pretrained_models/musiclm/janne/fine/epoch=0-step=171000-refactor.ckpt",  # noqa
        device=device,
    )

    batch_size = 3
    text_embs, audio_embs, prompts, categories = gather_prompts(
        mulan_semantic_model.mulan_model,
        "google",
        "Acoustic guitar. Funky jazz.",
        batch_size=batch_size,
        device=device,
    )

    batch_idx = int(os.getenv("ARNOLD_ID", 0))
    num_rounds = int(os.getenv("NUM_ROUNDS", 1))

    text_batch = torch.split(text_embs, batch_size)
    audio_batch = torch.split(audio_embs, batch_size)
    batched_prompts = [
        prompts[idx : idx + batch_size] for idx in range(0, len(prompts), batch_size)
    ]
    batched_categories = [
        categories[idx : idx + batch_size] for idx in range(0, len(prompts), batch_size)
    ]

    dir_fp = os.path.join(
        "/mnt/bn/audio-diffusion/results", os.getenv("ARNOLD_TRIAL_ID", "")
    )
    os.makedirs(dir_fp, exist_ok=True)

    if batch_idx > len(text_batch):
        print("No more examples to generate")
        exit(0)

    for rd in range(num_rounds):
        mulan_token_ids = text_batch[batch_idx]
        audio_embeds_batch = audio_batch[batch_idx]

        with torch.no_grad():
            semantic_token_ids = mulan_semantic_model.sample_with_conditioning(
                mulan_token_ids, temperature=0.8, data_type="mulan"
            )

        sampled_audio = inference_pipeline(
            mulan_token_ids,
            semantic_token_ids,
            coarse_model,
            fine_model,
            coarse_temperature=0.94,
            fine_temperature=0.9,
        )

        obj = {
            "audio": sampled_audio.cpu(),
            "mulan_token_ids": mulan_token_ids.cpu(),
            "semantic_token_ids": semantic_token_ids.cpu(),
            "prompt": batched_prompts[batch_idx],
            "categories": batched_categories[batch_idx],
        }
        fp = os.path.join(dir_fp, f"generation-{batch_idx}-{rd}.pkl")
        with open(fp, "wb") as f:
            pickle.dump(obj, f)

import torch
from tqdm import tqdm

from recipes.musiclm.inference.utils import load_config, load_model
from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.acoustic_modeling.semantic_acoustic_model import (
    SemanticAcousticModel,
)


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
        fine_model.model.max_sequence_length, dim=2
    )

    for (
        s
    ) in (
        split_coarse_tokens
    ):  # TODO: We drop last as it's (usually) not a full sequence, bug in offsetting.
        fine_token_ids.append(
            fine_model.sample(
                cond_token_ids=s.clone(),
                temperature=fine_temperature,
                batch_size=s.shape[0],
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

    cfg = load_config(
        "recipes/musiclm/conf/acoustic_modeling/musiclm_mulan_semantic_coarse.yaml"
    )
    pl_datamodule = cfg.pl_datamodule
    val_loader = pl_datamodule.val_dataloader()
    coarse_model: SemanticAcousticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/logs/403979/trials/2492184/musiclm_mulan_semantic_coarse/baseline_v4/checkpoints/epoch=0-step=129000.ckpt",  # noqa
        device=device,
    )
    coarse_model.on_train_epoch_start()  # cast to devices

    max_samples = 100
    all_tokens = []
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        if batch_idx > max_samples:
            break
        with torch.no_grad():
            source_audio = batch[0].to(coarse_model.device)

            tokens = {}
            mulan_token_ids = coarse_model.mulan_model(source_audio, data_type="music")
            semantic_token_ids = coarse_model.semantic_model(source_audio)

            tokens["mulan_token_ids"] = mulan_token_ids.cpu()
            tokens["semantic_token_ids"] = semantic_token_ids.cpu()

            all_tokens.append(tokens)

    breakpoint()

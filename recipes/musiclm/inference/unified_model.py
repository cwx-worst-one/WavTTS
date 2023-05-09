import torch
import torchaudio
from pytorch_lightning import seed_everything

from recipes.musiclm.inference.utils import load_config, load_model
from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.unified_model import UnifiedModel


def inference_pipeline(
    input_audio,
    target_audio,
    unified_model: UnifiedModel,
    fine_model: AcousticModel,
    unified_temperature: float,
    fine_temperature: float,
):
    coarse_token_ids = unified_model.sample_ssa(
        input_audio, target_audio, temperature=unified_temperature
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
    seed_everything(52)

    cfg = load_config("recipes/musiclm/conf/unified_model.yaml")
    pl_datamodule = cfg.pl_datamodule
    val_loader = pl_datamodule.val_dataloader()
    unified_model = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/logs/480161/trials/2601662/unified_model/baseline_v6/checkpoints/epoch=0-step=33000.ckpt",  # noqa
        device=device,
    )
    unified_model.on_train_epoch_start()  # cast to devices

    cfg = load_config("./recipes/musiclm/conf/acoustic_modeling/musiclm_fine.yaml")
    fine_model = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/logs/401551/trials/2601657/musiclm_fine/baseline_v6/checkpoints/epoch=0-step=18000.ckpt",  # noqa
        device=device,
    )

    for idx, batch in enumerate(val_loader):
        input_audio = batch[0]
        target_audio = batch[1]
        input_audio = input_audio[:1]
        target_audio = target_audio[:1]

        sampled_audio = inference_pipeline(
            input_audio,
            target_audio,
            unified_model,
            fine_model,
            unified_temperature=0.94,
            fine_temperature=0.9,
        )

        torchaudio.save(f"{idx}-sampled.mp3", sampled_audio[0].cpu(), 24000)
        torchaudio.save(f"{idx}-input.mp3", input_audio[0].cpu(), 24000)
        torchaudio.save(f"{idx}-target.mp3", target_audio[0].cpu(), 24000)

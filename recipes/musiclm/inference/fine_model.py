import torch
import torchaudio
from pytorch_lightning import seed_everything

from recipes.musiclm.inference.utils import load_config, load_model
from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel


def inference_pipeline(input_audio, fine_model: AcousticModel, fine_temperature: float):
    input_audio = input_audio.to(fine_model.device)

    with torch.no_grad():
        acoustic_tokens = fine_model.audio_model(input_audio)

    coarse_tokens = acoustic_tokens[:, fine_model.hparams.cond_quantizers]

    pred_tokens = fine_model.sample_with_audio_conditioning(
        acoustic_token_ids=coarse_tokens, temperature=fine_temperature
    )

    pred_acoustic_tokens = torch.cat((coarse_tokens, pred_tokens), dim=1)

    with torch.no_grad():
        sampled_audio = fine_model.audio_model.decode(pred_acoustic_tokens)
    return sampled_audio


if __name__ == "__main__":
    device = "cuda"
    seed_everything(52)

    cfg = load_config("./recipes/musiclm/conf/acoustic_modeling/musiclm_fine.yaml")
    pl_datamodule = cfg.pl_datamodule
    val_loader = pl_datamodule.val_dataloader()
    fine_model = load_model(
        cfg.pl_module,
        ckpt_path="hdfs://harunava/home/byte_arnold_va/data/lab/audio/musiclm/tasks/401551/trials/2628455/musiclm_fine/baseline_v6/checkpoints/epoch=0-step=171000-refactor.ckpt",  # noqa
        device=device,
    )

    for idx, batch in enumerate(val_loader):
        input_audio = batch[0]
        input_audio = input_audio[:1]
        torchaudio.save(f"{idx}-input.mp3", input_audio[0].cpu(), 24000)

        sampled_audio = inference_pipeline(
            input_audio, fine_model, fine_temperature=0.4
        )

        torchaudio.save(f"{idx}-sampled.mp3", sampled_audio[0].cpu(), 24000)

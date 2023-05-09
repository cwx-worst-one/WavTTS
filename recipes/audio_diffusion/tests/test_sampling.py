import torch
from hyperpyyaml import load_hyperpyyaml

from samantha.utils.hparams import DotDict
from tests.helpers.runif import RunIf


@RunIf(min_cuda_gpus=1)
def test_sampling():
    hparams_file = (
        "./recipes/audio_diffusion/conf/"
        "140223-singsong/singsong_semantic2semantic-coarse.yaml"
    )

    with open(hparams_file, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)

    cfg = DotDict(hparams)
    pl_module = cfg.pl_module.to("cuda")

    val_dataloader = cfg.pl_datamodule.val_dataloader()
    audio_acc, audio_vocal_ids, semantic_acc, semantic_vocal_ids = next(
        iter(val_dataloader)
    )
    audio_vocal_ids = audio_vocal_ids.to(pl_module.device)
    semantic_vocal_ids = semantic_vocal_ids.to(pl_module.device)

    sequence_length = 800

    # acoustic_token_ids = pl_module.cuda_transforms.quantize(
    #     pl_module.cuda_transforms.encode(audio_vocal_ids)
    # )

    with torch.no_grad():
        sampled_acoustic_token_ids = pl_module.generate_samples(
            cond_embeddings=semantic_vocal_ids,
            labels=None,
            temperature=1.0,
            num_outputs=1,
            sequence_length=sequence_length,
        )

    assert sampled_acoustic_token_ids.shape == (1, 2, sequence_length)

import argparse
import os
import pickle
from typing import Optional

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


@torch.no_grad()
def sample_semantic(
    semantic_model: MulanSemanticModel,
    mulan_token_ids: torch.LongTensor,
    temperature: float,
    seq_len: Optional[int],
    overlap: float,
):
    if seq_len is None:
        seq_len = semantic_model.hparams.max_sequence_length
    batch_size = mulan_token_ids.shape[0]
    semantic_token_ids = torch.empty(
        batch_size, 0, device=semantic_model.device, dtype=torch.long
    )

    window_size = semantic_model.hparams.max_sequence_length
    prefix_size = int(window_size * overlap)
    assert (seq_len - window_size) % (window_size - prefix_size) == 0
    while semantic_token_ids.shape[1] < seq_len:
        this_prefix = None
        if semantic_token_ids.shape[1] > 0 and prefix_size > 0:
            this_prefix = semantic_token_ids[..., -prefix_size:]
        this_semantic_token_ids = semantic_model.sample_with_conditioning(
            mulan_token_ids,
            temperature=temperature,
            data_type="mulan",
            prefix=this_prefix,
        )
        semantic_token_ids = torch.cat(
            (semantic_token_ids, this_semantic_token_ids), dim=1
        )
    return semantic_token_ids[..., :seq_len]


@torch.no_grad()
def sample_coarse(
    mulan_token_ids: torch.LongTensor,
    semantic_token_ids: torch.LongTensor,
    coarse_model: SemanticAcousticModel,
    temperature: float,
    overlap: float,
):
    # HACK: assume coarse frame rate is twice semantic frame rate
    RATIO = 2
    seq_len = semantic_token_ids.shape[1] * RATIO
    batch_size = mulan_token_ids.shape[0]
    coarse_token_ids = torch.empty(
        batch_size,
        len(coarse_model.input_quantizers),
        0,
        device=coarse_model.device,
        dtype=torch.long,
    )

    window_size = coarse_model.hparams.max_sequence_length
    assert window_size % 2 == 0
    semantic_window_size = int(window_size / RATIO)
    prefix_size = int(window_size * overlap)
    assert (seq_len - window_size) % (window_size - prefix_size) == 0
    while coarse_token_ids.shape[2] < seq_len:
        this_prefix = None
        if coarse_token_ids.shape[2] == 0:
            st_semantic = 0
        else:
            st_coarse = coarse_token_ids.shape[2] - prefix_size
            st_semantic = int(st_coarse / RATIO)
            if prefix_size > 0:
                this_prefix = coarse_token_ids[..., -prefix_size:]
        en_semantic = st_semantic + semantic_window_size
        this_coarse_token_ids = coarse_model.sample_with_semantic_conditioning(
            mulan_token_ids,
            semantic_token_ids=semantic_token_ids[..., st_semantic:en_semantic],
            temperature=temperature,
            prefix=this_prefix,
        )
        coarse_token_ids = torch.cat((coarse_token_ids, this_coarse_token_ids), dim=2)
    return coarse_token_ids[..., :seq_len]


@torch.no_grad()
def sample_fine(
    coarse_token_ids: torch.LongTensor,
    fine_model: AcousticModel,
    temperature: float,
    overlap: float,
):
    seq_len = coarse_token_ids.shape[2]
    batch_size = coarse_token_ids.shape[0]
    fine_token_ids = torch.empty(
        batch_size,
        len(fine_model.input_quantizers),
        0,
        device=fine_model.device,
        dtype=torch.long,
    )

    window_size = fine_model.hparams.max_sequence_length
    prefix_size = int(window_size * overlap)
    assert (seq_len - window_size) % (window_size - prefix_size) == 0
    while fine_token_ids.shape[2] < seq_len:
        this_prefix = None
        if fine_token_ids.shape[2] == 0:
            st = 0
        else:
            st = fine_token_ids.shape[2] - prefix_size
            if prefix_size > 0:
                this_prefix = fine_token_ids[..., -prefix_size:]
        this_fine_token_ids = fine_model.sample_with_audio_conditioning(
            acoustic_token_ids=coarse_token_ids[..., st : st + window_size],
            temperature=temperature,
            prefix=this_prefix,
        )
        fine_token_ids = torch.cat((fine_token_ids, this_fine_token_ids), dim=2)
    return fine_token_ids[..., :seq_len]


@torch.no_grad()
def inference_pipeline(
    mulan_token_ids: torch.LongTensor,
    semantic_token_ids: torch.LongTensor,
    coarse_model: SemanticAcousticModel,
    fine_model: AcousticModel,
    coarse_temperature: float,
    fine_temperature: float,
    coarse_overlap: float,
    fine_overlap: float,
):
    coarse_token_ids = sample_coarse(
        mulan_token_ids,
        semantic_token_ids,
        coarse_model,
        temperature=coarse_temperature,
        overlap=coarse_overlap,
    )
    fine_token_ids = sample_fine(
        coarse_token_ids, fine_model, temperature=fine_temperature, overlap=fine_overlap
    )
    acoustic_token_ids = torch.cat((coarse_token_ids, fine_token_ids), dim=1)
    sampled_audio = fine_model.audio_model.decode(acoustic_token_ids)
    return sampled_audio


def main(args):
    device = "cuda"
    seed_everything(52)

    print("Loading semantic model...")
    cfg = load_config(
        "recipes/musiclm/conf/semantic_modeling/musiclm_mulan_semantic.yaml"
    )
    pl_datamodule = cfg.pl_datamodule
    _ = pl_datamodule.val_dataloader()
    mulan_semantic_model: MulanSemanticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/pretrained_models/musiclm/janne/semantic/last.ckpt",  # noqa
        device=device,
    )
    mulan_semantic_model.on_train_epoch_start()  # cast to devices
    mulan_semantic_model.hparams.max_sequence_length = (
        mulan_semantic_model.semantic_model.n_frames
    )

    print("Loading coarse model...")
    cfg = load_config(
        "recipes/musiclm/conf/acoustic_modeling/musiclm_mulan_semantic_coarse.yaml"
    )
    coarse_model: SemanticAcousticModel = load_model(
        cfg.pl_module,
        ckpt_path="/mnt/bn/audio-diffusion/pretrained_models/musiclm/janne/coarse/last.ckpt",  # noqa
        device=device,
    )
    coarse_model.on_train_epoch_start()  # cast to devices

    print("Loading fine model...")
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
        _ = audio_batch[batch_idx]

        semantic_token_ids = sample_semantic(
            semantic_model=mulan_semantic_model,
            mulan_token_ids=mulan_token_ids,
            temperature=0.8,
            seq_len=args.seq_len,
            overlap=args.semantic_overlap,
        )

        sampled_audio = inference_pipeline(
            mulan_token_ids,
            semantic_token_ids,
            coarse_model,
            fine_model,
            coarse_temperature=0.94,
            fine_temperature=0.9,
            coarse_overlap=args.coarse_overlap,
            fine_overlap=args.fine_overlap,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Generated sequence length in terms of semantic tokens. If not "
        "specified, use the training sequence length.",
    )
    parser.add_argument(
        "--semantic_overlap",
        type=float,
        default=0.5,
        help="Amount of overlap in strided generation for semantic model.",
    )
    parser.add_argument(
        "--coarse_overlap",
        type=float,
        default=0.5,
        help="Amount of overlap in strided generation for coarse model.",
    )
    parser.add_argument(
        "--fine_overlap",
        type=float,
        default=0.5,
        help="Amount of overlap in strided generation for fine model.",
    )
    args = parser.parse_args()
    main(args)

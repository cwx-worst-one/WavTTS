import argparse
import itertools
import os

import torch
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml
from pytorch_lightning.loggers import TensorBoardLogger

from samantha.utils.hparams import DotDict


def load_config_from_file(config_fp: str):
    with open(config_fp, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides="")
    return DotDict(hparams)


def load_model(config_fp, ckpt_path, device):
    cfg = load_config_from_file(config_fp)
    pl_module = cfg.pl_module
    pl_datamodule = cfg.pl_datamodule
    encoder = cfg.encoder_transform.to(device)
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    pl_module.load_state_dict(state_dict=state_dict)
    pl_module = pl_module.to(device)
    return pl_module, pl_datamodule, encoder


root_dir = "/mlx_devbox/users/janne.spijkervet/repo/44/samantha"
logs_dir = os.path.join(root_dir, "logs")
configs = {
    "karaoke_acc": {
        "coarse_yaml": os.path.join(
            root_dir,
            "recipes/audio_diffusion/conf/140223-singsong/",
            "singsong_semantic2semantic-coarse.yaml",
        ),
        "coarse_ckpt": (
            "/mlx_devbox/users/janne.spijkervet/repo/44/samantha/"
            "logs/2320967/singsong/s2sa_large/checkpoints/epoch=0-step=22000.ckpt"
        ),
        "fine_yaml": os.path.join(
            root_dir,
            "recipes/audio_diffusion/conf/080223-coarse-fine/",
            "musiclm_fine_v2_karaoke_acc.yaml",
        ),
        "fine_ckpt": (
            "/mnt/bn/audio-diffusion/logs/ducle/musiclm_fine/karaoke_acc_uncond_v2/"
            "checkpoints/last.ckpt"
        ),
    }
}


def generate_fine_from_coarse(fine_model, fine_temp, coarse_ids, fine_step_size):
    window_size = 240
    assert fine_step_size <= window_size, "Step size can't be larger than window size!"
    assert (
        coarse_ids.shape[2] - window_size
    ) % fine_step_size == 0, "For now, remaining length must be multiples of step size"
    prefix_size = window_size - fine_step_size

    if fine_model.training:
        print("Fine model is in training mode, setting to eval!")
        fine_model = fine_model.eval()
    # Fine generation
    with torch.no_grad():
        fine_ids = None
        start_idx = 0
        samples_to_generate = window_size
        seed_sequence = None
        while fine_ids is None or fine_ids.shape[2] < coarse_ids.shape[2]:
            print(f"Generating {samples_to_generate} fine samples...")
            sub_labels = coarse_ids[:, :, start_idx : start_idx + window_size]
            sub_fine_ids = fine_model.generate_samples(
                cond_embeddings=None,
                labels=sub_labels,
                temperature=fine_temp,
                num_outputs=1,
                sequence_length=window_size,
                seed_sequence=seed_sequence,
            )
            selected_fine_ids = sub_fine_ids[:, :, -samples_to_generate:]
            if fine_ids is None:
                fine_ids = selected_fine_ids
            else:
                fine_ids = torch.cat([fine_ids, selected_fine_ids], dim=2)
            # Prepare for next round
            start_idx += fine_step_size
            samples_to_generate = fine_step_size
            if prefix_size > 0:
                dummy_coarse_ids = coarse_ids[:, :, -prefix_size:]
                prefix_fine_ids = sub_fine_ids[:, :, -prefix_size:]

                fn = (
                    fine_model.acoustic_token_model.extract_channel_group_and_interleave
                )
                seed_sequence = fn(
                    labels=torch.cat(
                        [dummy_coarse_ids, prefix_fine_ids], dim=1
                    ).permute(0, 2, 1),
                    start_channel=2,
                    end_channel=6,
                    apply_offset=True,
                )
            else:
                seed_sequence = None
    return fine_ids


def generate_coarse(
    coarse_model,
    coarse_temp,
    audio,
    audio_vocal,
    semantic_vocal,
    encoder,
    coarse_guided,
    coarse_step_size,
    sample_length,
):
    if coarse_guided > 0:
        assert audio is not None, "Can't do coarse guided if audio is None!"
    window_size = 800
    assert sample_length >= window_size
    assert (
        sample_length - window_size
    ) % coarse_step_size == 0, (
        "For now, remaining length must be multiples of step size"
    )
    assert (
        coarse_step_size <= window_size
    ), "Step size can't be larger than window size!"
    prefix_size = window_size - coarse_step_size

    if coarse_model.training:
        print("Coarse model is in training mode, setting to eval!")
        coarse_model = coarse_model.eval()
    # Coarse generation
    fn = coarse_model.acoustic_token_model.extract_channel_group_and_interleave
    with torch.no_grad():
        coarse_ids = None
        samples_to_generate = window_size
        seed_sequence = None
        while coarse_ids is None or coarse_ids.shape[2] < sample_length:
            print(f"Generating {samples_to_generate} coarse samples...")
            if coarse_ids is None and coarse_guided > 0:
                # Ground-truth IDs
                ref_codes = encoder.quantize(encoder.encode(audio))[
                    :, :, :coarse_guided
                ].permute(0, 2, 1)
                # Flatten and interleave

                seed_sequence = fn(
                    labels=ref_codes, start_channel=0, end_channel=2, apply_offset=True
                )
            # cond_embeddings = encoder.quantize(encoder.encode(audio_vocal))
            # TODO: we only use semantic here

            assert semantic_vocal.shape[1] == 250
            assert semantic_vocal.max() < 1024
            assert semantic_vocal.min() >= 0
            sub_coarse_ids = coarse_model.generate_samples(
                cond_embeddings=semantic_vocal,
                labels=None,
                temperature=coarse_temp,
                num_outputs=1,
                sequence_length=window_size,
                seed_sequence=seed_sequence,
            )
            selected_coarse_ids = sub_coarse_ids[:, :, -samples_to_generate:]
            if coarse_ids is None:
                coarse_ids = selected_coarse_ids
            else:
                coarse_ids = torch.cat([coarse_ids, selected_coarse_ids], dim=2)
            # Prepare for next round
            samples_to_generate = coarse_step_size
            fn = coarse_model.acoustic_token_model.extract_channel_group_and_interleave
            if prefix_size > 0:
                prefix_coarse_ids = coarse_ids[:, :, -prefix_size:]
                seed_sequence = fn(
                    labels=prefix_coarse_ids.permute(0, 2, 1),
                    start_channel=0,
                    end_channel=2,
                    apply_offset=True,
                )
            else:
                seed_sequence = None
    return coarse_ids


def save_audio(logger, fname, coarse_ids, fine_ids, encoder, step, audio_vocal):
    quant_ids = torch.cat([coarse_ids, fine_ids], dim=1)

    encoder = encoder.cuda()
    with torch.no_grad():
        quant_ids_codebook_dim = (
            quant_ids % 1024
        )  # TODO: is this correct? (shouldn't we be shifting by 1 too?)
        demo = encoder.decode(quant_ids_codebook_dim).cpu().squeeze(0)
    demo = demo / demo.abs().max()
    logger.experiment.add_audio(fname, demo, step, sample_rate=24000)
    logger.experiment.add_audio(
        fname + "_REMIXED", demo + audio_vocal.cpu(), step, sample_rate=24000
    )


def generate_impl(
    name,
    audio,
    audio_vocal,
    semantic_vocal,
    encoder,
    coarse_model,
    fine_model,
    logger,
    sample_id,
    coarse_step_size,
    fine_step_size,
    coarse_temp,
    fine_temp,
    step,
    coarse_guided,
    sample_length,
):
    suffix = f"{sample_id}"
    suffix = f"{suffix}_ct-{coarse_temp}_ft-{fine_temp}"
    suffix = f"{suffix}_css-{coarse_step_size}_fss-{fine_step_size}"
    suffix = f"{suffix}_guided-{coarse_guided}_sl-{sample_length}"
    sample_name = f"{step}_{name}_{suffix}"
    print(f"Sample {sample_name}...")

    print("Generating coarse IDs...")
    demo_coarse_ids = generate_coarse(
        coarse_model,
        coarse_temp,
        audio,
        audio_vocal,
        semantic_vocal,
        encoder,
        coarse_guided,
        coarse_step_size,
        sample_length,
    )

    demo_coarse_ids_for_fine = demo_coarse_ids.clone()
    print("Generating demo fine IDs...")
    demo_fine_ids = generate_fine_from_coarse(
        fine_model, fine_temp, demo_coarse_ids_for_fine, fine_step_size
    )

    save_audio(
        logger, sample_name, demo_coarse_ids, demo_fine_ids, encoder, step, audio_vocal
    )


def generate(
    log_dir,
    log_name,
    prefix,
    dataset,
    encoder,
    coarse_model,
    fine_model,
    coarse_step_sizes,
    fine_step_sizes,
    coarse_temps,
    fine_temps,
    num_samples,
    step,
    coarse_guideds,
    sample_lengths,
):
    logger = TensorBoardLogger(save_dir=logs_dir, name="demo", version=prefix)

    sample_id = 0
    while sample_id < num_samples:
        audio, audio_vocal, semantic_acc, semantic_vocal, audio_mix = next(
            iter(dataset.train_dataloader())
        )
        audio = audio.to(coarse_model.device)
        for i in range(audio.shape[0]):
            this_audio = audio[i : i + 1]
            this_audio_vocal = audio_vocal[i : i + 1].to(coarse_model.device)
            this_semantic_vocal = semantic_vocal[i : i + 1].to(coarse_model.device)

            print(
                f"this audio shape: {this_audio.shape} this audio vocal shape:"
                f" {this_audio_vocal.shape}"
            )
            # assert this_audio_vocal.ndim == this_audio.ndim
            logger.experiment.add_audio(
                f"demo/ref_{step}_{sample_id}_mix",
                (this_audio + this_audio_vocal) / 2,
                0,
                sample_rate=24000,
            )
            logger.experiment.add_audio(
                f"demo/ref_{step}_{sample_id}_acc", this_audio, 0, sample_rate=24000
            )
            logger.experiment.add_audio(
                f"demo/ref_{step}_{sample_id}_vocal",
                this_audio_vocal,
                0,
                sample_rate=24000,
            )
            for css, fss, ft, ct, cg, sl in itertools.product(
                coarse_step_sizes,
                fine_step_sizes,
                fine_temps,
                coarse_temps,
                coarse_guideds,
                sample_lengths,
            ):
                generate_impl(
                    this_audio,
                    this_audio_vocal,
                    this_semantic_vocal,
                    encoder,
                    coarse_model,
                    fine_model,
                    logger,
                    sample_id,
                    css,
                    fss,
                    ct,
                    ft,
                    step,
                    cg,
                    sl,
                )
            sample_id += 1
        if num_samples is not None and sample_id >= num_samples:
            break


def resolve_cfg(args):
    cfg = {
        "coarse_yaml": None,
        "coarse_ckpt": None,
        "fine_yaml": None,
        "fine_ckpt": None,
    }
    if args.config_name in configs:
        cfg = configs[args.config_name]
    if args.coarse_yaml is not None:
        cfg["coarse_yaml"] = args.coarse_yaml
    if args.coarse_ckpt is not None:
        cfg["coarse_ckpt"] = args.coarse_ckpt
    if args.fine_yaml is not None:
        cfg["fine_yaml"] = args.fine_yaml
    if args.fine_ckpt is not None:
        cfg["fine_ckpt"] = args.fine_ckpt
    for key in cfg:
        assert cfg[key] is not None, f"Missing {key}!"
    return cfg


def main(args):
    cfg = resolve_cfg(args)
    print(f"cfg: {cfg}")
    coarse_model, datamodule, _ = load_model(
        cfg["coarse_yaml"], cfg["coarse_ckpt"], args.device
    )
    fine_model, _, encoder = load_model(cfg["fine_yaml"], cfg["fine_ckpt"], args.device)

    if args.text_prompts is None:
        print("Reading from built-in dataloader...")
        dataloader = datamodule.train_dataloader()

        def dataset_fn():
            for audio, emb in dataloader:
                yield ["audio_demo/sampled"] * audio.shape[0], audio, emb

        dataset = dataset_fn()
    else:
        print(f"Reading from {args.text_prompts}...")
        dataloader = (
            wds.WebDataset(args.text_prompts).decode().to_tuple("text.txt", "emb.npy")
        )

        def dataset_fn():
            for text, emb in dataloader:
                # Avoid super long names
                words = text.split()[:30]
                yield (
                    ["text_demo/sampled_" + "_".join(words)],
                    [None],
                    torch.from_numpy(emb).unsqueeze(0),
                )

        dataset = dataset_fn()

    generate(
        args.log_dir,
        args.log_name,
        args.config_name,
        dataset,
        encoder,
        coarse_model,
        fine_model,
        args.coarse_step_sizes,
        args.fine_step_sizes,
        args.coarse_temps,
        args.fine_temps,
        args.num_samples,
        args.step,
        args.coarse_guideds,
        args.sample_lengths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Config name")
    parser.add_argument(
        "--coarse_step_sizes",
        nargs="+",
        type=int,
        default=[400],
        help="Step size in samples of coarse decoder, set to 800 for no overlap",
    )
    parser.add_argument(
        "--fine_step_sizes",
        nargs="+",
        type=int,
        default=[80],
        help="Step size in samples of fine decoder, set to 240 for no overlap",
    )
    parser.add_argument("--coarse_temps", nargs="+", type=float, default=[1.0])
    parser.add_argument("--fine_temps", nargs="+", type=float, default=[0.4])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--coarse_guideds",
        nargs="+",
        type=int,
        default=[0],
        help="Number of ground-truth coarse samples to use as seed",
    )
    parser.add_argument(
        "--sample_lengths",
        nargs="+",
        type=int,
        default=[800],
        help="Length of audio (in samples) to generate",
    )
    parser.add_argument(
        "--text_prompts",
        default=None,
        help=(
            "Path to tar file containing text embeddings. If specified, "
            "use this instead of the built-in dataloaders."
        ),
    )
    parser.add_argument(
        "--log_dir", default=os.path.join(root_dir, "logs"), help="TensorBoard log dir"
    )
    parser.add_argument("--log_name", default="demo2", help="TensorBoard log name")
    parser.add_argument("--coarse_yaml", default=None)
    parser.add_argument("--coarse_ckpt", default=None)
    parser.add_argument("--fine_yaml", default=None)
    parser.add_argument("--fine_ckpt", default=None)
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_model


DEFAULT_CKPT = (
    "/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia/"
    "F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1-"
    "emilia-8gpus-19200sample_per_gpu-bf16/ckpts/model_1000000.pt"
)
DEFAULT_CONFIG = "src/f5_tts/configs/WavTTS_scale_8_16k.yaml"
DEFAULT_VOCAB = "data/Emilia_ZH_EN_pinyin/vocab.txt"


def main():
    parser = argparse.ArgumentParser(description="Smoke-load a trained WavTTS waveform checkpoint.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action="store_true", help="Also run a tiny sample pass.")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    config = Path(args.config)
    vocab = Path(args.vocab)
    for path in (ckpt, config, vocab):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(config)
    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    cfm_kwargs = getattr(cfg.model, "cfm", {}) or {}

    model = load_model(
        model_cls,
        cfg.model.arch,
        str(ckpt),
        mel_spec_type=cfg.model.mel_spec.mel_spec_type,
        vocab_file=str(vocab),
        device=args.device,
        cfm_kwargs=cfm_kwargs,
        mel_spec_kwargs=cfg.model.mel_spec,
    )

    assert model.wav_input_only is True
    assert model.wav_frame_len == int(cfg.model.mel_spec.wav_frame_len)
    assert model.num_channels == int(cfg.model.mel_spec.wav_frame_len)
    assert model.mel_spec is None

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {ckpt}")
    print(f"Device: {args.device}")
    print(f"Parameters: {total:,}")
    print(f"wav_input_only={model.wav_input_only}, wav_frame_len={model.wav_frame_len}")

    if args.sample:
        # Keep this intentionally short: the full model is large, so this is a functional smoke only.
        sr = int(cfg.model.mel_spec.target_sample_rate)
        ref_len = model.wav_frame_len * 4
        duration = model.wav_frame_len * 6
        ref = torch.zeros(1, ref_len, device=args.device)
        with torch.inference_mode():
            wav, _ = model.sample(
                cond=ref,
                text=["test"],
                duration=duration,
                lens=torch.tensor([ref_len], device=args.device),
                steps=args.steps,
                cfg_strength=0.0,
                sway_sampling_coef=None,
                use_epss=False,
            )
        assert wav.shape == (1, duration)
        assert torch.isfinite(wav).all()
        print(f"Sample OK: shape={tuple(wav.shape)}, sample_rate={sr}")


if __name__ == "__main__":
    main()

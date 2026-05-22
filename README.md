<div align="center">
  <h1>
  WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling
  </h1> 

  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10-brightgreen.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/arXiv-2605.xxxxx-blueviolet.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="#"><img src="https://img.shields.io/badge/🌐%20Demo-Page-orange.svg" alt="Demo"></a>
    <a href="#model-checkpoints"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg" alt="HuggingFace"></a>
  </p>

  <p align="center">
    <i>An exploration of end-to-end zero-shot TTS directly in the raw waveform space.</i>
  </p>

</div>

## Introduction

WavTTS is an end-to-end zero-shot TTS framework that generates speech directly in the raw waveform space. By bypassing traditional intermediate representations—such as mel-spectrograms, VAE latents, or codec tokens—WavTTS significantly simplifies the synthesis pipeline. Powered by flow matching with DiT, it combines waveform patchification, multi-scale mel-spectrogram supervision, and optimized noise scheduling to achieve high-fidelity waveform generation. 

<div align="center">
  <img src="docs/static/images/wavtts_pipeline.png" alt="WavTTS pipeline" width="85%">
  <br>
  <!-- <sub>Figure 1: Overview of the WavTTS raw-waveform generation pipeline.</sub> -->
</div>

For more details, please refer to our paper [WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling]().

## Installation

We recommend using Conda to manage your environment.

```bash
# 1. Clone the repository
git clone https://github.com/cwx-worst-one/WavTTS
cd WavTTS

# 2. Create and activate a virtual environment
conda create -n wavtts python=3.10
conda activate wavtts

# 3. Install PyTorch >= 2.2.0 with CUDA support, e.g.,
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 4. Install WavTTS in editable mode
pip install -e .
```

## Model Checkpoints

We provide the official WavTTS checkpoint on Hugging Face: TODO. It uses `src/wavtts/configs/WavTTS_scale_9_16k.yaml` and supports 16 kHz zero-shot TTS inference with `wavtts_infer-cli`.

If you use a locally downloaded checkpoint, pass its path explicitly through the inference configuration or CLI options.

## Inference

The primary supported inference interface is the command-line tool:

```bash
wavtts_infer-cli --model WavTTS_scale_8_16k \
  --ref_audio "provide_prompt_wav_path_here.wav" \
  --ref_text "The content, subtitle, or transcription of the reference audio." \
  --gen_text "The text you want WavTTS to synthesize."
```

You can also run the default example or provide a custom TOML file:

```bash
# Run with default settings.
wavtts_infer-cli

# Run with the included basic example.
wavtts_infer-cli -c src/wavtts/infer/examples/basic/basic.toml

# Run with your own configuration.
wavtts_infer-cli -c custom.toml
```

For available flags, run:

```bash
wavtts_infer-cli --help
```

Inference notes:

- Use a clean reference audio clip. Short prompt audio with a small amount of trailing silence usually works best.
- Provide `--ref_text` when possible. Leaving it empty may require an ASR model and extra GPU memory, depending on the inference path.
- Use punctuation and spaces in `--gen_text` to make intended pauses explicit.
- If generated audio is blank or silent, first check FFmpeg and checkpoint paths.

## Training

Training is the main supported workflow in this repository.

### 1. Prepare data

Dataset preparation scripts are provided under `src/wavtts/train/datasets/`. Download the corresponding dataset first and update paths as needed.

```bash
# Prepare Emilia.
python src/wavtts/train/datasets/prepare_emilia.py

# Prepare WenetSpeech4TTS.
python src/wavtts/train/datasets/prepare_wenetspeech4tts.py

# Prepare LibriTTS.
python src/wavtts/train/datasets/prepare_libritts.py

# Prepare LJSpeech.
python src/wavtts/train/datasets/prepare_ljspeech.py
```

For custom data described by a metadata CSV, use:

```bash
python src/wavtts/train/datasets/prepare_csv_wavs.py
```

TODO: document the expected metadata format and dataset directory layout for WavTTS release training.

### 2. Choose a config

The current primary training config is:

```text
src/wavtts/configs/WavTTS_scale_9_16k.yaml
```

Other retained configs:

```text
src/wavtts/configs/WavTTS_scale_8_16k.yaml
src/wavtts/configs/WavTTS_scale_10_16k.yaml
src/wavtts/configs/WavTTS_scale_8_16k_libritts.yaml
```

### 3. Start training

Use the main launcher for the current WavTTS baseline:

```bash
bash src/wavtts/train/run_main_train.sh
```

The retained LibriTTS launcher is:

```bash
bash src/wavtts/train/run_train_libritts.sh
```

The launcher scripts define their default values near the top of each file. Edit those values directly, or launch `train.py` with Hydra overrides for one-off changes. Example:

```bash
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  src/wavtts/train/train.py \
  --config-name WavTTS_scale_9_16k.yaml \
  ++datasets.batch_size_per_gpu=19200 \
  ++hydra.run.dir=./exp/nar_wav_tts
```

You can also launch training directly with Accelerate:

```bash
accelerate config
accelerate launch src/wavtts/train/train.py --config-name WavTTS_scale_9_16k.yaml
```

More training details are available in `src/wavtts/train/README.md`.

## Evaluation

Install evaluation dependencies first:

```bash
pip install -e ".[eval]"
```

To generate samples for evaluation:

```bash
accelerate config
bash src/wavtts/eval/eval_infer_batch.sh --infer-only
```

To run batch inference together with the corresponding evaluation pipeline:

```bash
bash src/wavtts/eval/eval_infer_batch.sh
```

Objective evaluation examples:

```bash
# WER on Seed-TTS test set.
python src/wavtts/eval/eval_seedtts_testset.py --eval_task wer --lang zh --gen_wav_dir <GEN_WAV_DIR> --gpu_nums 8

# Speaker similarity on LibriSpeech-PC test-clean.
python src/wavtts/eval/eval_librispeech_test_clean.py --eval_task sim --gen_wav_dir <GEN_WAV_DIR> --librispeech_test_clean_path <TEST_CLEAN_PATH>

# UTMOS.
python src/wavtts/eval/eval_utmos.py --audio_dir <WAV_DIR> --ext wav
```

See `src/wavtts/eval/README.md` for dataset preparation and evaluation checkpoint requirements.


## Acknowledgements

WavTTS is developed from the F5-TTS codebase. We thank the authors and contributors of the following projects and resources:

- [F5-TTS](https://github.com/SWivid/F5-TTS) for the original flow-matching TTS framework.
- [E2-TTS](https://arxiv.org/abs/2406.18009) for the simple and effective TTS formulation.
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), and [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) for valuable datasets.
- [lucidrains](https://github.com/lucidrains) and [bfs18](https://github.com/bfs18) for the initial CFM structure and discussions.
- [SD3](https://arxiv.org/abs/2403.03206) and [Hugging Face diffusers](https://github.com/huggingface/diffusers) for DiT and MMDiT code structure references.
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq), [Vocos](https://huggingface.co/charactr/vocos-mel-24khz), and [BigVGAN](https://github.com/NVIDIA/BigVGAN) for related audio generation tooling.
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), and [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools.
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech editing evaluation support.

## Citation

TODO: add the WavTTS paper citation.

```bibtex
@article{todo2026wavtts,
  title={WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

## License

The code is released under the MIT License. Pre-trained model licensing should follow the licenses of the training data and released checkpoints.

TODO: confirm the final license statement for public WavTTS checkpoints.

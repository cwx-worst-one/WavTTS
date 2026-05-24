<div align="center">
  <h1>
  WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling
  </h1> 

  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10-brightgreen.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/Arxiv-2605.xxxxx-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    <a href="#"><img src="https://img.shields.io/badge/🌐%20Demo-Page-orange.svg" alt="Demo"></a>
    <a href="#model-checkpoints"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow.svg" alt="HuggingFace"></a>
  </p>

  <p align="center">
    <i>End-to-end zero-shot TTS directly in the raw waveform space.</i>
  </p>

</div>

## Introduction

WavTTS is an end-to-end zero-shot TTS framework that generates speech directly in the raw waveform space, without relying on intermediate acoustic representations such as mel-spectrograms, VAE latents, or codec tokens. Built on flow matching with DiT, WavTTS combines waveform patchification, multi-scale mel-spectrogram supervision, and optimized noise scheduling to achieve high-fidelity waveform generation.

<div align="center">
  <img src="docs/static/images/wavtts_pipeline.png" alt="WavTTS pipeline" width="85%">
</div>

You can find details in the paper [WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling]().

**Note:** This repository is based on [F5-TTS](https://github.com/SWivid/F5-TTS). For general usage guidance, please refer to the original repository. The following sections summarize the main WavTTS usage workflows.

## Installation

We recommend using Conda to manage the environment.

```bash
# 1. Clone the repository
git clone https://github.com/cwx-worst-one/WavTTS
cd WavTTS

# 2. Create and activate a virtual environment
conda create -n wavtts python=3.10
conda activate wavtts

# 3. Install PyTorch >= 2.2.0 with CUDA support, e.g.,
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install WavTTS in editable mode
pip install -e .
```

## Model Checkpoints

We provide the official WavTTS checkpoint on Hugging Face: [WavTTS 🤗](). The released checkpoint uses `src/wavtts/configs/WavTTS_scale_9_16k.yaml` and supports 16 kHz zero-shot TTS inference.


## Inference

WavTTS supports both command-line inference and script-based inference.

### CLI Inference

To synthesize speech from a reference audio prompt, run:

```bash
wavtts_infer-cli --model WavTTS_scale_9_16k \
  --ckpt_file "/path/to/model.pt" \
  --ref_audio "provide_prompt_wav_path_here.wav" \
  --ref_text "The content, subtitle, or transcription of the reference audio." \
  --gen_text "The text you want WavTTS to synthesize."
```

You can also run inference with a TOML configuration file:

```bash
# Use the provided example config
wavtts_infer-cli -c src/wavtts/infer/examples/basic.toml

# Use your own custom config
wavtts_infer-cli -c custom.toml
```

### Script-based Inference

To run single-sample inference with the script, modify the paths and text in `src/wavtts/infer/infer.sh`, then execute:

```bash
bash src/wavtts/infer/infer.sh
```

## Training

The training workflow mainly includes preparing training-data metadata and running the main training process.

### 1. Prepare data

Dataset preparation scripts are provided under `src/wavtts/train/datasets/`. Download the corresponding dataset first and update paths as needed.

```bash
# Prepare Emilia.
python src/wavtts/train/datasets/prepare_emilia.py
```

More dataset preparation details, including other datasets and custom data, are available in `src/wavtts/train/datasets/README.md`.

### 2. Start training

Use the main launcher for `WavTTS_scale_9_16k`:

```bash
bash src/wavtts/train/run_main_train.sh
```

The retained LibriTTS launcher is:

```bash
bash src/wavtts/train/run_train_libritts.sh
```

The launcher scripts define their default values near the top of each file. Edit those values directly, or launch `train.py` with Hydra overrides for one-off changes:

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

To run batch inference together with the evaluation pipeline:

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

WavTTS is built upon the awesome [F5-TTS](https://github.com/SWivid/F5-TTS) codebase. with references to the implementations of [DAC](https://github.com/descriptinc/descript-audio-codec) and [JiT](https://github.com/LTH14/JiT). We sincerely thank the authors for their valuable open-source contributions.

If you encounter any issues, we recommend first checking the [F5-TTS issue tracker](https://github.com/SWivid/F5-TTS/issues), where many common questions may have already been discussed or resolved.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{chen2026wavtts,
  title={WavTTS: Towards High-Fidelity Zero-Shot TTS via Direct Raw Waveform Modeling},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

## License

The codebase of this repository is released under the MIT License. Due to the license restrictions of the Emilia training data, our pre-trained models are released under the CC BY-NC 4.0 license.
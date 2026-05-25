# Evaluation

## 1. Installation

First, install the optional evaluation dependencies from the WavTTS root directory:

```bash
pip install -e .[eval]
```

## 2. Prepare Evaluation Models

Objective evaluation requires ASR models and a speaker similarity model.

1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. Speaker Similarity Model: WavLM (download `wavlm_large_finetune.pth` from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view))

> **⚠️ Important Checkpoint Setup:**
> - **ASR Models:** By default, the ASR models will be downloaded automatically from Hugging Face. If you are running in an offline environment with the `--local` flag, download them manually and update `asr_ckpt_dir` in `eval_librispeech_test_clean.py` and `eval_seedtts_testset.py`.
> - **WavLM Model:** This model **MUST** be downloaded manually. Pass the downloaded `wavlm_large_finetune.pth` path with `--wavlm_ckpt_dir` for SIM metric scripts.

## 3. Prepare Test Datasets

We recommend using the following standard datasets for evaluation:

1. Seed-TTS Testset (ZH/EN): Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. LibriSpeech test-clean (EN): Download from [OpenSLR](http://www.openslr.org/12/).

Unzip the downloaded datasets and place them into your local `data/` directory.

## 4. Running Evaluations

You can either run the full generation and evaluation pipeline automatically, or compute metrics step by step on existing audio files.

### A. All-in-One Batch Inference & Evaluation

To run batch inference and evaluation, execute the following commands:

```bash
# Set up Accelerate if you have not configured it yet.
accelerate config

# Generate audio only, skipping metric calculation.
bash src/wavtts/eval/eval_infer_batch.sh --infer-only

# Generate audio and calculate objective metrics automatically.
bash src/wavtts/eval/eval_infer_batch.sh --full-eval --wavlm-ckpt-dir "<WAVLM_CKPT_PATH>"
```

### B. Calculate Metrics Manually

If you have already generated `.wav` files via batch inference, you can evaluate them independently by pointing the scripts to your `<GEN_WAV_DIR>`.

```bash
# Evaluation [WER] for Seed-TTS test [ZH] set
python src/wavtts/eval/eval_seedtts_testset.py --eval_task wer --lang zh --gen_wav_dir "<GEN_WAV_DIR>" --gpu_nums 8

# Evaluation [SIM] for LibriSpeech-PC test-clean (cross-sentence)
python src/wavtts/eval/eval_librispeech_test_clean.py --eval_task sim --gen_wav_dir "<GEN_WAV_DIR>" --librispeech_test_clean_path "<TEST_CLEAN_PATH>" --wavlm_ckpt_dir "<WAVLM_CKPT_PATH>"

# Evaluation [UTMOS] for any directory containing audio files
python src/wavtts/eval/eval_utmos.py --audio_dir "<GEN_WAV_DIR>" --ext wav
```

> **💡 Tip:**
> Once evaluation is completed, detailed results will be saved as `_*_results.jsonl` files directly within your `<GEN_WAV_DIR>` directory.

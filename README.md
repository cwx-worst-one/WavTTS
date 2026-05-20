# WavTTS

WavTTS is a training-first fork derived from F5-TTS, focused on the waveform-based training path used in this repository.

At the current cleanup stage, the repository branding is `WavTTS`, while some internal package paths, config names, and CLI commands still retain legacy `f5_tts` / `F5-TTS` naming for compatibility.

Current maintenance priority:
- training
- CLI inference
- data preparation and core model code

Not prioritized for now:
- Gradio App
- Docker / Triton runtime
- Development workflow documentation
- evaluation and finetune CLI polish

## Installation


```bash
# We recommend using conda to create a new environment.
conda create -n wavtts python=3.10
conda activate wavtts

# Clone the repository
git clone <your-wavtts-repo-url>
cd WavTTS_final_release

# Install PyTorch first. The previously used environment used torch 2.9.1 / torchaudio 2.9.1.
pip install torch==2.9.1 torchaudio==2.9.1 ema-pytorch==0.7.9 torchcodec==0.9.1 torchdiffeq==0.2.5

# Install editable version of WavTTS
pip install -e .
```

### Docker image

Docker / Triton runtime support is currently not a maintenance priority in this repository snapshot. The training and CLI inference paths are the primary supported workflows.

### Current main training config

The current primary training config for this repository is:

```text
src/f5_tts/configs/F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml
```

## Inference

This repository currently prioritizes CLI inference.

During the current transition period, the exposed CLI command remains `f5-tts_infer-cli` for compatibility.

```bash
# Run with flags
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio "provide_prompt_wav_path_here.wav" \
  --ref_text "The content, subtitle or transcription of reference audio." \
  --gen_text "Some text you want TTS model generate for you."

# Run with default setting
f5-tts_infer-cli

# Or with your own .toml file
f5-tts_infer-cli -c custom.toml
```

For more details, see `src/f5_tts/infer/`.

## Training

Training is the main supported workflow.

### Main entry

```bash
bash src/f5_tts/train/run_main_train.sh
```

### Related training scripts

- `src/f5_tts/train/run_main_train.sh`
- `src/f5_tts/train/train.py`
- `src/f5_tts/train/runs_emilia/run_large_scale_9_aux_mel_w_0_05_dropout_0_joint_drop_0_1.sh`

### Example overrides

```bash
NUM_PROCESSES=8 \
BATCH_SIZE_PER_GPU=19200 \
OUTPUT_ROOT=./exp/nar_wav_tts \
MASTER_PORT=29500 \
bash src/f5_tts/train/run_main_train.sh
```

### Notes

- Gradio-based finetuning is not a current maintenance priority.
- Evaluation scripts remain in the repository, but they are not part of the current cleanup priority.
- Docker and Triton runtime support are intentionally not documented as the primary path for now.
- This repository remains a derived work of F5-TTS, so legacy names may still appear in configs, scripts, and compatibility-facing interfaces during the cleanup process.

## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)
- [Yuekai Zhang](https://github.com/yuekaizhang) Triton and TensorRT-LLM support ~

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
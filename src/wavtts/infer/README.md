# Inference

WavTTS provides command-line inference for zero-shot TTS from a reference audio prompt. Use a reference audio shorter than about 12 seconds and keep a short silence at the end to avoid truncating the prompt mid-word.

## CLI Inference

Run the provided example config:

```bash
wavtts_infer-cli -c src/wavtts/infer/examples/basic.toml
```

Or pass the main options directly:

```bash
wavtts_infer-cli \
  --model WavTTS_scale_9_16k \
  --ckpt_file /path/to/model.pt \
  --vocab_file infer/examples/vocab.txt \
  --ref_audio infer/examples/basic_ref_en.wav \
  --ref_text "Some call me nature, others call me mother nature." \
  --gen_text "The text you want WavTTS to synthesize."
```

Use `--model_cfg` instead of `--model` when you want to provide an explicit YAML model config path. Use `--ckpt_file` with a local checkpoint path or a `cached_path` URI.

## TOML Config

A `.toml` file stores the same runtime options accepted by `wavtts_infer-cli`, such as checkpoint path, vocabulary path, reference prompt, generated text, output path, and sampling settings. The example config is `src/wavtts/infer/examples/basic.toml`; `custom.toml` in examples means a user-created config file.

```toml
model = "WavTTS_scale_9_16k"
ckpt_file = "/path/to/model.pt"
vocab_file = "infer/examples/vocab.txt"
ref_audio = "infer/examples/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "The text you want WavTTS to synthesize."
output_dir = "output"
output_file = "infer_cli_basic.wav"
remove_silence = false

nfe_step = 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
timestep_mapping = "power"
timestep_power = 2.0
shift = 3.0
speed = 1.0
```

## Script-based Inference

For single-sample inference, edit paths and text in `src/wavtts/infer/infer.sh`, then run:

```bash
bash src/wavtts/infer/infer.sh
```

For batch inference, edit checkpoint, task, and dataset paths in `src/wavtts/eval/eval_infer_batch.sh`, then run:

```bash
bash src/wavtts/eval/eval_infer_batch.sh --infer-only
```

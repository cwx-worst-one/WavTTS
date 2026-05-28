# Inference

WavTTS supports zero-shot TTS from a reference audio prompt via either command-line or script-based inference.

> **💡 Tip for Reference Audio:**
> For optimal results, use a short reference clip, preferably under 12 seconds. We also recommend leaving a brief silence at the end of the clip to avoid truncating the prompt mid-word.

The pretrained checkpoint and matching vocabulary are available on [🤗 Hugging Face](https://huggingface.co/worstchan/WavTTS) and the checkpoint will be downloaded automatically when using the default configuration. To use a local checkpoint, specify its path with the `ckpt_file` parameter.


## CLI Inference

> **Note:** `wavtts_infer-cli` is an alias for `python src/wavtts/infer/infer_cli.py`. You can use either command interchangeably.

CLI inference can be run either with direct arguments or with a TOML configuration file. Command-line arguments always override values defined in the TOML config.

### A. Direct Arguments

Run inference by passing the required arguments directly:

```bash
wavtts_infer-cli \
  --model WavTTS \
  --ref_audio "infer/examples/basic_ref_en.wav" \
  --ref_text "Some call me nature, others call me mother nature." \
  --gen_text "The text you want WavTTS to synthesize."
```
*(Optional: Use `--model_cfg` instead of `--model` to provide an explicit YAML model config path.)*

### B. TOML Configuration

For reproducible runs, we recommend storing inference settings in a `.toml` file. If no config is provided, `wavtts_infer-cli` uses the default example config at `src/wavtts/infer/examples/basic.toml`.

```bash
# Use the default example config
wavtts_infer-cli -c src/wavtts/infer/examples/basic.toml

# Use a custom config with optional argument overrides
wavtts_infer-cli -c custom.toml --gen_text "Override text here."
```

**Example `custom.toml`:**
```toml
# Model settings
model = "WavTTS"
ckpt_file = "" # Leave empty to use the Hugging Face default
vocab_file = "infer/examples/vocab.txt"

# Prompt and generation
ref_audio = "infer/examples/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "The text you want WavTTS to synthesize."

# Output settings
output_dir = "output"
output_file = "infer_cli_basic.wav"
remove_silence = false

# Generation hyperparameters
nfe_step = 50
cfg_strength = 3.0
timestep_mapping = "power"
timestep_power = 2.0
shift = 3.0
speed = 1.0
```

## Script-based Inference

For customized pipelines or evaluation, you can modify and run the provided bash scripts.

### Single-sample Inference
Edit the paths and text in `src/wavtts/infer/infer.sh`, then execute:

```bash
bash src/wavtts/infer/infer.sh
```

### Batch Inference
For batch inference, configure the task and dataset paths in `src/wavtts/eval/eval_infer_batch.sh`, then run:

```bash
bash src/wavtts/eval/eval_infer_batch.sh --infer-only
```

> **💡 Tip:**
> If you notice obvious background noise in the synthesized speech, try lowering the `shift` value to mitigate it (e.g., set `shift = 1.0`).
# Research Model

## Installation
```bash
bash ./recipes/research/bootstrap.sh
```

## Vocoder
1. Train vocoder (150M parameters, 44.1kHz, 21Hz, 64dim)
```bash
bash ./recipes/research/bootstrap.sh fit --conf ./recipes/research/audio_codec/conf/default.yaml 
```

2. Compile the vocoder to TorchScript. You can obtain the `commit_hash` and `ckpt_path` from the WanDB/TensorBoard logs.
```bash
python3 recipes/research/audio_codec/scripts/compile.py --commit_hash [COMMIT_HASH] --ckpt_path [CKPT_PATH]
```

3. Add the compiled vocoder to the audio codec model zoo: `recipes/research/audio_codec.zoo.py`

## Latent diffusion model

1. Set the newly trained/compiled vocoder in `recipes/research/diff/diff_instrumental.py`

2. Make sure that the latents from the vocoder (approximately) have mean 0 and std 1.0 (check the logs: `feature/mean`, `feature/std`).

3. Pre-compute vocoder latents (Optional, but you'll get a large speedup)
```bash
./recipes/research/torchrun [NUM_GPU_DEVICES] recipes/research/dataset/featextract/featextract.py
```

4. Train latent diffusion model:
```bash
bash ./recipes/research/bootstrap.sh fit --conf ./recipes/research/diff/conf/instrumental.yaml 
```

5. Run inference using Gradio server:
```bash
python3 ./recipes/research/diff/prod/gradio_server.py
```

## PromptGPT (Optional)

1. Train PromptGPT model on a ParquetIndexDataset (~approx 1 hour for ShutterStock text descriptions):
```bash
bash ./recipes/research/bootstrap.sh fit --conf ./recipes/research/prompt_gpt/conf/default.yaml 
```

2. Run inference after adjusting `commit_hash` of trained model:
```bash
python3 recipes/research/prompt_gpt/prod/main.py
```
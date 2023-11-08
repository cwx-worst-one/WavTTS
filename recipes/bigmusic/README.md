# Unified modeling for musiclm and lyrics_to_song (instrumental + mixture)

# Firstime Setup
```bash
bash recipes/bigmusic/bootstrap.sh
```

* If you are on Merlin, add the following to your .bashrc file:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/pypetrel/pypetrel/lib/
export PYTHONPATH="${PYTHONPATH}:/opt/tiger/pypetrel/pypetrel"
```

## Training

`python3 -m samantha.main fit --config recipes/bigmusic/conf/Q4_2023/semantic/vocal/semantic_baseline_30s.yaml --extra_params.log_dir /mnt/bn/lyrics-to-song/YOUR_USERNAME/logs`

## Inference

### Inference Semantic
`python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml --extra_params.output_dir ./assets/generated_outputs`

Note: by default, inference will run with eval metrics - CER (character error rate) and MCS (mulan cosine similarity score) as well as generate video outputs.
To disable this and save only the outputs, set `--additional_callbacks []`

See `recipes/bigmusic/scripts/run_inference_semantic.sh` for latest baseline inference commands 

### Inference CSV lists (aka prompt_paths)

Master [Doc](https://bytedance.us.feishu.cn/sheets/DjQCskPkJhkpy5t1BU7ua7vUsje?sheet=bYaKbY)
CSV Paths: `/mnt/bn/audio-diffusion/data/bigmusic_text_prompts`
```
1. /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv # Default vocal music prompt
```
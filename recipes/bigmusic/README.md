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

`python3 -m samantha.main fit --config recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml --extra_params.log_dir /mnt/bn/lyrics-to-song/YOUR_USERNAME/logs`

## Inference

### Inference Semantic
`python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml --extra_params.output_dir ./generated_outputs`

Note: by default, inference will run with eval metrics - CER (character error rate) and MCS (mulan cosine similarity score) as well as generate video outputs.
To disable this and save only the outputs, set `--additional_callbacks []`

See `recipes/bigmusic/scripts/run_inference_semantic.sh` for latest inference commands

### Batch Inference from model checkpoint list

*Must* be run from merlin devbox.

1. Create a model list text file. This can be a list of model log folders or a list of exact checkpoint paths. For an example, see: `recipes/bigmusic/dev/qq/scripts/list_model_mos.txt`
2. Update `model_list_path` in `recipes/bigmusic/scripts/run_inference_from_model_list.sh`
3. Run `recipes/bigmusic/scripts/run_inference_from_model_list.sh`


### Inference GT

For checking ground truth reconstruction for different token2audio models

See `recipes/bigmusic/scripts/run_inference_gt.sh` for latest GT inference commands


### Inference CSV lists (aka prompt_paths)

Master [Doc](https://bytedance.us.feishu.cn/sheets/DeAEsx9iNhoJdHtq1GmurUyNsyg?sheet=gTfPGB)
```
1. /mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv # default inference eval set. Use this for metrics eval
2. /mnt/bn/audio-diffusion/data/mixture_prompts/sanity-check-30s-50.csv # sanity check eval set
3. /mnt/bn/audio-diffusion/data/mixture_prompts/goodcase_prompts.csv # good case easy eval set. Use this for showcase demos
```
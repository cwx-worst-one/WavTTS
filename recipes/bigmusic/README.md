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

#### Baseline models:
`
Mulan 110: /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-median_rank_1=110-kaggle-minimal.ckpt
Diffusion 30s: /mnt/bn/audio-diffusion/wtl/diffusion/model_14_30s_finetune/checkpoints/last-minimal.ckpt
Diffusion 2min: /mnt/bn/audio-diffusion/wtl/diffusion/model_14_120s_finetune/checkpoints/last-minimal.ckpt
Vocoder: /mnt/bn/audio-diffusion/ducle/recipes/diffusion/assets/soundstream-step=374999-val_sdr=12.9557-minimal.ckpt
UMM: /mnt/bn/audio-diffusion/ducle/recipes/diffusion/assets/umm_stage3_music_chroma_vq32768x32/step=0030000-minimal.ckpt
`

`
30s:
Q3: /mnt/bn/lyrics-to-song/baseline_models/q3/20231102-online-demo/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt
Q4: /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt

2-Minutes: 
GroupB: /mnt/bn/lyrics-to-song/baseline_models/q4/20231106-2min_varlen/checkpoints/step=216000-tr_loss=0.0000-val_accu_0=17.27.ckpt
GroupA Finetune: /mnt/bn/lyrics-to-song/baseline_models/q4/20231106-2min_varlen_groupA/checkpoints/step=016000-tr_loss=0.0000-val_loss_0=4.1951.ckpt
`

### Inference CSV lists (aka prompt_paths)

Master [Doc](https://bytedance.us.feishu.cn/sheets/DjQCskPkJhkpy5t1BU7ua7vUsje?sheet=bYaKbY)
CSV Paths: `/mnt/bn/audio-diffusion/data/bigmusic_text_prompts`
```
1. /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv # Default vocal music prompt
```

### MIR Offline Callback

Run MIR models locally to obtain MIR related metrics.

1. Run `bootstrap_sami_models.sh` to install
   ```sh
   bash recipes/bigmusic/scripts/bootstrap_sami_models.sh
   ```
2. Add the following line under `additional_callbacks` in your inference yaml
   ```yaml
   !new:recipes.bigmusic.callbacks.mir_offline_metrics.MIROfflineCallback
   ```
3. Set environment variable
   ```sh
   export RDMAV_FORK_SAFE=1
   ```
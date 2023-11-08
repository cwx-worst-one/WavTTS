#!/bin/bash
# OUTPUT_DIR=$1 # enable 

# MOS comparison

# Q3 Baseline
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/baseline_models/q3/20231102-online-demo/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt \
    --extra_params.output_dir assets/generated_outputs/baseline_q3 \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --run_opts.precision 32

# Q4 Baseline
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt \
    --extra_params.output_dir assets/generated_outputs/baseline_q4 \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv

# Q4 Baseline + Rerank
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt \
    --extra_params.output_dir assets/generated_outputs/baseline_q4_rerank \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4

# Convert all to mp3
find assets/generated_outputs -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \;

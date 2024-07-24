#!/bin/bash

CWD=$(pwd)

# prerequisites:
pip3 install -q ffmpeg-normalize logger

# download repo:
git clone git@code.byted.org:lab-audio/bigmusic_eval.git
cd bigmusic_eval

base_dir="/mnt/bn/janne-research-xl/mos_tests"
experiment_name="bigmusic-20240715-v8_pretrain_norerank-diffv2stereo"
method_tag_1="v8_pretrain_norerank"
method_tag_2="v2_diffusion_stereo_5c5f7aa"
vocal_prompt="/mnt/bn/audio-diffusion/peng/sstk_random_30_prompts.csv"

# python3 util_normalize_samples.py --path_method "${base_dir}/${method_tag_1}" --sr 44100 
python3 util_normalize_samples.py --path_method "${base_dir}/${method_tag_2}" --sr 44100 


python3 util_upload_data.py \
    --path_method "${base_dir}/${method_tag_1}" \
    --filename_prompt "$vocal_prompt" \
    --format ".mp3"

python3 util_upload_data.py \
    --path_method "${base_dir}/${method_tag_2}" \
    --filename_prompt "$vocal_prompt" \
    --format ".mp3"

python3 util_generate_sail_dataset_csv.py \
    --path_methods \
        "${base_dir}/${method_tag_1}" \
        "${base_dir}/${method_tag_2}" \
    --path_experiment "${experiment_name}" \
    --eval_mode "ab_test" \
    --generation_mode "instrumental"
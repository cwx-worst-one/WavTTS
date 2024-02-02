#!/bin/bash
# OUTPUT_DIR=$1 # enable 



# MOS comparison
OUTPUT_DIR=results/generated_outputs
PROMPT_PATH=/mnt/bn/audio-diffusion/weituo/vclone_test.csv # latest rewrite
# PROMPT_PATH=/mnt/bn/audio-diffusion/weituo/singsong_test_ly.csv # latest rewrite
FOLDER_NAME=$(basename $PROMPT_PATH .csv)
mkdir -p $OUTPUT_DIR

# Diffusion example
CUDA_VISIBLE_DEVICES=0 python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s_vocalapp.yaml \
    --extra_params.output_dir $OUTPUT_DIR \
    --extra_params.inference_conditions style_text,lyrics_tokens,vocal_audio \
    --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.voice_clone_duration 6 \
    --extra_params.app_type vclone \
    --extra_params.token2wav_type diffusion \
    --extra_params.semantic_ckpt /mnt/bn/audio-diffusion/semantci_label_134/semantic_model_mulan_text_07B/v00/checkpoints/step=000020-val_accu_0=0.00.ckpt
# step=008000-val_accu_0=15.18.ckpt
# step=008000-val_accu_0=20.69.ckpt
# 3AR example
#python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_31s.yaml \
#    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_1.9b_3tag_diffusion \
#    --extra_params.prompt_path $PROMPT_PATH \
#    --extra_params.token2wav_type ar

# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd

pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd

# python3 random_code/track_combine.py $OUTPUT_DIR
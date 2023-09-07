#!/bin/bash
# OUTPUT_DIR=$1 # enable 

# MOS comparison
OUTPUT_DIR=results/generated_outputs
# PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/final_genres-10s.csv 
# PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/rich-10s.csv 
# PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/karaoke-10s.csv
PROMPT_PATH=/mnt/bn/audio-diffusion/data/mixture_prompts/multi-tag-30s.csv
FOLDER_NAME=$(basename $PROMPT_PATH .csv)
mkdir -p $OUTPUT_DIR

python3 -m  pdb -c continue -m  samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_chroma_03B_30s_b20w6_140k \
    --extra_params.inference_conditions style_text,lyrics_tokens \
    --extra_params.prompt_path $PROMPT_PATH  \
    --extra_params.lyrics_max_seq_len 400 --extra_params.max_items null --run_opts.batch_size 16 --extra_params.duration 30 \
    --extra_params.semantic_cls_path recipes.bigmusic.dev.qq.lightning.semantic_modules_qq.MixSemanticModule \
    --extra_params.token2wav_type diffusion \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/qq/logs/chroma_mulan_vocal/varlen_20_30_bs20_multi_03B_6w/checkpoints/step=140000-val_accu_0=18.91.ckpt
    
# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd




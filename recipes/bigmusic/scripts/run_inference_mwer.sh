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

# 0.7b - lyrics filter
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_lyrics_filter_8gpu \
    --extra_params.inference_conditions style_text,lyrics_tokens \
    --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 32 \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_mulan/semantic_700m_30s_8gpu/checkpoints/step=039000-val_accu_0=9.61.ckpt \
    --extra_params.token2wav_type diffusion

# bestrq - text train / text inference - 1.9B model
# 3-tag
python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_1.9b_3tag_diffusion \
    --extra_params.inference_conditions style_text,lyrics_tokens \
    --extra_params.prompt_path $PROMPT_PATH \
    --extra_params.max_items null --run_opts.batch_size 32 \
    --extra_params.semantic_cls_path recipes.bigmusic.dev.v0.lightning.semantic_modules_v0.SemanticModule \
    --extra_params.lyrics_max_seq_len 150 \
    --extra_params.semantic_ckpt /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/semantic_model_mulan/semantic_chroma_text_from_scratch_l24_h16_b8_g32_2em4/checkpoints/step=137000-val_accu_0=8.35.ckpt \
    --extra_params.token2wav_type diffusion


python3 -m  pdb -c continue -m  samantha.main predict --config recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/bestrq_chroma_03B_30s_b20w6_140k \
    --extra_params.inference_conditions style_text,lyrics_tokens \
    --extra_params.prompt_path $PROMPT_PATH  \
    --extra_params.lyrics_max_seq_len 400 --extra_params.max_items null --run_opts.batch_size 16 --extra_params.duration 30 \
    --extra_params.semantic_cls_path recipes.bigmusic.dev.qq.lightning.semantic_modules_qq.MixSemanticModule \
    --extra_params.token2wav_type diffusion \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/qq/logs/chroma_mulan_vocal/varlen_20_30_bs20_multi_03B_6w/checkpoints/step=140000-val_accu_0=18.91.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/coarse_flash_llama_wav2vec/bestrq_chroma_coarse_0.7b/checkpoints/step=195000-tr_loss=3.6975.ckpt
    
# # bestrq - text train / text inference - t5 model
# python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml \
#     --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/t5_text_train_text_inference \
#     --extra_params.inference_conditions style_text,lyrics_tokens,style_tokens \
#     --extra_params.prompt_path $PROMPT_PATH \
#     --extra_params.max_items null --run_opts.batch_size 32 \
#     --extra_params.semantic_cls_path recipes.bigmusic.lightning.semantic_modules.SemanticT5Module \
#     --extra_params.semantic_ckpt /mnt/bn/jt/logs/vocalmusic/semantic_model_t5/semantic_chroma_text_from_scratch_T5_gender_l24_h16_b8_g32_2em4/checkpoints/step=026000-val_accu_0=7.53.ckpt \
#     --extra_params.token2wav_type diffusion

# ### Instrumental - TODO: (AS) update ckpt path
# python3 -m samantha.main predict --config recipes/bigmusic/conf/inference_semantic.yaml \
#     --extra_params.output_dir $OUTPUT_DIR/$FOLDER_NAME/instrumental \
#     --extra_params.inference_conditions style_text \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/google_prompts/text_prompt_collection_20230713.csv  \
#     --extra_params.semantic_ckpt /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/semantic_model_mulan/semantic_chroma_text_from_scratch_g80_2em3/checkpoints/step=122000-val_accu_0=8.82.ckpt \
#     --extra_params.coarse_ckpt /mnt/bn/audio-diffusion/jt/model_archive/vocalmusic/coarse_flash_llama_wav2vec/bestrq_chroma_coarse_0.7b/checkpoints/step=195000-tr_loss=3.6975.ckpt


# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd




#!/bin/bash
OUTPUT_DIR=$1
# OUTPUT_DIR=/mnt/bn/lyrics-to-song/ashaw/results/l2s/generated_samples
mkdir -p $OUTPUT_DIR


# Vocal Mulan rank180.

# # Pop.json - audio prompt
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/audio_prompt_pop_2 \
    --extra_params.max_items 256 \
    --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/pop.json \
    --extra_params.inference_type audio_prompt \
    --extra_params.conditions audio_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 1

# # Pop.json - text prompt
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_pop \
    --extra_params.max_items 64 \
    --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/pop.json \
    --extra_params.inference_type text_prompt_combinations \
    --extra_params.conditions text_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 2

# Pop gpt json prompt
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_pop \
    --extra_params.max_items 64 \
    --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/pop_gpt.json \
    --extra_params.inference_type text_prompt_combinations \
    --extra_params.conditions text_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 2

# # Jukebox.json
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_jukebox \
    --extra_params.max_items 64 \
    --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/jukebox.json \
    --extra_params.inference_type text_prompt \
    --extra_params.conditions text_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 2


# Default.json prompts
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_combinations \
    --extra_params.max_items 64 \
    --extra_params.inference_type text_prompt_combinations \
    --extra_params.conditions text_prompt,lyrics \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
    --extra_params.num_rounds 2

# # Default.json prompts
# mlx worker launch --cpu 8 --memory 32 --type t4 -- python3 -m samantha.main predict \
#     --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
#     --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_v180/text_prompt_combinations \
#     --extra_params.max_items 64 \
#     --extra_params.inference_type text_prompt_combinations \
#     --extra_params.conditions text_prompt,lyrics \
#     --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
#     --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180/checkpoints/last.ckpt \
#     --run_opts.batch_size 8 \
#     --run_opts.precision "16-mixed"


# Accompaniment prompts
mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_conditional/text_prompt_combinations \
    --extra_params.max_items 64 \
    --extra_params.inference_type validation_prompt \
    --extra_params.conditions text_prompt,lyrics,mulan_vocals,chroma \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180_conditional_vocals_v2/checkpoints/last.ckpt 

mlx worker launch --cpu 8 --memory 32 -- python3 -m samantha.main predict \
    --config recipes/l2v/conf/inference_mulan_phoneme.yaml \
    --extra_params.output_dir $OUTPUT_DIR/vocal_mulan_300m_conditional/audio_prompt_condition \
    --extra_params.max_items 64 \
    --extra_params.inference_type validation_prompt \
    --extra_params.conditions audio_prompt,lyrics,mulan_vocals,chroma \
    --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_0705_mmme/checkpoints/mulan-step=005000-median_rank_0=180-kaggle.ckpt \
    --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/300m_vocal_mulan_rank180_conditional_vocals_v2/checkpoints/last.ckpt 



# Convert all to mp3
pushd $OUTPUT_DIR && find . -name "*.wav" -exec ffmpeg -y -i {} {}.mp3 \; -exec rm {} \; && popd
pushd $OUTPUT_DIR/.. && zip -r $(basename $OUTPUT_DIR).zip  $(basename $OUTPUT_DIR) && popd

# # Instrumental Mulan Models
# # old v127
# mlx worker launch --cpu 16 --memory 64 -- python3 -m samantha.main predict --config recipes/l2v/conf/inference_mulan_phoneme.yaml --extra_params.output_dir ./generated_samples/vocal_mulan_v228/text_prompt_combinations --extra_params.max_items 64 --extra_params.inference_type text_prompt_combinations --extra_params.conditions text_prompt,lyrics --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_mme_0701_supcon/checkpoints/mulan-step=006600-median_rank_0=228-kaggle.ckpt --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/embed_mulan_coarse_vocal_v2/checkpoints/last.ckpt

# # Acc v127
# mlx worker launch --cpu 16 --memory 64 -- python3 -m samantha.main predict --config recipes/l2v/conf/inference_mulan_phoneme.yaml --extra_params.output_dir ./generated_samples/mulan_acc_v127/text_prompt_combinations --extra_params.max_items 64 --extra_params.inference_type text_prompt_combinations --extra_params.conditions text_prompt,lyrics --pl_module.required_modules.mulan.hpath hdfs://harunava/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/gpt_all/checkpoints/mulan-step=036000-median_rank_0=127-kaggle.ckpt --pl_module.required_modules.mulan_centers.hpath hdfs://harunava/home/byte_speech_sv/dongguo/mulan/codebook/kmeans_minibatch_codebook-mulan1b_g4_mix_127-1024x12.npy --extra_params.coarse_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/l2s/coarse/mulan_phoneme_coarse_conditional/mulan_acc_v0/checkpoints/last.ckpt --extra_params.use_continuous_embedding False
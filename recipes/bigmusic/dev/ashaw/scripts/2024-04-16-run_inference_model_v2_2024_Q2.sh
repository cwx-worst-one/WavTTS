

#### 30s ####

# pretrain
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/pretrain_30s_v0/checkpoints/step=040000-tr_loss=4.6478-val_loss_0=4.8342.save.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/30s/pretrain/pretrain_30s_v0 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_no_chorus_mcc.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# stage 1
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/sft_30s_v0/checkpoints/step=015000-tr_loss=4.6786-val_accu_0=13.33.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/30s/sft_stage1/sft_30s_v0 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_chorus.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# rl

#### 2m ####
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/pretrain_2min_v0/checkpoints/step=020000-tr_loss=4.2225-val_accu_0=14.85.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m/pretrain/pretrain_2min_v0_20k \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen_mcc_map_no_chorus.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# SFT
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/sft_2min_v1_48k/checkpoints/step=013000-tr_loss=4.2190-val_accu_0=15.17.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m/sft_stage1/sft_2min_v1_48k_13k \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# RL
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/rl_2min_v1_48k_billboard/checkpoints/step=000100.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m/rl/rl_2min_v1_48k_billboard_100s \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9



##### Audio prompting ####
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_rl_deepchorus_groupB_stage1_v8_year_filter_gt_intensity/checkpoints/step=000250.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m_audio/sft/baseline_2min_rl_deepchorus_groupB_stage1_v8_year_filter_gt_intensity_250_pop \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240229_2m_gen_lyrics_sections_v5_audio_pop.csv \
    --extra_params.inference_conditions style_audio,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9 \
    --pl_module.required_modules.reranker.hpath.rewards.mulan_sim 1.0 --pl_module.required_modules.reranker.hpath.rewards.style_text 0.0

# SFT
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/sft_2min_v1_48k/checkpoints/step=013000-tr_loss=4.2190-val_accu_0=15.17.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m_audio/sft/sft_2min_v1_48k_13k_ap_v3 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240229_2m_gen_lyrics_sections_v5_audio_pop.csv \
    --extra_params.inference_conditions style_audio,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9 \
    --pl_module.required_modules.reranker.hpath.rewards.style_audio 1.0 --pl_module.required_modules.reranker.hpath.rewards.style_text 0.0

## RL
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/rl_2min_v1_48k_billboard/checkpoints/step=000100.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m_audio/rl/rl_2min_v1_48k_billboard_100s \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240229_2m_gen_lyrics_sections_v5_audio_pop.csv \
    --extra_params.inference_conditions style_audio,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9


## RL - baseline - category
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_2024_Q2/rl_2min_v1_48k_billboard/checkpoints/step=000100.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v17_Q4_2023/2m_audio/rl/rl_2min_v1_48k_billboard_100s_category \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240229_2m_gen_lyrics_sections_v5_audio_pop.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9


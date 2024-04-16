## latest datasets
## ARTIST - /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240229_2m_gen_lyrics_sections_v3.csv
## GENRE BOCHEN - /mnt/bn/lyrics-to-song/bochen/test_prompts/vocal_prompts_20240318_en_vocal_sft.csv
## 30s GENRE - /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/20240301_suno100_chorus.csv


# /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_artist.csv 
# /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen_mcc_map.csv 
# /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv 
# /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_chorus.csv 
# /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_no_chorus_mcc.csv

## 2 Minutes
# pretrain
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_v7/checkpoints/step=064000-tr_loss=4.2294-val_loss_0=4.3048.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m/pretrain/baseline_2min_v7 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen_mcc_map.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9
# stage 1
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_sft_deepchorus_groupB_stage1_v4_2min/checkpoints/step=024000-tr_loss=4.4284-val_accu_0=15.77.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m/sft_stage1/baseline_2min_sft_deepchorus_groupB_stage1_v4_2min \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv  \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# stage 2
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_sft_deepchorus_groupB_stage2_v4_2min_s24k/checkpoints/step=001000-tr_loss=3.7889-val_accu_0=14.42.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m/sft_stage2/baseline_2min_sft_deepchorus_groupB_stage1_v4_2min \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv  \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode gumbel  --extra_params.sample_thresh 0.9

# RL
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_rl_deepchorus_groupB_stage2_v4_2min/checkpoints/step=000550.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m/rl/baseline_2min_rl_deepchorus_groupB_stage2_v4_2min_550s \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_bochen.csv  \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9


# Stage2 - artist
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_sft_deepchorus_groupB_stage2_v4_2min_s24k/checkpoints/step=001000-tr_loss=3.7889-val_accu_0=14.42.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m_artist/sft_stage2/baseline_2min_sft_deepchorus_groupB_stage1_v4_2min \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_artist.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode gumbel  --extra_params.sample_thresh 0.9


# RL - Artist
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_2m.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/baseline_2min_rl_deepchorus_groupB_stage2_v4_2min/checkpoints/step=000550.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/2m_artist/rl/baseline_2min_rl_deepchorus_groupB_stage2_v4_2min_550 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_2m_artist.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 1 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

### 30s

# pretrain
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/mcc60m_vocalB_style_mixed_cat_audio_v8_loss_mask/checkpoints/step=112000-tr_loss=4.5971-val_accu_0=13.87.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/30s/pretrain/mcc60m_vocalB_style_mixed_cat_audio_v8_loss_mask \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_no_chorus_mcc.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# stage 1
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/mcc60m_vocalB_style_mixed_cat_audio_v11_sft_stage1/checkpoints/step=015000-tr_loss=4.8572-val_accu_0=22.63.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/30s/sft_stage_1/mcc60m_vocalB_style_mixed_cat_audio_v11_sft_stage1_15k \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_chorus.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# stage 2
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/mcc60m_vocalB_style_mixed_cat_audio_v11_sft_stage2/checkpoints/step=000750-tr_loss=3.5835-val_accu_0=21.02.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/30s/sft_stage_2/mcc60m_vocalB_style_mixed_cat_audio_v11_sft_stage2 \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_chorus.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9


# rl
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/mcc60m_vocalB_style_mixed_cat_audio_v11_rl_v2_3ds/checkpoints/step=002000.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/30s/rl/mcc60m_vocalB_style_mixed_cat_audio_v11_rl_v2_3ds \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_chorus.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9

# rl - artist
python3 -m samantha.main predict --config recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_202402_baseline/mcc60m_vocalB_style_mixed_cat_audio_v11_rl_v2_3ds/checkpoints/step=002000.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/dataset_study_v16_demo/30s_artist/rl/mcc60m_vocalB_style_mixed_cat_audio_v11_rl_v2_3ds \
    --extra_params.prompt_path /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/dev/ashaw/scripts/inference_scripts/vocal_prompts_20240325_demo_30s_suno100_artist.csv \
    --extra_params.inference_conditions style_category,lyrics_tokens  --extra_params.max_items 60 \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 12 --extra_params.sample_mode top_p  --extra_params.sample_thresh 0.9


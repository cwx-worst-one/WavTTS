# MCC VocalB dataset study

# Old baseline
recipes/bigmusic/bootstrap.sh samantha.main fit -c recipes/bigmusic/conf/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml \
    --run_opts.log_name semantic_model_mulan_text_07B_groupB --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version baseline

# 300k
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_300k_vocal_mixed_text_audio_v2 --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 100000 --extra_params.dataset_types mcc60m_300k_vocal_mixed_text_audio

# 500k
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_500k_vocal_mixed_text_audio --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_500k_vocal_mixed_text_audio

# 1M
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_1M_vocal_mixed_text_audio --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_1M_vocal_mixed_text_audio

# 2M
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_2M_vocal_mixed_text_audio_v3 --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_2M_vocal_mixed_text_audio

# New baseline - WARNING: uses CTIGA not flash_llama
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_ctiga.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_vocalB_style_mixed_text_audio__ctiga --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_vocalB_style_mixed_text_audio

# Vocal + Instrumental + Speech. WARNING - must be run on bigmusic/300k_study branch. Needs separate audio_type embedder
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2 --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 150000 --extra_params.dataset_types mcc60m_300k_vocal_mixed_text_audio,mcc40m_instrumental_style_text,speech

recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_baseline.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_500k_vocal_mixed_text_audio --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_500k_vocal_mixed_text_audio


# 2 Minute model
recipes/bigmusic/bootstrap.sh fit -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/semantic_model_token_bestrq_30s_ctiga_varlen_2m.yaml \
    --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal --run_opts.version mcc60m_vocal_style_text__ctiga_varlen_2m_v3 --run_opts.log_name semantic_model_dataset_v4_baseline \
    --run_opts.max_steps 250000 --extra_params.dataset_types mcc60m_vocal_style_mixed_text_audio \
    --extra_params.pretrained_path /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_v2/checkpoints/step=124000-tr_loss=0.0000-val_accu_0=16.20.ckpt \
    --run_opts.num_workers 6  --extra_params.duration "[40, 60, 80, 100, 120]" --run_opts.batch_size 4
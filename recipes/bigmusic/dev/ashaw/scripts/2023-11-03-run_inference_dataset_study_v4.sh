## Commenting out launch "launch --noupdate --type a100-80g --cpu 4 --memory 32 -- "


## latest prompts

# Dataset size

# switch to higher checkpoint - switch to mcc60m_300k_vocal_mixed_text_audio_v2_continue when ready

# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio_v2_continue/checkpoints/step=064000-tr_loss=3.9307-val_accu_0=23.73.ckpt \
#     --extra_params.output_dir assets/dataset_study_v4/mcc60m_300k_vocal_mixed_text_audio_v2_64k \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation True


# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio_v2/checkpoints/step=064000-tr_loss=3.9307-val_accu_0=23.73.ckpt \
#     --extra_params.output_dir assets/dataset_study_v4/mcc60m_300k_vocal_mixed_text_audio_v2_64k \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_500k_vocal_mixed_text_audio/checkpoints/step=136000-tr_loss=3.6913-val_loss_0=4.1324.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/mcc60m_500k_vocal_mixed_text_audio_136k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_1M_vocal_mixed_text_audio/checkpoints/step=196000-tr_loss=4.0754-val_accu_0=24.33.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/mcc60m_1M_vocal_mixed_text_audio_184k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio_v3/checkpoints/step=144000-tr_loss=3.9391-val_loss_0=4.0735.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/mcc60m_2M_vocal_mixed_text_audio_v3_132k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True


# # vocal + intrumental + speech
# # switch to higher checkpoint when ready
# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2/checkpoints/step=096000-tr_loss=4.0025-val_accu_0=23.71.ckpt \
#     --extra_params.output_dir assets/dataset_study_v4/mcc60m_300k_vocal_mixed_text_audio_v2_64k \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation True


# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio__speech_instrumental__with_flag/checkpoints/step=236000-tr_loss=3.9266-val_accu_0=24.28.ckpt \
#     --extra_params.output_dir assets/dataset_study_v4/mcc60m_300k_vocal_mixed_text_audio_v2_64k \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation True

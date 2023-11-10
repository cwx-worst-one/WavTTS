## Commenting out launch "launch --noupdate --type a100-80g --cpu 4 --memory 32 -- "


## latest prompts

# Dataset size
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio_v2_continue/checkpoints/step=076000-tr_loss=4.0772-val_accu_0=23.73.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_300k_vocal_mixed_text_audio_v2_76k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_500k_vocal_mixed_text_audio/checkpoints/step=136000-tr_loss=3.6913-val_loss_0=4.1324.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_500k_vocal_mixed_text_audio_136k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_1M_vocal_mixed_text_audio/checkpoints/step=228000-tr_loss=3.7890-val_accu_0=24.39.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_1M_vocal_mixed_text_audio_228k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio_v3/checkpoints/step=204000-tr_loss=3.9501-val_accu_0=24.45.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_2M_vocal_mixed_text_audio_v3_204k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True


# Dataset size - Rerank
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio_v2_continue/checkpoints/step=076000-tr_loss=4.0772-val_accu_0=23.73.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_300k_vocal_mixed_text_audio_v2_76k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_500k_vocal_mixed_text_audio/checkpoints/step=136000-tr_loss=3.6913-val_loss_0=4.1324.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_500k_vocal_mixed_text_audio_136k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_1M_vocal_mixed_text_audio/checkpoints/step=228000-tr_loss=3.7890-val_accu_0=24.39.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_1M_vocal_mixed_text_audio_228k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio_v3/checkpoints/step=204000-tr_loss=3.9501-val_accu_0=24.45.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_2M_vocal_mixed_text_audio_v3_204k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4






# # vocal + intrumental + speech
# # switch to higher checkpoint when ready

git checkout bigmusic/300k_study

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2/checkpoints/step=148000-tr_loss=3.9090-val_accu_0=23.90.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2_148k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True


python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio__speech_instrumental__with_flag/checkpoints/step=236000-tr_loss=3.9266-val_accu_0=24.28.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/greedy/mcc60m_2M_vocal_mixed_text_audio__speech_instrumental__with_flag_236k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True


python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2/checkpoints/step=148000-tr_loss=3.9090-val_accu_0=23.90.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_300k_vocal_mixed_text_audio__speech_instrumental__with_flag_v2_148k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4


python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_2M_vocal_mixed_text_audio__speech_instrumental__with_flag/checkpoints/step=236000-tr_loss=3.9266-val_accu_0=24.28.ckpt \
    --extra_params.output_dir assets/dataset_study_v4/rerank/mcc60m_2M_vocal_mixed_text_audio__speech_instrumental__with_flag_236k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4

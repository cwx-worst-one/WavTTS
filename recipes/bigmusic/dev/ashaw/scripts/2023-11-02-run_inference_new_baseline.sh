## Commenting out launch "launch --noupdate --type a100-80g --cpu 4 --memory 32 -- "


## latest prompts

# new baseline - no rerank
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocalB_style_mixed_text_audio/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt \
    --extra_params.output_dir assets/generated_outputs_dataset_study_v2_new_baseline/mcc60m_vocalB_style_mixed_text_audio_196k_steps \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True --run_opts.precision 16


# new baseline - ctiga + mixed training + punctuation
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocalB_style_mixed_text_audio/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt \
    --extra_params.output_dir assets/generated_outputs_dataset_study_v2_new_baseline/mcc60m_vocalB_style_mixed_text_audio_196k_steps_rerank \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True --run_opts.precision 16 \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4


# old baseline - online demo + rerank + top_p
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt \
    --extra_params.output_dir assets/generated_outputs_dataset_study_v2_new_baseline/varlen30_tag3_bs12_07B_6w_v1 \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation False \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1 \
    --extra_params.use_reranker True --extra_params.mulan_ckpt /mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=024600-kaggle.ckpt --extra_params.beam_size 4 --run_opts.batch_size 4

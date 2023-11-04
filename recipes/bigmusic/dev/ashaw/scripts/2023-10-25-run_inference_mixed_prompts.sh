## Commenting out launch "launch --noupdate --type a100-80g --cpu 4 --memory 32 -- "

# Mulan

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_mixed_tag_audio/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_mixed_tag_audio \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True
    
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_mixed_text_audio/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_mixed_text_audio \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_mixed_text_audio_warmstart/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_mixed_text_audio_warmstart \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True

# # bf16 test
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_mixed_text_audio_warmstart/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_mixed_text_audio_warmstart_bf16 \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True --run_opts.precision bf16-mixed

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_text \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True


# # Needs ctiga fixed noise update
python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_text__ctiga_sampled_noise \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True --run_opts.precision 16 --extra_params.sample_mode gumbel

python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga/checkpoints/last.ckpt \
    --extra_params.output_dir assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_text__ctiga_fixed_noise \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --predict_dataset.enable_punctuation True --run_opts.precision 16 --extra_params.sample_mode gumbel_fixed_noise


# # # Run using varlen branch
# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_cls_path recipes.bigmusic.lightning.semantic_varlen_modules.SemanticModuleVarlen \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__phoneme_loss/checkpoints/last.ckpt \
#     --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/generated_outputs_mixed_prompts/mcc60m_vocal_style_text__phoneme_loss \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation True --run_opts.batch_size 1



# # Online demo
# python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
#     --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt \
#     --extra_params.output_dir assets/generated_outputs_mixed_prompts/varlen30_tag3_bs12_07B_6w_v1 \
#     --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
#     --predict_dataset.enable_punctuation False

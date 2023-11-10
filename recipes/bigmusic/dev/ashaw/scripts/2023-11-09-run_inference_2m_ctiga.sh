## 2m demo
# VocalB
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_v4/checkpoints/step=248000-tr_loss=0.0000-val_accu_0=17.27.ckpt \
    --extra_params.output_dir assets/dataset_study_v4_ctiga/mcc60m_vocal_style_text__ctiga_varlen_2m_v4/248k_default \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231031_mixed35_2min.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4


# VocalA
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_vocalA/checkpoints/step=016000-tr_loss=0.0000-val_loss_0=4.1951.ckpt \
    --extra_params.output_dir assets/dataset_study_v4_ctiga/vocalA/16k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231031_mixed35_2min.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4

python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_vocalA/checkpoints/step=056000-tr_loss=0.0000-val_accu_0=18.05.ckpt \
    --extra_params.output_dir assets/dataset_study_v4_ctiga/vocalA/56k \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231031_mixed35_2min.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4

# Detaild desc
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_vocalA/checkpoints/step=056000-tr_loss=0.0000-val_accu_0=18.05.ckpt \
    --extra_params.output_dir assets/dataset_study_v4_ctiga/vocalA/56k_detailed \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/mixture_prompts/20231106-detailed_desc-2min.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4

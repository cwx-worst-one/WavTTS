## Commenting out launch "launch --noupdate --type a100-80g --cpu 4 --memory 32 -- "


## latest prompts

# Dataset size - Rerank
python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_100k_vocal_mixed_text_audio__ctiga/checkpoints/step=032000-tr_loss=4.0840-val_accu_0=22.36.ckpt \
    --extra_params.output_dir assets/dataset_study_v5/rerank/mcc60m_100k_vocal_mixed_text_audio__ctiga \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4


python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_150k_vocal_mixed_text_audio__ctiga/checkpoints/step=044000-tr_loss=4.1302-val_accu_0=22.76.ckpt \
    --extra_params.output_dir assets/dataset_study_v5/rerank/mcc60m_150k_vocal_mixed_text_audio__ctiga \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4


python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_200k_vocal_mixed_text_audio__ctiga/checkpoints/step=064000-tr_loss=4.1277-val_accu_0=23.13.ckpt \
    --extra_params.output_dir assets/dataset_study_v5/rerank/mcc60m_200k_vocal_mixed_text_audio__ctiga \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4


python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio__ctiga/checkpoints/step=088000-tr_loss=3.7338-val_accu_0=23.53.ckpt \
    --extra_params.output_dir assets/dataset_study_v5/rerank/mcc60m_300k_vocal_mixed_text_audio__ctiga \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4


# pretrained model paths
# /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/speech_instrumental_pretrain/checkpoints/step=144000-tr_loss=3.6574-val_loss_0=10.0371.ckpt

# speech model paths
# /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/speech_pretrain/checkpoints/step=128000-tr_loss=3.2997-val_loss_0=10.0506.ckpt



python3 -m samantha.main predict -c recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_300k_vocal_mixed_text_audio__ctiga/checkpoints/step=088000-tr_loss=3.7338-val_accu_0=23.53.ckpt \
    --extra_params.output_dir assets/dataset_study_v5/rerank/mcc60m_300k_vocal_mixed_text_audio__ctiga_test_refactor \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231018_mixed135.csv \
    --extra_params.use_reranker True --extra_params.beam_size 4 --run_opts.batch_size 4 --extra_params.max_items 8

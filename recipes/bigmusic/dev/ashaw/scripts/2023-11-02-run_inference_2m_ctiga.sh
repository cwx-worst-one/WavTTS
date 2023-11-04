## 2m demo
python3 -m samantha.main predict -c recipes/bigmusic/conf/semantic_modeling/dataset_v2/inference_semantic_2m_ctiga_varlen.yaml \
    --extra_params.semantic_cls_path recipes.bigmusic.lightning.semantic_varlen_modules.SemanticModuleVarlen \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_dataset_v4_baseline/mcc60m_vocal_style_text__ctiga_varlen_2m_v3/checkpoints/step=200000-tr_loss=0.0000-val_loss_0=4.2434.ckpt \
    --extra_params.output_dir /mnt/bn/ashaw-us/repos/samantha/assets/mcc60m_vocal_style_text__ctiga_varlen_2m_v3/200k_steps_rerank \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/bigmusic_text_prompts/vocal_prompts_20231031_mixed35_2min.csv \
    --extra_params.sample_mode top_p --extra_params.semantic_temperature 1.1 --extra_params.use_reranker True --extra_params.beam_size 4



launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/ashaw/logs/bigmusic/semantic_vocal/semantic_model_mulan_tagging/mulan_110_style_mixed_mulan_tag_punctuation_varlen15s/checkpoints/step=200000-tr_loss=4.2448-val_accu_0=20.20.ckpt \
    --extra_params.output_dir assets/generated_outputs_new_baseline_v2/mulan_110_style_mixed_mulan_tag_punctuation_varlen15s__step=200000-tr_loss=4.2448-val_accu_0=20.20.ckpt \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv \
    --predict_dataset.enable_punctuation True --predict_dataset.use_mulan_v2_genres True

launch --noupdate --type a100-80g --cpu 4 --memory 32 -- python3 -m samantha.main predict -c recipes/bigmusic/conf/inference_semantic_30s.yaml \
    --extra_params.semantic_ckpt /mnt/bn/lyrics-to-song/qq/logs/semantic_model_mulan_text_07B/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt \
    --extra_params.output_dir assets/generated_outputs_new_baseline_v2/varlen30_tag3_bs12_07B_6w_v1__step=234000-val_accu_0=21.02.ckpt \
    --extra_params.prompt_path /mnt/bn/audio-diffusion/data/mixture_prompts/suno100.csv \
    --predict_dataset.enable_punctuation False --predict_dataset.use_mulan_v2_genres False
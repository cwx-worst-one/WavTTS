# Setup

Request access to bytenas drive:
* https://cloud-i18n.bytedance.net/bytenas/detailcreated/1958?psm=bytenas.volume.lyrics-to-song
* https://cloud-i18n.bytedance.net/bytenas/detailcreated/1397?psm=bytenas.volume.ashaw-us (optional)

# Training

## Phoneme Training
```bash
# On MLX
mlx worker launch --cpu 16 --memory 128 --type a100-80g -- python3 -m samantha.main fit --conf python3 -m samantha.main fit --config recipes/l2v/conf/semantic_modeling/mulan_phoneme_coarse_embed.yaml

# On Arnold
bash recipes/musiclm/bootstrap.sh fit --conf recipes/l2v/conf/semantic_modeling/mulan_phoneme_coarse_embed.yaml --run_opts.version mulan_arnold_v0 --log_dir /mnt/bn/lyrics-to-song/ashaw/logs/arnold
```

### Options:
* Change mulan version (Mulan v1.3 -> Vocal Mulan): --pl_module.required_modules.mulan.hpath /mnt/bn/audio-diffusion/mulan_exp/MuLan_large/mulan_mme_0701_supcon/checkpoints/mulan-step=006600-median_rank_0=228-kaggle.ckpt
* Model size (110M -> 300M): --model_config.num_hidden_layers 24 --model_config.num_attention_heads 16
* Log dir: --run_opts.log_dir /mnt/bn/lyrics-to-song/ashaw/logs/arnold

# Inference

## Mulan-Phoneme Inference (Coarse model)

TLDR: run latest inference script.
```bash
bash recipes/l2v/scripts/run_inference.sh /mnt/bn/lyrics-to-song/ashaw/results/l2s/generated_samples
```

Run config directly:
```bash
# Infer with Audio Prompt + Lyrics
mlx worker launch -- python3 -m samantha.main predict --config recipes/l2v/conf/inference_mulan_phoneme.yaml --extra_params.output_dir ./generated_output --extra_params.inference_type audio_prompt --extra_params.conditions audio_prompt,lyrics

# Infer with Text Prompt + Lyrics
mlx worker launch -- python3 -m samantha.main predict --config recipes/l2v/conf/inference_mulan_phoneme.yaml --extra_params.output_dir ./generated_output --extra_params.inference_type text_prompt --extra_params.conditions text_prompt,lyrics

# inference_type can be text_prompt or audio_prompt
# optionally specify --extra_params.prompt_path recipes/l2v/datasets/inference_prompts/default.json
```

## Ground Truth Inference
```bash
mlx worker launch --noupdate -- python3 recipes/l2v/scripts/run_groundtruth_inference.py
```

# Dataset List - Index List TSVs:

## Mixture datasets (tsv, audio_key, audio_format)
* Karaoke: 
    * /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_train.tar_to_index.tsv, full.mp3, mp3
    * /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_valid.tar_to_index.tsv, full.mp3, mp3
* MCC: /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc9m.tar_to_index.tsv, mp3, mp3
* Resso: /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso.tar_to_index.tsv, mp3, m4a
* MCC 60M vocal: /mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv, mp3, mp3

## Accompaniment - Vocal datasets (tsv, audio_key, audio_format, vocal_key)
* Karaoke
    * /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/karaoke_train.tar_to_index.tsv, full.mp3, mp3, vocal.mp3
* Resso:
    * /mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv, mss_acc, mp3, mss_vocal

## Baseline Models:
* Mulan:
    * Mulan 1.2 - Currently works better for vocals
        * Model: hdfs://harunava/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/g4/checkpoints/mulan-step=044800-median_rank_1=170-kaggle.ckpt
        * Centers: hdfs://harunava/home/byte_speech_sv/zongyu.yin/ckpts/mulan/kmeans_minibatch_codebook-mulan1b_g4_170-1024x12.npy
    * Mulan 1.3 - 
        * Model: hdfs://harunava/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/gpt_all/checkpoints/mulan-step=036000-median_rank_0=127-kaggle.ckpt
        * Centers: hdfs://harunava/home/byte_speech_sv/dongguo/mulan/codebook/kmeans_minibatch_codebook-mulan1b_g4_mix_127-1024x12.npy
    * Mulan Vocal -
        * Model: hdfs://harunava/home/byte_speech_sv/weitsung.lu/mulan_exp/MuLan_large/gpt_all/checkpoints/mulan-step=036000-median_rank_0=127-kaggle.ckpt
        * Centers: None. Use embeddings
* Soundstream:
    * hdfs://harunava/home/byte_speech_sv/zongyu.yin/ckpts/soundstream/190k
* Fine Decoder - Vocal MCC60m:
    * /mnt/bn/lyrics-to-song/ashaw/logs/l2s/fine/fine_flash_llama_mcc_vocals/vocal_finetune_60m/checkpoints/last.ckpt

* Semantic:
    * Wav2Vec:
        * hdfs://harunava/home/byte_speech_sv/zongyu.yin/ckpts/w2v/2.1/semantic.jit.pt
        * hdfs://harunava/home/byte_speech_sv/zongyu.yin/ckpts/w2v/2.1/centroids_epoch_10.npy

# Tensorboard Logs (Baseline results)
```
cd /mnt/bn/lyrics-to-song/ashaw/logs/l2s/
tensorboard --logdir . --bind_all
```
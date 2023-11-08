# Training and inference configs for Q3 2023

* Baseline Model:
```
Model: Flash Llama - 0.7B
Dataset: MCC60M - MCC style text rewrite. Lossy dataset with no punctuation. /mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt
Training Config: recipes/bigmusic/conf/Q3_2023/semantic_modeling/semantic_model_mix_varlen30_07B_us.yaml
Path: /mnt/bn/lyrics-to-song/baseline_models/q3/20231102-online-demo/varlen30_tag3_bs12_07B_6w_v1/checkpoints/step=234000-val_accu_0=21.02.ckpt
Inference: recipes/bigmusic/conf/Q3_2023/inference_semantic_30s.yaml
```

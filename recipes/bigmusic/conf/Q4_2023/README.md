# Training and inference configs for Q4 2023

* Baseline Model:
```
Model: cTIGA Llama - 0.7B
Dataset: MCC60M - 30s Lossless + Punctuation. hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/mcc/mcc60_lossless_asr
Training Config: recipes/bigmusic/conf/Q4_2023/semantic/vocal/semantic_baseline_30s.yaml
Path: /mnt/bn/lyrics-to-song/baseline_models/q4/20231102-new-baseline-mixed-ctiga-196k/checkpoints/step=196000-tr_loss=3.9746-val_accu_0=24.90.ckpt
Inference: /mnt/bn/ashaw-us/repos/samantha/recipes/bigmusic/conf/Q4_2023/inference/inference_vocal_30s.yaml
```

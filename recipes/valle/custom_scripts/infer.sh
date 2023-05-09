#!/bin/bash -ex

python3 recipes/llm_asr/scripts/infer_valle.py \
      --ar_config recipes/llm_asr/conf/valle_phoneme_infer_2.yaml \
      --ar_ckpt_path /mnt/bn/jcong5/logs/samantha/logs/valle_phoneinput_1400h/fp16-ds3ol/checkpoints/epoch=67-step=109004-valid_token_acc=0.00.ckpt/checkpoint_fp32.pth \
      --config /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/bytegen/valle/exp_libri_light-1400h_ar-A100-80GB-1-8/config.yaml \
      --nar_ckpt_path /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/bytegen/valle/exp_libri_light-1400h_nar-A100-80GB-1-8/checkpoints/latest_ckpt.pyt \
      --codec_ckpt /mnt/bn/jcong5/workspace/models/soundstream/2023-01-17_causal_x300_1024_6book_doubleG/latest_ckpt.pyt \
      --meta_file /mnt/bn/jcong5/data/kat_test/4-10s_libri/thread-00.lst \
      --device cuda \
      --out_dir test

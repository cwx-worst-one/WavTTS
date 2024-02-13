#!/bin/bash


for d in genre_1 mood scene vocal_gender; do
    echo Copying $d
    # hdfs dfs -cp \
    # hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/logs/m1_music_sft_{$d}/v0_umm_lr5.0e-5_precision32_1layer_mcc30k_ummcn/checkpoints/step=0030000.ckpt \
    # hdfs://harunasg/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1/m1_music_sft_${d}_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_ummcn_step=0030000.ckpt

    hdfs dfs -cp \
    hdfs://harunasg/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1/m1_music_sft_${d}_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_ummcn_step=0030000.ckpt \
    hdfs://haruna/home/byte_speech_sv/janne.spijkervet/js/models/m1/m1_music_sft_${d}_v0_umm_lr5.0e-5_precision32_1layer_mcc30k_ummcn_step=0030000.ckpt

done;
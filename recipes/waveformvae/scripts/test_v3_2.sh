
config_path='hdfs://haruna/home/byte_speech_sv/user/cjw/WaveformVAE/hparams.yaml'
ckpt_path='hdfs://haruna/home/byte_speech_sv/user/cjw/WaveformVAE/v3_2/epoch=00-step=258000.ckpt'
input_dir=$1
output_dir=$2

hdfs dfs -get $config_path
hdfs dfs -get $ckpt_path
mkdir -p logs/v3_2
mv hparams.yaml logs/
mv epoch=00-step=258000.ckpt logs/v3_2/
mkdir -p $output_dir

config_path='logs/hparams.yaml'
ckpt_path='logs/v3_2/epoch=00-step=258000.ckpt'

CUDA_VISIBLE_DEVICES=0 python3 recipes/waveformvae/scripts/test_v3_2.py \
    --config_path $config_path \
    --ckpt_path $ckpt_path \
    --input_dir $input_dir \
    --output_dir $output_dir

#WAV_RES_REF_TEXT_PATH=/mnt/bn/cjw-lq-1/project/jiawei/temp/samantha/logs/v3_2/wav_res_ref_text_librispeech_test
#YOUR_WAV_DIR2=${output_dir}/recon_wavs
#cd /mnt/bn/cjw-lq-1/project/zhiying/bigtts-eval
#asr_type=open_source # internal or open_source
#lang=en # en or zh
#bash run-eval.sh /mnt/bn/cjw-lq-1/project/zhiying/bigtts-eval \
#$asr_type ${WAV_RES_REF_TEXT_PATH} ${YOUR_WAV_DIR2} $lang

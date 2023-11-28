
config_path='/mnt/bn/cjw-lq-1/work_dir/samantha_waveformvae/finetuning/waveformvae/1.0/hparams.yaml'
ckpt_path='/mnt/bn/cjw-lq-1/work_dir/samantha_waveformvae/finetuning/finetune/uniKL5.0/checkpoints/last.ckpt'
input_dir='/mnt/bn/cjw-lq-1/data/high_quality_wav'
out_dir='/mnt/bn/cjw-lq-1/work_dir/samantha_waveformvae/finetuning/finetune/uniKL5.0/recon_0806/'
#WAV_RES_REF_TEXT_PATH=/mnt/bn/cjw-lq-1/project/zhiying/bigtts-eval/wav_res_ref_text_librispeech_test_finetune_uniKL

CUDA_VISIBLE_DEVICES=0 python3 recipes/waveformvae/scripts/test.py \
    --config_path $config_path \
    --ckpt_path $ckpt_path \
    --input_dir $input_dir \
    --output_dir $out_dir


#YOUR_WAV_DIR2=${out_dir}recon_wavs
#cd /mnt/bn/cjw-lq-1/project/zhiying/bigtts-eval
#asr_type=open_source # internal or open_source
#lang=en # en or zh
#bash run-eval.sh /mnt/bn/cjw-lq-1/project/zhiying/bigtts-eval \
#$asr_type ${WAV_RES_REF_TEXT_PATH} ${YOUR_WAV_DIR2} $lang

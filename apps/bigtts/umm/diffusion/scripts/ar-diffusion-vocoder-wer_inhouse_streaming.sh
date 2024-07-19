#!/bin/bash

set -x

src_lang=zh
tgt_lang=zh
seed=1996
prompt_wav_dir=
meta_lst=
out_umm_dir=
out_wav_dir=
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/consistency_distill_fixed/CD_block32_8H800_setting0_UMM062_UMMnoDrop_LR8e-6_guid1to6_tDiff0.02_X0Prob0.1_L2SSIM/checkpoints/epoch=00-step=300000-loss=0.06.ckpt
umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.master.ckpt
vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt
only_use_global_prompt=False

nfe=4
sampler=consistency
cfg=1

. scripts/parse_options.sh


mkdir -p $out_wav_dir

bash launch.sh predict \
    -c apps/bigtts/umm/diffusion/conf/infer_ar_diffusion_streaming.yaml \
    --predict_dataset.npy_path $out_umm_dir \
    --predict_dataset.meta_file $meta_lst \
    --predict_dataset.wav_dir $prompt_wav_dir \
    --run_opts.output_dir $out_wav_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
    --run_opts.seed ${seed} \
    --run_opts.num_workers 1 \
    --run_opts.infer_type ar-diffusion-vocoder \
    --pl_module.umm_type UMM \
    --pl_module.diffusion_precision bf16 \
    --pl_module.diffusion_nfe $nfe \
    --pl_module.diffusion_sampler $sampler \
    --bn_config.bn_norm_std 2 \
    --bn_config.bn_padding -5 \
    --pl_module.text_cfg_w $cfg \
    --pl_module.use_wvae_vocoder True \
    --bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
    --bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
    --pl_module.only_use_global_prompt ${only_use_global_prompt} || exit 1



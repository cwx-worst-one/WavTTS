#!/bin/bash

config=apps/bigtts/umm/ar/conf/ar_diffusion_inference.yaml

seed=
src_lang=
tgt_lang=
meta_lst=

umm_ckpt=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.master.ckpt
umm_version=0.6.2

ar_ckpt_path=
ar_precision=fp16-mixed
ar_output_path=
temperature=
thresh=
mode=
step_out_blank=
max_blank_length=
max_repeat_times=
ar_model_version=
icl_mode=
eos_weight=

diffusion_ckpt_path=
diffusion_output_path=
diffusion_precision=bf16-mixed
wvae_encoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt
wvae_decoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt

echo "Infer CMD:"
echo "$0 $@" | sed 's=--=\\\n    --=g'
echo
. scripts/parse_options.sh

# check configurations
all_args="config seed src_lang tgt_lang meta_lst ar_ckpt_path ar_output_path temperature thresh mode step_out_blank max_repeat_times ar_model_version icl_mode eos_weight diffusion_ckpt_path diffusion_output_path"
empty_args=
for arg in $all_args; do
    value=`eval echo \\$$arg`
    if [ -z $value ]; then
        empty_args="$empty_args --$arg"
    fi
done
if [ ! -z "$empty_args" ]; then
    echo "These args are empty, please specify:"
    echo "    ${empty_args:1}"
    exit 1
fi

run_opts="--run_opts.seed $seed \
    --run_opts.src_lang $src_lang \
    --run_opts.tgt_lang $tgt_lang \
    --run_opts.num_workers 1 \
    --run_opts.meta_lst $meta_lst"

umm_opts="--umm_opts.ckpt_path ${umm_ckpt} \
    --umm_opts.version ${umm_version} \
    --umm_opts.frame_rate 25"

ar_opts="--ar_opts.ckpt_path $ar_ckpt_path \
    --ar_opts.precision $ar_precision \
    --ar_opts.output_path $ar_output_path \
    --ar_opts.temperature $temperature \
    --ar_opts.thresh $thresh \
    --ar_opts.mode $mode \
    --ar_opts.step_out_blank $step_out_blank \
    --ar_opts.max_blank_length $max_blank_length \
    --ar_opts.max_repeat_times $max_repeat_times \
    --ar_opts.version $ar_model_version \
    --ar_opts.icl_mode $icl_mode \
    --ar_opts.eos_weight $eos_weight"

diffusion_opts="--diffusion_opts.output_path $diffusion_output_path \
    --diffusion_opts.ckpt_path $diffusion_ckpt_path \
    --diffusion_opts.mel_frame_rate 40 \
    --diffusion_opts.precision $diffusion_precision \
    --diffusion_opts.nfe 10 \
    --diffusion_opts.sampler ddim \
    --diffusion_opts.text_cfg_w 4 \
    --diffusion_opts.bn_config.bn_norm_std 2 \
    --diffusion_opts.bn_config.bn_padding -5 \
    --diffusion_opts.bn_config.wvae_encoder_path $wvae_encoder_path \
    --diffusion_opts.bn_config.wvae_decoder_path $wvae_decoder_path"

echo "Infer Options: $run_opts $umm_opts $ar_opts $diffusion_opts" | sed 's=--=\\\n    --=g'
echo

export NCCL_DEBUG=WARN
bash launch.sh predict -c $config $run_opts $umm_opts $ar_opts $diffusion_opts || exit 1

bash apps/bigtts/umm/diffusion/scripts/analysis.sh $meta_lst $(realpath $diffusion_output_path) $tgt_lang || (echo "error on analysis"; exit 1)

#!/bin/bash

# 配置
ar_config=apps/bigtts/umm/ar/conf/infer_bigtts_ar_shortform.yaml
diffusion_config=apps/bigtts/umm/diffusion/conf/infer_ar_diffusion_streaming.yaml

# 公共参数
seed=1997
src_lang=
tgt_lang=
meta_lst=
output_path=

umm_ckpt=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.master.ckpt
umm_version=0.6.2
vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt
wvae_encoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt
wvae_decoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt

# AR 相关参数
ar_ckpt_path=
temperature=0.5
thresh=0.9
mode=naive
eos_weight=0.8
step_out_blank=v2  # 对应decoding版本
max_blank_length=10
max_repeat_times=1

ar_model_version=merge_v2 # run_opts.version
ar_precision=fp16  # 确认对应模型的精度
use_pre_utt=True

# diffusion 相关参数
diffusion_nfe=4
diffusion_sampler=consistency
text_cfg_w=1
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/consistency_distill_fixed/CD_block32_8H800_setting0_UMM062_UMMnoDrop_LR8e-6_guid1to6_tDiff0.02_X0Prob0.1_L2SSIM/checkpoints/epoch=00-step=300000-loss=0.06.ckpt
diffusion_precision=bf16

echo "Infer CMD:"
echo "$0 $@" | sed 's=--=\\\n    --=g'
echo
. scripts/parse_options.sh

# 以下参数需要根据语种判断
if [ $src_lang = $tgt_lang ]; then
    icl_mode="continuation"
    only_use_global_prompt=False
else
    icl_mode="non-continuation"
    only_use_global_prompt=True
fi

# 检查配置
all_args="src_lang tgt_lang meta_lst output_path ar_ckpt_path diffusion_ckpt_path"
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

mkdir -p $output_path

# RUN AR Inference
echo ">>>>> Start AR Inference"
ar_output_umm_path="${output_path}/umm"
mkdir -p $ar_output_umm_path

ar_opts="\
    --pl_module.semantic_precision ${ar_precision} \
    --pl_module.eos_weight $eos_weight \
    --run_opts.seed $seed \
    --run_opts.num_workers 1 \
    --run_opts.meta_lst $meta_lst \
    --run_opts.output_path $ar_output_umm_path \
    --run_opts.semantic_model_path $ar_ckpt_path \
    --run_opts.umm_ckpt_path $umm_ckpt \
    --run_opts.umm_version ${umm_version} \
    --run_opts.src_lang $src_lang \
    --run_opts.tgt_lang $tgt_lang \
    --run_opts.temperature $temperature \
    --run_opts.thresh $thresh \
    --run_opts.mode $mode \
    --run_opts.step_out_blank $step_out_blank \
    --run_opts.max_blank_length $max_blank_length \
    --run_opts.max_repeat_times $max_repeat_times \
    --run_opts.version ${ar_model_version} \
    --run_opts.icl_mode $icl_mode \
    --run_opts.use_pre_utt ${use_pre_utt} \
    "

echo "Infer AR Options: $ar_opts" | sed 's=--=\\\n    --=g'
echo

export NCCL_DEBUG=WARN
bash launch.sh predict -c $ar_config $ar_opts || exit 1
echo ">>>>> END AR Inference"

# RUN Diffusion Inference
echo ">>>>> Start Diffusion Inference"
diffusion_output_path="${output_path}/split_wav"
mkdir -p $diffusion_output_path

cat $output_path/meta_split.lst | sort | uniq > $output_path/meta_split.lst.sort.uniq
mv $output_path/meta_split.lst.sort.uniq $output_path/meta_split.lst
split_meta_lst=$output_path/meta_split.lst

diffusion_opts="\
    --pl_module.umm_type UMM \
    --pl_module.diffusion_precision $diffusion_precision \
    --pl_module.diffusion_nfe $diffusion_nfe \
    --pl_module.diffusion_sampler $diffusion_sampler \
    --pl_module.only_use_global_prompt ${only_use_global_prompt} \
    --pl_module.text_cfg_w $text_cfg_w \
    --pl_module.use_wvae_vocoder True \
    --bn_config.bn_norm_std 2 \
    --bn_config.bn_padding -5 \
    --bn_config.wvae_encoder_path $wvae_encoder_path \
    --bn_config.wvae_decoder_path $wvae_decoder_path \
    --predict_dataset.npy_path $ar_output_umm_path \
    --predict_dataset.meta_file $split_meta_lst \
    --predict_dataset.wav_dir $prompt_wav_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
    --run_opts.seed ${seed} \
    --run_opts.num_workers 1 \
    --run_opts.infer_type ar-diffusion-vocoder \
    --run_opts.output_dir $diffusion_output_path \
    "

echo "Infer Diffusion Options: $diffusion_opts" | sed 's=--=\\\n    --=g'
echo

export NCCL_DEBUG=WARN
bash launch.sh predict -c $diffusion_config $diffusion_opts || exit 1
echo ">>>>> End Diffusion Inference"

echo ">>>>> Start Merge&Norm Wav"
output_merged_wav_path="${output_path}/merged_wav"
mkdir -p $output_merged_wav_path
python3 apps/bigtts/umm/scripts/merge_split_wavs.py $split_meta_lst $diffusion_output_path $output_merged_wav_path
output_wav_path="${output_path}/wav"
mkdir -p $output_wav_path
python3 apps/bigtts/umm/scripts/norm_volume.py ${output_merged_wav_path} $output_wav_path
echo ">>>>> End Merge&Norm Wav "
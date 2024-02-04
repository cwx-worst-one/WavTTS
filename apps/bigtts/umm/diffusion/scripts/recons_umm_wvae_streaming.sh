#!/bin/bash

set -e

function get_file_from_branch() {
    # usage: get_file_from_branch <branch_name> <file_or_folder_path>
    branch_name=$1
    target_file=$2
    git fetch origin $branch_name
    git checkout origin/$branch_name -- $target_file
}

rm -rf recipes/umm
get_file_from_branch speech-diffusion-streaming recipes/umm_062
ln -s -r recipes/umm_062 recipes/umm

exp=${exp:-PrefixLDM4_300M_8H800_setting0_40hzWVAE_MixedUMMv1_textDrop0.25_Norm2}
ckpt=${ckpt:-"epoch=00-step=425000-loss=0.52.ckpt"}
step=${step:-425000}

diffusion_ckpt_path=${diffusion_ckpt_path:-"hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt"}

lang=${lang:-en}  # zh, en
token_chunk=${token_chunk:-8000}
chunk_overlap=${chunk_overlap:-0}

out_dir=${out_dir:-/mnt/bn/jdy-lq-2/bigtts-nar/output}
meta_lst=${meta_lst:-/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst}

out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps_TextCFG4_chunk${token_chunk}_overlap${chunk_overlap}_test
mkdir -p $out_dir

umm_ckpt_path=${umm_ckpt_path:-/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt}


# python3 samantha/main.py predict \
bash launch.sh predict \
	-c apps/bigtts/umm/diffusion/conf/infer_reconstruction_40hzMel_streaming.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
	--run_opts.seed ${SEED:=1996} \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
    --run_opts.infer_type diffusion-vocoder \
	--pl_module.umm_type UMM \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe 10 \
	--pl_module.diffusion_sampler ddim \
	--pl_module.use_wvae_vocoder True \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding -5 \
	--pl_module.text_cfg_w 4 \
	--pl_module.token_chunk_size $token_chunk \
	--pl_module.token_chunk_overlap $chunk_overlap

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $lang

# use example:
# export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/xieshuangyi/voicebox/logdir/PrefixLDM4_block32_300M_8A100_setting0_40hzWVAE_UMMv062/checkpoints/epoch=00-step=205000-loss=0.53.ckpt
# export exp=PrefixLDM4_block32_300M_8A100_setting0_40hzWVAE_UMMv062
# export ckpt=epoch=00-step=205000-loss=0.53.ckpt
# export step=205000

# export out_dir=/opt/tiger/samantha/output

# export token_chunk=8000
# export chunk_overlap=0
# export lang=en

# bash -x apps/bigtts/umm/diffusion/scripts/recons_umm_wvae_streaming.sh
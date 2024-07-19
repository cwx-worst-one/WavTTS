#!/bin/bash


############################ Common Options ######################################
seed=1997
hdfs_root=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bigtts/umm_ar_outputs
output_root=/mnt/bn/cjw-lq-1/bigtts/ar/output
meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
umm_ckpt=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpt_repo/umm/v0.6.2.master.ckpt
note=r1 # will add to output folder to indicate which situation we run on.

################################# AR Options ######################################
thresh=0.9
mode=naive
eos_weight=0.8
step_out_blank=v2 # v0(不使用发愣) or v1(5个重复帧发愣) or v2(滑窗发愣) or v3(滑窗发愣+发愣逻辑)
temperature=0.5
max_blank_length=10
max_repeat_times=1
ar_model_version=merge_v2
ar_precision=fp16-mixed

############################# AR CHECKPOINT CONFIGURATION ###################################
# We assume that checkpoint saved in a path as following:
# ${ar_ckpt_root}/${ar_log_name}/${ar_log_version}/checkpoints/xxx.ckpt
ar_ckpt_root=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei
ar_log_name=umm_ar
ar_log_version=merge2_master_umm0.6.2_data2609_0dur90_cycle400k_lr1_agb4_size0.7B_24H800_fix2
ar_step=500k

################################ Diffusino Options ##########################################
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpt_repo/diffusion/epoch=00-step=900000-loss=0.52.master.ckpt
diffusion_precision=bf16-mixed
wvae_encoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt
wvae_decoder_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt

. scripts/parse_options.sh

# check configurations
if [ -z $ar_ckpt_root ]; then
    echo "ar_ckpt_root must be provided, please specify it by --ar_ckpt_root xxx"
    exit 1
fi
if [ -z $ar_step ]; then
    echo "ar_step must be provided, please specify it by --ar_step xxx"
    exit 1
fi

ar_ckpt_path=$(hdfs dfs -ls ${ar_ckpt_root}/${ar_log_name}/${ar_log_version}/checkpoints/ | grep ${ar_step//k/000} | awk '{print $NF}')

if [ -z $ar_ckpt_path ]; then
    echo "No ckpt found"
    echo "  ar_ckpt_root=${ar_ckpt_root}"
    echo "  ar_log_name=${ar_log_name}"
    echo "  ar_log_version=${ar_log_version}"
    echo "  ar_step=${ar_step}"
    exit 1
fi


info=ar_model_version-${ar_model_version}_blank-${step_out_blank}_win-${max_blank_length}_eos-${eos_weight}_seed-${seed}_note-${note}
output_root=${output_root}/${ar_log_name}/${ar_log_version}/${ar_step}/${info}
log_dir=${output_root}/infer_logs/
mkdir -p $log_dir

echo "log dir: ${log_dir}"

echo "---------------------- $(date "+%Y-%m-%d %H:%M:%S") Inference Start ----------------------"

# --diffusion_output_path $diffusion_output_path \
# --ar_output_path ${ar_output_path} \
# --src_lang
# --tgt_lang
# --meta_lst
# --icl_mode
common_args="--seed ${seed} \
    --ar_ckpt_path ${ar_ckpt_path} \
    --ar_precision ${ar_precision} \
    --temperature ${temperature} \
    --thresh ${thresh} \
    --mode ${mode} \
    --step_out_blank ${step_out_blank} \
    --max_blank_length ${max_blank_length} \
    --max_repeat_times ${max_repeat_times} \
    --ar_model_version ${ar_model_version} \
    --eos_weight ${eos_weight} \
    --diffusion_ckpt_path ${diffusion_ckpt_path} \
    --diffusion_precision ${diffusion_precision}"


#############################################################################################
###################################### EN TO EN #############################################
#############################################################################################
src_lang=en
tgt_lang=en
meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst
icl_mode=continuation
ar_output_path=${output_root}/${src_lang}2${tgt_lang}/ar
diffusion_output_path=${output_root}/${src_lang}2${tgt_lang}/diffusion
ARNOLD_WORKER_0_PORT=10000 bash apps/bigtts/umm/ar/scripts/infer_ar_diffusion_one.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --meta_lst ${meta_lst} \
    --icl_mode ${icl_mode} \
    --ar_output_path ${ar_output_path} \
    --diffusion_output_path ${diffusion_output_path} \
    ${common_args} |& tee ${log_dir}/${src_lang}2${tgt_lang}.log &


#############################################################################################
###################################### ZH to ZH #############################################
#############################################################################################
src_lang=zh
tgt_lang=zh
meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst
icl_mode=continuation
ar_output_path=${output_root}/${src_lang}2${tgt_lang}/ar
diffusion_output_path=${output_root}/${src_lang}2${tgt_lang}/diffusion
ARNOLD_WORKER_0_PORT=10001 bash apps/bigtts/umm/ar/scripts/infer_ar_diffusion_one.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --meta_lst ${meta_lst} \
    --icl_mode ${icl_mode} \
    --ar_output_path ${ar_output_path} \
    --diffusion_output_path ${diffusion_output_path} \
    ${common_args} |& tee ${log_dir}/${src_lang}2${tgt_lang}.log &


#############################################################################################
###################################### ZH to EN #############################################
#############################################################################################
src_lang=zh
tgt_lang=en
meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
icl_mode=non-continuation
ar_output_path=${output_root}/${src_lang}2${tgt_lang}/ar
diffusion_output_path=${output_root}/${src_lang}2${tgt_lang}/diffusion
ARNOLD_WORKER_0_PORT=10002 bash apps/bigtts/umm/ar/scripts/infer_ar_diffusion_one.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --meta_lst ${meta_lst} \
    --icl_mode ${icl_mode} \
    --ar_output_path ${ar_output_path} \
    --diffusion_output_path ${diffusion_output_path} \
    ${common_args} |& tee ${log_dir}/${src_lang}2${tgt_lang}.log &


#############################################################################################
###################################### EN to EN #############################################
#############################################################################################
src_lang=en
tgt_lang=zh
meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
icl_mode=non-continuation
ar_output_path=${output_root}/${src_lang}2${tgt_lang}/ar
diffusion_output_path=${output_root}/${src_lang}2${tgt_lang}/diffusion
ARNOLD_WORKER_0_PORT=10003 bash apps/bigtts/umm/ar/scripts/infer_ar_diffusion_one.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --meta_lst ${meta_lst} \
    --icl_mode ${icl_mode} \
    --ar_output_path ${ar_output_path} \
    --diffusion_output_path ${diffusion_output_path} \
    ${common_args} |& tee ${log_dir}/${src_lang}2${tgt_lang}.log &
wait

echo "---------------------- $(date "+%Y-%m-%d %H:%M:%S") Inference Done ----------------------"
echo
echo "---------------------- Metrics Overall ----------------------"
grep WER ${log_dir}/*.log
echo "-------------------------------------------------------------"
grep ASV ${log_dir}/*.log
echo "-------------------------------------------------------------"
echo

hdfs_output_path=${hdfs_root}/${ar_log_name}/${ar_log_version}/${ar_step}
echo "Uploading outputs to hdfs ${hdfs_output_path}"
set -x
tar -cf ${info}.tar ${output_root}
hdfs dfs -mkdir -p ${hdfs_output_path}
hdfs dfs -put -f ${info}.tar ${hdfs_output_path}
rm ${info}.tar
set +x

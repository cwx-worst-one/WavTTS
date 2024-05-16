#!/bin/bash

hdfs_root=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/bigtts/umm_ar_outputs
############################ AR DECODING CONFIGURATION ######################################
thresh=0.9
mode=naive
eos_weight=0.8
step_out_blank=v2 # v0(不使用发愣) or v1(5个重复帧发愣) or v2(滑窗发愣) or v3(滑窗发愣+发愣逻辑)
temperature=0.9
max_blank_length=5
max_repeat_times=1
ar_seed=1997

############################# AR CHECKPOINT CONFIGURATION ###################################
# We assume that checkpoint saved in a path as following:
# ${ckpt_root}/${log_name}/${log_version}/checkpoints/xxx.ckpt
ckpt_root=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei
log_name=umm_ar
log_version=merge2_master_umm0.6.2_data2609_0dur90_cycle400k_lr1_agb4_size0.7B_24H800_fix2
ar_step=500k
note=r1
############################### AR MODEL CONFIGURATION ######################################
model_version=merge_v2

diffusion_seed=2023
output_root=/mnt/bn/data-storage/bigtts/ar/output
meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
# umm_ckpt=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpt_repo/umm/v0.6.2.master.ckpt

. scripts/parse_options.sh

output_root=${output_root}/${log_name}/${log_version}/${ar_step}/${note}
log_dir=${output_root}/infer_logs/
mkdir -p ${log_dir}/{ar,diffusion}
echo "log dir: ${log_dir}"

echo "$(date): AR Inference Start."

ar_common_args="--thresh ${thresh} \
    --mode ${mode} \
    --eos_weight ${eos_weight} \
    --step_out_blank ${step_out_blank} \
    --temperature ${temperature} \
    --max_blank_length ${max_blank_length} \
    --max_repeat_times ${max_repeat_times} \
    --seed $ar_seed \
    --step $ar_step \
    --ckpt_root $ckpt_root \
    --log_name $log_name \
    --log_version $log_version \
    --model_version ${model_version} \
    --note $note \
    --meta_root ${meta_root} \
    --output_root ${output_root}"

ARNOLD_WORKER_0_PORT=10000 bash apps/bigtts/umm/ar/scripts/infer-umm-ar-spkid-wer_v3.sh --src_lang en --tgt_lang en ${ar_common_args} &> ${log_dir}/ar/en2en.log &
ARNOLD_WORKER_0_PORT=10001 bash apps/bigtts/umm/ar/scripts/infer-umm-ar-spkid-wer_v3.sh --src_lang zh --tgt_lang zh ${ar_common_args} &> ${log_dir}/ar/zh2zh.log &
ARNOLD_WORKER_0_PORT=10002 bash apps/bigtts/umm/ar/scripts/infer-umm-ar-spkid-wer_v3.sh --src_lang en --tgt_lang zh ${ar_common_args} &> ${log_dir}/ar/en2zh.log &
ARNOLD_WORKER_0_PORT=10003 bash apps/bigtts/umm/ar/scripts/infer-umm-ar-spkid-wer_v3.sh --src_lang zh --tgt_lang en ${ar_common_args} &> ${log_dir}/ar/zh2en.log &
wait

echo "$(date): AR Inference Done."


echo "$(date): Diffusion Inference Start."

diffusion_common_args="--seed ${diffusion_seed} \
    --log_name ${log_name} \
    --log_version $log_version \
    --ar_step $ar_step \
    --meta_root ${meta_root} \
    --output_root ${output_root}"

tag=umm_t0.5_p0.9_ar${ar_step}_en_v0.6.2_${model_version}_continuation_blankv2_win10_200wh_eos${eos_weight}_1pad1_copy3_wer_seed$ar_seed
ARNOLD_WORKER_0_PORT=20000 bash apps/bigtts/umm/diffusion/scripts//ar-diffusion-vocoder-wer_v3.sh --src_lang en --tgt_lang en --tag ${tag} ${diffusion_common_args} &> ${log_dir}/diffusion/en2en.log &

tag=umm_t0.5_p0.9_ar${ar_step}_zh_v0.6.2_${model_version}_continuation_blankv2_win10_200wh_eos${eos_weight}_1pad1_copy3_wer_seed$ar_seed
ARNOLD_WORKER_0_PORT=20001 bash apps/bigtts/umm/diffusion/scripts//ar-diffusion-vocoder-wer_v3.sh --src_lang zh --tgt_lang zh --tag ${tag} ${diffusion_common_args} &> ${log_dir}/diffusion/zh2zh.log &

tag=umm_t0.5_p0.9_ar${ar_step}_en2zh_v0.6.2_${model_version}_non-continuation_blankv2_win10_200wh_eos${eos_weight}_1pad1_copy3_wer_seed$ar_seed
ARNOLD_WORKER_0_PORT=20002 bash apps/bigtts/umm/diffusion/scripts//ar-diffusion-vocoder-wer_v3.sh --src_lang en --tgt_lang zh --tag ${tag} ${diffusion_common_args} &> ${log_dir}/diffusion/en2zh.log &

tag=umm_t0.5_p0.9_ar${ar_step}_zh2en_v0.6.2_${model_version}_non-continuation_blankv2_win10_200wh_eos${eos_weight}_1pad1_copy3_wer_seed$ar_seed
ARNOLD_WORKER_0_PORT=20003 bash apps/bigtts/umm/diffusion/scripts//ar-diffusion-vocoder-wer_v3.sh --src_lang zh --tgt_lang en --tag ${tag} ${diffusion_common_args} &> ${log_dir}/diffusion/zh2en.log &
wait

echo "$(date): Diffusion Inference Done."

echo "----------------------Metrics Overall---------------"
grep WER ${log_dir}/diffusion/*.log
echo "----------------------------------------------------"
grep ASV ${log_dir}/diffusion/*.log

echo "Uploading outputs to hdfs ${hdfs_root}"
set -x
tar -cf output.tar ${output_root}
hdfs_output_path=${hdfs_root}/${log_name}/${log_version}/${ar_step}/${note}
hdfs dfs -mkdir -p ${hdfs_output_path}
hdfs dfs -put output.tar ${hdfs_output_path}
rm output.tar
set +x

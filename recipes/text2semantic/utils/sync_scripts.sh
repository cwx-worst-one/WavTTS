#! /usr/bin/env bash
#for x in run_pretrain_WFVAE_v2_merge20231008.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_v2.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_sft.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_sft20231011.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_nobpe.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_sft20231014.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_nobpe_sft20231014.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_nobpe_sft20231011.sh;do
#for x in run_pretrain_WFVAE_v2_merge20231008_sft20231014-v2.sh;do
for x in run_pretrain_WFVAE_v2_merge20231008_sft20231016.sh;do
    hdfs dfs -mkdir -p hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge20231008/recipes/text2semantic/custom_scripts
    for file in `ls /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge20231008/recipes/text2semantic/custom_scripts/$x`;do
        echo ${file}
        name=`basename $file`
        echo "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge20231008/recipes/text2semantic/custom_scripts/$name"
        hdfs dfs -put -f ${file} hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge20231008/recipes/text2semantic/custom_scripts/
    done
done

#! /usr/bin/env bash
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-1879h_wd0.1_dp0.0_0.9b_bpe.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-1879h_wd0.1_dp0.0_0.9b_bpe_textloss.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-1879h_wd0.0_dp0.0_0.9b_bpe.sh;do
#for x in run_pretrain_yuanzhe_sft_spkid_labv3.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss_tv31.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss_tv32.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-1879h_wd0.1_dp0.0_0.9b_bpe_scratch.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-1879h_wd0.1_dp0.0_0.9b_bpe_textloss_sft_full_spk_nospkid.sh;do
for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss_stop.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss_stop_tv31.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b_textloss_stop_tv32.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-others_wd0.1_dp0.0_0.9b_textloss.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others_wd0.1_dp0.0_0.9b.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-others_wd0.1_dp0.0_0.9b.sh;do
#for x in run_pretrain_WFVAE_v2_labv3_punc_ll-rp-fq-others-enx2_wd0.1_dp0.0_0.9b_textloss.sh;do
    hdfs dfs -mkdir -p hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge/recipes/text2semantic/custom_scripts
    for file in `ls /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_merge/recipes/text2semantic/custom_scripts/$x`;do
        echo ${file}
        name=`basename $file`
        echo "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge/recipes/text2semantic/custom_scripts/$name"
        hdfs dfs -put -f ${file} hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_merge/recipes/text2semantic/custom_scripts/
    done
done

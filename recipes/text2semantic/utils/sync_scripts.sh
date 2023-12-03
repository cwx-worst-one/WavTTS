#! /usr/bin/env bash
for x in run_pretrain_WFVAE_v2_lang_spk_tag_pq_ali_sft.sh;do
    hdfs dfs -mkdir -p hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_ref_enc/recipes/text2semantic/custom_scripts
    for file in `ls /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_ref_enc/recipes/text2semantic/custom_scripts/$x`;do
        echo ${file}
        name=`basename $file`
        echo "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_ref_enc/recipes/text2semantic/custom_scripts/$name"
        hdfs dfs -put -f ${file} hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/code/samantha_bigtts_ref_enc/recipes/text2semantic/custom_scripts/
    done
done

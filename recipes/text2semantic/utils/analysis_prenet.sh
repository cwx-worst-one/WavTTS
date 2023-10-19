# hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_crosslingual/text2semantic/pretrain_labv3_punc_152_bt17500_16A100_accu2_lang_spk_concat_gr/checkpoints/epoch=00-step=55000-kl_loss=0.79.ckpt
for ckpt_name in epoch=00-step=30000-kl_loss=0.85.ckpt;do
    hdfs_path=hdfs://haruna/home/hcache/centralize_lq/gpt_java/speech/user/huangzhiying.92/exp/samantha_bigtts_crosslingual/text2semantic/pretrain_labv3_punc_152_bt17500_16A100_accu2_lang_spk_concat_gr_gl2-256/checkpoints/$ckpt_name
    local_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_crosslingual/text2semantic/pretrain_labv3_punc_152_bt17500_16A100_accu2_lang_spk_concat_gr_gl2-256/checkpoints/

    [ ! -d $local_dir ] && mkdir -p $local_dir

    ckpt_name=`echo $hdfs_path | awk -F"/" '{print $NF}'`
    step=`echo $ckpt_name | cut -d "-" -f 2 | cut -d "=" -f 2`
    suffixes=bn_prenet_gr${step}

    local_path=$local_dir/$ckpt_name
    local_dir=`dirname $local_path`
    [ ! -f $local_path ] && hdfs dfs -get $hdfs_path $local_dir/

    work_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_crosslingual
    cd $work_dir
    python3 recipes/text2semantic/utils/analysis_prenet.py $local_path /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/for_analysis/zh_en/WFVAE_melvq_ss_en $local_dir/WFVAE_melvq_ss_en $suffixes
    python3 recipes/text2semantic/utils/analysis_prenet.py $local_path /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/for_analysis/zh_en/WFVAE_melvq_ss_zh $local_dir/WFVAE_melvq_ss_zh $suffixes

    python3 recipes/text2semantic/utils/plot_bn.py $local_dir/WFVAE_melvq_ss_zh $local_dir/WFVAE_melvq_ss_en $suffixes $local_dir/plot
    cd -
done
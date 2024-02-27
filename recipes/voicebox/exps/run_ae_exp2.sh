#!/bin/bash -ex

set -x

SPARSE_GPT=TRUE
DYN_BATCH_SIZE=TRUE

mode=$1
batch_total_tokens=30000

log_name="voicebox"
version="0.0.2"
data_lst="hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/data_store/BigTTS/librilight_mos3.8_sim0.0_snr7_rms-13_asr0.85_internal_1_5_10s/package/wav_1.0_web_dataset_1/data/*/*.tar"
log_dir="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/congjian/logs/${log_name}/${version}"

if [[ ${mode} == "train" ]];then
    base_config="--config recipes/voicebox/conf/voicebox.yaml \
        --run_opts.urls ${data_lst} \
        --run_opts.log_dir ./logs \
        --run_opts.hdfs_path ${log_dir} \
        --run_opts.log_name ${log_name} \
        --run_opts.version ${version} \
        --run_opts.batch_total_tokens ${batch_total_tokens} \
        --run_opts.precision bf16 \
        --run_opts.checkpointing False \
        --run_opts.n_step_save 2000 \
        --run_opts.num_epochs 2000 \
        --run_opts.weight_decay 0.0 \
        --run_opts.num_workers 1 \
        --trainer.log_every_n_steps 100"
    bash launch.sh fit $base_config
fi


# ICL fighting spk enc with dur model
# hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_18wh_cn_3wh_hifiganmel_spkenc_0.7B_4/checkpoints/epoch=00-step=30000-mel_loss=0.31.ckpt
if [[ ${mode} == "ICL" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_18wh_cn_3wh_hifiganmel_spkenc_0.7B_4/checkpoints/epoch=00-step=30000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output_1004/en_test_0.7B
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting/meta.lst.106.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types zero_shot

    mkdir -p $output_dir/wav
    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir/wav \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


if [[ ${mode} == "ICL_CN" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/cn_3wh_hifiganmel_spkenc/checkpoints/epoch=00-step=24000-mel_loss=0.31.ckpt
    # ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_18wh_cn_3wh_hifiganmel_spkenc_0.7B_4/checkpoints/epoch=00-step=30000-mel_loss=0.31.ckpt
    # ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_4wh_6scrop/checkpoints/epoch=00-step=50000-mel_loss=0.30.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output_1004/cn_test_cn3w
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting_CN/meta.lst.138.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --pl_module.lang cn \
        --run_opts.inference_types zero_shot

    mkdir -p $output_dir/wav
    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir/wav \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


if [[ ${mode} == "infer_zero_shot_spkenc_test" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_4wh_6scrop_lsa/checkpoints/epoch=00-step=10000-mel_loss=0.34.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output_1003/10k_lsa
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting/meta.lst.different.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    mkdir -p $output_dir/wav
    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir/wav \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi



if [[ ${mode} == "infer_zero_shot_spkenc_80hz" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_3wh_spkenc_24k_hop300/checkpoints/epoch=00-step=95000-mel_loss=0.27.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output_1004/test_80hz
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting/meta.lst.different_80hz.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc_24k_hop300.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    mkdir -p $output_dir/wav
    cd recipes/voicebox/vocoder/iBigVGAN/
    python3 infer_torchscript.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir/wav \
        --torchscript_pth /mnt/bn/cyz-lq-nas/ibigvgan80_160k.pt
    # cd recipes/voicebox/vocoder/hifi-gan/
    # python3 inference_e2e.py \
    #     --input_mels_dir $output_dir \
    #     --output_dir $output_dir/wav \
    #     --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


if [[ ${mode} == "infer_fake_zero_shot_lsaspkenc" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_3wh_hifiganmel_local_spkattn/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/en_4wh_cn_3wh_hifiganmel_local_spkattn_15kdebug2
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst.different.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    mkdir -p $output_dir/wav
    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir/wav \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


# 训练集内编辑测试
if [[ ${mode} == "infer_librilight" ]];then
    ckpt_path=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/ckpt/0.0.2/epoch=00-step=46000-mel_loss=0.55.ckpt
    output_dir=/mnt/bn/jcong5/logs/voicebox/$version/librilight-editing/
    mkdir -p $output_dir
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/librilight/meta.out
    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types content_editing

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


# 集外 fake-zero-shot 测试
if [[ ${mode} == "infer_fake_zero_shot" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_hifiganmel/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_15k_2
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst.different.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi

# bigvgan
if [[ ${mode} == "infer_fake_zero_shot_bigvgan" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_hifiganmel/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_15k_bigvgan
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst.different.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/BigVGAN/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/cyz-lq-nas/bigvgan_22khz_80band/g_05000000.zip
fi


# # 集外 fake-zero-shot 测试
if [[ ${mode} == "infer_fake_zero_shot_40hz" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_40hzmel/checkpoints/epoch=00-step=30000-mel_loss=0.30.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_40hz_clip
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting/meta.lst.different_40hz.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_40hz.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/iBigVGAN/
    python3 infer_torchscript.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --torchscript_pth /mnt/bn/cyz-lq-nas/ibigvgan_3191.pt
fi





# 集外 fake-zero-shot 测试
if [[ ${mode} == "infer_fake_zero_shot_CNtoEN" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_3wh_hifiganmel/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_cn3wh_15k_CNtoEN
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting_CN/meta.lst.normal2.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


# 集外 fake-zero-shot 测试
if [[ ${mode} == "infer_fake_zero_shot_ENtoCN" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_3wh_hifiganmel/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_cn3wh_15k_ENtoCN
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting/meta.lst.en2cn.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi



if [[ ${mode} == "infer_fake_zero_shot_CN" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_cn_3wh_hifiganmel/checkpoints/epoch=00-step=15000-mel_loss=0.31.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_cn3wh_15k_CN
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting_CN/meta.lst.normal.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


if [[ ${mode} == "infer_fake_zero_shot_spkenc_CN" ]];then
    ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenyuanzhe/voicebox/en_4wh_hifiganmel_spkenc/checkpoints/epoch=00-step=15000-mel_loss=0.27.ckpt
    output_dir=/mnt/bn/cyz-lq-nas/output_dir/vb_output/test_en4wh_spkenc_15k_CN3
    mkdir -p $output_dir
    # meta_lst=/mnt/bn/jcong5/workspace3/samantha-vb-zhuo/recipes/voicebox/testset/icl_fighting/meta.lst
    meta_lst=/mnt/bn/cyz-lq-nas/project/samantha/recipes/voicebox/testset/icl_fighting_CN/meta.lst.normal2.text

    bash launch.sh predict \
        -c recipes/voicebox/conf/voicebox_infer_spkenc.yaml \
        --run_opts.meta_lst $meta_lst  \
        --run_opts.ckpt_path $ckpt_path \
        --run_opts.output_dir $output_dir \
        --run_opts.num_workers 1 \
        --run_opts.inference_types fake_zero_shot

    cd recipes/voicebox/vocoder/hifi-gan/
    python3 inference_e2e.py \
        --input_mels_dir $output_dir \
        --output_dir $output_dir \
        --checkpoint_file /mnt/bn/jcong5/workspace3/samantha-voicebox/.module_cache/hifigan/universal_v1/g_02500000 
fi


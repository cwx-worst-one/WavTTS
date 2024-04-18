#!/bin/bash
# run example:
# bash recipes/diffusion/scripts/infer-umm-ar-wer_v3.sh en en 1996

set -x

# link umm
rm -fr recipes/umm  # recipes/umm_062
git fetch origin cjw/bigmusic/bigtts_wfvae && git checkout origin/cjw/bigmusic/bigtts_wfvae -- recipes/umm_062
ln -s -r recipes/umm_062 recipes/umm

src_lang=${1:-"zh"}
tgt_lang=${2:-"zh"}
seed=${3:-1997}
ckpt_type=$4
step=$5
flag=${6:+_$6}


version=v0.6.2
thresh=0.9
mode=naive
max_blank_length=5
model_version="spkid_v3_3"
# icl_mode="non-continuation" # 需要修改续写或者非续写模式 ["continuation", "non-continuation"]

if [ $src_lang = $tgt_lang ]; then
    icl_mode="continuation"
else
    icl_mode="non-continuation"
fi

step_out_blank=true
temperature=0.9

suffix=${model_version}_${icl_mode}_200w_daze_${step_out_blank}_seed$seed

# step=100k
# ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei/llama/$exp/checkpoints/$ckpt
# ckpt_path=/mnt/bn/cjw-lq-1/project/wangxin.colin/infer/ckpts/opt_bucket/step=100000-accu=14.00.ckpt
# ckpt_path=/mnt/bn/cjw-lq-1/project/wangxin.colin/infer/ckpts/opt_all/step=100000-accu=14.87.ckpt
# ckpt_path=/mnt/bn/cjw-lq-1/project/wangxin.colin/infer/ckpts/baseline/step=100000-accu=14.23.ckpt
ckpt_root=/mnt/bn/cjw-lq-1/project/wangxin.colin/infer/ckpts
# ckpt_type=mix_dataloader
ckpt_path=$(ls ${ckpt_root}/${ckpt_type}/* | grep ${step//k/000})

echo ckpt_path=$ckpt_path
echo model_version=$model_version
echo icl_mode=$icl_mode

meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
out_root=/mnt/bn/cjw-lq-1/project/wangxin.colin/infer/output/ar/${ckpt_type}/${step}${flag}/
echo out_root=$out_root
mkdir -p $out_root

if [ $src_lang == "en" ] && [ $tgt_lang == "en" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst

elif [ $src_lang == "zh" ] && [ $tgt_lang == "zh" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst

elif [ $src_lang == "en" ] && [ $tgt_lang == "zh" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst

elif [ $src_lang == "zh" ] && [ $tgt_lang == "en" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst

### another en2zh
elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "zh" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/en2${tgt_lang}_meta.lst
    tgt_lang=$src_lang

### another zh2en
elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "en" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    save_npy_name=speech_umm_sami_t${temperature}_p${thresh}_ar${step}_${src_lang}2${tgt_lang}_${version}_${suffix}
    meta_lst=${meta_root}/icl_testset_2.0/zh2${tgt_lang}_meta.lst
    tgt_lang=$src_lang

else
    echo "Unsupported Language!"
    exit
fi

output_path=${out_root}/${save_npy_name}/
mkdir -p $output_path

TORCHRUN samantha/main.py predict \
        -c apps/bigtts/umm/ar/conf/infer_bigtts_ar.yaml \
        --pl_module.semantic_precision fp16 \
        --run_opts.num_workers 1 \
        --run_opts.meta_lst $meta_lst \
        --run_opts.umm_ckpt_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt \
        --run_opts.semantic_model_path $ckpt_path \
        --run_opts.output_path $output_path \
        --run_opts.src_lang $src_lang \
        --run_opts.tgt_lang $tgt_lang \
        --run_opts.temperature $temperature \
        --run_opts.thresh $thresh \
        --run_opts.mode $mode \
        --run_opts.step_out_blank $step_out_blank \
        --run_opts.max_blank_length $max_blank_length \
        --run_opts.seed $seed \
        --run_opts.version $model_version \
        --run_opts.icl_mode $icl_mode || exit 1

echo $output_path

set +x

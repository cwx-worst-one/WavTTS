#!/bin/bash
# run example:
# bash recipes/diffusion/scripts/infer-umm-ar-wer_v3.sh en en 1996


# link umm
mv recipes/umm recipes/umm_bak
mv recipes/umm_062 recipes/umm_062_bak
rm -rf recipes/umm
rm -rf recipes/umm_062
git fetch bigmusic/bigtts_wfvae
git checkout bigmusic/bigtts_wfvae -- recipes/umm_062
ln -s -r recipes/umm_062 recipes/umm

src_lang=${1:-"zh"}
tgt_lang=${2:-"zh"}
seed=${3:-1996}

version=v0.6.2
temperature=0.9
thresh=0.9
mode=naive
step_out_blank=true
max_blank_length=5

suffix=langid_stable_seed$seed
exp=umm_v0.6.2_ctiga_lang_wvaefe_punc_data772_dur60_cycle400k_lr6_agb2
ckpt=step=120000-accu=17.22.ckpt
step=120k
# ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/chenjiawei/llama/$exp/semantic_model_0.7B/$exp/checkpoints/$ckpt
# ckpt_path=/home/tiger/ref/step=120000-accu=17.22.ckpt  # ref
ckpt_path=/home/tiger/align/step=120000-accu=17.00.ckpt  # align

meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
# out_root=/mnt/bn/cjw-lq-1/project/kainan/umm/llama/out_npys
# out_root=/opt/tiger/out_npys
# out_root=/home/tiger/ref  # ref
out_root=/home/tiger/align7

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

bash launch.sh predict \
        -c apps/bigtts/umm/ar/conf/infer_bigtts_ar.yaml \
        --pl_module.semantic_precision fp16 \
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
        --run_opts.seed $seed

echo $output_path

rm -rf recipes/umm
rm -rf recipes/umm_062
mv recipes/umm_bak recipes/umm
mv recipes/umm_062_bak recipes/umm_062

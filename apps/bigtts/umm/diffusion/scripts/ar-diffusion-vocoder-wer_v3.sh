#!/bin/bash
# run example:

src_lang=zh
tgt_lang=zh
tag=
seed=2023
log_version=
ar_step=
log_name=
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpt_repo/diffusion/epoch=00-step=900000-loss=0.52.master.ckpt
diffusion_step=900k
umm_ckpt=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.new.ckpt

meta_root=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset
output_root=/mnt/bn/data-storage/bigtts/ar/output
vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

echo "cmd: $0 $@"

. scripts/parse_options.sh



pip3 install -q jiwer

# 修改这里为 AR 对应输出
ar_predict_token_path=${output_root}/${tag}
out_root=${output_root}/diffusion/seed${seed}

mkdir -p ${out_root}

if [ $src_lang == "en" ] && [ $tgt_lang == "en" ]; then
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/${tgt_lang}/

elif [ $src_lang == "zh" ] && [ $tgt_lang == "zh" ]; then
    meta_lst=${meta_root}/icl_testset_2.0/${tgt_lang}/meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/${tgt_lang}/

elif [ $src_lang == "en" ] && [ $tgt_lang == "zh" ]; then
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/

elif [ $src_lang == "zh" ] && [ $tgt_lang == "en" ]; then
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/

elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "zh" ]; then
    src_lang="en"
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/

elif [ $src_lang == "zh_en" ] && [ $tgt_lang == "en" ]; then
    echo "src = $src_lang, tgt = $tgt_lang"
    src_lang="zh"
    meta_lst=${meta_root}/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst
    prompt_wav_dir=${meta_root}/icl_testset_2.0/

else
    echo "Unsupported Language!"
    exit
fi

out_dir=${out_root}/$tag/diff${diffusion_step}
mkdir -p $out_dir

echo "CONFIGURATION:"
echo "    src_lang=$src_lang"
echo "    tgt_lang=$tgt_lang"
echo "    seed=$seed"
echo "    diffusion_ckpt_path=$diffusion_ckpt_path"
echo "    diffusion_step=$diffusion_step"
echo "    log_name=$log_name"
echo "    log_version=$log_version"
echo "    meta_root=$meta_root"
echo "    output_path=$out_dir"
echo "    meta_lst=$meta_lst"

TORCHRUN samantha/main.py predict \
    -c apps/bigtts/umm/diffusion/conf/infer_ar_diffusion.yaml \
    --predict_dataset.npy_path $ar_predict_token_path \
    --predict_dataset.meta_file $meta_lst \
    --predict_dataset.wav_dir $prompt_wav_dir \
    --predict_dataset.prompt_lang $src_lang \
    --predict_dataset.syn_lang $tgt_lang \
    --run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
    --run_opts.seed ${seed} \
    --run_opts.num_workers 1 \
    --run_opts.infer_type ar-diffusion-vocoder \
    --pl_module.umm_type UMM \
    --pl_module.diffusion_precision bf16 \
    --pl_module.diffusion_nfe 10 \
    --pl_module.diffusion_sampler ddim \
    --mel_config.mel_norm_mean -2.5 \
    --mel_config.mel_norm_std 6 \
    --bn_config.bn_norm_std 2 \
    --bn_config.bn_padding -5 \
    --pl_module.text_cfg_w 4 \
    --pl_module.use_wvae_vocoder True \
    --bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
    --bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt || exit 1
    # --pl_module.save_prompt True \
    # --pl_module.use_phone_lang True 

bash apps/bigtts/umm/diffusion/scripts//analysis.sh $meta_lst $out_dir $tgt_lang || exit 1


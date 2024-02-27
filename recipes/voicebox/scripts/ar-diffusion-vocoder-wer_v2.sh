# link umm
rm -r recipes/umm
ln -s -r recipes/umm_062 recipes/umm

# # OLD
# exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv062_textDrop0.25_Norm2
# ckpt="epoch=00-step=300000-loss=0.53.ckpt"
# step=300000

# NEW
exp=PrefixLDM4_300M_32H800_setting3a_40hzWVAE_UMMv062_textDrop0.25_Norm2
# ckpt="epoch=00-step=200000-loss=0.53.ckpt"
ckpt="epoch=00-step=510000-loss=0.51.ckpt"
step=510000

tag=speech_umm_sami_t0.9_p0.9_ar180k_en_v0.6.2_lang_stable
ar_predict_token_path=/mnt/bn/cjw-lq-1/project/kainan/umm/llama/out_npys/$tag

if [ $1 == 'en' ]; then

    echo 'src = en, tgt = en'
    lang=en
    prompt_wav_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/
    meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/meta.lst 

elif [ $1 == 'zh' ] ; then

    echo 'src = zh, tgt = zh'
    lang=zh
    prompt_wav_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/
    meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/meta.lst 

elif [ $1 == 'en2zh' ] ; then

    echo 'src = en, tgt = zh'
    src_lang=en
    tgt_lang=zh
    lang=${tgt_lang}
    prompt_wav_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/
    meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst 

elif [ $1 == 'zh2en' ] ; then

    echo 'src = zh, tgt = zh'
    src_lang=zh
    tgt_lang=en
    lang=${tgt_lang}
    prompt_wav_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/
    meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${src_lang}2${tgt_lang}_meta.lst 

else

    echo 'invalid conversion'
    exit

fi

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt

vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

# bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
# exit


out_dir=/mnt/bn/cjw-lq-1/project/kainan/umm/llama/out_wavs/$tag/wavs
mkdir -p $out_dir
bash launch.sh predict \
	-c recipes/voicebox/conf/infer_ar_diffusion.yaml \
	--predict_dataset.npy_path $ar_predict_token_path \
    --predict_dataset.meta_file $meta_lst \
    --predict_dataset.wav_dir $prompt_wav_dir \
    --predict_dataset.prompt_lang $src_lang \
    --predict_dataset.syn_lang $tgt_lang \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
	--run_opts.seed 1996 \
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
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--pl_module.use_phone_lang True \
    # --pl_module.save_prompt True

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang




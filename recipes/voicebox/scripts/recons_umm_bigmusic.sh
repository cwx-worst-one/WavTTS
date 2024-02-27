

exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv062_textDrop0.25_Norm2_testNewCode_Prefetch_OldImage
ckpt="epoch=00-step=60000-loss=0.53.ckpt"
step=60000


lang=en

# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_wo_vc.lst
meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_vc.lst
out_dir=/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp
mkdir -p $out_dir


# hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm_mix/umm_stage3_preclipped_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0330000.ckpt
umm_ckpt_path=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/Stage3_Multilingua_Music-TTS_MKii/step=0330000.ckpt

# wvae_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/vocoder/20240110/wavevae_decoder.pt
# h800_ds1198_without_prompt_0119
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1198_without_prompt_0119/checkpoints/epoch=00-step=690000-loss=0.13.ckpt
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1198_without_prompt_0119/checkpoints/epoch=00-step=1230000-loss=0.13.ckpt
# h800_ds1198_with_prompt_lr1e_5_0120
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1198_with_prompt_lr1e_5_0120/checkpoints/epoch=00-step=800000-loss=0.14.ckpt


    # --run_opts.infer_type diffusion-vocoder \
    # --run_opts.infer_type vocoder \
	# --bn_config.wvae_encoder_path /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/vocoder/20240112/wavevae_encoder.pt \
	# --bn_config.wvae_decoder_path /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/vocoder/20240112/wavevae_decoder.pt \
bash launch.sh predict \
	-c recipes/voicebox/conf/infer_reconstruction_25hzMel.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
	--run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.umm_ckpt_path $umm_ckpt_path \
	--run_opts.umm_frame_rate 25 \
	--run_opts.mel_frame_rate 40 \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--run_opts.infer_type diffusion-vocoder \
	--pl_module.text_cfg_w 0

# bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

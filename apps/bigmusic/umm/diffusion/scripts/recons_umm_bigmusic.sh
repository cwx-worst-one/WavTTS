


# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_wo_vc.lst
meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_vc2.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_clip.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_no_vc.lst
# meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/voice_condition_valsets/test2.lst
# meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/samples_sstk_30s/test.lst



# ====== tokenizer version ====== 
# | tokenizer type | local path                                                                                                                                             | ckpt                                                                                                                                                                  |
# | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | Conformer mix  | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/Stage3_Multilingua_Music-TTS_MKii/step=0330000.ckpt                                             | hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm_mix/umm_stage3_preclipped_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0330000.ckpt                   |
# | Conformer ZH   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/step=070000.ckpt                             | hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/checkpoints/step=070000.ckpt                                 |
# | Conv           | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/step=0160000.ckpt | hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt |
# | Dual Conv      | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0208_dualummv2_fullinp_1/step=0970000.ckpt                                                      | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0208_dualummv2_fullinp_1/checkpoints/step=0970000.ckpt                               |



# Conformer ZH + Music 125hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1641_prompt_16xH800_0224/checkpoints/epoch=00-step=160000-loss=0.38.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_125hzSS.yaml
# sub_dir=exp1_160k

# Conformer ZH + Music 40hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1614_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.19.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_40hzSS.yaml
# sub_dir=exp13_500k
# sub_dir=exp2_500k

# DualConv + Music 125hz
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1609_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_125hzSS.yaml
sub_dir=exp4_430k

# DualConv + Music 40hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1588_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.18.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_40hzSS.yaml
# sub_dir=exp5_500k
# sub_dir=exp14_500k

# Conv + Music 125hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1617_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_125hzSS.yaml
# sub_dir=exp7_430k

# Conv + Music 40hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1611_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.20.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_40hzSS.yaml
# sub_dir=exp8_500k




    # --run_opts.infer_type diffusion-vocoder \
    # --run_opts.infer_type vocoder \

# cleanup vocoder model in case of freq mismatch
# rm ./.module_cache/wavevae_*pt
# rm ./data/tmp/*.wav
out_dir=/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/without_prefix/${sub_dir}
mkdir -p $out_dir


bash launch.sh predict \
	-c $cfg_path \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
	--run_opts.diffusion_ckpt_path $diffusion_ckpt_path \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--run_opts.infer_type diffusion-vocoder \
	--bn_config.wav_norm True \
	--pl_module.text_cfg_w 0


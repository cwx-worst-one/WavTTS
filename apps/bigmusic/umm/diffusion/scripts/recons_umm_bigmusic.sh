


# ====== tokenizer version ====== 
# | tokenizer type | local path                                                                                                                                             | ckpt                                                                                                                                                                  |
# | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | Conformer mix  | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/Stage3_Multilingua_Music-TTS_MKii/step=0330000.ckpt                                             | hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm_mix/umm_stage3_preclipped_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0330000.ckpt                   |
# | Conformer ZH   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/step=070000.ckpt                             | hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/checkpoints/step=070000.ckpt                                 |
# | Conv           | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/step=0160000.ckpt | hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt |
# | Dual Conv V3   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0219_dualummv1_novocadv_1/step=0710000.ckpt                                                     | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0219_dualummv1_novocadv_1/checkpoints/step=0710000.ckpt                              |
# | Dual Conv V2   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0208_dualummv2_fullinp_1/step=0970000.ckpt                                                      | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0208_dualummv2_fullinp_1/checkpoints/step=0970000.ckpt                               |
# | Dual Conv V1   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0131_dualummv2_noref_2/step=0580000.ckpt                                                        | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0131_dualummv2_noref_2/checkpoints/step=0580000.ckpt                                 |


# cleanup vocoder model in case of freq mismatch
# rm ./.module_cache/soundstream*
# rm ./data/tmp/*.wav
current_time=$(date  "+%Y%m%d-%H%M%S")
out_dir=${output_dir}/${current_time}
mkdir -p $out_dir

cp $cfg_path $out_dir
cp $0 $out_dir

if [ -z $batch_size ]; then
	echo "no environment variant 'batch_size' exsists, will use 'batch_size=1'"
	export batch_size=1
fi

if [ -z $seed ]; then
	echo "no environment variant 'batch_size' exsists, will use 'batch_size=1'"
	export seed=1996
fi

if [ -z $diffusion_nfe ]; then
	echo "no environment variant 'diffusion_nfe' exsists, will use 'diffusion_nfe=10'"
	export diffusion_nfe=10
fi

if [ -z $tokenizer_version ]; then
	echo "no environment variant 'tokenizer_version' exsists, will use 'tokenizer_version=ConformerUMM_baseline'"
	export tokenizer_version=ConformerUMM_baseline
fi

if [ -z $vocoder_version ]; then
	echo "no environment variant 'vocoder_version' exsists, will use 'vocoder_version=44.1k_vocal'"
	export vocoder_version=44.1k_vocal
fi


bash launch.sh predict \
	-c $cfg_path \
	--run_opts.meta_lst $meta_lst \
	--run_opts.output_dir $out_dir \
	--run_opts.diffusion_ckpt_path $diffusion_ckpt_path \
	--run_opts.batch_size $batch_size \
	--run_opts.tokenizer_version $tokenizer_version \
	--run_opts.vocoder_version $vocoder_version \
	--pl_module.diffusion_nfe ${diffusion_nfe} \
	--run_opts.seed $seed
	
cp lightning_logs/version_0/config.yaml $out_dir
echo "====== cfg_path="$cfg_path
echo "====== meta_lst="$meta_lst
echo "====== diffusion_ckpt_path="$diffusion_ckpt_path
echo "====== diffusion_nfe="$diffusion_nfe
echo "====== all assets is saved in "$out_dir


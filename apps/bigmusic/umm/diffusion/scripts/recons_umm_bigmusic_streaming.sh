export https_proxy=http://bj-rd-proxy.byted.org:3128 http_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org
bash scripts/reinstall_s3a.sh 68
sudo apt install -y bc && pip install thop
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_wo_vc.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_vc2.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/voice_condition_valsets/test_vc2_without_prompt_x.lst
meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/1min_zh_vocal/test_no_vc.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_clip.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test.lst
# meta_lst=/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/test_no_vc.lst
# meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/voice_condition_valsets/test2.lst
# meta_lst=/mnt/bn/bigspeech-lf-nas/user/zhangshuo/data/assets/samples_sstk_30s/test.lst

# dur= x/ umm_freq
token_chunk_size=200
token_chunk_overlap=10
prompt_drop=0.0

# ====== tokenizer version ====== 
# | tokenizer type | local path                                                                                                                                             | ckpt                                                                                                                                                                  |
# | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | Conformer mix  | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/Stage3_Multilingua_Music-TTS_MKii/step=0330000.ckpt                                             | hdfs:///home/byte_speech_sv/zongyu.yin/logs/umm_mix/umm_stage3_preclipped_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0330000.ckpt                   |
# | Conformer ZH   | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/step=070000.ckpt                             | hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/umm_stage3_zh_dw1-1-0_wordpiece_vq32768x16-layer12/checkpoints/step=070000.ckpt                                 |
# | Conv           | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/step=0160000.ckpt | hdfs:///home/byte_speech_sv/hanoi.hantrakul/logs/umm_dual/umm_stage3_pitch_baseline_conv_1D_bert-base-multilingual-uncased_None32768x32/checkpoints/step=0160000.ckpt |
# | Dual Conv      | /mnt/bn/data-storage-hl/user/zhangshuo/data/assets/umm/0208_dualummv2_fullinp_1/step=0970000.ckpt                                                      | hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/umm_mix/0208_dualummv2_fullinp_1/checkpoints/step=0970000.ckpt                               |

# Conformer ZH + Music 125hz
dataset_id=1641
exp_name="25hzUMM_125hzSS"
chunk_size=1000 #100,500,1000
version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0303"
step=200000
loss=0.37
cfg_name="25hzConformer_125hzSS"


# consistency
version=${version}_consistency_0307
prompt_drop=0.1
step=490000
loss=0.06

# # prompt bn dropout
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_prompt_drop${prompt_drop}_streaming_chunk${chunk_size}_0308"
# dataset_id=1907
# prompt_drop=0.1
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_prompt_drop${prompt_drop}_streaming_chunk${chunk_size}_0315"
# version=${version}_consistency_0320
# step=90000
# loss=0.12


# Conformer ZH + Music 40hz
# dataset_id=1614
# exp_name="25hzUMM_40hzSS"
# chunk_size=320 #100,160,320
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0301"
# step=450000
# loss=0.19
# cfg_name="25hzConformer_40hzSS"
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1614_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.19.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConformer_40hzSS.yaml
# sub_dir=exp13_500k
# sub_dir=exp2_500k

# DualConv + Music 125hz
# dataset_id=1609
# exp_name="50hzUMM_125hzSS"
# chunk_size=1000 #100,500,1000
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0301"
# step=550000
# loss=0.37
# cfg_name="50hzDualConv_125hzSS"
# # consistence
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0301_consistency_0305"
# step=600000
# loss=0.09


# DualConv + Music 40hz
# dataset_id=1588
# exp_name="50hzUMM_40hzSS"
# chunk_size=320 #100,160,320
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0301"
# step=470000
# loss=0.18
# cfg_name="50hzDualConv_40hzSS"
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1588_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.18.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_50hzDualConv_40hzSS.yaml
# sub_dir=exp5_500k
# sub_dir=exp14_500k

# Conv + Music 125hz
# dataset_id=1616
# exp_name="25hzUMM_125hzSS"
# chunk_size=1000 #100,500,1000
# version="h800_ds${dataset_id}_prompt_16xH800_${exp_name}_streaming_chunk${chunk_size}_0301"
# step=370000
# loss=0.35
# cfg_name="25hzConv_125hzSS"
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1617_prompt_16xH800_0223/checkpoints/epoch=00-step=430000-loss=0.36.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_125hzSS.yaml
# sub_dir=exp7_430k

# Conv + Music 40hz
# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion/h800_ds1611_prompt_16xH800_0223/checkpoints/epoch=00-step=500000-loss=0.20.ckpt
# cfg_path=apps/bigmusic/umm/diffusion/conf/infer_reconstruction_25hzConv_40hzSS.yaml
# sub_dir=exp8_500k
wav_norm=false

# nfe=10 # 4,10
# sampler="ddim" # consistency, ddim
# text_cfg_w=0

nfe=4 # 4,10
sampler="consistency" # consistency, ddim
text_cfg_w=1

diffusion_ckpt_base_dir="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/unified_diffusion/unified_diffusion"
diffusion_ckpt_path="${diffusion_ckpt_base_dir}/${version}/checkpoints/epoch=00-step=${step}-loss=${loss}.ckpt"
cfg_path="apps/bigmusic/umm/diffusion/conf/infer_reconstruction_${cfg_name}_streaming.yaml"
sub_dir="ds${dataset_id}_${exp_name}_prompt_drop${prompt_drop}_streaming/chunk=${chunk_size}/sampler=${sampler}-nfe=${nfe}-text_cfg_w=${text_cfg_w}-norm=${wav_norm}/step=${step}"

without_prompt_bn=false
if [ $(echo "$prompt_drop > 0.0" | bc -l ) -eq 1 ]; then
	without_prompt_bn=true
fi
echo "without_prompt_bn: ${without_prompt_bn}"
if [ $without_prompt_bn ]; then
	sub_dir="without_prompt_bn/${sub_dir}"
fi

# cleanup vocoder model in case of freq mismatch
# rm ./.module_cache/soundstream*.ckpt
# rm ./data/tmp/*.wav
without_prefix=true # true, false
if [ ${without_prefix} = true  ] ; then
	out_dir=/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/without_prefix/${sub_dir}
else
	out_dir=/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/${sub_dir}
fi
echo "out_dir: ${out_dir}"
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
	--bn_config.wav_norm $wav_norm \
	--pl_module.text_cfg_w $text_cfg_w \
	--pl_module.token_chunk_size $token_chunk_size \
	--pl_module.token_chunk_overlap $token_chunk_overlap \
	--pl_module.without_prefix $without_prefix \
	--pl_module.diffusion_sampler $sampler \
	--pl_module.diffusion_nfe $nfe
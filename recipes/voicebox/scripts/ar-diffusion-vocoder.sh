#exp=PrefixLDM4_300M_48A100_token30000_setting2_40hzMel_UMMv02
#ckpt="epoch=00-step=500000-loss=0.13.ckpt"
#step=500000

#exp=PrefixLDM4_300M_16H800_token30000_setting0_40hzWVAE_UMMv07
#ckpt="epoch=00-step=190000-loss=0.14.ckpt"
#step=190000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv02_textDrop0.25_Norm2
#ckpt="epoch=00-step=60000-loss=0.53.ckpt"
#step=60000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv062_textDrop0.25_Norm2
#ckpt="epoch=00-step=300000-loss=0.53.ckpt"
#step=300000

exp=PrefixLDM4_300M_32H800_setting3a_40hzWVAE_UMMv062_textDrop0.25_Norm2
ckpt="epoch=00-step=430000-loss=0.52.ckpt"
step=430000

tag=speech_umm_sami_t09_p09_ar150k_zh_v0.6.2

lang=zh
test_wav_dir=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/
text_file=/mnt/bn/mam/kainan/umm/llama/infer/tacolabel_${lang}_synth_2.0.pyt

pickle_file=/mnt/bn/jdy-lq-2/bigtts-nar/ar_output/${tag}.pickle

#tag=exp1_AR480k_UMMv02_icl_test_2.0_en
#ar_predict_token_path=/mnt/bn/jdy-lq-2/bigtts-nar/ar_output/$tag

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
#out_dir=$out_dir/$exp/${tag}/$step/DDIM_20steps_textCFG4
out_dir=$out_dir/$exp/${tag}/$step/plms_10steps_textCFG4
mkdir -p $out_dir

#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt
umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt

vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/meta.lst 

#bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
#exit

bash launch.sh predict \
	-c recipes/voicebox/conf/infer_ar_diffusion.yaml \
	--predict_dataset.pickle_file $pickle_file \
    --predict_dataset.wav_dir $test_wav_dir \
    --predict_dataset.text_file $text_file \
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
	--pl_module.diffusion_sampler plms \
	--mel_config.mel_norm_mean -2.5 \
	--mel_config.mel_norm_std 6 \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding -5 \
	--pl_module.text_cfg_w 4 \
	--pl_module.use_wvae_vocoder True \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	#--pl_module.save_prompt True \

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

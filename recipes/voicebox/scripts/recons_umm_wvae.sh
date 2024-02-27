#exp=PrefixLDM4_300M_16H800_token30000_setting0_40hzWVAE_UMMv06
#ckpt="epoch=00-step=80000-loss=0.14.ckpt"
#step=80000

#exp=PrefixLDM4_300M_16H800_token30000_setting0_40hzWVAE_UMMv07
#ckpt="epoch=00-step=80000-loss=0.14.ckpt"
#step=80000

#exp=PrefixLDM4_300M_16H800_token30000_setting0_40hzWVAE_UMMv0.8.1
#ckpt="epoch=00-step=80000-loss=0.14.ckpt"
#step=80000

#exp=PrefixLDM4_300M_8A100_setting0_40hzWVAE_UMMv02
#ckpt="epoch=00-step=60000-loss=0.14.ckpt"
#step=60000

#exp=PrefixLDM4_300M_8A100_setting0_40hzWVAE_UMMv02_textDrop0.25
#ckpt="epoch=00-step=60000-loss=0.15.ckpt"
#step=60000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv02_textDrop0.25_Norm2
#ckpt="epoch=00-step=60000-loss=0.53.ckpt"
#step=60000
#ckpt="epoch=00-step=180000-loss=0.52.ckpt"
#step=180000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv02_textDrop0.25_Norm2_L2
#ckpt="epoch=00-step=70000-loss=0.47.ckpt"
#step=70000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv02_textDrop0.25_Norm2_L2_NewMirror
#ckpt="epoch=00-step=60000-loss=0.47.ckpt"
#step=60000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv2.0_textDrop0.25_Norm2
#ckpt="epoch=00-step=60000-loss=0.53.ckpt"
#step=60000
#ckpt="epoch=00-step=190000-loss=0.53.ckpt"
#step=190000

#exp=PrefixLDM4_300M_48A100_Fixedsetting2_40hzWVAE_UMMv02_textDrop0.25_Norm2
#ckpt="epoch=00-step=280000-loss=0.52.ckpt"
#step=280000

#exp=PrefixLDM4_300M_32H800_setting3a_40hzWVAE_UMMv02_textDrop0.25_Norm2
#ckpt="epoch=00-step=300000-loss=0.51.ckpt"
#step=300000

#exp=PrefixLDM4_300M_32H800_setting3a_40hzWVAE_UMMv062_textDrop0.25_Norm2
#ckpt="epoch=00-step=600000-loss=0.52.ckpt"
#step=600000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv062_textDrop0.25_Norm2
#ckpt="epoch=00-step=60000-loss=0.53.ckpt"
#step=60000
#ckpt="epoch=00-step=300000-loss=0.53.ckpt"
#step=300000
#ckpt="epoch=00-step=420000-loss=0.52.ckpt"
#step=420000

exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv062_textDrop0.25_Norm2_testNewCode_Prefetch_OldImage
ckpt="epoch=00-step=60000-loss=0.53.ckpt"
step=60000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv02_textDrop0.25_Norm2_LangPhone
#ckpt="epoch=00-step=180000-loss=0.53.ckpt"
#step=180000
#ckpt="epoch=00-step=60000-loss=0.53.ckpt"
#step=60000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv0.2.2_textDrop0.25_Norm2
#ckpt="epoch=00-step=200000-loss=0.52.ckpt"
#step=200000

#exp=PrefixLDM4_300M_8H800_setting0_40hzWVAE_UMMv0.2.3_textDrop0.25_Norm2
#ckpt="epoch=00-step=60000-loss=0.54.ckpt"
#step=60000

lang=en

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps_TextCFG4
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps
mkdir -p $out_dir

#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V2.0.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt
umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.2.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.3.ckpt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

#bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
#exit

bash launch.sh predict \
	-c recipes/voicebox/conf/infer_reconstruction_40hzMel.yaml \
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
	--pl_module.umm_type UMM \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe 10 \
	--pl_module.diffusion_sampler ddim \
	--pl_module.use_wvae_vocoder True \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding -5 \
	--pl_module.text_cfg_w 4 \
	#--pl_module.use_phone_lang True \
	#--pl_module.save_prompt True
	#--pl_module.umm_codebook_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.codebook \

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

#exp=PrefixLDM3_2.8B_64H800_lr4e-5_setting3a_CFG0.1_FixText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=760000-loss=0.525.ckpt"
#step=760000

#exp=PrefixLDM4_1.3B_64H800_lr2e-5FT_setting3a_CFG0.1_FixText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=760000-loss=0.525.ckpt"
#step=760000

#exp=PrefixLDM4_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=180000-loss=0.527.ckpt"
#step=180000

exp=DIT_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
ckpt="epoch=00-step=680000-loss=0.510.ckpt"
step=680000
#ckpt="epoch=00-step=270000-loss=0.524.ckpt"
#step=270000

#exp=PrefixLDM4_300M_fp16_8H800_lr5e-5_setting0_CFG0.1_FixText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=90000-loss=0.522.ckpt"
#step=90000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_NoToken_SegEmb_PosEmb_ctcL11Lam1e-1
#ckpt="epoch=00-step=90000-loss=0.520.ckpt"
#step=90000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_NoToken_SegEmb_PosEmb_ctcL11Lam1
#ckpt="epoch=00-step=90000-loss=0.560.ckpt"
#step=90000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_NoToken_SegEmb_PosEmb_ctcFinalLam1e-1
#ckpt="epoch=00-step=90000-loss=0.654.ckpt"
#step=90000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_SegEmb_PosEmb_ctcBeforePostLam1e-1
#ckpt="epoch=00-step=290000-loss=0.516.ckpt"
#step=290000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_SegEmb_PosEmb_MaskText0.2Lam1
#ckpt="epoch=00-step=80000-loss=0.693.ckpt"
#step=80000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_SegEmb_PosEmb_MaskText0.1Lam0.1
#ckpt="epoch=00-step=90000-loss=0.561.ckpt"
#step=90000

#exp=PrefixLDM4_300M_8H800_lr5e-5_setting0_CFG0.1_FixText_SegEmb_PosEmb_MaskText0.2Lam0.1
#ckpt="epoch=00-step=200000-loss=0.547.ckpt"
#step=200000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_FixText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=150000-loss=0.525.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=460000-loss=0.517.ckpt"
#step=460000
#ckpt="epoch=00-step=150000-loss=0.515.ckpt"
#step=150000
#ckpt="epoch=00-step=90000-loss=0.548.ckpt"
#step=90000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
#ckpt="epoch=00-step=290000-loss=0.522.ckpt"
#step=290000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_NoToken_SegEmb_bnPad0
#ckpt="epoch=00-step=150000-loss=0.522.ckpt"
#step=150000

lang=zh
syn_lang=zh
nfe=25
cfg=4
cg=4
sampler=ddim
length_factor=1 #zh2en: 0.8, en2zh: 1.25
#length_reference_path=/mnt/bn/jdy-lq-2/bigtts-nar/ar_output/icl_testset_demo_8.0/merge_v2_en2zh_demo0301/diff900k
length_reference_path=/mnt/bn/jdy-lq-2/bigtts-nar/ar_output/icl_testset_demo_8.0/merge_v2_zh2zh_demo0301/diff900k
#length_reference_path=/mnt/bn/jdy-lq-2/bigtts-nar/ar_output/icl_testset_demo_8.0/merge_v2_en2en_demo0301/diff900k
prompt_mode=post

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_demo_3.0/meta_${lang}.txt
#out_dir=$out_dir/$exp/icl_testset_demo_3.0_${lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_length${length_factor}_${prompt_mode}

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/meta.lst
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_length${length_factor}_${prompt_mode}
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

meta_lst=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_8.0/meta_${syn_lang}_cmos.txt
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_RefLength_${prompt_mode}
out_dir=$out_dir/$exp/icl_testset_demo_8.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}_cg${cg}_maxt0.999_Global

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/cn_hardcase/meta.lst
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_RefLength_${prompt_mode}
#out_dir=$out_dir/$exp/cn_hardcase/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

#meta_lst=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_8.0/meta_${lang}2${syn_lang}_cmos.txt
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${lang}2${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_RefLength_${prompt_mode}
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${lang}2${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_length${length_factor}_${prompt_mode}

#meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/meta.lst.listen_all_new_zh_select_v2.temp_taozi_conv_talk_0124
#out_dir=$out_dir/$exp/taozi_zh/$step/${sampler}_${nfe}steps_CFG${cfg}_${prompt_mode}
#meta_lst=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/meta.lst.listen_all_new_zh_select_v2.temp_taozi_conv_talk_0124
#out_dir=$out_dir/$exp/taozi_zh/$step/${sampler}_${nfe}steps_CFG${cfg}_${prompt_mode}
mkdir -p $out_dir


rm -r recipes/umm
ln -s -r recipes/umm_062 recipes/umm

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

bash launch.sh predict \
	-c recipes/voicebox/conf/infer_tts.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--predict_dataset.syn_lang $syn_lang \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding 0 \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.norm_cfg False \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.length_factor $length_factor \
	--pl_module.length_reference_path $length_reference_path \
	--pl_module.use_adaptive_length True \
	--pl_module.classifier_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/umm_classifier/logdir/emo_for_diffusion/Cfm100M_data2031_bz3600_4A100_100kstep_lr1e4_beta0_neu02_newDataCode/checkpoints/step=040000.ckpt \
	--pl_module.cg_w ${cg} \
	--pl_module.only_use_global_prompt True \
	--trainer.inference_mode False

	#--pl_module.save_prompt False

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $syn_lang

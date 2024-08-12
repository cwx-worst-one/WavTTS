#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

#exp=DIT_2.8B_64H800accum4_lr6e-5_setting3a_CFG0.1_QKNormHead_NoOverlap_NewMask
#ckpt="epoch=00-step=40000-loss=0.561.ckpt"
#step=40000

#exp=DIT_2.8B_64H800FSDP_lr6e-5_setting3a_CFG0.1_QKNormHead_NoOverlap
#ckpt="epoch=00-step=40000-loss=0.554.ckpt"
#step=40000

#exp=DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap
#ckpt="epoch=00-step=300000-loss=0.509.ckpt"
#ckpt="epoch=00-step=430000-loss=0.527.ckpt"
#ckpt="last.ckpt"
#step=430000

exp=DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap_NewMask
#ckpt="epoch=00-step=300000-loss=0.534.ckpt"
#step=300000
#ckpt="epoch=00-step=600000-loss=0.524.ckpt"
#step=600000
ckpt="epoch=00-step=770000-loss=0.523.ckpt"
step=770000

#exp=PrefixLDM4_1.3B_64H800_lr2e-5FT_setting3a_CFG0.1_FixText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=760000-loss=0.525.ckpt"
#step=760000

#exp=PrefixLDM4_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_NoToken_SegEmb_PosEmb
#ckpt="epoch=00-step=180000-loss=0.527.ckpt"
#step=180000

#exp=DIT_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
#ckpt="epoch=00-step=680000-loss=0.510.ckpt"
#step=680000
#ckpt="epoch=00-step=270000-loss=0.524.ckpt"
#step=270000

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
#ckpt="epoch=00-step=150000-loss=0.554.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_NoToken_SegEmb_bnPad0
#ckpt="epoch=00-step=150000-loss=0.522.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_promptLoss0.001
#ckpt="epoch=00-step=150000-loss=0.527.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_SilenceBN
#ckpt="epoch=00-step=150000-loss=0.524.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_KeepText_SegEmb_bnBOSEOS_bnPad0
#ckpt="epoch=00-step=150000-loss=0.513.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_QKNormHead
#ckpt="epoch=00-step=150000-loss=0.539.ckpt"
#step=150000
#ckpt="epoch=00-step=300000-loss=0.521.ckpt"
#step=300000
#ckpt="epoch=00-step=690000-loss=0.512.ckpt"
#step=690000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_QKNormHead_NoOverlap
#ckpt="epoch=00-step=150000-loss=0.537.ckpt"
#step=150000
#ckpt="epoch=00-step=800000-loss=0.527.ckpt"
#step=800000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_QKNormChannel
#ckpt="epoch=00-step=150000-loss=0.533.ckpt"
#step=150000
#ckpt="epoch=00-step=290000-loss=0.524.ckpt"
#step=290000
#ckpt="epoch=00-step=690000-loss=0.499.ckpt"
#step=690000


lang=zh
syn_lang=zh
nfe=25
cfg=4
sampler=ddim
length_factor=1 #zh2en: 0.8, en2zh: 1.25
prompt_mode=post

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/openai_demo/en2zh.lst
#out_dir=$out_dir/$exp/openai_demo/$step/${sampler}_${nfe}steps_CFG${cfg}_${prompt_mode}_NewRule

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/meta.lst
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_length${length_factor}_${prompt_mode}
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/debug

#meta_lst=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_8.0/meta_${syn_lang}_cmos.txt
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
#out_dir=$out_dir/$exp/icl_testset_demo_8.0_${syn_lang}/$step/debug

meta_lst=/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_3.0/meta_zh_4x.txt
out_dir=$out_dir/$exp/icl_testset_3.0_${syn_lang}_4x/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

#meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/zh/meta.lst.200.aug.setting2
#out_dir=$out_dir/$exp/repeat_test_${syn_lang}_4x/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

#meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/zh/meta.lst.200.aug.raokouling
#out_dir=$out_dir/$exp/raokouling_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/cn_hardcase/meta.lst
#out_dir=$out_dir/$exp/hard_prompt_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/cn_hardcase/zbs.lst
#out_dir=$out_dir/$exp/zbs/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icassp_demo/meta.lst
#out_dir=$out_dir/$exp/icassp_demo/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
mkdir -p $out_dir

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

#bash recipes/voicebox/scripts/analysis.sh $meta_lst $out_dir $syn_lang
#exit

#bash launch.sh predict \
python3 samantha/main.py predict \
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
	--pl_module.use_adaptive_length True \
	--pl_module.random_seed True

#bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $syn_lang
bash recipes/voicebox/scripts/analysis.sh $meta_lst $out_dir $syn_lang

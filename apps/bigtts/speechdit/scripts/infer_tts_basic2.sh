set -e

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export ARNOLD_WORKER_GPU=4

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_FixPromptBN_transfered
#ckpt="epoch=00-step=150000-loss=0.535.ckpt"
#step=150000
#ckpt="epoch=00-step=290000-loss=0.534.ckpt"
#step=290000
#ckpt="epoch=00-step=300000-loss=0.552.ckpt"
#step=300000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_FixPromptBN_transfered_attn2
#ckpt="epoch=00-step=300000-loss=0.535.ckpt"
#step=300000
#ckpt="epoch=00-step=450000-loss=0.536.ckpt"
#step=450000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_EqualPromptBN_PDropCtx0.5
#ckpt="epoch=00-step=820000-loss=0.522.ckpt"
#step=820000
#ckpt="epoch=00-step=300000-loss=0.537.ckpt"
#step=300000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_PDropCtx0.5_logitnorm
#ckpt="epoch=00-step=80000-loss=0.460.ckpt"
#step=80000
#ckpt="epoch=00-step=300000-loss=0.441.ckpt"
#step=300000

#exp=DIT_1.3B_64H800_lr8e-5_setting4_CFG0.1_QKNormHead_PDropCtx0.4
#ckpt="epoch=00-step=840000-loss=0.536.ckpt"
#step=840000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_PDropCtx0.5_logitnorm_pseudohuber
#ckpt="epoch=00-step=300000-loss=0.045.ckpt"
#step=300000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_PDropCtx0.5_pseudohuber1
#ckpt="epoch=00-step=300000-loss=0.207.ckpt"
#step=300000

#exp=CD_350MDIT_EMA0.999_lr8e-6_guide4_N50_Huber1
#ckpt="epoch=00-step=220000-loss=0.005.ckpt"
#step=220000

#exp=CD_350MDIT_EMA0.97_lr8e-6_guide4_N50_Huber1
#ckpt="epoch=00-step=220000-loss=0.006.ckpt"
#step=220000

#exp=CD_350MDIT_EMA0.99995_lr8e-6_guide4_N50_Huber1
#ckpt="epoch=00-step=280000-loss=0.006.ckpt"
#step=280000

#exp=CD_350MDIT_EMA0.99995_lr8e-6_guide4_N100_Huber1
#ckpt="epoch=00-step=160000-loss=0.002.ckpt"
#step=160000

#exp=CD_350MDIT_EMA0.99995_lr8e-6_guide3to5_N50_Huber1
#ckpt="epoch=00-step=90000-loss=0.006.ckpt"
#step=90000

#exp=AD_verify
#ckpt="epoch=00-step=10000-loss=0.000.ckpt"
#step=10000

#exp=test
#ckpt="epoch=00-step=10000-loss=0.000.ckpt"
#step=10000

lang=en
syn_lang=en
nfe=10
cfg=1
sampler=consistency
length_factor=1 #zh2en: 0.8, en2zh: 1.25
prompt_mode=post

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/meta.lst
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}_logSNR_order2_multi
out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
alignment_path=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/alignment

#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}_logSNR

#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_demo_3.0/meta_${lang}2${syn_lang}.txt
#out_dir=$out_dir/$exp/icl_testset_demo_3.0_${lang}2${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}_Global
mkdir -p $out_dir
mkdir -p $alignment_path

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/$exp/checkpoints/$ckpt

#bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $syn_lang
#exit

#python3 samantha/main.py predict \
bash launch.sh predict \
	-c apps/bigtts/speechdit/conf/infer_tts.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--predict_dataset.syn_lang $syn_lang \
	--predict_dataset.alignment_path $alignment_path \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.length_factor $length_factor \
	--pl_module.use_adaptive_length True \
	--pl_module.only_use_global_prompt False \
	--pl_module.training_type advdistill \
	#--pl_module.training_type pretrain \
	#--pl_module.training_type consistencydistill \

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $syn_lang

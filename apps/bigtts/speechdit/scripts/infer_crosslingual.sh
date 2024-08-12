set -e

cd /mnt/bn/lxdnas-lq/jacobdunn/cla/codes/samantha_dit/
#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

# 6-lang
# exp=DIT_300M_4H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_EqualPromptBN_PDropCtx0.5
# ckpt="epoch=00-step=820000-loss=0.540.ckpt"
# step=820000


# 8-lang
exp=DIT_300M_8A100_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_EqualPromptBN_PDropCtx1.0_frde_0617
ckpt="epoch=00-step=800000-loss=0.544.ckpt"
step=800000


lang=${SRC_LANG}
syn_lang=${TGT_LANG}
nfe=25
cfg=4
sampler=ddim
prompt_mode=post

out_dir=${OUTPUT_DIR}
meta_lst=${METALST}
mkdir -p $out_dir

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jacobdunn/speechdit/logdir/$exp/checkpoints/$ckpt

bash launch.sh predict \
	-c apps/bigtts/speechdit/conf/infer_tts.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--run_opts.infer_mode "cla" \
	--predict_dataset.lang $lang \
	--predict_dataset.syn_lang $syn_lang \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.training_type pretrain \
	--pl_module.only_use_global_prompt True 

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $syn_lang

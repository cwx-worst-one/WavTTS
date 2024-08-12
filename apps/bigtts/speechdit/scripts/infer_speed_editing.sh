set -e

#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_EqualPromptBN_PDropCtx0.5
ckpt="epoch=00-step=820000-loss=0.522.ckpt"
step=820000

#exp=DIT_750M_32H800_lr1e-4_setting4_CFG0.1_QKNormHead_PDropCtx0.4
#ckpt="epoch=00-step=600000-loss=0.530.ckpt"
#step=600000

#exp=DIT_1.3B_64H800_lr8e-5_setting4_CFG0.1_QKNormHead_PDropCtx0.4
#ckpt="epoch=00-step=920000-loss=0.533.ckpt"
#step=920000

lang=zh
syn_lang=${lang}
nfe=20
cfg=4
sampler=ddim
length_factor=2
prompt_mode=post

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output

meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_tr/meta_${syn_lang}_speed.txt
out_dir=$out_dir/$exp/icl_testset_tr_speed_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_${prompt_mode}_global_speed${length_factor}
alignment_path=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_tr/${syn_lang}/alignment

mkdir -p $out_dir
mkdir -p $alignment_path

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/$exp/checkpoints/$ckpt

python3 samantha/main.py predict \
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
	--pl_module.only_use_global_prompt True \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.training_type pretrain \
	--pl_module.speed_editing_mode True

bash apps/bigtts/speechdit/scripts/eval.sh $meta_lst $out_dir $syn_lang

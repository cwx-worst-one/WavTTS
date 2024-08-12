set -e

#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

exp1=DIT_750M_32H800_lr1e-4_setting4_QKNormHead_PDropCtx0.4
ckpt1="epoch=00-step=1000000-loss=0.517.ckpt"
step1=1000000

exp2=DIT_150M_8GPU_lr1e-4_setting4_QKNormHead_Uncond
ckpt2="epoch=00-step=260000-loss=0.602.ckpt"
step2=260000

lang=en
syn_lang=en
nfe=25
cfg=3.5
sampler=ddim
length_factor=1 #zh2en: 0.8, en2zh: 1.25
prompt_mode=post

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/meta.lst
out_dir=$out_dir/${exp1}_${exp2}/icl_testset_2.0_${syn_lang}/${step1}_${step2}/${sampler}_${nfe}steps_CFG${cfg}_AdaLength_${prompt_mode}
mkdir -p $out_dir

diffusion_ckpt_path1=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/$exp1/checkpoints/$ckpt1
diffusion_ckpt_path2=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/$exp2/checkpoints/$ckpt2

#python3 samantha/main.py predict \
bash launch.sh predict \
	-c apps/bigtts/speechdit/conf/infer_tts_dual.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path1\
    --run_opts.uncond_diffusion_ckpt_path $diffusion_ckpt_path2\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--predict_dataset.syn_lang $syn_lang \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.length_factor $length_factor \
	--pl_module.use_adaptive_length True \
	--pl_module.training_type pretrain

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $syn_lang

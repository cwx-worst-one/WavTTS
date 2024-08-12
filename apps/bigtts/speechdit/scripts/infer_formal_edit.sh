#export CUDA_VISIBLE_DEVICES=0 #export ARNOLD_WORKER_GPU=1

#exp=DIT_1.3B_64H800_lr8e-5_setting4_CFG0.1_QKNormHead_PDropCtx0.4
#ckpt="epoch=00-step=840000-loss=0.536.ckpt"
#step=840000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_QKNormHead_NewMask_EqualPromptBN_PDropCtx0.5
#ckpt="epoch=00-step=820000-loss=0.522.ckpt"
#step=820000

# exp=CD_350MDIT_EMA0.99_lr8e-6_guide4_N25_L2SSIMFix_Px0.1_CFG0.1_PDropBN0.4
# ckpt="epoch=00-step=50000-loss=0.229.ckpt"
# step=50000

exp=DIT_300M_24H800_lr1e-4_setting4_CFG0.1_QKNormHead_PDropCtx0.4_L1
#ckpt="epoch=00-step=1220000-loss=0.571.ckpt"
#step=1220000
ckpt="epoch=00-step=1380000-loss=0.568.ckpt"
step=1380000

syn_lang=zh
nfe=20
cfg=4
sampler=ddim

 out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
# meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/edit_test/ZH_edit_meta.txt
#meta_lst=/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/meta10_demo_ZH_EN.lst
meta_lst=/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/dit_tesetset_0729_meta_split.lst
#meta_lst=/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/dit_tesetset_0729_meta_split_EN.lst
#meta_lst=/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/dit_vs_voco_0801_meta_ZH.lst
#meta_lst=/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/dit_vs_voco_0801_meta_ZH.lst
#out_dir=/mnt/bn/jdy-lq-2/caorong/bigtts-nar/output
out_dir=$out_dir/$exp/dit_edit_0729/$step/Edit_${sampler}_${nfe}steps_CFG${cfg}_FullPrompt
mkdir -p $out_dir

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/speechdit/logdir/$exp/checkpoints/$ckpt

python3 samantha/main.py predict \
	-c apps/bigtts/speechdit/conf/infer_formal_edit.yaml \
	--predict_dataset.meta_lst $meta_lst  \
	--predict_dataset.lang $syn_lang \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--pl_module.cfg_w ${cfg} \
	--pl_module.use_all_for_prompt True


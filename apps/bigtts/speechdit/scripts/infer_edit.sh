#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

#exp=DIT_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
#ckpt="epoch=00-step=680000-loss=0.510.ckpt"
#step=680000

#exp=DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap
#ckpt="epoch=00-step=300000-loss=0.509.ckpt"
#ckpt="epoch=00-step=430000-loss=0.527.ckpt"
#ckpt="last.ckpt"
#step=430000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
#ckpt="epoch=00-step=150000-loss=0.554.ckpt"
#step=150000

#exp=DIT_300M_8H800_lr1e-4_setting0_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0_QKNormHead
#ckpt="epoch=00-step=150000-loss=0.539.ckpt"
#step=150000

exp=DIT_1.3B_64H800_lr8e-5_setting3a_CFG0.1_QKNormHead_NoOverlap_NewMask
#ckpt="epoch=00-step=150000-loss=0.542.ckpt"
#step=150000
#ckpt="epoch=00-step=300000-loss=0.534.ckpt"
#step=300000
ckpt="epoch=00-step=500000-loss=0.520.ckpt"
step=500000

syn_lang=en
nfe=25
cfg=4
sampler=ddim
mask_ratio=0.4

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output

meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/Edit_${sampler}_${nfe}steps_CFG${cfg}_ratio${mask_ratio}_CropPrompt
#out_dir=$out_dir/$exp/icl_testset_2.0_${syn_lang}/$step/Edit_${sampler}_${nfe}steps_CFG${cfg}_ratio${mask_ratio}
mkdir -p $out_dir

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

#bash launch.sh predict \
python3 samantha/main.py predict \
	-c recipes/voicebox/conf/infer_edit.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $syn_lang \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.mask_part ${mask_ratio} \
	--pl_module.use_all_for_prompt False

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $syn_lang

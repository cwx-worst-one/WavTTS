set -e

# example:
# exp=PrefixLDM4_300M_48A100_setting2_40hzWVAE_UMMv02_textDrop0.25_Norm2
# ckpt="epoch=00-step=50000-loss=0.52.ckpt"
# step=50000

exp=migrate_v3
ckpt="epoch=00-step=60000-loss=0.54.ckpt"
step=60000

# exp=migrate_v3
# ckpt="epoch=00-step=200000-loss=0.53.ckpt"
# step=200000

# lang=zh
lang=en

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
out_dir=/opt/tiger/samantha/output  # change to your path, or comment it, must use absolute path
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps_TextCFG4
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps
mkdir -p $out_dir

# for umm/v0.2.2.ckpt, you should use specific version of umm code.
# you can replace umm version like this: 
#     git checkout speech-diffusion -- recipes/umm_022
#     mv recipes/umm recipes/umm_old
#     mv recipes/umm_022 recipes/umm

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.2.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.6.2.ckpt

# diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt
diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/xieshuangyi/voicebox/logdir/$exp/checkpoints/$ckpt

bash launch.sh predict \
	-c apps/bigtts/umm/diffusion/conf/infer_reconstruction_40hzMel.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.umm_frame_rate 40 \
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
	--pl_module.text_cfg_w 4
	#--pl_module.save_prompt True
	#--pl_module.umm_codebook_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.codebook

bash apps/bigtts/umm/diffusion/scripts/eval.sh $meta_lst $out_dir $lang

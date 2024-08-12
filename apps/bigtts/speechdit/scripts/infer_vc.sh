#export CUDA_VISIBLE_DEVICES=0
#export ARNOLD_WORKER_GPU=1

exp=DIT_1.3B_64H800_lr5e-5_setting3a_CFG0.1_DropText_SegEmb_bnBOSEOS_bnPad0
ckpt="epoch=00-step=570000-loss=0.509.ckpt"
step=570000

lang=zh
syn_lang=zh
nfe=25
cfg=4
sampler=ddim
prompt_mode=post
backward_t=0.95

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${syn_lang}/non_para_reconstruct_meta.lst
out_dir=$out_dir/$exp/nonpara_icl_testset_2.0_${syn_lang}/$step/${sampler}_${nfe}steps_CFG${cfg}_backT${backward_t}_${prompt_mode}
mkdir -p $out_dir


rm -r recipes/umm
ln -s -r recipes/umm_062 recipes/umm

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt


bash launch.sh predict \
	-c recipes/voicebox/conf/infer_vc.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.mel_frame_rate 40 \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
	--predict_dataset.syn_lang $syn_lang \
    --run_opts.infer_type diffusion-vocoder \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe ${nfe} \
	--pl_module.diffusion_sampler ${sampler} \
	--pl_module.use_wvae_vocoder True \
	--bn_config.wvae_encoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_encoder_z1_%d.pt \
	--bn_config.wvae_decoder_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/wvae_3.1/wavevae_decoder_z1_%d.pt \
	--bn_config.bn_norm_std 2 \
	--bn_config.bn_padding 0 \
	--pl_module.cfg_w ${cfg} \
	--pl_module.cfg_text True \
	--pl_module.norm_cfg False \
	--pl_module.prompt_mode $prompt_mode \
	--pl_module.backward_t $backward_t

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $syn_lang

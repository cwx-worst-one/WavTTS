
exp=DualCondNet2_small_250M_fp32_32H800_token8000_setting1_40hzMel_UMMv02
ckpt="epoch=00-step=240000-loss=0.13.ckpt"
step=240000

#exp=PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02_MaskWOAlignment
#ckpt="epoch=00-step=330000-loss=0.10.ckpt"
#step=330000


lang=en
out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta_subset1.lst
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/sampler_test/10step_dpmsolver_cosine_timeuniform_order2_multistep
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/sampler_test/10step_ddim
mkdir -p $out_dir

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt

vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

bash launch.sh predict \
	-c recipes/voicebox/conf/infer_reconstruction_40hzMel.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 40 \
    --run_opts.mel_frame_rate 40 \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
    --run_opts.infer_type crop_prompt \
	--pl_module.umm_type UMM \
	--pl_module.diffusion_precision fp32 \
	--pl_module.diffusion_nfe 10 \
	--pl_module.diffusion_sampler ddim \
	--mel_config.mel_norm_mean -2.5 \
	--mel_config.mel_norm_std 6 \
	--pl_module.save_prompt True
	#--pl_module.umm_codebook_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.codebook \

#bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

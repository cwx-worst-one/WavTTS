#CUDA_LAUNCH_BLOCKING=1
#ARNOLD_WORKER_GPU=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARNOLD_WORKER_GPU=4

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_80hzMel
#ckpt="epoch=00-step=145000-loss=0.12.ckpt"
#step=145000

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_80hzMel_UMMv0
#ckpt="epoch=00-step=80000-loss=0.12.ckpt"
#step=80000
#ckpt="epoch=00-step=240000-loss=0.13.ckpt"
#step=240000

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_80hzMel_UMMv01
#ckpt="epoch=00-step=80000-loss=0.13.ckpt"
#step=80000

#exp=LDM3_400M_16A100_18wEN_10wCN_40hzMel_UMMvMinus1
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_80hzMel_UMMv02
#ckpt="epoch=00-step=80000-loss=0.12.ckpt"
#step=80000

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_40hzMel_UMMv02
#ckpt="epoch=00-step=160000-loss=0.11.ckpt"
#step=160000

#exp=voicebox_LDM3_400M_16A100_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=160000-loss=0.11.ckpt"
#step=160000

#exp=PrefixLDM3_400M_16A100_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=240000-loss=0.11.ckpt"
#step=240000

#exp=PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02_rerun
#ckpt="epoch=00-step=80000-loss=0.12.ckpt"
#step=80000

#exp=PrefixLDM3a_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02Vector
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=PrefixLDM3a_ummdropout02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02Vector
#ckpt="epoch=00-step=160000-loss=0.11.ckpt"
#step=160000

#exp=PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv02_MaskWOAlignment
#ckpt="epoch=00-step=330000-loss=0.10.ckpt"
#step=330000

#exp=PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_UMMv02_MaskWOAlignment_Mask1
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

exp=PrefixLDM4_300M_48A100_token30000_setting2_40hzMel_UMMv02_Fixed
ckpt="epoch=00-step=550000-loss=0.13.ckpt"
step=550000

#exp=DualCondNet2_small_250M_fp32_32H800_token8000_setting1_40hzMel_UMMv02
#ckpt="epoch=00-step=400000-loss=0.13.ckpt"
#step=400000

#exp=DualCondNet2_small_250M_16A100_token25000_setting1_40hzMel_UMMv02
#ckpt="epoch=00-step=80000-loss=0.14.ckpt"
#step=80000

#exp=PrefixLDM3_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=200000-loss=0.11.ckpt"
#step=200000

#exp=PrefixLDM3_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv03
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=PrefixLDM3_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv03
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=PrefixLDM3a_ummdrop02_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv03_MaskWOAlignment
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=PrefixLDM3a_400M_16A100_18wEN_10wCN_40hzMel_NormWav_UMMv03Vector
#ckpt="epoch=00-step=80000-loss=0.10.ckpt"
#step=80000

#exp=DualCondNet2_250M_16A800_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=160000-loss=0.11.ckpt"
#step=160000

#exp=DualCondNet2_250M_16mixed_16A800_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=DualCondNet2_250M_16mixed_16H800_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=80000-loss=0.12.ckpt"
#step=80000

#exp=PrefixLDM4_300M_fp16_16H800_token30000_setting1_40hzMel_UMMv02
#ckpt="epoch=00-step=65000-loss=0.14.ckpt"
#step=65000

#exp=PrefixLDM4_300M_16H800_token30000_setting1_40hzMel_UMMv02
#ckpt="epoch=00-step=220000-loss=0.12.ckpt"
#step=220000

lang=zh
out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
#meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_${lang}_1000/reconstruct_meta.lst
#out_dir=$out_dir/$exp/icl_testset_${lang}_1000/$step/copy_prompt
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/DDIM_10steps
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/dpm_10steps_concat
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/10step_dpmsolver_cosine_timeuniform_order2_multistep
#out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/dpm_order2_cosine_timeUniform_singlestep_5step
mkdir -p $out_dir

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/USM/V1.4.1.ckpt

#vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/80hz_ibigvgan/ibigvgan80_510k.pt
#vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ibigvgan/ibigvgan_325k.pt
vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

#bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
#exit

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
    --run_opts.infer_type diffusion-vocoder \
	--pl_module.umm_type UMM \
	--pl_module.diffusion_precision bf16 \
	--pl_module.diffusion_nfe 10 \
	--pl_module.diffusion_sampler ddim \
	--mel_config.mel_norm_mean -2.5 \
	--mel_config.mel_norm_std 6 \
	#--pl_module.save_prompt True
	#--pl_module.umm_codebook_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.codebook \

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

#exp=PrefixLDM3a_400M_16H800_18wEN_10wCN_40hzMel_NormWav_UMMv141_MaskWOAlignment
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

#exp=PrefixLDM3a_400M_16A800_18wEN_10wCN_40hzMel_NormWav_UMMv141_MaskWOAlignment
#ckpt="epoch=00-step=80000-loss=0.12.ckpt"
#step=80000

exp=PrefixLDM3a_ummdropout02_400M_16A800_18wEN_10wCN_40hzMel_NormWav_UMMv141_MaskWOAlignment
ckpt="epoch=00-step=80000-loss=0.12.ckpt"
step=80000

#exp=PrefixLDM3a_400M_16A800_18wEN_10wCN_40hzMel_NormWav_UMMv1.3_MaskWOAlignment
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

lang=en
out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/DDIM_10steps_ununiform
mkdir -p $out_dir

#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.ckpt
umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/USM/V1.4.1.ckpt
#umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/USM/V1.3.ckpt

#vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/80hz_ibigvgan/ibigvgan80_510k.pt
vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ibigvgan/ibigvgan_325k.pt
#vocoder_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/ihifigan/500k.pt

diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/jiadongya/voicebox/logdir/$exp/checkpoints/$ckpt

bash launch.sh predict \
	-c recipes/voicebox/conf/infer_reconstruction_40hzMel.yaml \
	--run_opts.meta_lst $meta_lst  \
	--run_opts.output_dir $out_dir \
    --run_opts.diffusion_ckpt_path $diffusion_ckpt_path\
    --run_opts.umm_ckpt_path $umm_ckpt_path \
    --run_opts.vocoder_ckpt_path $vocoder_ckpt_path \
    --run_opts.umm_frame_rate 25 \
    --run_opts.mel_frame_rate 40 \
	--run_opts.seed 1996 \
	--run_opts.num_workers 1 \
	--predict_dataset.lang $lang \
    --run_opts.infer_type copy_prompt \
	--pl_module.umm_type USM \
	--pl_module.diffusion_precision bf16 \
	#--pl_module.umm_codebook_path /mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.codebook \


bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

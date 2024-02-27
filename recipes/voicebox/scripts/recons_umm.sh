#exp=DualCondNet2_250M_16mixed_16A800_18wEN_10wCN_40hzMel_UMMv03
#ckpt="epoch=00-step=80000-loss=0.11.ckpt"
#step=80000

exp=DualCondNet2_250M_16mixed_16H800_18wEN_10wCN_40hzMel_UMMv03
ckpt="epoch=00-step=80000-loss=0.12.ckpt"
step=80000


lang=en
out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output
meta_lst=/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
out_dir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/DDIM_10steps_ununiform
mkdir -p $out_dir

umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.3.ckpt

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
    --run_opts.infer_type copy_prompt \
	--pl_module.umm_type UMM \
	--pl_module.diffusion_precision fp16 \
	--pl_module.diffusion_nfe 10 \
	--pl_module.diffusion_sampler ddim \

bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang

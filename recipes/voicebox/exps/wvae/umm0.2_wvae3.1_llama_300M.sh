mode=$1

exp=PrefixLDM4_Wvae31_UMM02_300M_Llama

logdir=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/congjian/voicebox/logdir/
logdir=$logdir/$exp/

#  for debug. ARNOLD_WORKER_GPU=1
if [[ ${mode} == "train" ]];then
    bash launch.sh fit --config recipes/voicebox/conf/PrefixLDM4.yaml \
            --run_opts.data_id 470 \
            --run_opts.log_dir ./logs \
            --run_opts.hdfs_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/congjian/voicebox/logdir/PrefixLDM4_Wvae31_UMM02_300M_Llama \
            --run_opts.log_name umm_diffusion_wvae \
            --run_opts.version PrefixLDM4_Wvae31_UMM02_300M_Llama \
            --run_opts.batch_total_tokens 28000 \
            --run_opts.checkpointing False \
            --run_opts.n_step_save 5000 \
            --trainer.accumulate_grad_batches 1 \
            --scheduler_cls.cycle_steps 500000 \
            --run_opts.num_workers 8 \
            --trainer.log_every_n_steps 100 \
            --pl_module.umm_dropout 0.2 \
            --run_opts.target bn \
            --run_opts.prompt_feature bn \
            --run_opts.ctx_feature bn \
            --model_config.in_channels 64 \
            --model_config.out_channels 64 \
            --model_config.prompt_mel_dim 64 \
            --model_config.local_cond_dim 256 \
            --model_config.time_embed_dim 256 \
            --model_config.encoder_dim 1024 \
            --model_config.encoder_n_layers 24 \
            --model_config.encoder_n_heads 16 \
            --ckpt_path hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/congjian/voicebox/logdir/PrefixLDM4_Wvae31_UMM02_300M_Llama/checkpoints/epoch=00-step=20000-loss=0.15.ckpt
fi

if [[ ${mode} == "infer" ]];then
    workdir=$PWD    
    umm_ckpt_path=/mnt/bn/jdy-lq-2/bigtts-nar/pretrain_model/UMM/V0.2.ckpt
    root_out_dir=/mnt/bn/jcong5/workspace4/logs/nar/diffusion/
    diffusion_ckpt_path=$logdir/checkpoints/epoch=00-step=20000-loss=0.15.ckpt
    echo $diffusion_ckpt_path
    step=`echo $diffusion_ckpt_path| grep -o -E 'step=[0-9]+' | cut -d'=' -f2`
    for lang in zh;
    do
        cd $workdir
        meta_lst=/mnt/bn/jcong5/workspace4/bigtts_testset/icl_testset_2.0/${lang}/reconstruct_meta.lst
        out_dir=$root_out_dir/$exp/icl_2.0/$lang/$step/
        mkdir -p $out_dir

        ARNOLD_WORKER_GPU=1  bash launch.sh predict \
            -c recipes/voicebox/conf/infer_reconstruction_40hzMel.yaml \
            --run_opts.meta_lst $meta_lst  \
            --run_opts.output_dir $out_dir \
            --run_opts.diffusion_ckpt_path $diffusion_ckpt_path \
            --run_opts.umm_ckpt_path $umm_ckpt_path \
            --run_opts.umm_frame_rate 40 \
            --run_opts.mel_frame_rate 40 \
            --run_opts.infer_type crop_prompt \
            --run_opts.use_wvae_vocoder True \
            --run_opts.lang $lang \
            --bn_config.wvae_version 2.1

        bash recipes/voicebox/scripts/eval.sh $meta_lst $out_dir $lang
    done
fi
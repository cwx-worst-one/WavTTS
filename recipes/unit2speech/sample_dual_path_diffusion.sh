SAMPLE_FLAGS="--batch_size 1 --diffusion_steps 10 --classifier_free_guidance 1 --use_ema True --test_prompt False"

autoencoder=autoencoder
autoencoder_path=./logs_${autoencoder}_en_16dim/checkpoints/1350k_ckpt.pyt
autoencoder_path=./logs_${autoencoder}_en_16dim/checkpoints/500k_ckpt.pyt
autoencoder_config=./logs_${autoencoder}_en_16dim/config.yaml
autoencoder_path=./logs_${autoencoder}_en_8dim/checkpoints/850k_ckpt.pyt
autoencoder_path=./logs_${autoencoder}_en_8dim/checkpoints/1000k_ckpt.pyt
autoencoder_config=./logs_${autoencoder}_en_8dim/config.yaml
#autoencoder_path=./logs_${autoencoder}_en_16dim_with_zeros/checkpoints/1000k_ckpt.pyt
#autoencoder_config=./logs_${autoencoder}_en_16dim_with_zeros/config.yaml

export OPENAI_LOGDIR=./logs_20230604_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10
#logs_20230603_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-192_rmsnorm_1400h

MODEL_FLAGS=`cat $OPENAI_LOGDIR/settings`

model_ckpt_dir=$OPENAI_LOGDIR/ckpt100000
rm -rf $model_ckpt_dir/samples
wav_list=metadata/librilight_test_clean_4-10.list
#wav_list=metadata/all_english_4-10.list
#wav_list=metadata/en_1400h.list

accelerate launch --multi_gpu \
	--config_file recipes/unit2speech/acc_ddp_conf.yaml \
	--num_processes ${ARNOLD_WORKER_GPU} \
	--main_process_port 13668 \
	recipes/unit2speech/sample_dual_path_diffusion.py \
	--autoencoder $autoencoder \
	--autoencoder_path $autoencoder_path \
	--autoencoder_config $autoencoder_config \
	--wav_list $wav_list \
	--model_ckpt_dir $model_ckpt_dir \
	$MODEL_FLAGS $SAMPLE_FLAGS

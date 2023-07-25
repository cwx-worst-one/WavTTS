# bash install_requirements.sh
# export CUDA_VISIBLE_DEVICES=0,1
# ARNOLD_WORKER_NUM=1

# hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/logs_autoencoder_en_8dim
# hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/metadata/en_1400h.list

export BYTED_TORCH_C10D_LOG_LEVEL=ERROR
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0

export MASTER_PORT=`echo $METIS_WORKER_0_PORT | awk -F, '{print $1}'`
export MASTER_ADDR=${METIS_WORKER_0_HOST}
export NODE_RANK=${ARNOLD_ID}
export WORLD_SIZE=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
export ARNOLD_OUTPUT=${ARNOLD_OUTPUT}

echo "TOTAL WORKERS    :   ${ARNOLD_WORKER_NUM}"
echo "CURRENT WORKER ID:   ${ARNOLD_ID}"
echo "#GPU PER-WORKER  :   ${ARNOLD_WORKER_GPU}"
echo "MASTER NODE IP   :   ${METIS_WORKER_0_HOST}"
echo "MASTER NODE PORT :   ${MASTER_PORT}"
echo "WORLD SIZE       :   ${WORLD_SIZE}"
echo "ARNOLD OUTPUT    :   ${ARNOLD_OUTPUT}"

export OMP_NUM_THREADS=8
export BYTED_TORCH_BYTECCL=O3

GPU_TYPE=`nvidia-smi -q | grep "Product Name" | head -n1 | awk '{print $NF}'`

#MODEL_FLAGS="--num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --attention_resolutions 32,16,8 --image_size 256"
MODEL_FLAGS="--input_dim 8 --feature_dim 512 --num_blocks 12 --dropout 0 --segment_size 64 --segment_stride 32"
DIFFUSION_FLAGS="--diffusion_steps 0 --num_chunks 1 --chunk_length 2500"
TRAIN_FLAGS="--lr 3e-4 --batch_size 3 --log_interval 10 --save_interval 2500 --use_fp16 False"

network=dprtnet
fine_model=sru
coarse_model=roformer
MODEL_FLAGS="--intra_seq2seq ${fine_model} --inter_seq2seq ${coarse_model} ${MODEL_FLAGS}"
cond_embedder=ssl
autoencoder=autoencoder
autoencoder_path=./logs_${autoencoder}_en_8dim/checkpoints/1000k_ckpt.pyt
autoencoder_config=./logs_${autoencoder}_en_8dim/config.yaml
#autoencoder_path=./logs_${autoencoder}_en_16dim/checkpoints/1000k_ckpt.pyt
#autoencoder_config=./logs_${autoencoder}_en_16dim/config.yaml
# wav_list=metadata/all_english.list
# wav_list=metadata/all_english_4-10.list
wav_list=./en_1400h.list
#wav_list=metadata/hifi_speech.list
#wav_list=metadata/libri_light.list
#wav_list=metadata/en_1400h.list

export OPENAI_LOGDIR=./logs_20230604_${fine_model}-${coarse_model}_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10

mkdir -p $OPENAI_LOGDIR
cp diffusion/dprtnet.py $OPENAI_LOGDIR
cp train_dual_path_diffusion.sh $OPENAI_LOGDIR
echo "$MODEL_FLAGS $DIFFUSION_FLAGS" > $OPENAI_LOGDIR/settings

accelerate launch \
	--multi_gpu \
	--mixed_precision no \
	--dynamo_backend no \
	--main_process_ip ${MASTER_ADDR} \
	--main_process_port ${MASTER_PORT} \
	--num_machines ${ARNOLD_WORKER_NUM} \
	--num_processes ${WORLD_SIZE} \
	--machine_rank ${NODE_RANK} \
	recipes/unit2speech/train_dual_path_diffusion.py \
	--cond_embedder $cond_embedder \
	--autoencoder $autoencoder \
	--autoencoder_path $autoencoder_path \
	--autoencoder_config $autoencoder_config \
	--wav_list $wav_list \
	$TRAIN_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS # \
	# --resume_ckpt_dir logs_20230603_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-192_rmsnorm_1400h/ckpt205000
	#--config_file acc_ddp_conf.yaml \
	#--main_process_port 13668 \
	#--resume_ckpt_dir logs_20230601_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_1400h/ckpt015000
	#--resume_ckpt_dir logs_20230601_sru-roformer_8in-12x512_64-32segs_aug-prompt-tokens_cross-attn_mse-loss_1400h/ckpt005000
	#--resume_ckpt_dir ./logs_20230526_sru-roformer_ssl_80-20segs_no_prompt/ckpt035000
	#--resume_ckpt_dir logs_20230414_dprtnet-sru-roformer_w2v-bert-mulan_autoencoder-4dim/ckpt475000
	#--resume_ckpt_dir logs_20230329_dprtnet-lstm-lstm_mulan-audio_autoencoder/ckpt080000
	#--resume_ckpt_dir ./logs_20230314_dptnet_mulan-audio_autoencoder/ckpt010000
	#--resume_ckpt_dir ./logs_20230307_t5_autoencoder/ckpt090000
	#--audio_only_wav_list $audio_only_wav_list
	#--context_dim 768 \

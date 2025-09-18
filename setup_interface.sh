#!/bin/bash

# Get parameters from command line
semantic_ckpt=$1

export RDMAV_FORK_SAFE=1

# Environment setup
export LITE_USE_PL_MODULE=1
pip3 install -U byted-lsdp
pip3 install -U byted-omnistore

moe_model=true
LOCAL_DIR="model"

#######################
# Download and convert model
#######################
echo "Downloading and converting model..."

python3 apps/mariana/tasks/audio/merge_omnistore_ckpt.py --omnistore_ckpt_path=$semantic_ckpt \
    --output_path=${LOCAL_DIR} \
    --model_only \
    --framework="lsdp"

#######################
# Split weights
#######################

LOCAL_MODEL=${LOCAL_DIR}/model.pt

echo "Splitting weights..."
python3 ./apps/bigmusic/mariana_tasks/utils/split_weight.py \
    --ckpt "${LOCAL_MODEL}" \
    --text_llm_ckpt ./llm.pt \
    --audio_ckpt ./emb.pt \
    --moe_model ${moe_model}

#######################
# Install dependencies
#######################
echo "Installing dependencies..."
pip3 install websockets emoji nest_asyncio sortedcontainers --no-deps
pip3 install confusables prettytable pretty_midi thop mido --no-deps
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/ --no-deps
pip3 install -U bytedance.easycycle -i https://bytedpypi.byted.org/simple --no-deps
pip3 uninstall -y triton pydantic || true
pip3 install --upgrade attrs

mkdir -p /opt/tiger/tokenizer/

hdfs dfs get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/bbpe155k-v6.4.3-ml.pret /opt/tiger/

#export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/bigspeech_workflow_autorun/_v/Diffusion/unified_diffusion/h800_ds5085_32xH800_25hzUMM_49hzSS_streaming_chunk392_1004_v2_133/checkpoints/epoch=00-step=570000-loss=0.70.ckpt
export vocoder_version="sacodec_umm_env"
export vocoder_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/andrew.shaw/sacodec_v5/sacodec_0408_44k_50h_128d_h1536_data8424/checkpoints/sacodec_checkpoint_epoch=0_step=400000_val_loss=1.1710.ckpt
export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/andrew.shaw/latent2wav/unified_diffusion_sacodec/0414_diffusion_sacodec_data8424_norm0_wvaeft/checkpoints/epoch=00-step=260000-loss=0.63.ckpt
export audio_norm_type="clip"
export diffusion_cfg=1.6
export diffusion_nfe=15
export HF_ENDPOINT=https://hf-mirror.com

#######################
# Launch prediction
#######################
# bash ./launch.sh predict \
#     --config "${config}" \
#     --extra_params.network_cfg "${network_cfg}" \
#     --data.predict_dataset_ids "${predict_dataset_ids}" \
#     --extra_params.output_dir "${out_dir}" \
#     --extra_params.emit_eos_thresh_secs -1 \
#     --extra_params.exclude_eos_thresh_secs -1 \
#     --extra_params.exclude_eos_first_secs 5 \

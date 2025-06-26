
# Environment setup
export LITE_USE_PL_MODULE=1
scm_install seed.speech.lsdp 1.1.0.2

# Configuration variables
HDFS_DIR="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/rui.xia/bigmusic/experiments/ar/2025050212_V5-D7b_H20X64/version_45198953/checkpoints"
STEP='epoch=0-step=6000'
output_dir="./results"
current_time=$(date "+%Y%m%d-%H%M%S%4N")
out_dir="${output_dir}/${current_time}"
LOCAL_DIR="./"
predict_dataset_ids="[10495]"
network_cfg="apps/bigmusic/mariana_tasks/conf/v5_dense_7b.yaml"
moe_model=false


# Download and convert model
echo "Downloading and converting model..."
hdfs dfs -get "${HDFS_DIR}/${STEP}_model_state_dict_dckpt" "${LOCAL_DIR}"

set -x
python3 ./apps/mariana/tasks/audio/merge_lsdp_model.py \
    --source "${LOCAL_DIR}${STEP}_model_state_dict_dckpt" \
    --output "${LOCAL_DIR}${STEP}.pt" \
    --bfloat16
set +x

LOCAL_MODEL="${LOCAL_DIR}${STEP}.pt"

# Split weights
echo "Splitting weights..."
python3 ./apps/bigmusic/mariana_tasks/utils/split_weight.py \
    --ckpt "${LOCAL_MODEL}" \
    --text_llm_ckpt ./llm.pt \
    --audio_ckpt ./emb.pt \
    --moe_model ${moe_model}

# Install dependencies
echo "Installing dependencies..."
pip3 install websockets emoji nest_asyncio sortedcontainers --no-deps
pip3 install confusables prettytable pretty_midi thop mido --no-deps
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/ --no-deps
pip3 install -U bytedance.easycycle -i https://bytedpypi.byted.org/simple --no-deps
pip3 uninstall -y triton pydantic || true
pip3 install --upgrade attrs


bash ./launch.sh predict \
    --config /opt/tiger/samantha/apps/bigmusic/mariana_tasks/conf/v5_inference_new_emb.yaml \
    --extra_params.network_cfg ${network_cfg} \
    --extra_params.output_dir ${out_dir} \
    --extra_params.emit_eos_thresh_secs 5 \
    --extra_params.exclude_eos_thresh_secs 1 \
    --extra_params.exclude_eos_first_secs 20 \
    --trainer.limit_predict_batches 5

echo "Script completed."
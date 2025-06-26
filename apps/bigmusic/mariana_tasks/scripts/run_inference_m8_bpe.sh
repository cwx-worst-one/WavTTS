export diffusion_ckpt_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/bigspeech_workflow_autorun/_v/Diffusion/unified_diffusion/h800_ds5085_32xH800_25hzUMM_49hzSS_streaming_chunk392_1004_v2_133/checkpoints/epoch=00-step=570000-loss=0.70.ckpt
# Environment setup
export LITE_USE_PL_MODULE=1
scm_install seed.speech.lsdp 1.1.0.2

# Configuration variables
HDFS_DIR="hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/ziqian/logs/m8/p004_222k/"
output_dir="./results"
current_time=$(date "+%Y%m%d-%H%M%S%4N")
out_dir="${output_dir}/${current_time}"
LOCAL_DIR="./"
predict_dataset_ids="[12498]"

echo "Downloading and converting model..."
hdfs dfs -get "${HDFS_DIR}/llm.pt" "${LOCAL_DIR}"
hdfs dfs -get "${HDFS_DIR}/emb.pt" "${LOCAL_DIR}"

# ========================================================

echo "Installing dependencies..."
pip3 install websockets emoji nest_asyncio sortedcontainers --no-deps
pip3 install confusables prettytable pretty_midi thop mido --no-deps
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/ --no-deps
pip3 uninstall -y triton pydantic || true

hdfs dfs get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhangshuo/bigmusic/assets/bbpe155k-v6.4.3-ml.pret /home/code

bash ./launch.sh predict \
 --config apps/bigmusic/mariana_tasks/conf/v5_inference_simple_bpe_v5_raw_cfg.yaml \
 --extra_params.network_cfg apps/bigmusic/mariana_tasks/conf/v5_m8_680m_bpe_simple.yaml \
 --extra_params.emit_eos_thresh_secs 50 \
 --extra_params.exclude_eos_thresh_secs 10 \
 --extra_params.exclude_eos_first_secs 20 \
 --data.predict_dataset_ids ${predict_dataset_ids} \
 --extra_params.output_dir ${out_dir} \
 --extra_params.use_controller_cfg False \
 --trainer.limit_predict_batches 1

hdfs dfs put ${out_dir} hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/ziqian/outputs/ci

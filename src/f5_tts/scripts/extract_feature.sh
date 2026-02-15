export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

nproc_per_node=4

SSL_MODEL_PATH=facebook/hubert-large-ll60k # or local path

OUTPUT_ROOT=./hubert_feature/ # output directory
mkdir -p $OUTPUT_DIR

AUDIO_ROOT_DIR=/path/to/LibriTTS

for subset in train-clean-100 train-clean-360 train-other-500;do
    AUDIO_ROOT=$AUDIO_ROOT_DIR/$subset
    OUTPUT_DIR=$OUTPUT_ROOT/$subset
    echo "Process $AUDIO_ROOT"
    torchrun \
    --nnodes 1 \
    --nproc_per_node $nproc_per_node \
    src/f5_tts/scripts/extract_ssl_features.py \
    --audio-root-dir $AUDIO_ROOT \
    --output-dir     $OUTPUT_DIR \
    --ssl-model-path $SSL_MODEL_PATH \
    --file-extension .wav \
    --layer-index -1 \
    --num-workers 16 \
    --log-every 500
done
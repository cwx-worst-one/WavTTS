# Start at samantha/
# sh recipes/unit2speech/cruise_train_dual_path_diffusion.sh

CRUISE_FLAGS=$@

hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/logs_autoencoder_en_8dim
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/metadata/en_1400h.list

MODEL_FLAGS="--input_dim 8 --feature_dim 512 --num_blocks 12 --dropout 0 --segment_size 64 --segment_stride 32"
DIFFUSION_FLAGS="--diffusion_steps 0 --num_chunks 1 --chunk_length 2500"

fine_model=sru
coarse_model=roformer
MODEL_FLAGS="--intra_seq2seq ${fine_model} --inter_seq2seq ${coarse_model} ${MODEL_FLAGS}"

pip3 install -r recipes/unit2speech/requirements.txt

export OPENAI_LOGDIR=./logs_20230604_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10

mkdir -p $OPENAI_LOGDIR

echo "$MODEL_FLAGS $DIFFUSION_FLAGS" > $OPENAI_LOGDIR/settings

hdfs dfs -mkdir hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_ckpt/$ARNOLD_TRIAL_ID

hdfs dfs -put $OPENAI_LOGDIR hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_ckpt/$ARNOLD_TRIAL_ID

TORCHRUN recipes/unit2speech/modules/cruisemodule.py --config=recipes/unit2speech/unit2speech.yaml --model.hdfs_path=hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_ckpt/$ARNOLD_TRIAL_ID/$OPENAI_LOGDIR $CRUISE_FLAGS
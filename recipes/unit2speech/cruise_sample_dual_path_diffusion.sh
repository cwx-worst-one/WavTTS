# Start at samantha/
# sh recipes/unit2speech/cruise_sample_dual_path_diffusion.sh 15347474

TRAIN_TRIAL_ID=$@

hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/logs_autoencoder_en_8dim

export OPENAI_LOGDIR=./logs_20230604_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10

mkdir -p $OPENAI_LOGDIR

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_ckpt/$TRAIN_TRIAL_ID/logs_20230604_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10/settings $OPENAI_LOGDIR
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_ckpt/$TRAIN_TRIAL_ID/logs_20230604_sru-roformer_8in-12x512_64-32segs_aug-prompt-mel_cross-attn_mse-loss_bsz-256_rmsnorm_alleng4-10/ckpt100000 $OPENAI_LOGDIR

mkdir metadata
cp /mnt/bn/sami-linyongye/git/speech-stable-diffusion/metadata/librilight_test_clean_4-10.list metadata/

mkdir assets
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/linyongye/speech-DPD/assets/hubert_v2 assets

TORCHRUN recipes/unit2speech/modules/cruiseinfmodule.py --config=recipes/unit2speech/unit2speechinf.yaml

hdfs dfs -mkdir hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_generation/$ARNOLD_TRIAL_ID
hdfs dfs -put $OPENAI_LOGDIR/ckpt100000/samples hdfs://haruna/home/byte_arnold_lq/user/zuo.lei/unit2speech_generation/$ARNOLD_TRIAL_ID
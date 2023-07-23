wav_res_ref_text_path=$1
GPU_ID=$2

export https_proxy=http://bj-rd-proxy.byted.org:3128 http_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org

export CUDA_VISIBLE_DEVICES=$GPU_ID

### cal_wer
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/utils/
python3 cal_wer_cn.py $wav_res_ref_text_path $wav_res_ref_text_path.wer
tail -n 1 $wav_res_ref_text_path.wer
echo
cd -
python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/analyse/get_IDS_by_wer.py $wav_res_ref_text_path.wer >> $wav_res_ref_text_path.wer
tail -n 1 $wav_res_ref_text_path.wer

### cal_asv
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/UniSpeech/downstreams/speaker_verification
# bash lazy_install.sh
python3 verification_pair_list_v2.py $wav_res_ref_text_path --model_name wavlm_large --checkpoint wavlm_large_finetune.pth --scores $wav_res_ref_text_path.asv --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1
tail -n 1 $wav_res_ref_text_path.asv
echo
cd -

# ### cal naturalness
# cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/NISQA
# bash run.sh $wav_res_dir
# cd -

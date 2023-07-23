wav_res_ref_text_path=$1
GPU_ID=$2

export https_proxy=http://bj-rd-proxy.byted.org:3128 http_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org
export CUDA_VISIBLE_DEVICES=$GPU_ID

# cal_wer
# python3 recipes/valle/eval/cal_wer_online.py $wav_res_ref_text_path $wav_res_ref_text_path.wer

python3 recipes/valle/eval/cal_wer.py $wav_res_ref_text_path $wav_res_ref_text_path.wer
### cal_asv
bash recipes/valle/eval/cal_asv.sh $wav_res_ref_text_path $wav_res_ref_text_path.asv

# wer_score=`cat $wav_res_ref_text_path.wer | grep "avg wer score"`
# echo $wer_score

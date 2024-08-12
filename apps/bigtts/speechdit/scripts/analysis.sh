# evaluation.
meta_lst=$1
output_dir=$2
lang=$3

# 需修改;
bigtts_eval_dir=/mnt/bn/cjw-lq-1/project/scripts/bigtts-eval

cd $bigtts_eval_dir
# pip install -r requirements.txt

python3 utils/get_wav_res_ref_text.py $meta_lst $output_dir $output_dir/wav_res_ref_text

echo $output_dir/wav_res_ref_text

bash eval/cal_wer.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.wer bigasr $lang

bash eval/cal_asv.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.asv

#bash eval/cal_dur.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.dur $lang

#bash eval/cal_pitch.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.pitch

cd -


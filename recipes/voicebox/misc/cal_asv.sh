# evaluation.
meta_lst=/mnt/bn/jcong5/workspace3/bigtts_testset/icl_testset_fighting/en_meta_better_studio.lst.106
# output_dir=/mnt/bn/jcong5/workspace3/voicebox-original/voicebox_m/icl-fighting/icl-fighting-0.7
output_dir=/mnt/bn/jcong5/logs/baseline/baseline_dropout0.1/step=85000-kl_loss=0.65/vip-demo

bigtts_eval_dir=/mnt/bn/jcong5/workspace2/bigtts-eval/
cd $bigtts_eval_dir
python3 utils/get_wav_res_ref_text.py $meta_lst $output_dir $output_dir/wav_res_ref_text
echo $output_dir/wav_res_ref_text
# bash eval/cal_wer.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.wer internal en
bash eval/cal_asv.sh ${output_dir}/wav_res_ref_text ${output_dir}/wav_res_ref_text.asv
tail ${output_dir}/wav_res_ref_text.asv

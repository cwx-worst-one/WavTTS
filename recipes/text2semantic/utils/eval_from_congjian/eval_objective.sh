coarse_dir=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/ar/fp16_lr5e-5_batch15000
fine_dir=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/nar/fp16_lr5e-5_batch15000
metalst=/mnt/bn/jcong5/data/bigtts/asr_sub_meta.lst

export https_proxy=http://bj-rd-proxy.byted.org:3128 http_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org

timestamp=$(date +%s)
for item in asr_sub emotion_sub podcast_sub;
do
    # metalst=/mnt/bn/jcong5/data/bigtts/${item}_sub_meta.lst
    # out_dir=$coarse_dir/${item}_sub/
    wav_res_ref_text_path=/mnt/bn/jcong5/logs/sami_ai_models/valle_phoneinput_1400_librilight_90-1492--1-8/ar/fp16_lr5e-5_batch15000/${item}/coarse_1485k_epoch13_fine_1505k_epoch14/genwav_promptwav_text.lst
    # bash recipes/valle/custom_scripts/eval_raw.sh $coarse_dir $fine_dir $metalst $out_dir 

done
# cal_wer
python3 recipes/valle/eval/cal_wer_online.py $wav_res_ref_text_path $wav_res_ref_text_path.wer

# python3 recipes/valle/eval/cal_wer.py $wav_res_ref_text_path $wav_res_ref_text_path.wer
### cal_asv
bash recipes/valle/eval/cal_asv.sh $wav_res_ref_text_path $wav_res_ref_text_path.asv
done

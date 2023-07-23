part1_name=sft_en_TTS_all-libritts_clean_460
part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/will_librispeech_test_clean_coarse_107k_epoch22_fine_1505k_epoch14_minlen_noprompt/wav_res_ref_text.wer
out_dir=$PWD/20230622/sft_noprompt

[ ! -d $out_dir ] && mkdir -p $out_dir

cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/analyse
python3 analyse_wer.py $part1_name $part1_file $out_dir &> $out_dir/wer_compare.txt
cd -
echo $out_dir/wer_compare.txt
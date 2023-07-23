# part1_name=sft_en_TTS_all-libritts_clean_460
# part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/librispeech_test_clean_norm_coarse_107k_epoch22_fine_1505k_epoch14_minlen_retry5/wav_res_ref_text.wer
# part2_name=sft_en_TTS
# part2_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000_sft_en_TTS/checkpoints/test_librispeech_test_clean_epoch43/coarse_108k_epoch43_fine_1505k_epoch14/wav_res_ref_text.wer
# out_dir=$PWD/20230620/sft_compare_B_A

# [ ! -d $out_dir ] && mkdir $out_dir
# python3 analyse_wer_listen.py $part2_name $part2_file $part1_name $part1_file $out_dir &> $out_dir/wer_compare.txt
# echo $out_dir/wer_compare.txt


# part1_name=sft_en_TTS_all-libritts_clean_460
# part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/will_librispeech_test_clean_coarse_107k_epoch22_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
# part2_name=sft_en_TTS
# part2_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000_sft_en_TTS/checkpoints/test_spkwill_epoch43_testlibrispeech_test_clean_thres0.9/coarse_108k_epoch43_fine_1505k_epoch14/wav_res_ref_text.wer
# out_dir=$PWD/20230620/sft_compare_A_B_will

# [ ! -d $out_dir ] && mkdir $out_dir
# python3 analyse_wer_listen.py $part1_name $part1_file $part2_name $part2_file $out_dir &> $out_dir/wer_compare.txt
# echo $out_dir/wer_compare.txt




# part1_name=sft_en_TTS_all-libritts_clean_460
# part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/will_librispeech_test_clean_coarse_107k_epoch22_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
# part2_name=sft_en_TTS_all
# part2_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all/wav_infer/will_librispeech_test_clean_coarse_107k_epoch32_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
# out_dir=$PWD/20230620/sft_compare_B_A_libritts_will

# [ ! -d $out_dir ] && mkdir $out_dir
# python3 analyse_wer_listen.py $part2_name $part2_file $part1_name $part1_file $out_dir &> $out_dir/wer_compare.txt
# echo $out_dir/wer_compare.txt


# part1_name=sft_en_TTS_all-libritts_clean_460
# part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/11labs_Adam_075_librispeech_test_clean_coarse_107k_epoch22_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
# part2_name=sft_en_TTS
# part2_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000_sft_en_TTS/wav_infer/11labs_Adam_075_librispeech_test_clean_coarse_108k_epoch43_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
# out_dir=$PWD/20230620/sft_compare_A_B_Adam_075

# [ ! -d $out_dir ] && mkdir $out_dir
# python3 analyse_wer_listen.py $part1_name $part1_file $part2_name $part2_file $out_dir &> $out_dir/wer_compare.txt
# echo $out_dir/wer_compare.txt



part1_name=sft_en_TTS_all-libritts_clean_460
part1_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/youtube_librispeech_test_clean_coarse_107k_epoch22_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
part2_name=sft_en_TTS
part2_file=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/bark/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch140000_sft_en_TTS/wav_infer/youtube_librispeech_test_clean_coarse_108k_epoch43_fine_1505k_epoch14_minlen/wav_res_ref_text.wer
out_dir=$PWD/20230620/sft_compare_A_B_youtube001

[ ! -d $out_dir ] && mkdir $out_dir
python3 analyse_wer_listen.py $part1_name $part1_file $part2_name $part2_file $out_dir &> $out_dir/wer_compare.txt
echo $out_dir/wer_compare.txt
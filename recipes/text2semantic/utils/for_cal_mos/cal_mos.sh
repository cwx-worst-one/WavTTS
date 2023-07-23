### mosnet
cp -r /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/for_mosnet/bytesep_data /home/tiger/

out_dir=/mnt/bd/huangzhiying-lq-valle-volume14/data/samantha/mosnet/libri_light/mosnet_by_shards
mkdir -p $out_dir
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/voice_cloning_pipeline_v2
for split in {0..31};do
    {
    cuda_id=`expr $split % 8`
    echo $cuda_id
    wav_list_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list.left.$split
    python3 code/external_tools/AcousticFrontends/test/test_quality_voice_list_v2.py -r $wav_list_path -d cuda:$cuda_id -o $out_dir
    }&
done
cd -
wait


cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/voice_cloning_pipeline_v2
for x in {1..5};do
python3 code/external_tools/AcousticFrontends/test/test_quality_voice_list.py -r /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/librispeech_test_clean_norm_coarse_107k_epoch22_fine_1505k_epoch14_minlen_retry$x.wav.list \
                                                                            -d cuda:$x \
                                                                            -f /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/librispeech_test_clean_norm_coarse_107k_epoch22_fine_1505k_epoch14_minlen_retry$x.wav.list.mos &
done
cd -


cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/voice_cloning_pipeline_v2
python3 code/external_tools/AcousticFrontends/test/test_quality_voice_list.py -r /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/librispeech_test_clean_norm_coarse_107k_epoch22_fine_1505k_epoch14_minlen.wav.list \
                                                                            -d cuda:0 \
                                                                            -f /mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha/valle/valle_phoneinput_1400_librilight_90-1492_mosnet2.8-A100-80GB-3-8/ar/fp16_lr4e-4_batch100000_sft_en_TTS_all-libritts_clean_460/wav_infer/librispeech_test_clean_norm_coarse_107k_epoch22_fine_1505k_epoch14_minlen.wav.list.mos &
cd -

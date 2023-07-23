### mosnet
cp -r /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/for_mosnet/bytesep_data /home/tiger/

out_dir=/mnt/bd/huangzhiying-lq-valle-volume14/data/samantha/mosnet/libri_light/mosnet_by_shards
mkdir -p $out_dir
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/voice_cloning_pipeline_v2
for split in {96..127};do
    {
    cuda_id=`expr $split % 8`
    echo $cuda_id
    wav_list_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list.left.$split
    python3 code/external_tools/AcousticFrontends/test/test_quality_voice_list_v2.py -r $wav_list_path -d cuda:$cuda_id -o $out_dir
    }&
done
cd -
wait

# cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/voice_cloning_pipeline_v2
# python3 code/external_tools/AcousticFrontends/test/test_quality_voice_list.py -r /mnt/bn/huangzhiying-nas-speech2speech-volume1/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list.0 \
#                                                                             -d cpu \
#                                                                             -f /mnt/bd/huangzhiying-lq-valle-volume13/data/samantha/mosnet/libri_light/mos.tsv.0
# cd -

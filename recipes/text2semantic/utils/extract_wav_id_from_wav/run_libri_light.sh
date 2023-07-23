# hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/code/samantha/recipes/valle/utils/extract_wav_id_from_wav/run_libri_light.sh

### 50hz_doubleG
for x in {0..7};do
    python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/extract_wav_id_from_wav/norm_trim_ts.py \
                            hdfs://haruna/home/byte_speech_sv/user/litang/vqgan/2023-06-19_x480_1024_12book_speech_doubleG \
                            /mnt/bd/huangzhiying-lq-librilight-ss50hz-part1/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list.$x \
                            $x \
                            /mnt/bd/huangzhiying-lq-librilight-ss50hz-part1/data/bytegen/soundstream/libri_light/wav_ids_norm_trim_50hz/ \
                            /mnt/bd/huangzhiying-lq-librilight-ss50hz-part1/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav2split.$x &
done

# ### 50hz_单G
# for x in 0;do
#     python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/extract_wav_id_from_wav/norm_trim_ts.py \
#                             hdfs://haruna/home/byte_speech_sv/user/litang/vqgan/2023-06-02_soundstream_x480_12book_export \
#                             /mnt/bd/huangzhiying-lq-valle-volume11/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav.list.$x.head100 \
#                             $x \
#                             /mnt/bd/huangzhiying-lq-valle-volume11/data/bytegen/soundstream/libri_light/wav_ids_norm_trim_50hz_singleG/ \
#                             /mnt/bd/huangzhiying-lq-valle-volume11/data/original_data/ASR/en/libri_light/data/wav_uniq_vad_asr_by_shards/wav2split.$x.head100
# done

meta_dir=hdfs://harunava/home/byte_speech_sv/user/huangzhiying.92/data/samantha/en_ASR
wds_dir=hdfs://harunava/home/byte_speech_sv/user/huangzhiying.92/data/samantha/en_ASR
out_dir=/mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/en_ASR

# meta_dir=hdfs://harunava/home/byte_speech_sv/user/huangzhiying.92/data/samantha/en_TTS
# wds_dir=hdfs://harunava/home/byte_speech_sv/user/huangzhiying.92/data/samantha/en_TTS
# out_dir=/mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/en_TTS

# meta_dir=hdfs://harunava/home/byte_speech_sv/user/huangzhiying.92/data/bytegen/valle/libri_light_dingchen/json/0523_merge_add_tacolab_mosnet
# wds_dir=hdfs://harunava/home/byte_speech_sv/audioGPT/data/speech/librilight/v2
# out_dir=/mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/libri_light

# meta_dir=hdfs://harunava/home/byte_speech_sv/data/resso_podcast/ground/part-00004
# wds_dir=hdfs://harunava/home/byte_speech_sv/dingchen.2101/exp/AudioGPT_data/resso_podcast_v2/part-00004
# out_dir=/mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/resso_podcast/part-00004

meta_dir=hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/data/samantha/zh_TTS
wds_dir=hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/data/samantha/zh_TTS
out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/zh_TTS

# meta_dir=hdfs://haruna/home/byte_speech_sv/user/dingchen.2101/AudioGPT_data/fanqie_v1_res_v1/part-00022
# wds_dir=hdfs://haruna/home/byte_speech_sv/user/dingchen.2101/AudioGPT_data/fanqie_v1/part-00022
# out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/fanqie_v1/part-00022

[ ! -d $out_dir ] && mkdir -p $out_dir

# ### get wds.lst and wds2meta.json
# cd /mnt/bn/huangzhiying-nas-volume1/code/samantha/recipes/valle/preprocess
# python3 gen_meta_lst.py --meta $meta_dir --wds $wds_dir --outputs $out_dir --jobs 50 #--chunks
# cd -

# ### split train valid
# cd /mnt/bn/huangzhiying-nas-volume1/code/samantha/recipes/valle/utils
# python3 split_train_test.py $out_dir/wds.lst $out_dir/wds.lst.shuf $out_dir/valid_wds.lst $out_dir/train_wds.lst
# cd -

### merge info
cd /mnt/bn/huangzhiying-nas-volume1/code/samantha/recipes/valle/utils
python3 merge_wdslst_idx2meta.py /mnt/bn/huangzhiying-nas-volume1/data/samantha/valle \
                                "en_ASR,en_TTS,libri_light,resso_podcast/part-00000,resso_podcast/part-00001,resso_podcast/part-00002,resso_podcast/part-00003,resso_podcast/part-00004" \
                                /mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/1400_libri_light_rp00000-4
cd -

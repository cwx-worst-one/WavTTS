data_dir=/mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/en_zh_TTS

# ### get wav_id
# wav_list_path=$data_dir/wav.list
# utt2split_path=$data_dir/wav2split
# cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bytegen/models/soundstream
# exp_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/bytegen/soundstream/2023-01-17_causal_x300_1024_6book_doubleG
# python3 extract_feature_list_by_wav2split.py --config $exp_dir/config_libri_causal.yaml \
#                         --ckpt_path $exp_dir/checkpoints/latest_ckpt.pyt \
#                         --wav_list_path $wav_list_path \
#                         --out_dir_prefix $data_dir \
#                         --world_size 8 \
#                         --utt2split_path $utt2split_path
# cd -


### get tacolab
# cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils
# cat $data_dir/spks | while read spk;do
#     if [ ! -d $data_dir/original_data/$spk/tacofrontend ]; then
#         cat $data_dir/original_data/$spk/text | awk '{print $1}' > $data_dir/original_data/$spk/temp
#         sed -i 's#$# temp#g' $data_dir/original_data/$spk/temp

#         num_job=16
#         num=`wc -l $data_dir/original_data/$spk/temp | awk -F' ' '{print $1}'`
#         num_per_thread=`expr $num / $num_job + 1`

#         split -l $num_per_thread -d -a2 $data_dir/original_data/$spk/temp $data_dir/original_data/$spk/temp.
#         split -l $num_per_thread -d -a2 $data_dir/original_data/$spk/text $data_dir/original_data/$spk/text.

#         for x in {00..15};do
#             python3 get_tacolab_en_by_split.py $data_dir/original_data/$spk/text.$x $data_dir/original_data/$spk/temp.$x $data_dir/original_data/$spk/tacofrontend &
#         done
#         wait

#         rm -f $data_dir/original_data/$spk/temp* $data_dir/original_data/$spk/text.*
#         mv $data_dir/original_data/$spk/tacofrontend/temp/*.lab $data_dir/original_data/$spk/tacofrontend/
#         rm -rf $data_dir/original_data/$spk/tacofrontend/temp
#     fi
# done
# wait


# ### get text_id
# ii=0
# job_num=64
# cat spks | while read line;do
#     {
#     mkdir $data_dir/data/$line/text_id -p
#     cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils
#     python3 get_text_id.py $data_dir/original_data/$line/tacofrontend $data_dir/data/$line/text_id /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/datasets/dict/metaid_to_textid.json
#     cd -
#     } &
#     ((ii++))
#     if [ $((ii%job_num)) -eq 0 ]; then
#         wait
#     fi
# done
# wait


# ### get metalen
# cd $data_dir
# paste -d "|" wav_id.list text_id.list > meta_list_all.txt
# num_job=64
# num=`wc -l meta_list_all.txt | awk -F' ' '{print $1}'`
# num_per_thread=`expr $num / $num_job + 1`
# split -l $num_per_thread -d -a2 meta_list_all.txt meta_list_all.txt.
# cd -

# cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils
# for x in {00..63};do
#     python3 get_utt2metalen.py $data_dir/meta_list_all.txt.$x $data_dir/utt2metalen_all.$x &
# done
# wait
# cd -

# for x in {00..63};do
#     cat $data_dir/utt2metalen_all.$x >> $data_dir/utt2metalen_all
# done
# rm -f $data_dir/utt2metalen_all.* $data_dir/meta_list_all.txt.*


# ### get meta_list_all.txt.add_metalen
# cat $data_dir/utt2metalen_all | awk '{print $2}' > $data_dir/metalen_all
# paste -d "|" $data_dir/meta_list_all.txt $data_dir/metalen_all > $data_dir/meta_list_all.txt.add_metalen


### filter 
cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils
# python3 split_train_test.py $data_dir/{meta_list_all.txt.add_metalen,meta_list_all.txt.add_metalen.shuf,meta_list_all.txt.add_metalen.shuf.valid,meta_list_all.txt.add_metalen.shuf.train} 0.001
# python3 filter_meta_list_by_len.py $data_dir/meta_list_all.txt.add_metalen.shuf.train $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_1492 90 1492
# python3 filter_meta_list_by_len.py $data_dir/meta_list_all.txt.add_metalen.shuf.valid $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_1492 90 1492

# python3 filter_meta_list_by_len.py $data_dir/meta_list_all.txt.add_metalen.shuf.train $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_2040 90 2040
# python3 filter_meta_list_by_len.py $data_dir/meta_list_all.txt.add_metalen.shuf.valid $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_2040 90 2040

sed '/libritts/d' $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_1492 > $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_1492.rm_libritts
sed '/libritts/d' $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_1492 > $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_1492.rm_libritts

sed '/libritts/d' $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_2040 > $data_dir/meta_list_all.txt.add_metalen.shuf.train.fiter_90_2040.rm_libritts
sed '/libritts/d' $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_2040 > $data_dir/meta_list_all.txt.add_metalen.shuf.valid.fiter_90_2040.rm_libritts
cd -


# ii=0
# job_num=32
# cat /mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/en_zh_TTS/spks | while read spk;do
#     {
#     python3 get_total_dur_with_wavdir.py /mnt/bd/huangzhiying-lq-valle-volume9/data/bytegen/valle/en_zh_TTS/original_data/$spk/{wav_24k,dur.txt}
#     } &
#     ((ii++))
#     if [ $((ii%job_num)) -eq 0 ]; then
#         wait
#     fi
# done

# ii=0
# job_num=32
# cat spks | while read spk;do
#     {
#     ls original_data/$spk/wav_24k/ | shuf | head -n 2 | while read line;do
#         cp original_data/$spk/wav_24k/$line for_listen/${spk}_$line
#     done
#     } &
#     ((ii++))
#     if [ $((ii%job_num)) -eq 0 ]; then
#         wait
#     fi
# done

# cat spks | while read line;do
#     python3 check_num.py original_data/$line/tacofrontend original_data/$line/wav_24k original_data/$line/text
# done
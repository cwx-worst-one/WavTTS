### gen wds_list and wds2meta.json
meta_dir=hdfs://harunava/home/byte_speech_sv/data/resso_podcast/ground/part-00004
wds_dir=hdfs://harunava/home/byte_speech_sv/dingchen.2101/exp/AudioGPT_data/resso_podcast_v2/part-00004
out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/resso_podcast/part-00004
[ ! -d $out_dir ] && mkdir -p $out_dir

### get wds.lst and wds2meta.json
python3 gen_meta_lst.py --meta $meta_dir --wds $wds_dir --outputs $out_dir --jobs 50 --chunks

### extract wav_id and get webdataset
local_rank=0
python3 get_wds_wav_id.py $out_dir/wds.lst $out_dir/wds2meta.json $out_dir/wav_id $local_rank
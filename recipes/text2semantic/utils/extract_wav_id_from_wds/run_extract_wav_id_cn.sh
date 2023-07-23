longform=$1
# env
pip3 install -r requirements.txt

# dir
# meta_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/audiogpt/fanqie_v1/part-00003
meta_dir=hdfs://haruna/home/byte_speech_sv/user/dingchen.2101/AudioGPT_data/fanqie_v1_res_v1/part-00003
wds_dir=hdfs://haruna/home/byte_speech_sv/user/dingchen.2101/AudioGPT_data/fanqie_v1/part-00003
out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/fanqie_v1/part-00003
# meta_dir=/mnt/bn/jeffus/data/valle/longform/rp-00003 
# wds_dir=hdfs://harunava/home/byte_speech_sv/dingchen.2101/exp/AudioGPT_data/resso_podcast_v2/part-00003
# out_dir=/mnt/bn/jeffus/data/valle/extrac_feat/rp-00005
[ ! -d $out_dir ] && mkdir -p $out_dir

# get wds.lst and wds2meta.json
python3 gen_meta_lst.py --meta $meta_dir --wds $wds_dir --outputs $out_dir --jobs 50 --chunks

# extract wav_id and get webdataset
local_rank=0
if [[ $longform == "" ]]; then
    echo "short form"
    python3 get_wds_wav_id_cn.py \
        --wds $out_dir/wds.lst \
        --meta $out_dir/wds2meta.json \
        --outputs $out_dir/wav_id \
        --rank $local_rank
else
    echo "long form"
    python3 get_wds_wav_id_cn.py  \
        --wds $out_dir/wds.lst \
        --meta $out_dir/wds2meta.json \
        --outputs $out_dir/wav_id \
        --rank $local_rank \
        --longform
fi
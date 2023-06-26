datasetname=$1
python gen_wds_lst.py --wds  "hdfs://harunava/home/byte_speech_sv/yanlin.chn/exp/BigTTS/librilight/v2/soundstream/longform/*/wav_id/*.tar" --outputs /mnt/bn/jeffus/data/metas/valle/${datasetname}/ll.lst
python gen_wds_lst.py --wds  "hdfs://harunava/home/byte_speech_sv/yanlin.chn/exp/BigTTS/resso_podcast/soundstream/longform/*/wav_id/*.tar" --outputs /mnt/bn/jeffus/data/metas/valle/${datasetname}/rp.lst
cat /mnt/bn/jeffus/data/metas/valle/${datasetname}/rp.lst /mnt/bn/jeffus/data/metas/valle/${datasetname}/ll.lst > /mnt/bn/jeffus/data/metas/valle/${datasetname}/wds.lst
wc -l  /mnt/bn/jeffus/data/metas/valle/${datasetname}/*.lst
echo /mnt/bn/jeffus/data/metas/valle/${datasetname}/wds.lst


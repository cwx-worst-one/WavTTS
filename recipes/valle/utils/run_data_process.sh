### filter with mosnet
in_meta_list_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/libri_light-1400h/meta_list_all.txt.filter_90_1492.train.add_metalen
utt2mos_path=/mnt/bd/huangzhiying-lq-valle-volume14/data/samantha/mosnet/libri_light/utt2mos
out_meta_list_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/bytegen/valle/libri_light-1400h/meta_list_all.txt.filter_90_1492.train.add_metalen.mosnet2.8

cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/
python3 for_mosnet/filter_meta_list_with_mos.py $in_meta_list_path $utt2mos_path $out_meta_list_path
cd -
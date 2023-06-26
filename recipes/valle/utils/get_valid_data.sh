for x in part-00001 part-00002;do
    python3 get_valid_data.py --wds2meta_path /mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/resso_podcast/$x/wds2meta.json &>/mnt/bn/huangzhiying-nas-volume1/data/samantha/valle/resso_podcast/$x/valid.log &
done
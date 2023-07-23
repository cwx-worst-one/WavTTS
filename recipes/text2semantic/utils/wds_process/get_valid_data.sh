for x in en_TTS en_ASR libri_light;do
    python3 get_valid_data.py --wds2meta_path /mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/$x/wds2meta.json &>/mnt/bn/huangzhiying-nas-speech2speech-volume1/data/samantha/valle/$x/valid.log &
done
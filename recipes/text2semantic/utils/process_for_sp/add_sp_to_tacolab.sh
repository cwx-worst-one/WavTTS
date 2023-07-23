job_num=64
ii=0
for x in 1955;do
    # {
    python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/process_for_sp/add_sp_to_tacolab.py \
        /mnt/bd/huangzhiying-lq-valle-volume5/data/bytegen/valle/libri_light/taco_labs/$x \
        /mnt/bd/huangzhiying-lq-valle-volume11/data/bytegen/valle/libri_light/meta_info/meta.$x.json \
        /mnt/bd/huangzhiying-lq-valle-volume11/data/bytegen/valle/libri_light/taco_labs_add_sp/$x/
    # } &
    # ((ii++))
    # if [ $((ii%job_num)) -eq 0 ]; then
    #     wait
    # fi
done

# uttname=`grep utt_name temp | awk '{print $2}'`
# grep "$uttname	" /mnt/bn/qq-nas-lq-a/from_zhiying/text_addpunc
# grep ".wav" temp

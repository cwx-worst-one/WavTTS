pip3 install bytedeuler --index-url=https://bytedpypi.byted.org/simple/

ii=0
job_num=64
for x in {0000..2540};do
    [ -f /mnt/bd/huangzhiying-lq-valle-volume10/data/bytegen/valle/libri_light/meta_info/meta.$x.json ] && continue
    echo $x
    {
    wav2taco=/mnt/bd/huangzhiying-lq-valle-volume10/data/bytegen/valle/libri_light/by_split/wav2taco2text.$x
    meta=/mnt/bd/huangzhiying-lq-valle-volume10/data/bytegen/valle/libri_light/meta_info/meta.$x.json
    python3 /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/process_for_sp/gen_meta_info.py $wav2taco $meta
    } &
    ((ii++))
    if [ $((ii%job_num)) -eq 0 ]; then
        wait
    fi
done
wait
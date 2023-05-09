#! /usr/bin/env bash
for x in run_libri_light-WenetSpeech-1400h-1000h.sh;do
    hdfs dfs -mkdir -p hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/code/samantha/recipes/speartts/custom_scripts/
    for file in `ls /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/speartts/custom_scripts/$x`;do
        echo ${file}
        name=`basename $file`
        echo "hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/code/samantha/recipes/speartts/custom_scripts/$name"
        hdfs dfs -put -f ${file} hdfs://haruna/home/byte_speech_sv/user/huangzhiying.92/code/samantha/recipes/speartts/custom_scripts/
    done
done

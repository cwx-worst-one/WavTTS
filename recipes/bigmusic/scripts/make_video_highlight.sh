#!/bin/bash

if [ $# != 2 ]; then
    echo "Usage: $0 <file_list> <output_dir>"
    exit 1
fi
set -ex
file_list=$1
output_dir=$2

mkdir -p $output_dir
rm -f $output_dir/*

declare -a arr=("green" "blue" "brown")
accum=1
readarray fnames < $file_list
for fname in "${fnames[@]}"; do
    basename=`echo "$fname" | sed -e "s/.*\///" -e "s/.generated.wav//"`
    meta=`echo "$fname" | sed -e "s/.generated.wav/.metadata.json/g"`

    text="$output_dir/${basename}.txt"
    video="$output_dir/${basename}.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text_highlight.py $meta $text --title $basename
    color=${arr[$((${accum}%3))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $fname -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${text}" $video
    #echo "$video" | sed -e 's/^/file /' >>$output_dir/fl.txt
    rm $text
done

#ffmpeg -f concat -safe 0 -i $output_dir/fl.txt -c copy $output_dir/output.mp4
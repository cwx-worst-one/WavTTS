#!/bin/bash

if [ $# != 3 ]; then
    echo "Usage: $0 <file_list> <input_dir> <output_dir>"
    exit 1
fi
set -ex
file_list=$1
input_dir=$2
output_dir=$3

mkdir -p $output_dir
rm -f $output_dir/*

declare -a arr=("green" "blue" "brown")
accum=1
readarray fnames < $file_list
for fname in "${fnames[@]}"; do
    basename=`echo "$fname" | sed -e "s/.generated.wav//"`

    text="$output_dir/${basename}.txt"
    video="$output_dir/${basename}.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text_highlight.py $input_dir/${basename}.metadata.json $text
    color=${arr[$((${accum}%3))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $input_dir/${basename}.generated.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${text}" $video
    echo "$video" | sed -e 's/^/file /' >>$output_dir/fl.txt
done

ffmpeg -f concat -safe 0 -i $output_dir/fl.txt -c copy $output_dir/output.mp4
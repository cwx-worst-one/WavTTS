#!/bin/bash

if [ $# != 4 ]; then
    echo "Usage: $0 <file_list> <a> <b> <output_dir>"
    exit 1
fi
set -ex
file_list=$1
a_dir=$2
b_dir=$3
output_dir=$4
a_str="MCC"
b_str="MCC + Shutterstock SFT"

mkdir -p $output_dir
rm -f $output_dir/*

declare -a arr=("green" "blue")
accum=1
readarray fnames < $file_list
for fname in "${fnames[@]}"; do
    category=`echo "$fname" | sed -e "s/\/.*//"`
    mkdir -p $output_dir/$category
    basename=`echo "$fname" | sed -e "s/.*\///" -e "s/.wav//"`

    a_text="$output_dir/$category/${basename}_a.txt"
    a_video="$output_dir/$category/${basename}_a.mp4"
    python3 recipes/musiclm/scripts/make_text.py "$accum : $a_str" $b_dir/$category/${basename}.txt $a_text
    color=${arr[$((${accum}%2))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $a_dir/$category/${basename}.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${a_text}" $a_video
    echo "$a_video" | sed -e 's/^/file /' >>$output_dir/fl.txt

    b_text="$output_dir/$category/${basename}_b.txt"
    b_video="$output_dir/$category/${basename}_b.mp4"
    python3 recipes/musiclm/scripts/make_text.py "$accum : $b_str" $b_dir/$category/${basename}.txt $b_text
    color=${arr[$((${accum}%2))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $b_dir/$category/${basename}.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${b_text}" $b_video
    echo "$b_video" | sed -e 's/^/file /' >>$output_dir/fl.txt
done

ffmpeg -f concat -safe 0 -i $output_dir/fl.txt -c copy $output_dir/output.mp4
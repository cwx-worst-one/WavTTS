#!/bin/bash

if [ $# != 10 ]; then
    echo "Usage: $0 <a_str> <b_str> <c_str> <d_str> <file_list> <a> <b> <c> <d> <output_dir>"
    exit 1
fi
set -ex
a_str=$1
b_str=$2
c_str=$3
d_str=$4
file_list=$5
a_dir=$6
b_dir=$7
c_dir=$8
d_dir=$9
output_dir=${10}

mkdir -p $output_dir
rm -f $output_dir/*

declare -a arr=("green" "blue" "brown" "purple")
accum=0
readarray fnames < $file_list
for fname in "${fnames[@]}"; do
    basename=`echo "$fname" | sed -e "s/.generated.wav//"`
    basename1=`echo "$basename" | sed -e "s/\//_/g"`

    a_text="${output_dir}/${basename1}_a.txt"
    a_video="$output_dir/${basename1}_a.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text.py "$a_str" $a_dir/${basename}.metadata.json $a_text
    color=${arr[$((${accum}%4))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $a_dir/${basename}.generated.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${a_text}" $a_video
    echo "$a_video" | sed -e 's/^/file /' >>$output_dir/fl.txt

    b_text="${output_dir}/${basename1}_b.txt"
    b_video="$output_dir/${basename1}_b.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text.py "$b_str" $b_dir/${basename}.metadata.json $b_text
    color=${arr[$((${accum}%4))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $b_dir/${basename}.generated.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${b_text}" $b_video
    echo "$b_video" | sed -e 's/^/file /' >>$output_dir/fl.txt

    c_text="${output_dir}/${basename1}_c.txt"
    c_video="$output_dir/${basename1}_c.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text.py "$c_str" $c_dir/${basename}.metadata.json $c_text
    color=${arr[$((${accum}%4))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $c_dir/${basename}.generated.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${c_text}" $c_video
    echo "$c_video" | sed -e 's/^/file /' >>$output_dir/fl.txt

    d_text="${output_dir}/${basename1}_d.txt"
    d_video="$output_dir/${basename1}_d.mp4"
    python3 /root/workspace/samantha/recipes/bigmusic/scripts/make_text.py "$d_str" $d_dir/${basename}.metadata.json $d_text
    color=${arr[$((${accum}%4))]}
    ((accum += 1))
    ffmpeg -f lavfi -i color=c=$color:s=800x800:d=0.5 -i $d_dir/${basename}.generated.wav -c:a aac  -vf "drawtext=fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:textfile=${d_text}" $d_video
    echo "$d_video" | sed -e 's/^/file /' >>$output_dir/fl.txt
done

ffmpeg -f concat -safe 0 -i $output_dir/fl.txt -c copy $output_dir/output.mp4
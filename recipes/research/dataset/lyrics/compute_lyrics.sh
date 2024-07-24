#!/bin/bash
export NODE_RANK=${ARNOLD_ID}
export NNODES=${ARNOLD_WORKER_NUM}
export NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

bash recipes/research/bootstrap.sh
pip3 install faster-whisper

let N_RANKS=$NPROC_PER_NODE-1

for rank in $(seq 0 $N_RANKS); do
    bb_file=billboard_hot_200-v2_$rank.txt
    bb_exists=$bb_file.exists
    
    # generate list of files:
    hdfs dfs -ls hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_normalised_-16LUFS_mp3 | awk '{print $8}' > $bb_file
    sed -i '1d' $bb_file

    hdfs dfs -ls hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/js/data/music/billboard_hot_200-v2_lyrics_wordlevel | awk '{print $8}' > $bb_exists
    sed -i '1d' $bb_exists

    python3 recipes/research/dataset/lyrics/compute_lyrics.py --rank $rank --index $bb_file --existing_files $bb_exists &
done

wait

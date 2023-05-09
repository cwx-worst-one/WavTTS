#!/bin/bash

# cd /opt/tiger/samantha/recipes/audio_diffusion/scripts/retrieval_scripts

if [ $# != 3 ]; then
	echo "Usage: $0 <data_indices> <data_prefix> <output_prefix>"
	echo -e "\nExamples:"
	echo -e "\tRun on a range of indices: $0 \"\`seq 1 8\`\""
	echo -e "\tRun on specified indices: $0 \"1 3 5 7\""
	echo -e "\nNote that this script is meant to be run with launch, e.g.:"
	echo -e "\tlaunch --cpu 16 --memory 32 --gpu 8 -- bash $0 \"\`seq 1 8\`\""
	echo -e "\t(make sure that the number of GPUs matches the number of indices)"
	exit 0
fi
set -e
data_indices=$1
data_prefix=$2
output_prefix=$3

for i in $data_indices; do
	echo "-- Launching job for data index $i --"
	./dump_one_mulan.sh $i $data_prefix $output_prefix &
	sleep 0.5
done
wait

#!/bin/bash -ex

cd $(dirname $0)/../../
echo "work dir: $(pwd)"

# Do something before lauching the main program
# e.g. Download some data, install some extra packages, etc.
pip3 install recipes/soundstream/torch-museval
pip3 install -q -r recipes/soundstream/requirements.txt

# Download gpt3 expansion
# hdfs_base="hdfs://harunava/home/byte_speech_sv/mulan"

# if [ ! -d "assets/" ]
# then
#     echo "Download ecals gpt3 expansion"
#     mkdir -p assets
#     hdfs dfs -get $hdfs_base/assets
# else
#     echo "Ecals gpt3 expansion exists, skip download"
# fi


sh launch.sh $@
